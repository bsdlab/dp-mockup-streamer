# Stream from files defined in the config/streaming.toml

import threading  # necessary to make it accessible via api
import time
import tomllib
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import mne
import numpy as np
import pylsl
import pyxdf
from dareplane_utils.general.time import sleep_s
from fire import Fire

from mockup_streamer.utils.logging import logger


class NoFilesFound(FileExistsError):
    pass


class EndOfDataError(KeyError):
    pass


class MockupStream:
    """A mockup streamer class to represent data from one source
    (file or random). Each such stream can have up to one markers stream
    associated which will be streamed in parallel with a separate name

    Attributes
    ----------
    name : str
        Name of the stream
    sfreq : float
        Target sampling frequency
    info : pylsl.StreamInfo
        pylsl StreamInfo object used to initialize the pylsl StreamOutlet
    outlet : pylsl.StreamOutlet
        pylsl StreamOutlet object data is pushed to
    buffer : np.ndarray
        The pre buffered data to be streamed from
    buffer_i : int
        index of the current position in the data buffer
    n_pushed : int
        number of samples pushed
    t_start_s : float
        timestamp of the start of the stream, required samples will be
        calculated relative to this

    """

    def __init__(
        self,
        cfg: dict,
        files: list[Path] = [],
    ):
        """
        Parameters
        ----------
        files : list[Path] (optional)
            if provided, data will be streamed from files, else random data
            will be generateed
        """
        self.cfg = cfg
        self.files = files  # can be used if multiple files should be
        self.file_i = 0
        self.outlet = None
        self.outlet_mrk = None  # will be populated once data is loaded and contains markers or if specified for random data  # noqa
        self.sfreq = cfg["sampling_freq"]

        self.load_next_data()

        # LSL - init after first data is loaded, as sfreq might be derived
        self.init_outlet(cfg)

    def init_buffer(self, data, markers: np.ndarray | None = None):
        self.buffer = data
        self.markers = markers
        self.buffer_i = 0
        self.n_pushed = 0
        self.t_start_s = pylsl.local_clock()

    def init_outlet(self):

        info = (
            pylsl.StreamInfo(
                self.cfg["stream_name"],
                self.cfg.get("stream_type", "EEG"),
                self.cfg["n_channels"],
                self.sfreq,
            ),
        )

        self.info = info
        self.outlet = pylsl.StreamOutlet(self.info)

    def init_outlet_mrk(self, name: str, sfreq: float):
        info = pylsl.StreamInfo(
            name,
            "Markers",
            1,
            pylsl.IRREGULAR_RATE,
            channel_format="string",
        )
        self.outlet_mrk = pylsl.StreamOutlet(info)

    def load_next_data(self):

        # Have this as a warning since loading will likely take a substantial
        # amount of time
        logger.warning(f"Loading new data for {self.name}")

        # random
        if self.files == []:

            self.sfreq = self.cfg["sfreq"]
            logger.debug("Loading new random data")
            data = np.random.randn(
                self.cfg["nchannels"],
                self.sfreq * self.cfg.get("pre_buffer_s", 300),
            )

            markers = None
            if self.cfg.get("markers", False):
                dt = self.cfg["markers"]["t_interval_s"]
                nmrk = int(self.cfg.get("pre_buffer_s", 300) / dt)
                # get markers as n_times x 2 (time, marker)
                markers = np.tile(
                    np.arange(nmrk, dtype="object") * dt * self.sfreq, (2, 1)
                ).T
                mval = self.cfg["markers"].get("values", "a")
                mval = mval if isinstance(mval, list) else [mval]
                mvals = np.tile(mval, nmrk // len(mval) + 1)
                markers[:, 1] = mvals[:nmrk]

        else:
            fl = self.files[self.file_i]
            data, markers, sfreq = load_data(fl, self.cfg)

            if self.sfreq == "derive":
                self.sfreq = sfreq

            if self.sfreq != sfreq:
                logger.warning(
                    f"Specified sampling rate {self.sfreq}Hz does not match"
                    f" sampling rate of file {fl} - {sfreq=}. Mockup stream"
                    " will provide output according to specified sampling rate"
                    " leading to a faster or slower replay."
                )

            self.file_i += 1

            if (
                self.file_i >= len(self.files)
                and self.cfg.get("mode", "") == "repeat"
            ):
                self.file_i = 0

        # put data to buffer and start indexing from zero
        self.init_buffer(data, markers=markers)

        # after loading, we know if there is markers in the file, if yes
        # create a marker stream
        if markers:
            default_name = self.cfg["stream_name"] + "_markers"
            self.init_outlet_mrk(
                self.cfg["markers"].get("marker_stream_name", default_name)
            )

    def push(self):
        n_required = (
            int((pylsl.local_clock() - self.t_start_s) * self.sfreq)
            - self.n_pushed
        )
        if n_required > 0:
            self.outlet.push_chunk(
                self.buffer[
                    self.buffer_i : self.buffer_i + n_required
                ].tolist()
            )

            # if marker stream is associated -> push as well
            if self.outlet_mrk is not None:
                self.push_markers(
                    idx_from=self.n_pushed, idx_to=self.n_pushed + n_required
                )

            self.buffer_i += n_required
            self.n_pushed += n_required

            if self.buffer_i >= self.buffer.shape[0]:
                self.load_next_data()

    def push_markers(self, idx_from: int, idx_to: int):
        """Check if there is a marker within the index range and push if yes"""
        msk = (self.mrks[:, 0] > idx_from) & (self.mrks[:, 0] < idx_to)
        if msk.any():
            for mrk in self.mrks[msk, 1]:
                self.outlet_mrk.push_sample([mrk])


def load_data(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray | None]:

    # load depending on suffix
    loaders = {
        "vhdr": load_bv,
        "fif": load_mne,
        "xdf": load_xdf,
    }

    data, markers, sfreq = loaders[fp.suffix](fp, **cfg)
    markers = None if len(markers) == 0 else markers

    return data, markers, sfreq


def load_bv(fp: Path, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
    """Load for brainvision files"""
    raw = mne.io.read_raw_brainvision(fp, preload=True)
    data, markers = mne_raw_to_data_and_markers(raw)

    return data, markers, raw.info["sfreq"]


def load_mne(fp: Path, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
    """Load for mne/fif files"""
    raw = mne.io.read_raw(fp, preload=True)
    data, markers = mne_raw_to_data_and_markers(raw)

    return data, markers, raw.info["sfreq"]


def load_xdf(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Load from xdf file - requires some configuration for which stream to
    use

    Parameters
    ----------
    fp : Path
        path to xdf file

    cfg : dict
        configuration dictionary specifying which stream to use and potential
        marker stream. Key value pairs are as follows:

        stream_name : str
            name of the stream to use

        marker_stream_name : str (optional)
            if given, look for stream and use to create markers. Markers are
            mapped to the nearest sample time point of the stream defined by
            `stream_name`

        pyxdf_kwargs : dict
            potential kwargs for pyxdf.load


    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
    """

    marker_stream = cfg.get("marker_stream_name", "")

    d = pyxdf.load_xdf(fp)
    snames = [s["info"]["name"][0] for s in d[0]]

    sdata = d[0][snames.index(cfg["stream_name"])]
    sfreq = float(sdata["info"]["nominal_srate"][0])
    data = sdata["time_series"]

    if marker_stream != "":
        # align the markers to match index of timepoints closest matching
        # do adjustments in loop as fully outer vectorized calculation
        # most likely is overkill (assuming n_markers << n_samples)
        td = sdata["time_stamps"]
        mdata = d[0][snames.index(marker_stream)]
        tm = mdata["time_stamps"]
        idx = [np.argmin(np.abs(td - t)) for t in tm]
        markers = np.asarray(
            [idx, [v[0] for v in mdata["time_series"]]], dtype="object"
        ).T

    else:
        markers = np.ndarray([])

    return data, markers, sfreq


def mne_raw_to_data_and_markers(
    raw: mne.io.BaseRaw,
) -> tuple[np.ndarray, np.ndarray]:

    # Use the keys in a marker array
    ev, evid = mne.events_from_annotations(raw, verbose=False)
    imap = {v: k for k, v in evid.items()}
    ev = ev.astype("object")
    ev[:, 2] = [imap[i] for i in ev[:, 2]]

    data = raw.get_data().T

    return data, ev[:, [0, 2]]


# # IGNORE the meta data for now
# def add_bv_ch_info(info: pylsl.StreamInfo, raw: mne.io.BaseRaw):
#     """Add channel meta data to a pylsl.StreamInfo object
#
#     Parameters
#     ----------
#     info : pylsl.StreamInfo
#         info object to add the channel info to
#
#     raw : mne.io.BaseRaw
#         mne raw object to derive the channel names from
#
#     """
#     info.desc().append_child_value("manufacturer", "MockupStream")
#     chns = info.desc().append_child("channels")
#
#     for chan_ix, channel in enumerate(raw.info["chs"]):
#         ch = chns.append_child("channel")
#         ch.append_child_value("label", channel["ch_name"])
#         ch.append_child_value("unit", str(channel["range"]))
#         ch.append_child_value("type", channel["kind"]._name)
#         ch.append_child_value("scaling_factor", "1")
#         loc = ch.append_child("location")
#
#         if not np.isnan(channel["loc"][:3]).all():
#             for name, pos in zip(["X", "Y", "Z"], channel["loc"][:3]):
#                 loc.append_child_value(name, float(pos))
#
#
# def add_channel_info(info: pylsl.StreamInfo, conf: dict, data: Any):
#     """Add channel info depending on what type of data was provided"""
#
#     info_add_funcs = {
#         "BV": add_bv_ch_info,
#         "mne": add_bv_ch_info,
#         "random": add_bv_ch_info,  # random did load to mne.io.RawArray -> the meta data is comparable to the BV load. # noqa
#     }
#
#     info_add_funcs[conf["sources"]["source_type"]](info, data)


def get_data_and_channel_names(
    conf: dict,
) -> tuple[list[mne.io.BaseRaw], list[str]]:
    # Prepare data and stream outlets
    data = load_data(**conf["sources"])

    # infer available channels from data[0] --> assume number of channels stay constant. Working with set intersections did shuffle names to much and nested list comprehensions would be ugly here.        # noqa
    ch_names = (
        data[0].ch_names
        if conf["streaming"]["channel_name"] == "all"
        else conf["streaming"]["channel_name"]
    )

    data = [r.pick_channels(ch_names) for r in data]

    return data, ch_names


def glob_path_to_path_list(cfg: dict) -> list[Path]:

    fp_str = cfg.get("file_path", "")
    if fp_str == "":
        files = []
    else:
        fp = Path(fp_str)
        sep = "\\" if isinstance(fp, PureWindowsPath) else "/"

        # index of where the glob starts
        idx = ["*" in s for s in str(fp).split(sep)].index(True)

        files = list(
            Path(f"{sep}".join(fp.parts[: idx - 1])).rglob(
                f"{sep}".join(fp.parts[idx - 1 :])
            )
        )

    return files


def run_stream(
    stop_event: threading.Event = threading.Event(),
    conf_pth: Path = Path("./config/streams.toml"),
) -> int:
    """
    Parameters
    ----------
    stop_event : threading.Event
        event used to stop the streaming

    conf : dict
        Configuration dictionary. If empty, will load the config specified at
        CONF_PATH (`./configs/streaming.yaml`).

    Returns
    -------
    int
        returns 0

    """

    conf = tomllib.load(open(conf_pth, "rb"))

    # Initialize streams as specified - first random streams
    streams = []
    for sname, scfg in conf.get("random", {}).items():
        streams += MockupStream(name=sname, cfg=scfg)

    # init streams from files
    for sname, scfg in conf.get("files", {}).items():
        files = glob_path_to_path_list(scfg)
        streams += MockupStream(name=sname, cfg=scfg, files=files)

    #
    # print("=" * 80)
    # print(conf)
    # print("=" * 80)
    #
    # data, ch_names = get_data_and_channel_names(
    #     conf,
    #     random_data,
    #     random_sfreq=random_sfreq,
    #     random_nchannels=random_nchannels,
    # )
    #
    # if conf["streaming"]["sampling_freq"] == "derive":
    #     conf["streaming"]["sampling_freq"] = data[0].info["sfreq"]
    # outlets = init_lsl_outlets(conf, data, ch_names)
    #
    # pushfunc = push_with_log if log_push else push_plain
    # # add capability to push markers if specified by wrapping pushfunc
    # pushfunc = push_factory(
    #     pushfunc,
    #     conf["streaming"]["stream_marker"],
    #     random_data_markers_s=random_data_markers_s,
    # )
    #
    # # initialize first block of data to stream
    # block_idx = 0
    # data_array, markers = load_next_block(
    #     block_idx, data, streaming_mode=conf["streaming"]["mode"]
    # )
    #
    # # Sending the data
    # print(f"Now sending data in {conf['streaming']['stream_name']=}")
    # sent_samples = 0
    # t_start = pylsl.local_clock()
    #
    # while not stop_event.is_set():
    #     elapsed_time = pylsl.local_clock() - t_start
    #     last_new_sample = int(
    #         conf["streaming"]["sampling_freq"] * elapsed_time
    #     )
    #     required_samples = last_new_sample - sent_samples
    #
    #     # check if a new block needs to be loaded
    #     if last_new_sample > data_array.shape[1]:
    #         # restart counting as a new file will be openened
    #         t_start = pylsl.local_clock()
    #         sent_samples = 0
    #
    #         block_idx += 1
    #         data_array, markers = load_next_block(
    #             block_idx, data, streaming_mode=conf["streaming"]["mode"]
    #         )
    #
    #         # restart counting - if not specified to repeat, load_next_block
    #         # will have thrown an error
    #         if block_idx >= len(data):
    #             block_idx == 0
    #
    #     if required_samples > 0:
    #         slc = slice(sent_samples, sent_samples + required_samples)
    #         chunk = data_array[:, slc]
    #         mchunk = markers[slc]
    #
    #         stamp = pylsl.local_clock()
    #
    #         # convert numpy to list of lists
    #         ldata = [list(r) for r in chunk.T]  # n_times x n_channels
    #
    #         pushfunc(
    #             outlets,
    #             ldata,
    #             stamp,
    #             markers=list(mchunk[mchunk != ""]),
    #             elapsed_time=elapsed_time,
    #         )
    #
    #         sent_samples += required_samples
    #
    #     sleep_s(0.001)
    #
    # print("Finished")
    #
    return 0


def run_mockup_streamer_thread(
    **kwargs,
) -> tuple[threading.Thread, threading.Event]:
    """Run the streaming within a separate thread and have a stop_event"""
    stop_event = threading.Event()
    stop_event.clear()

    thread = threading.Thread(
        target=run_stream,
        kwargs={"stop_event": stop_event, **kwargs},
    )

    thread.start()

    return thread, stop_event


if __name__ == "__main__":
    Fire(run_stream)
