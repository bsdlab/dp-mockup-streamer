# Stream from files defined in the config/streaming.toml


import threading  # necessary to make it accessible via api
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import pylsl
from fire import Fire

from mockup_streamer.utils.logging import logger

CONF_PATH = "./config/streaming.toml"

MRK_CNT = 0


class NoFilesFound(FileExistsError):
    pass


class EndOfDataError(KeyError):
    pass


def get_config() -> dict:
    """Load the config file at CONF_PATH into a dict"""
    with open(CONF_PATH, "rb") as f:
        conf = tomllib.load(f)

    return conf


def get_loader(source_type: str) -> Callable:
    """Loader factory, additional loaders will be implemented here"""
    loaders = {"BV": mne.io.read_raw_brainvision, "mne": mne.io.read_raw}

    return loaders[source_type]


def load_random(
    nchannels: int = 10, sfreq: float = 100
) -> list[mne.io.BaseRaw]:
    """Return a few random raw objects to loop through"""
    raws = []

    for i in range(10):
        data = np.random.randn(nchannels, sfreq * 100)
        info = mne.create_info(
            [f"ch_{i}" for i in range(nchannels)], sfreq, ch_types="eeg"
        )
        raws.append(mne.io.RawArray(data, info))

    return raws


def load_data(
    data_source: str, files_pattern: str, source_type: str
) -> list[mne.io.BaseRaw]:
    """
    Use `files_pattern` to glob the path at `data_source` and load with the
    appropriate loader
    """

    files = list(Path(data_source).rglob(files_pattern))

    if files == []:
        raise NoFilesFound(
            f"Did not find any files for pattern {files_pattern} at "
            f"{data_source}"
        )
    else:
        loader = get_loader(source_type)
        raws = []
        for f in files:
            raws.append(loader(f))

    return raws


def add_bv_ch_info(info: pylsl.StreamInfo, raw: mne.io.BaseRaw):
    """Add channel meta data to a pylsl.StreamInfo object

    Parameters
    ----------
    info : pylsl.StreamInfo
        info object to add the channel info to

    raw : mne.io.BaseRaw
        mne raw object to derive the channel names from

    """
    info.desc().append_child_value("manufacturer", "MockupStream")
    chns = info.desc().append_child("channels")

    for chan_ix, channel in enumerate(raw.info["chs"]):
        ch = chns.append_child("channel")
        ch.append_child_value("label", channel["ch_name"])
        ch.append_child_value("unit", str(channel["range"]))
        ch.append_child_value("type", channel["kind"]._name)
        ch.append_child_value("scaling_factor", "1")
        loc = ch.append_child("location")

        if not np.isnan(channel["loc"][:3]).all():
            for name, pos in zip(["X", "Y", "Z"], channel["loc"][:3]):
                loc.append_child_value(name, float(pos))


def add_channel_info(info: pylsl.StreamInfo, conf: dict, data: Any):
    """Add channel info depending on what type of data was provided"""

    info_add_funcs = {
        "BV": add_bv_ch_info,
        "mne": add_bv_ch_info,
        "random": add_bv_ch_info,  # random did load to mne.io.RawArray -> the meta data is comparable to the BV load. # noqa
    }

    info_add_funcs[conf["sources"]["source_type"]](info, data)


def push_plain(
    outlet: pylsl.StreamOutlet, data: list[list[float]], stamp: float
):
    # outlet.push_chunk(data, stamp)
    outlet.push_chunk(data)


def push_with_log(
    outlet: pylsl.StreamOutlet, data: list[list[float]], stamp: float
):
    print(f"Pushing n={len(data)} samples @ {stamp}")
    # outlet.push_chunk(data, stamp)
    outlet.push_chunk(data)


def get_data_and_channel_names(
    conf: dict,
    random_data: bool = False,
    random_sfreq: float = 100,
    random_nchannels: int = 10,
) -> tuple[list[mne.io.BaseRaw], list[str]]:
    # Prepare data and stream outlets
    logger.debug(f"Loading data >> {random_data=}")
    if random_data:
        data = load_random(sfreq=random_sfreq, nchannels=random_nchannels)
        ch_names = data[0].ch_names  # random is always all
        conf["sources"]["source_type"] = "random"

    else:
        data = load_data(**conf["sources"])

        # infer available channels from data[0] --> assume number of channels stay constant. Working with set intersections did shuffle names to much and nested list comprehensions would be ugly here.        # noqa
        ch_names = (
            data[0].ch_names
            if conf["streaming"]["channel_name"] == "all"
            else conf["streaming"]["channel_name"]
        )

        data = [r.pick_channels(ch_names) for r in data]

    return data, ch_names


def sleep_s(s: float):
    """Sleep for s seconds."""

    start = time.perf_counter_ns()
    if s > 0.1:
        # If not yet reached 90% of the sleep duration, sleep in 10% increments
        # The 90% threshold is somewhat arbitrary but when testing intervals
        # with 1 ms to 500ms this produced very accurate results with deviation
        # less than 0.1% of the desired target value. On Mac M1 with python 3.11
        while time.perf_counter_ns() - start < (s * 1e9 * 0.9):
            time.sleep(s / 10)

    # Sleep for the remaining time
    while time.perf_counter_ns() - start < s * 1e9:
        pass


def load_next_block(
    block_idx: int, data: list[mne.io.BaseRaw], streaming_mode: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    block_idx : int
        index of the block to load

    data : list[mne.io.BaseRaw]
        list of data to stream

    streaming_mode : str
        "repeat" or "" (default), if "repeat" it will loop over the files again
        after all data was streamed

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        data array and markers/annotations array as part of the mne.io.BaseRaw
        which is at data[block_idx]. Note, the markers array is aligned
        with the data to be selectable with the same slice

    """
    if block_idx >= len(data):
        if streaming_mode == "repeat":
            block_idx = 0
        else:
            raise EndOfDataError(f"All data streamed and {streaming_mode=}")

    print("Fetching next block of data")
    raw = data[block_idx]
    raw.load_data()  # note this might need to go into separate concurrent routine to prevent larger lags - [ ] TODO: investigate # noqa

    markers = np.asarray([""] * len(raw.times), dtype="object")
    ev, evid = mne.events_from_annotations(raw, verbose=False)
    # invert the map to use the string markers within the marker array
    imap = {v: k for k, v in evid.items()}

    # ev will have the time stamps with raw.first_samp included -> remove to align with the data as indexed from one # noqa
    markers[ev[:, 0] - raw.first_samp] = [imap[v] for v in ev[:, 2]]

    return raw.get_data(), markers


def init_lsl_outlets(
    conf: dict,
    data: list[mne.io.BaseRaw],
    ch_names: list[str],
) -> dict[pylsl.StreamOutlet]:
    outlets = {}
    # --- the data outlet
    info = pylsl.StreamInfo(
        name=conf["streaming"]["stream_name"],
        type=conf["streaming"]["stream_type"],
        channel_count=len(ch_names),
        nominal_srate=conf["streaming"]["sampling_freq"],
    )
    add_channel_info(info, conf, data[0])
    outlets["data"] = pylsl.StreamOutlet(info, 32, 360)

    # --- the marker outlet
    if conf["streaming"]["stream_marker"]:
        info_mrks = pylsl.StreamInfo(
            name=conf["streaming"]["marker_stream_name"],
            type="Markers",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format="string",
        )
        outlets["markers"] = pylsl.StreamOutlet(info_mrks)
    return outlets


def push_factory(
    pushfunc: Callable,
    stream_markers: bool = False,
    random_data_markers_s: float = 0,
) -> Callable:
    """Factory function to create a push function with the correct signature

    Parameters
    ----------
    pushfunc : Callable
        The push function to wrap
    stream_markers : bool
        If true, will push markers
    random_data_markers_s : foat
        If != 0 will add markers to the random data stream every every <value>s

    Returns
    -------
    Callable
        The wrapped push function
    """

    # Prepare two ways of pushing - see wrapper for signature
    def push_wo_markers(outlets, data, tstamp, markers=[], elapsed_time=0):
        pushfunc(outlets["data"], data, tstamp)

    def push_with_markers(outlets, data, tstamp, markers=[], elapsed_time=0):
        pushfunc(outlets["data"], data, tstamp)

        # Currently all markers will be bundeled into one tsample. For mockup
        # this can be ok if the sampling rate of the mockup stream is high
        if markers != []:
            for mrk in markers:
                outlets["markers"].push_sample([mrk], tstamp)

    def push_random_with_markers(
        outlets, data, tstamp, markers=[], elapsed_time=0
    ):
        pushfunc(outlets["data"], data, tstamp)

        global MRK_CNT

        mrk_cnt = elapsed_time // random_data_markers_s
        if mrk_cnt > MRK_CNT:
            outlets["markers"].push_sample([str(mrk_cnt)], tstamp)
            MRK_CNT = mrk_cnt

    if random_data_markers_s != 0:
        pfunc = push_random_with_markers
    elif stream_markers:
        pfunc = push_with_markers
    else:
        pfunc = push_wo_markers

    def push_wrapper(
        outlets: dict[pylsl.StreamOutlet],
        data: np.ndarray,
        tstamp: float,
        markers: list = [],
        elapsed_time: float = 0,
    ):
        """Push data to the outlets
        Parameters
        ----------
        outlets : dict[pylsl.StreamOutlet]
            The outlets to push to, needs a `data` and a `markers` outlet
        data : np.ndarray
            The data to push
        tstamp : float
            The timestamp of the data
        markers: list
            a list of markers
        """
        return pfunc(
            outlets, data, tstamp, markers=markers, elapsed_time=elapsed_time
        )

    return push_wrapper


def run_stream(
    stop_event: threading.Event = threading.Event(),
    stream_name: str = "",
    log_push: bool = False,
    random_data: bool = False,
    random_sfreq: float = 100,
    random_data_markers_s: float = 0,
    random_nchannels: int = 10,
    conf: dict = {},
) -> int:
    """

    Parameters
    ----------
    stop_event : threading.Event
        event used to stop the streaming

    stream_name : str
        Name of the data stream. This will overwrte the name defined in
        `./configs/streaming.yaml`

    log_push : bool
        If true, will print a log message for each push

    random_data : bool
        If true, will load random data instead of real data

    random_sfreq : float
        Sampling frequency of the random data

    random_data_markers_s : float
        if != 0 will add markers to the random data stream every every <value>s

    conf : dict
        Configuration dictionary. If empty, will load the config specified at
        CONF_PATH (`./configs/streaming.yaml`).

    Returns
    -------
    int
        returns 0

    """
    # -- loading config and overwrites if provided to CLI
    if conf == {}:
        conf = get_config()
    if stream_name != "":
        conf["streaming"]["stream_name"] = stream_name

    print("=" * 80)
    print(conf)
    print("=" * 80)

    data, ch_names = get_data_and_channel_names(
        conf,
        random_data,
        random_sfreq=random_sfreq,
        random_nchannels=random_nchannels,
    )

    if conf["streaming"]["sampling_freq"] == "derive":
        conf["streaming"]["sampling_freq"] = data[0].info["sfreq"]
    outlets = init_lsl_outlets(conf, data, ch_names)

    pushfunc = push_with_log if log_push else push_plain
    # add capability to push markers if specified by wrapping pushfunc
    pushfunc = push_factory(
        pushfunc,
        conf["streaming"]["stream_marker"],
        random_data_markers_s=random_data_markers_s,
    )

    # initialize first block of data to stream
    block_idx = 0
    data_array, markers = load_next_block(
        block_idx, data, streaming_mode=conf["streaming"]["mode"]
    )

    # Sending the data
    print(f"Now sending data in {conf['streaming']['stream_name']=}")
    sent_samples = 0
    t_start = pylsl.local_clock()

    while not stop_event.is_set():
        elapsed_time = pylsl.local_clock() - t_start
        last_new_sample = int(
            conf["streaming"]["sampling_freq"] * elapsed_time
        )
        required_samples = last_new_sample - sent_samples

        # check if a new block needs to be loaded
        if last_new_sample > data_array.shape[1]:
            # restart counting as a new file will be openened
            t_start = pylsl.local_clock()
            sent_samples = 0

            block_idx += 1
            data_array, markers = load_next_block(
                block_idx, data, streaming_mode=conf["streaming"]["mode"]
            )

            # restart counting - if not specified to repeat, load_next_block
            # will have thrown an error
            if block_idx >= len(data):
                block_idx == 0

        if required_samples > 0:
            slc = slice(sent_samples, sent_samples + required_samples)
            chunk = data_array[:, slc]
            mchunk = markers[slc]

            stamp = pylsl.local_clock()

            # convert numpy to list of lists
            ldata = [list(r) for r in chunk.T]  # n_times x n_channels

            pushfunc(
                outlets,
                ldata,
                stamp,
                markers=list(mchunk[mchunk != ""]),
                elapsed_time=elapsed_time,
            )

            sent_samples += required_samples

        sleep_s(0.001)

    print("Finished")

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
