#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20220902
#
# Stream from files defined in the config/streaming.toml
#
#
# TODOS:
# - [ ] stream markers alongside
# - [ ] include streaming from an xdf file


from typing import Callable, Any
from pathlib import Path
from fire import Fire

import time
import mne
import tomllib
import pylsl

import threading  # necessary to make it accessible via api

import numpy as np

CONF_PATH = "./config/streaming.toml"


class NoFilesFound(FileExistsError):
    pass


def get_config() -> dict:
    with open(CONF_PATH, "rb") as f:
        conf = tomllib.load(f)

    return conf


def get_loader(source_type: str) -> Callable:
    loaders = {"BV": mne.io.read_raw_brainvision, "mne": mne.io.read_raw}

    return loaders[source_type]


def load_random(nchannels: int = 10) -> list[mne.io.BaseRaw]:
    """Return a few random raw objects to loop through"""
    raws = []

    for i in range(10):
        data = np.random.randn(nchannels, 100000)
        info = mne.create_info(
            [f"ch_{i}" for i in range(nchannels)], 100, ch_types="eeg"
        )
        raws.append(mne.io.RawArray(data, info))

    return raws


def load_data(
    data_source: str, files_pattern: str, source_type: str, **kwargs
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
        "random": add_bv_ch_info,  # random did load to mne.io.RawArray -> the meta data is comparable to the BV load. # noqa
    }

    info_add_funcs[conf["sources"]["source_type"]](info, data)


def push_plain(
    outlet: pylsl.StreamOutlet, data: list[list[float]], stamp: float
):
    outlet.push_chunk(data, stamp)


def push_with_log(
    outlet: pylsl.StreamOutlet, data: list[list[float]], stamp: float
):
    print(f"Pushing n={len(data)} samples @ {stamp}")
    outlet.push_chunk(data, stamp)


def run_stream(
    stop_event: threading.Event = threading.Event(),
    stream_name: str = "",
    log_push: bool = False,
    random_data: bool = False,
    conf: dict = {},
    **kwargs,
) -> int:
    """Run a mockup stream"""

    if conf == {}:
        conf = get_config()

    pushfunc = push_with_log if log_push else push_plain

    # Prepare data and stream outlets
    if random_data:
        data = load_random()
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

    stream_name = (
        conf["streaming"]["stream_name"] if stream_name == "" else stream_name
    )

    info = pylsl.StreamInfo(
        name=stream_name,
        type=conf["streaming"]["stream_type"],
        channel_count=len(ch_names),
        nominal_srate=conf["streaming"]["sampling_freq"],
    )
    add_channel_info(info, conf, data[0])
    outlet = pylsl.StreamOutlet(info, 32, 360)

    # Sending the data
    print(f"now sending data in {stream_name=}")
    sent_samples = 0
    t_start = pylsl.local_clock()

    srate = conf["streaming"]["sampling_freq"]

    block_idx = 0
    data_array = data[0].get_data()

    while not stop_event.is_set():
        elapsed_time = pylsl.local_clock() - t_start
        last_new_sample = int(srate * elapsed_time)
        required_samples = last_new_sample - sent_samples

        # check if a new block needs to be loaded
        if last_new_sample > data_array.shape[1]:
            block_idx += 1
            if block_idx >= len(data) and conf["streaming"]["mode"] != "repeat":
                block_idx = 0
            else:
                break

            print("Fetching next block of data")
            data[
                block_idx
            ].load_data()  # note this might need to go into separate concurrent routine to prevent larger lags - [ ] TODO: investigate # noqa
            data_array = data[block_idx].get_data()

        if required_samples > 0:
            chunk = data_array[
                :, sent_samples : sent_samples + required_samples
            ]
            stamp = pylsl.local_clock()

            ldata = [list(r) for r in chunk.T]  # n_times x n_channels

            pushfunc(outlet, ldata, stamp)
            sent_samples += required_samples

        time.sleep(0.01)

    print("Finished")

    return 0


if __name__ == "__main__":
    Fire(run_stream)
