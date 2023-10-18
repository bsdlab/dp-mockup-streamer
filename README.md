# Mockup_streamer

This is a module for the [Dareplane](https://github.com/bsdlab/Dareplane) project which provides mockup streaming from recorded data (currently only EEG from BrainVision files) or randomly generated data.

## Running the module

As for all Dareplane modules, you can have this run standalone, via a TCP server or here also as a Docker container. The core interaction is meant to take place via the TCP server.

## Config

The configuration options can be found under `./config` in the `streaming.toml` or `streaming_docker.toml` depending whether you plan to run as a docker container or not. For most use cases, running in a `shell` is sufficient and should be the prefered way.
### Config options

```toml
[sources]
data_source='<PATH/TO/DATAFOLDER>'                         # Path to a folder containing data files (mne '*.fif' or brain vision '*.eeg, *.vhdr, *.vmrk' )
source_type='mne'                                          # this will specify the loader that is loaded - 'mne' or 'BV'
files_pattern='*concatenated_Fp1_raw.fif'                  # a pattern to look for. If multiple files are found, the streamer will provide their data one after another.

[streaming]
stream_name='mock_EEG_stream'
mode='repeat'                               # repeat or once - if every file should be streamed just once or if the mockup streamer should repeat after finishing the last file.
type='regular'                              # regular or irregular - the basic outlet type
sampling_freq='derive'                      # sampling rate to be simulated, can be float or `derive` to take the sfreq from the sources
stream_type='EEG'                           # meta data for LSL
channel_name=['Fp1']                        # if 'all' stream all channels from source, else provide a list of strings e.g. ['C3', 'Cz', 'C4']
# channel_name='all'
stream_marker=true                          # if true, will also stream markers if source provides them 
marker_stream_name='mock_EEG_stream_marker'
```

### Running in a shell

#### As standalone python

A simple CLI is implemented in `./mockup_streamer/main.py` for the following function

```python
def run_stream(stop_event: threading.Event = threading.Event(),
               stream_name: str = "",
               log_push: bool = False,
               random_data: bool = True,
               **kwargs) -> int:
```

So you can start the mockup stream by simply calling `python -m mockup_streamer.main`, `python -m mockup_streamer.main --stream_name="my_stream"` to overwrite the stream name, which is otherwise configured under `./config/streaming.toml`.

#### As TCP server

To spawn it, simply run from this directory

```bash
python -m api.server
```

Note: By default, this will bind to port `8080` on the local host. You might want to change this within `./api/server.py`.

### Running as docker

Start by building the container, e.g.

```bash
sudo docker build -t mockup_streamer .
```

Telling docker to build an image with name `mockup_streamer`. Next, you can run the container with

```bash
sudo docker run -ip 127.0.0.1:8080:8080 --rm --name my_mockup_streamer --mount type=bind,source="/home/md/workspace/data/bbciRaw/my_eeg_session/folder_with_vhdr_files",target=/var/eeg,readonly -t mockup_streamer
```

Here we are creating a interactive `-i` container with a port mapping `p` which will be removed once it is closed `--rm` with the specific name `--name` and based of the `-t` container we just built above.

### Makefile

Both of the above-mentioned commands are coded into the `Makefile` and can be run by `make build` and `make run` respectively. The `Makefile` is here only to simplify the docker interaction.

#### Connecting

Once the server is running (either in a shell or in a docker container), you can send the primary commands `START` and `STOP` to the server, which will start or stop streaming data via LSL.
To do so connect a socket, e.g. via `telnet` from your terminal as

```bash
telnet 127.0.0.1 8080
```

## Configure

All configuration can be found under `./config/`. Check e.g. `./config/streaming.toml` for setting of the mockup stream, and where the source files are globbed from.
Also note the `./config/streaming_docker.toml` file which would handle the config for running within a container.

## TODOs

- [ ] implement replaying `xdf`
- [x] include a markers stream
