# Mockup_streamer

This is a module for the [Dareplane](https://github.com/bsdlab/Dareplane) project which provides mockup streaming from recorded data (currently only EEG from BrainVision files) or randomly generated data.

## Running the module

As for all Dareplane modules, you can have this run standalone, via a TCP server or here also as a Docker container. The core interaction is meant to take place via the TCP server.

## Config

The configuration options can be found under `./config` in the `streams.toml` or `streaming_docker.toml` depending whether you plan to run as a docker container or not. For most use cases, running in a `shell` is sufficient and should be the prefered way.

### Config options

```toml

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

This will run the standard configuration located in `./config/streams.toml`.

To simply spawn a quick stream of random data with CLI control over the parameters, use this:

```bash
python -m mockup_streamer.random_cli
# or this for help about parameters
python -m mockup_streamer.random_cli --help
```

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
