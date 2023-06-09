#  Note this still follows the old setup structure before dareplane_utils was created
import socket
import orjson

import threading
from typing import Callable
from fire import Fire
from functools import partial

from mockup_streamer.main import run_stream
from mockup_streamer.utils.logging import logger


from dareplane_default_server.server import DefaultServer


def run_server(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)

    pcommand_map = {
        "START": run_stream,
        "START_RANDOM": partial(run_stream, random_data=True),
    }

    logger.debug("Initializing server")
    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="mockup_server"
    )

    # initialize to start the socket
    server.init_server()

    # start processing of the server
    logger.debug("starting to listen on server")
    server.start_listening()
    logger.debug("stopped to listen on server")

    return 0


if __name__ == "__main__":
    Fire(run_server)
