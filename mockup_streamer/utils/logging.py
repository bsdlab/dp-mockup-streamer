import os
import logging
from pathlib import Path

# specify the default value and an environment variable which would be
# overwriting
LOGFORMAT ='%(asctime)s|%(name)s|%(levelname)s|%(message)s'

logcfg = dict(
    filename=Path('./dareplane_control_room_logfile.log').resolve(),
    level=logging.WARNING,
    format=LOGFORMAT,
)

# if evironment variables are defined use them and overwrite default
os_envs = dict(
    filename='PYTHON_LOGGING_DIR',
    level='PYTHON_LOGGING_LEVEL'
)

for k, v in os_envs.items():
    try:
        logcfg[k] = os.environ[v]
    except KeyError:
        pass

logging.basicConfig(**logcfg)

logger = logging.getLogger("control_room")

# For now always add a stream handler
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(
    logging.Formatter(logcfg['format'])
)
logger.addHandler(streamhandler)


def set_log_file(logger: logging.Logger, fpath: Path):
    """ Overwrite the file handler in the logger """
    for hdl in logger.root.handlers:
        if isinstance(hdl, logging.FileHandler):
            logger.root.removeHandler(hdl)

            new_fh = logging.FileHandler(fpath)
            formatter = logging.Formatter(LOGFORMAT)
            new_fh.setFormatter(formatter)

            logger.root.addHandler(new_fh)


if __name__ == '__main__':
    logger.debug("Debugging message")
    logger.info("Info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
