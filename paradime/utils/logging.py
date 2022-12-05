"""Loggin utility for ParaDime.

The :mod:`paradime.utils.logging` module implements logging functionality used
by verbose ParaDime routines.
"""

import logging
import sys

LOGGING_FORMAT = "%(asctime)s: %(message)s"
LOGGER_NAME = "paradime"

logger = logging.getLogger(LOGGER_NAME)
formatter = logging.Formatter(LOGGING_FORMAT)

logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)

logger.addHandler(ch)


def log(message: str) -> None:
    """Calls the ParaDime logger to print a timestamp and a message.

    Args:
        message: The message string to print.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(message)


def set_logfile(
    filename: str,
    mode: str = "a",
    disable_stdout: bool = False,
    disable_other_files: bool = False,
) -> None:
    """Configure the ParaDime logger to write its output to a file.

    Args:
        filename: The path to the log file.
        mode: The mode to open the file.
        disable_stdout: Whether or not to disbale logging to stdout.
        disable_other_files: Whether or not to remove other file handlers from
            the ParaDime logger.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if disable_stdout:
        logger.removeHandler(ch)
    if disable_other_files:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
    fh = logging.FileHandler(filename=filename, mode=mode, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
