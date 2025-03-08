import logging
import os
from pathlib import Path
import sys


# https://lscsoft.docs.ligo.org/bilby/_modules/bilby/core/utils/log.html#setup_logger
def setup_logger(name="default", outdir=None, label=None, log_level='INFO', log_level_file='DEBUG', print_version=False):
    """ Setup logging output: call at the start of the script to use

    Parameters
    ==========
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    log_level_file: str, optional
        similar to log_level, log level for logfile
    print_version: bool
        If true, print version information
    """

    if type(log_level) is str:
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError('log_level {} not understood'.format(log_level))
    else:
        level = int(log_level)

    if type(log_level_file) is str:
        try:
            level_file = getattr(logging, log_level_file.upper())
        except AttributeError:
            raise ValueError('log_level_file {} not understood'.format(log_level_file))
    else:
        level_file = int(log_level_file)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)

    if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(name)s %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
        if label is not None and outdir is not None:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

            file_handler.setLevel(level_file)
            logger.addHandler(file_handler)

    # for handler in logger.handlers:
    #     handler.setLevel(level)

    return logger
