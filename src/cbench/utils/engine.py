import os
import logging

from cbench.utils.logger import setup_logger


class BaseEngine(object):
    """ Engine objects are objects that interacts 
        with input (from stdin, filesystems, etc)
        or output (to logger, filesystems, etc).

        Currently only some variables are kept.

    Args:
        object (_type_): _description_
    """    
    def __init__(self, *args,
                 output_dir=None,
                 logger=None,
                 **kwargs):

        self.setup_engine(*args,
            output_dir=output_dir,
            logger=logger,
            **kwargs
        )

    def setup_engine(self, *args,
                 output_dir=None,
                 logger=None,
                 **kwargs):
        """setup filesystem and logger output

        Args:
            output_dir ([type], optional): [description]. Defaults to None.
            logger ([type], optional): [description]. Defaults to None.
        """        
        # output dir
        self.output_dir = output_dir
        if output_dir:
            if not os.path.exists(output_dir):
                # NOTE: Sometimes FileExistsError is still thrown... dont know why...
                os.makedirs(output_dir, exist_ok=True)
        
        # global logger
        if logger is None:
            if output_dir:
                logger = setup_logger(self.__class__.__name__, outdir=self.output_dir, label="log")
            else:
                logger = setup_logger(self.__class__.__name__)
            logger.setLevel(logging.INFO)

        # if logger is not None:
        self.logger = logger

    def setup_engine_from_copy(self, other):
        if isinstance(other, BaseEngine):
            self.setup_engine(
                output_dir=other.output_dir,
                logger=other.logger,
            )

