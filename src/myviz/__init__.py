from .classes import GridData, TrackData, TimeSeriesData
import logging

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """ Turns on console logging for the package. """
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s', 
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)