import glob
import logging
from .errors import MissingFilepathError

logger = logging.getLogger(__name__)

def validate_filepaths(filepath, owner_name="System"):
    """
    Checks if filepath contains valid files. Returns list of files if successful, or raises error.
    """

    matched_files = sorted(glob.glob(filepath))

    if not matched_files:
        logger.error(f"[{owner_name}] File check failed. Zero files matched.")
        raise MissingFilepathError(
            f"No files found matching: '{filepath}'."
        )

    logger.debug(f"[{owner_name}] Found {len(matched_files)} file(s) matching pattern.")

    return matched_files
