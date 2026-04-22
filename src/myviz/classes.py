import glob
import logging
import xarray as xr
from .err import MissingVariableError, MissingFilepathError, GridFileFormatError, TrackFileFormatError
from .utils import validate_filepaths, parse_track, standardize_coordinates

"""
This parses and creates standardized sets of data based on known possible inputs
"""

logger = logging.getLogger(__name__)

# Contains 2D gridded data such as model output
class GridData:
    def __init__(self, filepath, varname, name="GridData"):
        self.name = name
        logger.info(f"[{self.name}] Initializing GridData")

        self.filelist = validate_filepaths(filepath, owner_name=self.name)
        self.varname  = varname

        self.da = None
        self._load_data()

    def _load_data(self):
        """
        Reads one or multiple netcdf files and returns a standardized DataArray for a specific variable
        """
        ds = xr.open_mfdataset(
                self.filelist, 
                combine='nested', 
                concat_dim='time',
                parallel=True,
                coords='minimal',
                data_vars='minimal',
                compat='override'
        )
        da = ds[self.varname]
        da = standardize_coordinates(da, owner_name=self.name)

        self.da = da
    
    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")


# Contains 1D data such as buoy tracks
class TrackData:
    def __init__(self, filepath, varname, name='TrackData'):
        self.name = name
        logger.info(f"[{self.name}] Initializing TrackData")

        self.varname = varname
        self.filepath = validate_filepaths(filepath, owner_name=self.name)[0]

        self.da = None
        self._load_data()
    
    def _load_data(self):
        logger.info(f"[{self.name}] Parsing track file: {self.filepath}")
        ds = parse_track(self.filepath, owner_name=self.name)

        da = ds[self.varname]
        da = standardize_coordinates(da, owner_name=self.name)

        self.da = da

    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")

# Contains single-point time series data
class TimeSeriesData:
    def __init__(self, filepath, name='TimeSeriesData'):
        self.name = name
        logger.info(f"[{self.name}] Initializing TimeSeriesData")

        self.filepath = validate_filepaths(filepath, owner_name=self.name)

        self.da = None
    
    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")