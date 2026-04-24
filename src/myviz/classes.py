import glob
import logging
import xarray as xr
from .err import MissingVariableError, MissingFilepathError, GridFileFormatError, TrackFileFormatError
from .utils import validate_filepaths, parse_track, standardize_coordinates
from .math import extract_trajectory, calculate_thickness_from_tends 

"""
This parses and creates standardized sets of data based on known possible inputs
"""

logger = logging.getLogger(__name__)

# Contains 2D gridded data such as model output
class GridData:
    def __init__(self, filepath, varname, name="GridData", dateshift=False):
        self.name = name
        logger.info(f"[{self.name}] Initializing GridData")

        if dateshift:
            logger.warning(f"[{self.name}] dateshift is TRUE: This will shift dates forward by 2 days!")
        else:
            logger.warning(f"[{self.name}] dateshift is FALSE: Don't forget to check whether dates need to be shifted!")
        self.dateshift = dateshift

        self.filelist = validate_filepaths(filepath, owner_name=self.name)
        self.varname  = varname

        self.da = None
        self._load_data()

        self.track = None

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
        da = standardize_coordinates(da, self.dateshift, owner_name=self.name)

        self.da = da
    
    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")

    def extract_trajectory(self, track, mode='moving', method='nearest'):
        if self.track is not None:
            logger.warn(f"[{self.name}] Object already contains track data. Overwriting...")
        
        self.track = extract_trajectory(self, track, mode=mode, method=method)
        self.track.name = f"{self.name}"

# Contains 1D data such as buoy tracks
class TrackData:
    def __init__(self, filepath, varname, name='TrackData'):
        self.name = name
        logger.info(f"[{self.name}] Initializing TrackData")

        self.varname = varname
        self.filepath = validate_filepaths(filepath, owner_name=self.name)[0]

        self.da = None
        self._load_data()

        self.synced_da = None
        self.synced_to = None
    
    def _load_data(self):
        logger.info(f"[{self.name}] Parsing track file: {self.filepath}")
        ds = parse_track(self.filepath, owner_name=self.name)

        da = ds[self.varname]
        da = standardize_coordinates(da, owner_name=self.name)

        self.da = da

    def sync_times(self, data):
        if self.synced_da is not None:
            logger.warn("f[{self.name}] Already contains time-synced data. Overwriting...")

        data_daily = data.da.resample(time='1D').mean()
        self.synced_da = self.da.sel(time=data_daily['time'], method='nearest')
        self.synced_to = data.name

    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")

# Contains ice tendency data 
class IceTendData:
    def __init__(self, filepath, trend_type='total', name='IceTendData', dateshift=False):
        self.name = name
        logger.info(f"[{self.name}] Initializing IceTendData")

        self.varname = trend_type
        self.dateshift = dateshift

        self.da = None
        self.track = None

        self.filelist = validate_filepaths(filepath, owner_name=self.name)

        self._load_data()

    def _load_data(self):
        logger.info(f"[{self.name}] Parsing ice tenencies from: {self.filelist}")

        ds = xr.open_mfdataset(
                self.filelist, 
                combine='nested', 
                concat_dim='time',
                parallel=True,
                coords='minimal',
                data_vars='minimal',
                compat='override'
        )

        da = calculate_thickness_from_tends(ds, trend_type=self.varname, owner_name=self.name)

        da = standardize_coordinates(da, self.dateshift, varname=self.varname, owner_name=self.name)

        self.da = da

    def extract_trajectory(self, track, mode='moving', method='nearest'):
        if self.track is not None:
            logger.warn(f"[{self.name}] Object already contains track data. Overwriting...")
        
        self.track = extract_trajectory(self, track, mode=mode, method=method)
        self.track.name = f"{self.name}"

    def get_info(self):
        logger.info(f"This is {self.name}. Data: {self.da}")
