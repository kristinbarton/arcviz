import logging
import xarray as xr
from .utils import validate_filepaths, parse_track, standardize_coordinates, parse_obs_points
from .math import calculate_thickness_from_tends 

logger = logging.getLogger(__name__)

class BaseDataset:
    """ Base class to handle common initialization and validation for datasets. """
    def __init__(self, filepath, varname, name="Dataset", dateshift=False):
        self.name = name
        self.varname = varname
        self.dateshift = dateshift

        logger.info(f"[{self.name}] Initializing...")
        if dateshift:
            logger.warning(f"[{self.name}] dateshift is TRUE: Dates shifted forward by 2 days!")
            self.filelist = validate_filepaths(filepath, owner_name=self.name)

        self.filelist = validate_filepaths(filepath, owner_name=self.name)

        self.da = self._load_data()

        if self.da is not None:
            self.da.name = self.name


# Contains 2D gridded data such as model output
class GridData(BaseDataset):
    def _load_data(self):
        ds = xr.open_mfdataset(
                self.filelist, 
                combine='nested', 
                concat_dim='time',
                parallel=True,
                coords='minimal',
                data_vars='minimal',
                compat='override'
        )
        return standardize_coordinates(ds[self.varname], self.dateshift, owner_name=self.name)
    
# Contains 1D track data such as buoy observations
class TrackData(BaseDataset):
    def _load_data(self):
        # filelist[0] used because TrackData expects a single file
        ds = parse_track(self.filelist[0], owner_name=self.name)
        return standardize_coordinates(ds[self.varname], self.dateshift, owner_name=self.name)

# Contains data at a single point
class ObsPointData(BaseDataset):
    def _load_data(self):
        da = parse_obs_points(self.filelist, self.varname, owner_name=self.name)
        return standardize_coordinates(da, dateshift=False, varname=self.varname, owner_name=self.name)

# Contains CAFS data already extracted at a single point
class CAFSPointData(BaseDataset):
    def _load_data(self):
        ds = xr.open_mfdataset(
                self.filelist[0], 
                combine='nested', 
                concat_dim='time',
                parallel=True,
                coords='minimal',
                data_vars='minimal',
                compat='override'
        )
        return  standardize_coordinates(ds[self.varname], self.dateshift, owner_name=self.name)

# Contains ice tendency data 
class IceTendData(BaseDataset):
    def __init__(self, filepath, trend_type='total', name='IceTendData', dateshift=False):
        super().__init__(filepath=filepath, varname=trend_type, name=name, dateshift=dateshift)

    def _load_data(self):
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
        return standardize_coordinates(da, self.dateshift, varname=self.varname, owner_name=self.name)
