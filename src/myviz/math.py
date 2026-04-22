import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)

def standardize_coordinates(data, owner_name="System"):
    """
    Universally fixes coordinate names, lat/lon 0-360 wrapping, and radian to degree conversions.
    """

    logger.debug("Standardizing spatial coordinates.")

    rename_dict = {
        'grid_yt':'lat', 'grid_xt':'lon', # atmf*.nc | sfcf*.nc | atm_diag_*.nc
             'yh':'lat',      'xh':'lon', # ocn_*.nc
           'TLAT':'lat',    'TLON':'lon', # iceh_*.nc
           'ULAT':'lat',    'ULON':'lon', # iceh_*.nc
           'NLAT':'lat',    'NLON':'lon', # iceh_*.nc
           'ELAT':'lat',    'ELON':'lon', # iceh_*.nc
    }

    if isinstance(data, (xr.Dataset, xr.DataArray)):

        # For CICE data, we need to get the lat/lon corresponding to the variable
        coords_string = data.encoding.get('coordinates') or data.attrs.get('coordinates')
        if coords_string:
            active_coords = data.encoding['coordinates'].split()
        else:
            active_coords = list(data.coords.keys())

        coords_to_rename = {coord: rename_dict[coord] for coord in active_coords if coord in rename_dict}
        data = data.rename(coords_to_rename)

        # Fix radians to degrees conversion
        for coord in ['lat', 'lon']:
            if coord in data.coords:
                units = data[coord].attrs.get('units', '').lower()
                if 'rad' in units:
                    data.coords[coord] = data.coords[coord] * (180.0/np.pi)
                    data.coords[coord].attrs['units'] = 'degrees'

        # Fix lon range
        if 'lon' in data.coords and data.coords['lon'].max() > 180.0:
            data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
            if data.coords['lon'].ndim == 1:
                data = data.sortby('lon')

        return data
    
    else:
        raise TypeError(f"Coordinate standardizer cannot handle type: {type(data)}")