import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

def validate_filepaths(filepath, owner_name="System"):
    """
    Checks if filepath contains valid files. Returns list of files if successful, or raises error.
    """

    matched_files = sorted(glob.glob(filepath))

    if not matched_files:
        logger.error(f"[{owner_name}] File check failed. Zero files matched.")
        raise ValueError( f"No files found matching: '{filepath}'.")

    logger.debug(f"[{owner_name}] Found {len(matched_files)} file(s) matching pattern.")

    return matched_files

def parse_track(filepath, owner_name="System"):
    logger.info(f"[{owner_name}] Parsing track file: {filepath}")
    col_names = ["Time", "Lat", "Lon", "Ice_Thickness"]
    mdata = np.genfromtxt(filepath, skip_header=2, names=col_names)

    origin = pd.to_datetime("2019-01-01 00:00:00")

    time_deltas = pd.to_timedelta(mdata["Time"] -1, unit='D')
    real_datetimes = origin + time_deltas

    ds = xr.Dataset(
        data_vars=dict(
            Ice_Thickness = (["time"], mdata["Ice_Thickness"]),
        ),
        coords=dict(
            time=(["time"], real_datetimes),
            lat=(["time"], mdata["Lat"]),
            lon=(["time"], mdata["Lon"]),
        ),
        attrs=dict(
            description="MOSAiC track data",
            original_file=filepath
        )
    )

    return ds

def parse_obs_points(filepath, varname, owner_name="System"):
    """"
    Parses text-based point observations and returns and xarray Dataset.
    """

    logger.info(f"[{owner_name}] Parsing point observation files.")

    # List coordinates as needed in 0-360deg format
    site_coords = {
        'BRW': (71.3230, 203.3886)
    }

    logger.info(f"[{owner_name}] Parsing track file: {filepath}")

    col_names = [ 'site', 'year', 'month', 'day', 'hour', 'wind_dir', 'wind_speed', 'wind_steadiness', 'pressure', 'tmp2m', 'tmp10m', 'tmp_top', 'rh', 'precip' ]
    na_vals = { 'wind_dir': [-999], 'wind_speed': [-999.9], 'wind_steadiness': [-9], 'pressure': [-999.90, -999.9], 'tmp2m': [-999.9], 'tmp10m': [-999.9], 'tmp_top': [-999.9], 'rh': [-99], 'precip': [-99], }
    unit_map = { 'wind_dir': 'degrees', 'wind_speed': 'm/s', 'wind_steadiness': '%', 'pressure': 'hPa', 'tmp2m': 'degC', 'tmp10m': 'degC', 'tmp_top': 'degC', 'rh': '%', 'precip': 'mm/hr', }

    dfs = []

    for f in filepath:
        df = pd.read_csv(
            f,
            sep=r'\s+',
            header=None,
            names=col_names,
            na_values=na_vals
        )
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.set_index('time').sort_index()

    site_code = data['site'].iloc[0]
    if site_code not in site_coords:
        raise ValueError(f"Site code: '{site_code}' not found.")
    
    lat, lon = site_coords[site_code]

    if varname not in data.columns:
        raise ValueError(f"Variable '{varname}' not found. Available: {col_names[5:]}")

    lat_arr = np.full(len(data.index), lat)
    lon_arr = np.full(len(data.index), lon)
    
    da = xr.DataArray(
        data=data[varname].values,
        coords={
            'time': data.index,
            'lat': ('time', lat_arr),
            'lon': ('time', lon_arr)
        },
        dims=['time'],
        name=varname
    )
    da = da.assign_coords(site=site_code)

    da.attrs['units'] = unit_map[varname]

    return da


def standardize_coordinates(data, dateshift=False, varname=None, owner_name="System"):
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

    # Check the coordinates needed for the given variable
    if isinstance(data, xr.Dataset):
        if varname is None:
            raise ValueError(f"[{owner_name}] When inputting xarray.Dataset type, varname must be specified.")
        active_coords = data[varname].encoding['coordinates'].split() if data[varname].encoding.get('coordinates') else list(data[varname].coords.keys())
    elif isinstance(data, xr.DataArray):
        active_coords = data.encoding['coordinates'].split() if data.encoding.get('coordinates') else list(data.coords.keys())
    else:
        raise TypeError(f"[{owner_name}] Input to standardize_coordinates must be xarray.Dataset or xarray.DataArray type")

    logger.debug(f"[{owner_name}] active coords = {active_coords}")

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

    # Enforce standard calendar type
    if 'time' in data.coords:
        if data['time'].dtype == 'O' or hasattr(data.indexes['time'], 'calendar'):
            logger.debug("Converting non-standard cftime calendar to standard datetime64[ns].")
            standard_times = pd.to_datetime(data.indexes['time'].astype(str))
            data = data.assign_coords(time=standard_times)

        if dateshift:
            logger.debug("Applying 2-day shift")
            new_time = data['time'] + pd.Timedelta(days=2)
            data = data.assign_coords(time=new_time)

    # If there are units of degC, convert it to units of K
    if data.attrs.get('units') in ['degC', 'C', 'celsius', 'Celsius']:
        data = data + 273.15
        data.attrs['units'] = 'K'


    return data
    
