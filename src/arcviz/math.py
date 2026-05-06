import xarray as xr
import numpy as np
import logging
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

def extract_trajectory(grid_da, track_da, mode='follow', method='nearest'):
    """
    Extracts model data along a point or moving observation track.
    Expects xarray.DataArray objects
    Mode can be: 'follow' (follows along with track, each time point grabs a single cell)
                 'all_neighbors' (grabs ALL cells along track at ALL time points)
    (For point observations, use 'follow')
    Method can be 'nearest' or 'linear'. Linear uses 4 nearest grid neighbors to track/point.
    """

    grid_name = grid_da.name or "Grid"
    track_name = track_da.name or "Track"

    logger.info(f"Extracting data from {grid_da.name} along {track_da.name} track...")

    freq = xr.infer_freq(track_da['time'])

    if freq is None:
        logger.warning(f"Could not infer frequency from track time, defaulting to 1D")
        freq = '1D'

    # Resample grid data to track frequency
    grid_resampled = grid_da.resample(time=freq).mean()

    # Grab only track times corresponding to grid data rangge
    track_sync = track_da.sel(time=grid_resampled['time'], method='nearest')

    lat_coord = grid_resampled['lat']
    lon_coord = grid_resampled['lon']

    # For 1D coordinates, xarray can handle the remapping
    if lat_coord.ndim == 1:
        if mode == 'follow':
            extracted = grid_resampled.sel(
                lat=track_sync['lat'],
                lon=track_sync['lon'],
                method=method
            )
        elif mode == 'all_neighbors':
            extracted = grid_resampled.sel(
                lat=track_sync['lat'].rename({'time': 'track_point'}),
                lon=track_sync['lon'].rename({'time': 'track_point'}),
                method=method
            )
    # Otherwise, we will need to build a KDTree
    elif lat_coord.ndim == 2:
        grd_lat_rad = np.radians(lat_coord.values)
        grd_lon_rad = np.radians(lon_coord.values)
        trk_lat_rad = np.radians(track_sync['lat'].values)
        trk_lon_rad = np.radians(track_sync['lon'].values)

        # Convert to cartesian coordinates
        x_grd = np.cos(grd_lat_rad) * np.cos(grd_lon_rad)
        y_grd = np.cos(grd_lat_rad) * np.sin(grd_lon_rad)
        z_grd = np.sin(grd_lat_rad)
        x_trk = np.cos(trk_lat_rad) * np.cos(trk_lon_rad)
        y_trk = np.cos(trk_lat_rad) * np.sin(trk_lon_rad)
        z_trk = np.sin(trk_lat_rad)

        valid = np.isfinite(x_grd) & np.isfinite(y_grd) & np.isfinite(z_grd)
        valid_idx_flat = np.where(valid.ravel())[0]

        # Build tree from flattened GRID coords
        tree = KDTree(np.c_[
            x_grd.ravel()[valid_idx_flat], 
            y_grd.ravel()[valid_idx_flat], 
            z_grd.ravel()[valid_idx_flat]
        ])
        dim_y, dim_x = lat_coord.dims

        # Query tree with TRACK coordinates and place into temp indices
#        _, idx_tmp = tree.query(np.c_[x_trk, y_trk, z_trk])

        k_neighbors = 4 if method == 'linear' else 1
        distances, idx_tmp = tree.query(np.c_[x_trk, y_trk, z_trk], k=k_neighbors)

        # Wrap indices as DataArrays so that xarray can natively return time series
        if method == 'nearest':
            real_idx_flat = valid_idx_flat[idx_tmp]
            idx_j, idx_i = np.unravel_index(real_idx_flat, lat_coord.shape)
            if mode == 'follow':
                njda = xr.DataArray(idx_j, dims=['time'], coords={'time': track_sync['time']})
                nida = xr.DataArray(idx_i, dims=['time'], coords={'time': track_sync['time']})
                ntda = xr.DataArray(np.arange(len(track_sync['time'])), dims=['time'], coords={'time': track_sync['time']} )
                extracted = grid_resampled.isel(time=ntda, **{dim_y: njda, dim_x: nida}) 
            elif mode == 'all_neighbors':
                njda = xr.DataArray(idx_j, dims=['track_point'])
                nida = xr.DataArray(idx_i, dims=['track_point'])
                extracted = grid_resampled.isel(**{dim_y: njda, dim_x: nida})

        # Generalizable to nearest neighbor when k=1, but keep it seperate to reduce overhead in nn case
        elif method == 'linear':
            distances = np.maximum(distances, 1e-12)
            weights = 1.0 / distances
            weights /= weights.sum(axis=1, keepdims=True)

            real_idx_flat = valid_idx_flat[idx_tmp]
            idx_j, idx_i = np.unravel_index(real_idx_flat, lat_coord.shape)

            if mode == 'follow':
                njda = xr.DataArray(idx_j, dims=['time', 'neighbor'])
                nida = xr.DataArray(idx_i, dims=['time', 'neighbor'])
                ntda = xr.DataArray(np.arange(len(track_sync['time'])), dims=['time'])
                weights_da = xr.DataArray(weights, dims=['time', 'neighbor'])

                extracted_neighbors = grid_resampled.isel(time=ntda, **{dim_y: njda, dim_x: nida})
            
            elif mode == 'all_neighbors':
                njda = xr.DataArray(idx_j, dims=['track_point', 'neighbor'])
                nida = xr.DataArray(idx_i, dims=['track_point', 'neighbor'])
                weights_da = xr.DataArray(weights, dims=['track_point', 'neighbor'])

                extracted_neighbors = grid_resampled.isel(**{dim_y: njda, dim_x: nida})

            extracted = (extracted_neighbors * weights_da).sum(dim='neighbor',  keep_attrs=True)

    else:
        raise ValueError(f"Unsupported coordinate dimensions: {lat_coord.ndim}D")

    extracted.name = grid_name

    logger.debug(f"[extract_trajectory] Extracted data shape: {extracted.shape}")

    return extracted

def calculate_thickness_from_tends(data, trend_type, owner_name="System"):
    """
    Takes in a Dataset containing appropriave variables and outputs the recalculated thickness
    based on the thermodynamic tendences
    """

    required_vars = ['hi_h', 'dvidtt_h', 'dvidtd_h']
    if not all(var in data for var in required_vars):
        raise ValueError(f"[{owner_name}] Missing required varaibles from dataset. Required varaibles are: {required_vars}")
    
    # Start with the inital thickness averaged over the cell -- tendencies are in cm/day
    dt = 3./24.
    cm2m = 1./100.
    h_init = data['hi_h'].isel(time=0)   # Ice thickness averaged over grid cell area

    if trend_type == 'thermo':
        tend = data['dvidtt_h'] * dt * cm2m 
        tend = xr.where(tend['time'] == tend['time'][0], 0.0, tend)
        tend_sum = tend.cumsum(dim='time')
    elif trend_type == 'dynamic':
        tend = data['dvidtd_h'] * dt * cm2m
        tend = xr.where(tend['time'] == tend['time'][0], 0.0, tend)
        tend_sum = tend.cumsum(dim='time')
    elif trend_type == 'total':
        ttend = data['dvidtt_h'] * dt * cm2m 
        ttend = xr.where(ttend['time'] == ttend['time'][0], 0.0, ttend)
        tsum = ttend.cumsum(dim='time')
        dtend = data['dvidtd_h'] * dt * cm2m
        dtend = xr.where(dtend['time'] == dtend['time'][0], 0.0, dtend)
        dsum = dtend.cumsum(dim='time')
        tend_sum = tsum + dsum 
    else:
        raise ValueError(f"[{owner_name}] Invalid valud for trend_type: must be 'thermo', 'dynamic', or 'total'")
         

    tend_sum = tend_sum + h_init
    tend_sum.name = trend_type
    tend_sum.encoding = data['hi_h'].encoding

    return tend_sum
