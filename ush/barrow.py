import logging
import xarray as xr
from datetime import datetime
import arcviz
from arcviz.plots import plot_track_comparison
from arcviz.math import extract_trajectory

def main(ufs_fpath, ufs_var, lnd_fpath, lnd_var, obs_fpath, obs_var, dateshift=False, date_str=None):

    # Grab temperature data for UFS-Arctic, CAFS, and Barrow observations
    ufs_temps = arcviz.GridData(filepath=ufs_fpath, varname=ufs_var, name="UFS-Arctic 2m Temperature")
    cafs_temps = arcviz.CAFSPointData(filepath=lnd_fpath, varname=lnd_var, name="CAFS 2m Temperature")
    obs_temps = arcviz.ObsPointData(filepath=obs_fpath, varname=obs_var, name="Obs 2m Temperature")

    # Grab observations corresponding to UFS-Arctic/CAFS time range
    obs_sync_da = obs_temps.da.sel(time=ufs_temps.da.resample(time='1h').mean()['time'], method='nearest')

    # Exctract UFS-Arctic data at the observation location
    ufs_track_da = extract_trajectory(ufs_temps.da, obs_temps.da, mode='follow', method='linear')
    ufs_track_da.name = "UFS-Arctic 2m Temp (Trajectory)"

    plot_track_comparison(
        track_da = obs_sync_da,
        grid_das = [ufs_track_da],
        point_da = cafs_temps.da,
        title = f'2m Temperature at Barrow ({date_str})',
        colors = ['#ffa500'],
    )

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    res  = 'C918'
    fhrs = '240'
    ufs_var = 'tmp2m'
    cafs_var = 'TSA'
    obs_var = 'tmp2m'

    target_dates = [
        ('2019', '10', '28'),
        ('2020', '02', '27'),
        ('2020', '07', '02'),
        ('2020', '07', '09')    
    ]

    for y, m, d in target_dates:
        date_str = f"{y}/{m}/{d}"

        # Sometimes necessary for shifting CAFS grid data calendar to match UFS
        # dateshift = (y == '2020')

        ufs_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/stmp/10day_runs/{res}_{y}{m}{d}_{fhrs}HRS/sfcf*.nc'
        cafs_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/CAFS/lnd_points/{y}-{m}-{d}/Barrow.nc'
        obs_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/2m_temp_obs/barrow/met_brw_insitu_1_obop_hour_*.txt'

        main(ufs_fpath, ufs_var, cafs_fpath, cafs_var, obs_fpath, obs_var, date_str=date_str)
