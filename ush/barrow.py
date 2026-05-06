import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging
import arcviz
from arcviz.plots import plot_track_comparison

def main(ufs_fpath, ufs_var, lnd_fpath, lnd_var, obs_fpath, obs_var, date, dateshift=False):
    ufs_temps = arcviz.GridData(filepath=ufs_fpath, varname=ufs_var, name="UFS-Arctic 2m Temperature")
    obs_temps = arcviz.PointObsData(filepath=obs_fpath, varname=obs_var, name="Obs 2m Temperature")
    lnd_temps = arcviz.CAFSPointData(filepath=lnd_fpath, varname=lnd_var, name="CAFS 2m Temperature")

    obs_temps.sync_times(ufs_temps)
    ufs_temps.extract_trajectory(obs_temps, mode='moving', method='linear')

    plot_track_comparison(
        track = obs_temps,
        grids = [ufs_temps],
        point = lnd_temps,
        title = f'2m Temperature at Barrow ({date})',
#        colors = ['#61a4ad'],
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
    lnd_var = 'TSA'
    obs_var = 'tmp2m'

    yyyy = ['2019','2020','2020','2020']
    mm   = ['10',  '02',  '07',  '07'  ]
    dd   = ['28',  '27',  '02',  '09'  ]
#    yyyy = ['2019']
#    mm   = ['10']
#    dd   = ['28']

    for i in range(len(yyyy)):
        y = yyyy[i]
        m = mm[i]
        d = dd[i]

        if y == '2020':
            dateshift = True
        else:
            dateshift = False

        ufs_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/stmp/10day_runs/{res}_{y}{m}{d}_{fhrs}HRS/sfcf*.nc'
        lnd_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/CAFS/lnd_points/{y}-{m}-{d}/Barrow.nc'
        obs_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/2m_temp_obs/barrow/met_brw_insitu_1_obop_hour_*.txt'

        main(ufs_fpath, ufs_var, lnd_fpath, lnd_var, obs_fpath, obs_var, date=f'{y}/{m}/{d}', dateshift=dateshift)
