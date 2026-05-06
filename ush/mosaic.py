import arcviz
from arcviz.plots import plot_track_comparison
from arcviz.math import extract_trajectory
import matplotlib.pyplot as plt
import logging
import xarray as xr

def main(ufs_fpath, ufs_var, cafs_fpath, cafs_var, mosaic_fpath, mosaic_var, dateshift=False, date_str=None):

    ufs_data = arcviz.GridData(filepath=ufs_fpath, varname=ufs_var, name="UFS-Arctic Ice Thickness")
    cafs_data = arcviz.GridData(filepath=cafs_fpath, varname=cafs_var, name="CAFS Ice Thickness", dateshift=dateshift)
    mosaic_data = arcviz.TrackData(filepath=mosaic_fpath, varname=mosaic_var, name="Ice Thickness")

    thermo_tends = arcviz.IceTendData(filepath=ufs_fpath, trend_type="thermo", name="Thermo Ice Tendencies")
    dyn_tends = arcviz.IceTendData(filepath=ufs_fpath, trend_type="dynamic", name="Dynamic Ice Tendencies")

    ufs_track_da = extract_trajectory(ufs_data.da, mosaic_data.da, mode='follow')
    cafs_track_da = extract_trajectory(cafs_data.da, mosaic_data.da, mode='follow')
    thermo_follow = extract_trajectory(thermo_tends.da, mosaic_data.da, mode='follow')
    thermo_all_neighbors = extract_trajectory(thermo_tends.da, mosaic_data.da, mode='all_neighbors')
    dyn_all_neighbors = extract_trajectory(dyn_tends.da, mosaic_data.da, mode='all_neighbors')

    mosaic_sync_da = mosaic_data.da.sel(time=ufs_data.da.resample(time='1D').mean()['time'], method='nearest')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    plot_track_comparison(
        track_da = mosaic_sync_da,
        grid_das = [cafs_track_da, ufs_track_da],
        title = f'Ice Thickness Along MOSAiC Track ({date_str})',
        colors = ['#808080', '#52a447'],
        ax = axes[0],
    )

    plot_track_comparison(
        track_da = mosaic_sync_da,
        grid_das = [dyn_all_neighbors, thermo_all_neighbors, thermo_follow],
        title = f'Recreated Ice Thickness for Nearest Neighbor Cells (UFS Thermo vs. Dynamic Tends)',
        colors = ['#0a9396', '#ffa500', '#9b2226'],
        ax = axes[1],
    )

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    res  = 'C918'
    fhrs = '240'
    ufs_var    = 'hi_h'
    cafs_var   = 'hi_h'
    mosaic_var = 'Ice_Thickness'

    target_dates = [
        ('2019', '10', '28'),
        ('2020', '02', '27'),
        ('2020', '07', '02'),
        ('2020', '07', '09')
    ]

    for y, m, d in target_dates:
        # Sometimes necessary to adjust CAFS grid data to match UFS calendar
        dateshift = (y == '2020')
        date_str = f"{y}/{m}/{d}"

        ufs_fpath    = f'/scratch4/BMC/ufs-artic/Kristin.Barton/stmp/10day_runs/{res}_{y}{m}{d}_{fhrs}HRS/history/iceh_03h.*.nc'
        cafs_fpath   = f'/scratch3/BMC/ufs-artic/amy/CAFS/REB2.{y}-{m}-{d}.nc'
        mosaic_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/mosaic/MOSAiC_Calc_IceThick.txt'


        main(ufs_fpath, ufs_var, cafs_fpath, cafs_var, mosaic_fpath, mosaic_var, dateshift=dateshift, date_str=date_str)
