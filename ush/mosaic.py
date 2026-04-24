import myviz
from myviz.plots import plot_track_comparison
import matplotlib.pyplot as plt
import logging

def main(ufs_fpath, ufs_var, cafs_fpath, cafs_var, mosaic_fpath, mosaic_var, dateshift=False, date=None):

    ufs_data = myviz.GridData(filepath=ufs_fpath, varname=ufs_var, name="UFS-Arctic Ice Thickness")
    cafs_data = myviz.GridData(filepath=cafs_fpath, varname=cafs_var, name="CAFS Ice Thickness", dateshift=dateshift)
    mosaic_data = myviz.TrackData(filepath=mosaic_fpath, varname=mosaic_var, name="MOSAiC Ice Thickness Track")
    thermo_tend_moving = myviz.IceTendData(filepath=ufs_fpath, trend_type="thermo", name="Thermo Ice Tendencies (along track)")
    thermo_tend_stationary = myviz.IceTendData(filepath=ufs_fpath, trend_type="thermo", name="Thermo Ice Tendencies (all nearest neighbors)")
    dyn_tend = myviz.IceTendData(filepath=ufs_fpath, trend_type="dynamic", name="Dynamic Ice Tendencies")

    mosaic_data.sync_times(ufs_data)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    mode = 'moving'
    ufs_data.extract_trajectory(mosaic_data, mode=mode)
    cafs_data.extract_trajectory(mosaic_data, mode=mode)
    thermo_tend_moving.extract_trajectory(mosaic_data, mode=mode)
    plot_track_comparison(
        track = mosaic_data,
        grids = [cafs_data, ufs_data],
        title = f'Ice Thickness Along MOSAiC Track ({date})',
        colors = ['#808080', '#52a447'],
        ax = axes[0],
    )

    mode = 'stationary'
    thermo_tend_stationary.extract_trajectory(mosaic_data, mode=mode)
    dyn_tend.extract_trajectory(mosaic_data, mode=mode)
    plot_track_comparison(
        track = mosaic_data,
        grids = [dyn_tend, thermo_tend_stationary, thermo_tend_moving],
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

    yyyy = ['2019','2020','2020','2020']
    mm   = ['10',  '02',  '07',  '07'  ]
    dd   = ['28',  '27',  '02',  '09'  ]

    for i in range(len(yyyy)):
        y = yyyy[i]
        m = mm[i]
        d = dd[i]

        if y == '2020':
            dateshift = True
        else:
            dateshift = False

        ufs_fpath    = f'/scratch4/BMC/ufs-artic/Kristin.Barton/stmp/10day_runs/{res}_{y}{m}{d}_{fhrs}HRS/history/iceh_03h.*.nc'
        cafs_fpath   = f'/scratch3/BMC/ufs-artic/amy/CAFS/REB2.{y}-{m}-{d}.nc'
        mosaic_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/mosaic/MOSAiC_Calc_IceThick.txt'


        main(ufs_fpath, ufs_var, cafs_fpath, cafs_var, mosaic_fpath, mosaic_var, dateshift=dateshift, date=f"{y}/{m}/{d}")
