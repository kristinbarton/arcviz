import myviz
import logging

def main(ufs_fpath, cafs_fpath, mosaic_fpath):
    ufs_data = myviz.GridData(filepath=ufs_fpath, varname='hi_h', name="UFS-Arctic Data")
    cafs_data = myviz.GridData(filepath=cafs_fpath, varname='hi_h', name="CAFS Data")
    mosaic_data = myviz.TrackData(mosaic_fpath, name="MOSAiC Track")
    print("Data setup complete!")

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
    )

    res  = 'C918'
    yyyy = '2019'
    mm   = '10'
    dd   = '28'
    fhrs = '240'

    ufs_fpath    = f'/scratch4/BMC/ufs-artic/Kristin.Barton/stmp/10day_runs/{res}_{yyyy}{mm}{dd}_{240}HRS/history/iceh_03h.*.nc'
    cafs_fpath   = f'/scratch3/BMC/ufs-artic/amy/CAFS/REB2.{yyyy}-{mm}-{dd}.nc'
    mosaic_fpath = f'/scratch4/BMC/ufs-artic/Kristin.Barton/files/comparisons/mosaic/MOSAiC_Calc_IceThick.txt'

    main(ufs_fpath, cafs_fpath, mosaic_fpath)