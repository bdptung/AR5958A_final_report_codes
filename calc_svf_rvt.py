import numpy as np
from rvt.vis import sky_view_factor
from utils.utils import read_raster, write_raster
import time

def svf_rvt(dem_array: np.ndarray, cell_size: float, n_dirs: int = 16, radius: int = 100):
    """
    Compute SVF using rvt-py (Relief Visualization Toolbox) directly on a NumPy DEM.

    Returns a float32 array in [0,1].
    """
    result = sky_view_factor(
        dem=dem_array.astype("float32", copy=False),
        resolution=float(cell_size),
        compute_svf=True,
        compute_asvf=False,
        compute_opns=False,
        svf_n_dir=int(n_dirs),
        svf_r_max=int(radius),
    )

    return result['svf']

if __name__ == '__main__':
    data_location = 'zurich'
    data_location = 'sg'

    bld_data, bld_transf, crs,  bld_nodata = read_raster(f"data/{data_location}.tif")
    mask_data, mask_transf, _, mask_nodata = read_raster(f"data/{data_location}_mask.tif")
    terr_data, terr_transf, _, terr_nodata = read_raster(f"data/{data_location}_terrain.tif")

    STEP_DEG = 1
    SCAN_RADIUS = 128
    h_raster = bld_data + terr_data
    start = time.time()
    print(h_raster.shape)
    calc_svf_all = sky_view_factor(
        dem=h_raster.astype("float32", copy=False),
        resolution=1,
        compute_svf=True,
        compute_asvf=False,
        compute_opns=False,
        svf_n_dir=int(360 / STEP_DEG),
        svf_r_max=SCAN_RADIUS,
    )
    svf = calc_svf_all * (1 - mask_data)
    print(svf)
    timing = time.time() - start
    with open('./timing', 'w') as f:
        f.write(f'SVF for {data_location} finished in {timing} seconds')

    write_raster(svf, './data/{data_location}_svf_1.tif', bld_transf, crs)