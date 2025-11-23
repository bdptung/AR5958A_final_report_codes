from utils.utils import read_raster, write_raster


if __name__ == '__main__':
    bld_data, bld_transf, crs,  bld_nodata = read_raster("data/sg.tif")
    mask_data = (bld_data > 0) * 1
    print(mask_data)
    write_raster(mask_data, 'data/sg_mask.tif', bld_transf, crs)

