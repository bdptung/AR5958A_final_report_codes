'''
======================================================================================================================
PROMPT: write a function to read a raster file (1 band of data) as numpy 2D array
======================================================================================================================
'''

import rasterio
import numpy as np

def read_raster_as_array(filepath: str):
    """
    Read a single-band raster file into a NumPy 2D array.

    Parameters
    ----------
    filepath : str
        Path to the raster file (e.g., GeoTIFF).

    Returns
    -------
    data : np.ndarray
        2D array of raster values (dtype preserved from the file).
    transform : affine.Affine
        Affine transform for converting array indices to coordinates.
    crs : dict or rasterio.crs.CRS
        Coordinate reference system of the raster.
    nodata : value
        The nodata value of the raster (may be None).
    """
    with rasterio.open(filepath) as src:
        # read first band (index starts at 1 in rasterio)
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return data, transform, crs, nodata

arr, transform, crs, nodata = read_raster_as_array("dem.tif")

print(arr.shape)        # 2D (rows, cols)
print(crs)              # projection info
print(nodata)           # nodata value


'''
======================================================================================================================
PROMPT: write a function to write a tif file with the above read data, with lzw compression
======================================================================================================================
'''

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

def write_array_to_tif(
    data: np.ndarray,
    out_path: str,
    transform: Affine,
    crs: CRS,
    nodata=None,
    dtype=None,
    compress: str = "LZW",
    bigtiff: str = "IF_SAFER",
    tiled: bool = True,
):
    """
    Write a 2D NumPy array to a single-band GeoTIFF with LZW compression.

    Parameters
    ----------
    data : np.ndarray
        2D array (rows, cols). Can be float or int. Masked arrays supported.
    out_path : str
        Output GeoTIFF path.
    transform : Affine
        Affine geotransform for the raster.
    crs : rasterio.crs.CRS
        Coordinate reference system of the raster.
    nodata : scalar, optional
        Nodata value to store in the file metadata. If data is a masked array
        and nodata is provided, masked elements will be written as nodata.
    dtype : str or np.dtype, optional
        Output dtype. Defaults to data.dtype if not provided.
    compress : {"LZW", ...}
        Compression codec. Default "LZW" (lossless).
    bigtiff : {"YES","NO","IF_NEEDED","IF_SAFER"}
        BigTIFF creation option. Default "IF_SAFER".
    tiled : bool
        Write tiled TIFF (usually better for I/O). Default True.
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")

    # Decide dtype and ensure C-contiguous for speed
    out_dtype = np.dtype(dtype) if dtype is not None else data.dtype
    arr = np.ascontiguousarray(data.astype(out_dtype, copy=False))

    # If it's a masked array and nodata is provided, fill with nodata
    if np.ma.isMaskedArray(arr) and nodata is not None:
        arr = np.ma.filled(arr, nodata)

    height, width = arr.shape

    # TIFF predictor improves LZW compression: 2 for integer, 3 for float
    predictor = 3 if np.issubdtype(out_dtype, np.floating) else 2

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,                     # single band
        "dtype": out_dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "predictor": predictor,
        "tiled": tiled,
        "bigtiff": bigtiff,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

        # If we got a boolean mask and no nodata was provided, we can store an alpha mask
        if np.ma.isMaskedArray(data) and nodata is None:
            # 255 = valid, 0 = masked
            mask = (~data.mask).astype("uint8") * 255
            dst.write_mask(mask)

    return out_path

# read
arr, transform, crs, nodata = read_raster_as_array("dem.tif")

# write (same metadata, LZW compressed)
write_array_to_tif(
    data=arr,
    out_path="dem_copy_lzw.tif",
    transform=transform,
    crs=crs,
    nodata=nodata,   # keep original nodata
)
