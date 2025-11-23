import os
import random
import numpy as np
import torch
import rasterio


class EarlyStopper:
    def __init__(self, patience=7, min_rel_delta=0.01):
        self.patience = patience
        self.min_rel_delta = min_rel_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, val_metric):  # lower is better (e.g., MAE)
        if self.best is None or (self.best - val_metric) / max(self.best, 1e-8) > self.min_rel_delta:
            self.best = val_metric
            self.bad_epochs = 0
            return False  # don't stop
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience


def _unwrap_model(model):
    # Works for nn.DataParallel and nn.DistributedDataParallel
    return model.module if hasattr(model, "module") else model

def save_training_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    epoch: int = 0,
    global_step: int = 0,
    prev_losses: list[list[float]] = None,
    ema = None,
    extra: dict = None,
):
    """
    Save everything needed to resume training exactly.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    # RNG states for reproducibility on resume
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda_all"] = torch.cuda.get_rng_state_all()

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "prev_losses": prev_losses,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema_model": None,
        "rng_state": rng_state,
        "extra": extra or {},
        "pytorch_version": torch.__version__,
    }
    if ema:
        ema.apply_shadow()
        try:
            payload["ema_model"] = _unwrap_model(model).state_dict()
        finally:
            ema.restore()


    # Use a temporary file then atomic move to avoid partial writes
    tmp = filepath + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, filepath)

def load_training_state(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler = None,
    scaler: torch.cuda.amp.GradScaler = None,
    map_location = "cuda",
    strict: bool = True,
    load_ema: bool = False,
):
    """
    Load a checkpoint and restore model/optimizer/scheduler/scaler and RNG states.
    Returns a small dict with resume info: epoch, global_step, prev_losses, extra
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # torch.load's weights_only=True exists in newer PyTorch; fall back if not
    try:
        ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=map_location)

    # Model first
    if load_ema and 'ema_model' in ckpt and ckpt["ema_model"] is not None:
        _unwrap_model(model).load_state_dict(ckpt["ema_model"], strict=strict)
    else:
        _unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)

    # Optimizer / scheduler / scaler if present
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # Restore RNG states (important for determinism when resuming mid-epoch)
    rng = ckpt.get("rng_state", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"].type(torch.ByteTensor))
    if torch.cuda.is_available() and rng.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all([x.type(torch.ByteTensor) for x in rng["torch_cuda_all"]])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "prev_losses": ckpt.get("prev_losses", None),
        "extra": ckpt.get("extra", {}),
        "pytorch_version": ckpt.get("pytorch_version"),
    }

def load_model(
    filepath: str,
    model_class: torch.nn.Module,
    strict: bool = True,
    load_ema: bool = True,
    map_location = "cuda",
    in_channels = None
):
    """
    Load a checkpoint and restore model/optimizer/scheduler/scaler and RNG states.
    Returns a small dict with resume info: epoch, global_step, prev_losses, extra
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # torch.load's weights_only=True exists in newer PyTorch; fall back if not
    try:
        ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=map_location)
    extra = ckpt.get("extra", {})
    ic = extra['in_channels'] if 'in_channels' in extra else 3
    if in_channels: ic = in_channels
    model = model_class(
        in_channels = ic, 
        out_channels = extra['out_channels'] if 'out_channels' in extra else 1, 
        M = extra['M'] if 'M' in extra else 32,
        sigmoid = extra['sigmoid'] if 'sigmoid' in extra else False
    )

    if load_ema and 'ema_model' in ckpt and ckpt["ema_model"] is not None:
        _unwrap_model(model).load_state_dict(ckpt["ema_model"], strict=strict)
    else:
        _unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)
    return {
        "model": model,
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "prev_losses": ckpt.get("prev_losses", None),
        "extra": ckpt.get("extra", {}),
        "pytorch_version": ckpt.get("pytorch_version"),
    }


def read_raster(filepath: str):
    with rasterio.open(filepath) as src:
        # read first band (index starts at 1 in rasterio)
        data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return data, transform, crs, nodata

def read_raster_multi(filepath: str, bands: int = 3):
    data = []
    with rasterio.open(filepath) as src:
        # read first band (index starts at 1 in rasterio)
        for i in range(1, bands+1):
            data.append(src.read(i))
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    return data, transform, crs, nodata

def write_raster(
    data: np.ndarray,
    out_path: str,
    transform,
    crs,
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
