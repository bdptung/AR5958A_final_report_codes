'''
======================================================================================================================
PROMPT:
======================================================================================================================
'''

# save_resume.py
import os, random
import numpy as np
import torch

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
    best_val: float = None,
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
        "best_val": best_val,
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state,
        "extra": extra or {},
        "pytorch_version": torch.__version__,
    }

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
    map_location = "cpu",
    strict: bool = True,
):
    """
    Load a checkpoint and restore model/optimizer/scheduler/scaler and RNG states.
    Returns a small dict with resume info: epoch, global_step, best_val, extra
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # torch.load's weights_only=True exists in newer PyTorch; fall back if not
    try:
        ckpt = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=map_location)

    # Model first
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
        torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda_all"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_val": ckpt.get("best_val", None),
        "extra": ckpt.get("extra", {}),
        "pytorch_version": ckpt.get("pytorch_version"),
    }


# --- training setup ---
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
scaler = torch.cuda.amp.GradScaler()

# --- resume if available ---
from save_resume import save_training_state, load_training_state

ckpt_path = "checkpoints/runA.pt"
start_epoch, global_step, best_val = 0, 0, float("inf")

if os.path.exists(ckpt_path):
    info = load_training_state(
        ckpt_path, model, optimizer, scheduler, scaler, map_location=device
    )
    start_epoch = info["epoch"] + 1          # continue *after* last saved epoch
    global_step = info["global_step"]
    best_val = info["best_val"] if info["best_val"] is not None else best_val
    print(f"Resumed from {ckpt_path} @ epoch {start_epoch}, step {global_step}")

# --- training loop ---
for epoch in range(start_epoch, num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss = compute_loss(model, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

    # validate…
    val_metric = evaluate(model, val_loader)

    # track best
    if val_metric < best_val:
        best_val = val_metric

    # save at end of each epoch (or also best-only)
    save_training_state(
        ckpt_path, model, optimizer, scheduler, scaler,
        epoch=epoch, global_step=global_step, best_val=best_val,
        extra={"note": "runA", "val_metric": float(val_metric)}
    )
