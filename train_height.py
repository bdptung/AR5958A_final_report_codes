# unet_512x512_minimal.py
import math
import os
from glob import glob
import random
from PIL import Image

import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.loss import LOSS_FN_H
from utils.model import SegDataset, UNet
from utils.utils import save_training_state, read_raster, read_raster_multi, write_raster

DAY = '2025_11_20'
M = 32

def seed_everything(deterministic: bool = True, seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN / kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)  # error if an op is nondeterministic
        # Make cuBLAS deterministic (needed for some GEMMs)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
        # For bitwise-stable math (optional—slower but safer)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    g = torch.Generator()
    g.manual_seed(42)
    return g
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = seed_everything(deterministic = True, seed=42)
# torch.backends.cudnn.benchmark = True

# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Paths (change to your folders)
    train_images = ["./data/zurich_mask.tif", "./data/zurich_osm.tif"]
    mask_images  = ["./data/zurich.tif"]
    train_rasters = []
    mask_rasters  = []
    for file in train_images:
        if file.endswith('img.tif'):
            data_rasters, _, _, _ = read_raster_multi(file)
            for data in data_rasters:
                train_rasters.append(torch.tensor(data, device=device, dtype=torch.float32))
        else:
            data, _, _, _ = read_raster(file)
            train_rasters.append(torch.tensor(data, device=device, dtype=torch.float32))
    for file in mask_images:
        data, _, _, _ = read_raster(file)
        mask_rasters.append(torch.tensor(data, device=device, dtype=torch.float32))
    data_shape = data.shape

    model = UNet(in_channels=len(train_rasters), out_channels=len(mask_images), M=M).to(device)
    # ema = EMA(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device=device, init_scale=2**10, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)
    last_loss = [[], [], []]

    epochs = 100
    size = 512
    scan = 128
    skip = 50
    batch_size = 8
    test_size = 0.3

    offsets = []
    for i in range(skip):
        for j in range(skip):
            offsets.append([i, j])
    random.shuffle(offsets)
    offsets = offsets[:epochs * 2 + 5]
    print(offsets)

    steps_per_epoch = 10000000
    for epoch in range(1, epochs+1):
        offset = offsets[epoch]
        count = 0
        i = offset[0]
        while (i + size) < data_shape[0]:
            j = offset[1]
            while (j + size) < data_shape[1]:
                count += 1
                j += skip
            i += skip
        steps_per_epoch = min(math.ceil(count / batch_size), steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr  = 7e-4,          # peak LR; scale with batch size
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start = 0.2,         # peak LR at 20%
        div_factor = 25,         # initial_lr = max_lr / div_factor 
        final_div_factor = 1e3,  # final_lr   = max_lr / final_div_factor
        anneal_strategy="cos",
        cycle_momentum=False     # no cycle
    )

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        offset_train = offsets[epoch * 2]
        offset_test = offsets[epoch * 2 + 1]
        print('_____', offset_train, offset_test, steps_per_epoch)
        train_idx = []
        i = offset_train[0]
        while (i + size) < data_shape[0]:
            j = offset_train[1]
            while (j + size) < data_shape[1]:
                train_idx.append([i, j])
                j += skip
            i += skip

        test_idx = []
        i = offset_test[0]
        while (i + size) < data_shape[0]:
            j = offset_test[1]
            while (j + size) < data_shape[1]:
                test_idx.append([i, j])
                j += skip * 4
            i += skip * 4

        train_ds = SegDataset(train_rasters, mask_rasters, train_idx, size=size, offset=offset_train, terrain_idx=None)
        val_ds = SegDataset(train_rasters, mask_rasters, test_idx, size=size, offset=offset_test, terrain_idx=None)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)

        # train
        count = 0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            inp, out = batch
            opt.zero_grad(set_to_none=True)


            with torch.autocast(device_type="cuda"):
                logits = model(inp)
                loss = LOSS_FN_H(logits, inp, out, scan)
                print(f'e{epoch} train loss', loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            # ema.update()
            scaler.update()
            scheduler.step()  # step every optimizer step

            running_loss += loss.item() * inp.size(0)
            count += inp.size(0)
        train_loss = running_loss / max(1, count)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= steps_per_epoch / 2:
                    break
                imgs, masks = batch
                logits = model(imgs)
                loss = LOSS_FN_H(logits, imgs, masks)
                print(f'e{epoch} test loss', loss.item())
                val_loss += loss.item() * imgs.size(0)
        val_loss /= max(1, len(val_loader.dataset))
        val_iou  /= max(1, len(val_loader))

        last_loss[0].append(train_loss)
        last_loss[1].append(val_loss)
        last_loss[2].append(val_iou)

        print(f"[{epoch:02d}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")


        if not os.path.exists(f'trained_model_height/{DAY}'):
            os.mkdir(f'trained_model_height/{DAY}')
        f = f'trained_model_height/{DAY}/model.{epoch}.pth'
        save_training_state(f, model, opt, scheduler, scaler, epoch, 0, last_loss, 
            # ema=ema, 
            extra={'in_channels': len(train_rasters), 'out_channels': len(mask_images), 'M': M}
        )

    # Quick inference demo on one validation sample
    if len(val_ds) > 0:
        img, _ = val_ds[100]
        model.eval()
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(device))
            pred = logit[0,0].cpu().numpy()
            prob  = torch.sigmoid(logit)[0,0].cpu().numpy()  # (512,512) float in [0,1]
            pred  = (prob > 0.5).astype(np.uint8)*255
            pred  = pred[scan : -scan, scan : -scan]
        Image.fromarray(pred).save("pred_example.png")
        print("Saved example prediction to pred_example.png")

if __name__ == "__main__":
    main()
