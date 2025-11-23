'''
======================================================================================================================
PROMPT: write a simplest u-net cnn training that take in 512x512x3 to predict 512x512x1
======================================================================================================================
'''


# unet_512x512_minimal.py
import os
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

# ---------------------------
# 1) Minimal U-Net definition
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch*8, base_ch*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch*2, base_ch)

        # Output
        self.out = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder (with skip connections)
        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        c5 = self.dec4(u4)
        u3 = self.up3(c5)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.dec3(u3)
        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.dec2(u2)
        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.dec1(u1)

        logits = self.out(c8)  # (N,1,512,512)
        return logits  # NOTE: raw logits (use BCEWithLogitsLoss)

# ---------------------------
# 2) Minimal dataset
# ---------------------------
class SegDataset(Dataset):
    """
    Expects:
      images_dir: /path/to/images/  (RGB images)
      masks_dir:  /path/to/masks/   (grayscale, {0,1} or 0..255)
    Pairs by filename stem (e.g., image 'abc.jpg' matches mask 'abc.png').
    """
    def __init__(self, images_dir, masks_dir, size=512):
        self.images = sorted(glob(os.path.join(images_dir, "*")))
        self.masks  = sorted(glob(os.path.join(masks_dir,  "*")))
        # naive pairing by stem
        by_stem = {}
        for p in self.images:
            by_stem[os.path.splitext(os.path.basename(p))[0]] = {"img": p}
        for p in self.masks:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in by_stem:
                by_stem[stem]["mask"] = p
        self.pairs = [v for v in by_stem.values() if "img" in v and "mask" in v]

        self.img_tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),                 # => [0,1], shape (3,H,W)
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # simple default
        ])
        self.mask_tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),                 # => [0,1], shape (1,H,W)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path = self.pairs[i]["img"]
        mask_path = self.pairs[i]["mask"]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img  = self.img_tf(img)
        mask = self.mask_tf(mask)
        # if mask is 0..255, binarize to 0/1
        mask = (mask > 0.5).float()
        return img, mask

# ---------------------------
# 3) Training / eval utilities
# ---------------------------
def iou_from_logits(logits, targets, thresh=0.5, eps=1e-7):
    # logits: (N,1,H,W) raw
    # targets: (N,1,H,W) {0,1}
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3))
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

# ---------------------------
# 4) Main: wire it up
# ---------------------------
def main():
    # Paths (change to your folders)
    train_images = "./data/train/images"
    train_masks  = "./data/train/masks"
    val_images   = "./data/val/images"
    val_masks    = "./data/val/masks"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SegDataset(train_images, train_masks, size=512)
    val_ds   = SegDataset(val_images,   val_masks,   size=512)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_val_iou = 0.0
    epochs = 10

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                val_iou  += iou_from_logits(logits, masks)

        val_loss /= max(1, len(val_loader.dataset))
        val_iou  /= max(1, len(val_loader))

        print(f"[{epoch:02d}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={val_iou:.4f}")

        # Save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"model": model.state_dict()}, "unet512_best.pt")

    print("Done. Best Val IoU:", best_val_iou)

    # Quick inference demo on one validation sample
    if len(val_ds) > 0:
        img, _ = val_ds[0]
        model.eval()
        with torch.no_grad():
            logit = model(img.unsqueeze(0).to(device))
            prob  = torch.sigmoid(logit)[0,0].cpu().numpy()  # (512,512) float in [0,1]
            pred  = (prob > 0.5).astype(np.uint8)*255
        Image.fromarray(pred).save("pred_example.png")
        print("Saved example prediction to pred_example.png")

if __name__ == "__main__":
    main()
