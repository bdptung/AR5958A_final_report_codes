import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---------------------------
# 1) Minimal U-Net definition
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), # inplace=True
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), # inplace=True
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, M=32, sigmoid=False):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, M)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(M, M*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(M*2, M*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(M*4, M*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(M*8, M*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(M*16, M*8, 2, stride=2)
        self.dec4 = DoubleConv(M*16, M*8)
        self.up3 = nn.ConvTranspose2d(M*8, M*4, 2, stride=2)
        self.dec3 = DoubleConv(M*8, M*4)
        self.up2 = nn.ConvTranspose2d(M*4, M*2, 2, stride=2)
        self.dec2 = DoubleConv(M*4, M*2)
        self.up1 = nn.ConvTranspose2d(M*2, M, 2, stride=2)
        self.dec1 = DoubleConv(M*2, M)

        # Output
        self.out = nn.Conv2d(M, out_channels, 1)
        self.sigmoid = sigmoid

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
        if self.sigmoid: return torch.sigmoid(logits) # sigmoid
        return logits  # NOTE: raw logits (use BCEWithLogitsLoss)


# ---------------------------
# 2) make dataset from raster
# ---------------------------
class SegDataset(Dataset):
    """
    Expects:
      raster_files: /path/to/images/  (RGB images)
    """
    def __init__(self, inp_raster, out_rasters, indices, size=512, offset=[0, 0], terrain_idx=None):
        self.inputs = inp_raster
        self.outputs = out_rasters
        self.size = size
        self.offset = offset
        self.indices = indices
        self.ylen = math.floor((inp_raster[0].shape[0] - self.offset[0]) / self.size)
        self.xlen = math.floor((inp_raster[0].shape[1] - self.offset[1]) / self.size)
        self.terrain_idx = terrain_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, v):
        # i = self.indices[v]
        # y = (i % self.ylen) * self.size + self.offset[0]
        # x = math.floor(i / self.ylen) * self.size + self.offset[1]
        y, x = self.indices[v]

        inp_stack = []
        for raster in self.inputs:
            inp_stack.append(raster[y : y+self.size, x : x+self.size])
        if self.terrain_idx is not None:
            inp_stack[self.terrain_idx] = inp_stack[self.terrain_idx] - torch.min(inp_stack[self.terrain_idx])
        inp = torch.stack(inp_stack, axis=0)  # (3, H, W)

        out_stack = []
        for raster in self.outputs:
            out_stack.append(raster[y : y+self.size, x : x+self.size])
        out = torch.stack(out_stack, axis=0)  # (3, H, W)
        return inp, out
