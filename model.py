import torch
import torch.nn as nn


# ==========================
# Model Blocks
# ==========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # in_ch because skip + up = in_ch

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)         # concat on channel dim
        return self.conv(x)


# ==========================
# UNet Model
# ==========================

class UNet(nn.Module):
    """
    Mini UNet for star segmentation.

    Input:  (B, 1, H, W)  grayscale star image
    Output: (B, 1, H, W)  segmentation map, values in [0, 1]
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()

        self.inc        = DoubleConv(in_ch,       base_ch)
        self.down1      = Down(base_ch,            base_ch * 2)
        self.down2      = Down(base_ch * 2,        base_ch * 4)

        # Bottleneck also downsamples so skip sizes align in decoder
        self.bottleneck = Down(base_ch * 4,        base_ch * 8)

        self.up1        = Up(base_ch * 8,          base_ch * 4)
        self.up2        = Up(base_ch * 4,          base_ch * 2)
        self.up3        = Up(base_ch * 2,          base_ch)

        self.outc       = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.bottleneck(x3)

        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)

        return torch.sigmoid(self.outc(x))


# ==========================
# MobileUNet Blocks
# ==========================

class InvertedResidual(nn.Module):
    """
    MobileNetV2-style block. Three steps:
      1. Expand   — 1x1 conv, multiplies channels by expand_ratio
      2. Depthwise — 3x3 conv, one filter per channel (very cheap)
      3. Project  — 1x1 conv, squeezes back down to out_ch

    Depthwise conv is the key saving: instead of every output channel
    looking at every input channel, each filter only looks at ONE channel.
    Much fewer multiplications.
    """
    def __init__(self, in_ch, out_ch, expand_ratio=4):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.block = nn.Sequential(
            # 1. Expand
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # 2. Depthwise (groups=mid_ch means one filter per channel)
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # 3. Project
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = (in_ch == out_ch)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out   # skip connection when shapes match
        return out


class MobileDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            InvertedResidual(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class MobileUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = InvertedResidual(in_ch, out_ch)   # in_ch after skip concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ==========================
# MobileUNet Model
# ==========================

class MobileUNet(nn.Module):
    """
    Same UNet structure as UNet, but DoubleConv replaced with
    InvertedResidual blocks — much fewer parameters.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)  values in [0, 1]
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=16):
        super().__init__()

        self.inc        = InvertedResidual(in_ch,      base_ch)
        self.down1      = MobileDown(base_ch,          base_ch * 2)
        self.down2      = MobileDown(base_ch * 2,      base_ch * 4)

        self.bottleneck = MobileDown(base_ch * 4,      base_ch * 8)

        self.up1        = MobileUp(base_ch * 8,        base_ch * 4)
        self.up2        = MobileUp(base_ch * 4,        base_ch * 2)
        self.up3        = MobileUp(base_ch * 2,        base_ch)

        self.outc       = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.bottleneck(x3)

        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)

        return torch.sigmoid(self.outc(x))


# ==========================
# ELUNet Blocks
# ==========================

class SingleConv(nn.Module):
    """
    One conv instead of two. Even lighter than DoubleConv.
    Adds BatchNorm for training stability.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ELUDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ELUUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = SingleConv(in_ch, out_ch)    # in_ch after skip concat

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ==========================
# ELUNet Model
# ==========================

class ELUNet(nn.Module):
    """
    Lightest of the three. Single conv per block + smaller base channels.
    Trains fastest, fewest parameters.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)  values in [0, 1]
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=8):
        super().__init__()

        self.inc        = SingleConv(in_ch,        base_ch)
        self.down1      = ELUDown(base_ch,          base_ch * 2)
        self.down2      = ELUDown(base_ch * 2,      base_ch * 4)

        self.bottleneck = ELUDown(base_ch * 4,      base_ch * 8)

        self.up1        = ELUUp(base_ch * 8,        base_ch * 4)
        self.up2        = ELUUp(base_ch * 4,        base_ch * 2)
        self.up3        = ELUUp(base_ch * 2,        base_ch)

        self.outc       = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.bottleneck(x3)

        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)

        return torch.sigmoid(self.outc(x))


# ==========================
# Quick test
# ==========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 1, 256, 256).to(device)

    for name, Model in [("UNet", UNet), ("MobileUNet", MobileUNet), ("ELUNet", ELUNet)]:
        model  = Model().to(device)
        with torch.no_grad():
            out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:<12} output={out.shape}  params={params:>8,}  ({params*4/1024/1024:.2f} MB)")
