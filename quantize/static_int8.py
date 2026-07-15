"""
Produce a standard dynamic INT8 model from a pretrained ELUNet checkpoint.
Usage: python -m quantize.static_int8
Produces: checkpoints/elunet_int8.pth  (state_dict of the quantized model)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from model import ELUNet

CKPT_IN  = "checkpoints/elunet_sim_fp32.pth"
CKPT_OUT = "checkpoints/elunet_sim_int8.pth"


def make_int8(ckpt_in: str = CKPT_IN) -> nn.Module:
    model = ELUNet()
    model.load_state_dict(torch.load(ckpt_in, map_location="cpu", weights_only=True))
    model.eval()
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d}, dtype=torch.qint8
    )
    return model_int8


if __name__ == "__main__":
    m = make_int8()
    torch.save(m.state_dict(), CKPT_OUT)
    print(f"Saved {CKPT_OUT}")

    dummy = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = m(dummy)
    print(f"Output shape: {out.shape}")
