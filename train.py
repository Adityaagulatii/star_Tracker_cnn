"""
train.py — train one model and save the best checkpoint.

ELUNet uses dual output (seg + dist map) with DualLoss for sub-pixel centroiding.
UNet and MobileUNet use single output with SegLoss for comparison.

Usage:
    python train.py --model elunet       # best model — dual output
    python train.py --model unet
    python train.py --model mobileunet
"""

import argparse, csv, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import StarDataset
from loss    import SegLoss, DualLoss
from model   import UNet, MobileUNet, ELUNet

MODELS = {'unet': UNet, 'mobileunet': MobileUNet, 'elunet': ELUNet}


def train(args):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dual_out = (args.model == 'elunet')   # ELUNet uses dual output + DualLoss
    print(f"\nModel: {args.model}  |  Device: {device}  |  Dual-output: {dual_out}")

    train_ds = StarDataset('data/train')
    val_ds   = StarDataset('data/val')

    if args.overfit:
        train_ds = Subset(train_ds, list(range(10)))
        val_ds   = Subset(val_ds,   list(range(10)))
        epochs   = 100
        print("Overfit mode: 10 samples, 100 epochs")
    else:
        epochs = args.epochs

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DualLoss(dist_weight=2.5) if dual_out else SegLoss()

    ckpt = f"checkpoints/{args.model}_best.pth"
    log  = f"checkpoints/{args.model}_log.csv"
    Path("checkpoints").mkdir(exist_ok=True)
    best_val = float('inf')

    with open(log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, segs, dists in train_loader:
            images = images.to(device)
            segs   = segs.to(device)
            dists  = dists.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, segs, dists)[0] if dual_out else criterion(pred, segs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, segs, dists in val_loader:
                images = images.to(device)
                segs   = segs.to(device)
                dists  = dists.to(device)
                pred = model(images)
                loss = criterion(pred, segs, dists)[0] if dual_out else criterion(pred, segs)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f}  val={val_loss:.4f}  {time.time()-t0:.1f}s")

        with open(log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            print(f"  >> checkpoint saved (val={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f} -> {ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      choices=list(MODELS), required=True)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--overfit',    action='store_true')
    train(parser.parse_args())
