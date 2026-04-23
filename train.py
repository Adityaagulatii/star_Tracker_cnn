"""
train.py — train one model and save the best checkpoint.

Usage:
    python train.py --model unet
    python train.py --model mobileunet
    python train.py --model elunet
    python train.py --model unet --overfit     # sanity check on 10 samples
"""

import argparse, csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import StarDataset
from loss    import SegLoss
from model   import UNet, MobileUNet, ELUNet

MODELS = {'unet': UNet, 'mobileunet': MobileUNet, 'elunet': ELUNet}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nModel: {args.model}  |  Device: {device}")

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
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model     = MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = SegLoss()

    ckpt = f"checkpoints/{args.model}_best.pth"
    log  = f"checkpoints/{args.model}_log.csv"
    Path("checkpoints").mkdir(exist_ok=True)
    best_val = float('inf')

    with open(log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for images, segs in train_loader:
            images, segs = images.to(device), segs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), segs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, segs in val_loader:
                images, segs = images.to(device), segs.to(device)
                val_loss += criterion(model(images), segs).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f}  val={val_loss:.4f}")

        with open(log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)
            print(f"  >> checkpoint saved  (val={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val:.4f} -> {ckpt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      choices=list(MODELS), required=True)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--overfit',    action='store_true')
    train(parser.parse_args())
