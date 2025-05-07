import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau



# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path().resolve()
PROCESSED_DIR  = BASE_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PROCESSED_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 32768      #adjust for OOM
BIG_EPOCHS    = 50         #many mini-epochs via sampling
LEARNING_RATE = 1e-3
NUM_SAMPLES   = 200_000    #patches per epoch

DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
print(f"Using device: {DEVICE} — {device_name}")

BANDS = (
    [f"B{i}" for i in range(1, 13)]
    + ["B13lo", "B13hi", "B14lo", "B14hi"]
    + [f"B{i}" for i in range(15, 37)]
)

loss_log = []


#Dataset: memmap lists, NumPy index array to minimize RAM
#─────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, folders: list[Path], patch_size: int = 5):
        self.half = patch_size // 2
        self.maps   = []
        self.labels = []
        idx_list    = []
        for fi, folder in enumerate(folders):
            maps = [np.load(folder / f"{b}.npy", mmap_mode='r') for b in BANDS]
            self.maps.append(maps)
            lab = np.load(folder / "x.npy", mmap_mode='r')
            self.labels.append(lab)
            H, W = lab.shape
            ys, xs = np.argwhere(~np.isnan(lab)).T
            mask = (ys >= self.half) & (ys < H - self.half) & (xs >= self.half) & (xs < W - self.half)
            ys, xs = ys[mask], xs[mask]
            if ys.size:
                fi_arr = np.full_like(ys, fi)
                idx_list.append(np.stack([fi_arr, ys, xs], axis=1))
        self.indices = np.concatenate(idx_list, axis=0).astype(np.int32)
    def __len__(self):
        return self.indices.shape[0]
    def __getitem__(self, idx: int):
        fi, y, x = self.indices[idx]
        vol = self.maps[int(fi)]
        patch = np.stack([m[y-self.half:y+self.half+1, x-self.half:x+self.half+1] for m in vol], axis=0)
        inp = torch.from_numpy(patch[None].astype(np.float32))
        tgt = torch.tensor(self.labels[int(fi)][y, x], dtype=torch.float32)
        return inp, tgt


#Model definition (no change)
#─────────────────────────────────────────────────────────────────────────────
class CubeRegressor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,        64, kernel_size=3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64,       128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))  #maybe lets try no sigmoid
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


#Loss & Eval
#─────────────────────────────────────────────────────────────────────────────
loss_fn = nn.MSELoss()

def evaluate(model, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            losses.append(loss_fn(model(xb), yb).item())
    return float(np.mean(losses))


#Data discovery & loaders with subsampling
#─────────────────────────────────────────────────────────────────────────────
all_dirs = ([PROCESSED_DIR] if all((PROCESSED_DIR / f"{b}.npy").exists() for b in BANDS)
            and (PROCESSED_DIR / "x.npy").exists()
            else sorted(d for d in PROCESSED_DIR.iterdir() if d.is_dir()))
complete = [d for d in all_dirs if all((d / f"{b}.npy").exists() for b in BANDS)
            and (d / "x.npy").exists()]
val_count   = 5
train_count = min(100, len(complete) - val_count)
train_dirs  = complete[:train_count]
val_dirs    = complete[train_count:train_count+val_count]

train_ds, val_ds = PatchDataset(train_dirs), PatchDataset(val_dirs)
train_sampler = RandomSampler(train_ds, replacement=True, num_samples=NUM_SAMPLES)
train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                           num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=1)


#Compile model & setup optimizer + scheduler
#─────────────────────────────────────────────────────────────────────────────
model     = torch.compile(CubeRegressor().to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


#Training loop
#─────────────────────────────────────────────────────────────────────────────
for epoch in range(1, BIG_EPOCHS+1):
    model.train(); total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            loss = loss_fn(model(xb), yb)
        loss.backward(); optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)
    avg_val   = evaluate(model, val_loader)
    scheduler.step(avg_val)
    loss_log.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val})
    print(f"Epoch {epoch}/{BIG_EPOCHS}: train={avg_train:.6f}, val={avg_val:.6f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch}.pth")
    df = pd.DataFrame(loss_log)
    plt.figure(); plt.plot(df['epoch'], df['train_loss'], label='Train'); plt.plot(df['epoch'], df['val_loss'], '--', label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.savefig(CHECKPOINT_DIR / 'loss_plot.png'); plt.close()


#Post-training eval
#─────────────────────────────────────────────────────────────────────────────
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        preds = model(xb).cpu().numpy(); all_preds.append(preds); all_targets.append(yb.cpu().numpy())
all_preds   = np.concatenate(all_preds); all_targets = np.concatenate(all_targets)
mse = np.mean((all_preds - all_targets)**2); mae = np.mean(abs(all_preds - all_targets)); r2 = 1 - mse/np.var(all_targets)
print(f"Validation MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")
