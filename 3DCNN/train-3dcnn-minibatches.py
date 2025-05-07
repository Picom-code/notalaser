import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BANDS = [f"B{i}" for i in range(1, 13)] + ["B13lo", "B13hi", "B14lo", "B14hi"] + [f"B{i}" for i in range(15, 37)]
NUM_BANDS = len(BANDS)
BIG_EPOCHS = 3
SUB_EPOCHS = 3
BATCH_SIZE = 128
FOLDER_BATCH_SIZE = 5
FOLDERS_PER_BIG_EPOCH = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\U0001F680 Using device: {DEVICE} — {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU only'}")

loss_log = []

class CubeRegressionDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        print(f"Loading {folder.name}")
        self.inputs, self.labels = self.load_data()

    def load_data(self):
        band_arrays = [np.load(self.folder / f"{band}.npy").astype(np.float32) for band in BANDS]
        cube = np.stack(band_arrays, axis=0)
        label = np.load(self.folder / "x.npy").astype(np.float32)

        mask = ~np.isnan(label)
        indices = np.argwhere(mask)
        samples, targets = [], []

        H, W = label.shape
        for y, x in indices:
            if y - 2 < 0 or y + 3 > H or x - 2 < 0 or x + 3 > W:
                continue
            patch = cube[:, y-2:y+3, x-2:x+3]
            if patch.shape != (NUM_BANDS, 5, 5) or np.isnan(patch).any():
                continue
            samples.append(np.expand_dims(patch, axis=0))
            targets.append(label[y, x])

        X = np.stack(samples)
        y = np.array(targets)
        return X, y

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])
class CubeRegressor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

class WeightedMSELoss(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, preds, targets):
        weights = targets ** self.power
        return torch.mean(weights * (preds - targets) ** 2)

def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            losses.append(loss.item())
    return np.mean(losses)

# Load all folder paths
all_folders = sorted([f for f in PROCESSED_DIR.iterdir() if f.is_dir()])
train_folders = all_folders[:100]
val_folders = random.sample(all_folders[100:116], 5)
val_dataset = ConcatDataset([CubeRegressionDataset(f) for f in val_folders])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
model = CubeRegressor().to(DEVICE)
loss_fn = WeightedMSELoss(power=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for big_epoch in range(1, BIG_EPOCHS + 1):
    print(f"\nBig Epoch {big_epoch}/{BIG_EPOCHS}")
    selected_folders = random.sample(train_folders, FOLDERS_PER_BIG_EPOCH)
    folders_remaining = selected_folders.copy()
    random.shuffle(folders_remaining)

    batches_processed = 0
    total_batches = (FOLDERS_PER_BIG_EPOCH + FOLDER_BATCH_SIZE - 1) // FOLDER_BATCH_SIZE

    while folders_remaining:
        batch_folders = [folders_remaining.pop() for _ in range(min(FOLDER_BATCH_SIZE, len(folders_remaining)))]
        datasets = [CubeRegressionDataset(f) for f in batch_folders]
        batch_dataset = ConcatDataset(datasets)
        train_loader = DataLoader(batch_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        total_loss = 0

        for epoch in range(SUB_EPOCHS):
            epoch_loss = 0
            print(f"Sub-Epoch {epoch+1}/{SUB_EPOCHS} starting...")
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Train Loss (Sub-Epoch {epoch+1}): {avg_epoch_loss:.6f}")
            total_loss += epoch_loss

            if epoch+1 != SUB_EPOCHS:
                loss_log.append({
                "epoch": f"{big_epoch}.{batches_processed+1}.{epoch+1}",
                "train_loss": avg_epoch_loss,
                "val_loss": None
            })

        # Average over all sub-epochs
        avg_train_loss = total_loss / (SUB_EPOCHS * len(train_loader))
        avg_val_loss = evaluate(model, val_loader)

        loss_log.append({
            "epoch": f"{big_epoch}.{batches_processed+1}",
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"Finished Batch {batches_processed+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        batches_processed += 1

        if batches_processed == total_batches // 2:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"big_epoch_{big_epoch}_middle.pth")
            print(f"Saved checkpoint: big_epoch_{big_epoch}_middle.pth")

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"big_epoch_{big_epoch}_finished.pth")
    print(f"Saved checkpoint: big_epoch_{big_epoch}_finished.pth")

# Save loss log and final plot
loss_df = pd.DataFrame(loss_log)
loss_df.to_csv(BASE_DIR / "loss_log.csv", index=False)
plt.figure()
plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Final Loss Progression")
plt.legend()
plt.grid(True)
plt.savefig(CHECKPOINT_DIR / "loss_plot.png")
plt.close()
print("Training complete. Loss plot and CSV saved.")
