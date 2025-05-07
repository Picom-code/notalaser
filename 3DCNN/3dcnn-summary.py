import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    precision_score, f1_score, r2_score
)
import sys

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "big_epoch_4_finished.pth"
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)
LOG_FILE = PREDICTIONS_DIR / "evaluation_log.txt"

# === Logging setup ===
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_FILE)

# === Model & Data Config ===
BANDS = [f"B{i}" for i in range(1, 13)] + ["B13lo", "B13hi", "B14lo", "B14hi"] + [f"B{i}" for i in range(15, 37)]
NUM_BANDS = len(BANDS)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Dataset ===
class CubeRegressionDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        self.inputs, self.labels, self.coords = self.load_data()

    def load_data(self):
        band_arrays = [np.load(self.folder / f"{band}.npy").astype(np.float32) for band in BANDS]
        cube = np.stack(band_arrays, axis=0)
        label = np.load(self.folder / "x.npy").astype(np.float32)
        mask = ~np.isnan(label)
        indices = np.argwhere(mask)

        samples, targets, coords = [], [], []
        H, W = label.shape
        for y, x in indices:
            if y - 2 < 0 or y + 3 > H or x - 2 < 0 or x + 3 > W:
                continue
            patch = cube[:, y-2:y+3, x-2:x+3]
            if patch.shape != (NUM_BANDS, 5, 5) or np.isnan(patch).any():
                continue
            samples.append(np.expand_dims(patch, 0))
            targets.append(label[y, x])
            coords.append((y, x))
        return np.stack(samples), np.array(targets), coords

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])

# === Model ===
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)

# === Evaluation ===
def evaluate_model_on_test_folders():
    model = CubeRegressor().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_folders = sorted([f for f in PROCESSED_DIR.iterdir() if f.is_dir()])

    for folder in all_folders:
        dataset = CubeRegressionDataset(folder)
        if len(dataset) == 0:
            continue
        dataloader = DataLoader(dataset, batch_size=64)

        preds, labels, coords = [], [], dataset.coords
        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                out = model(xb)
                preds.append(out.cpu().numpy())
                labels.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        coords = np.array(coords)

        mse = mean_squared_error(labels, preds)
        r2 = r2_score(labels, preds)
        weighted_mse = np.mean((labels**2) * (preds - labels)**2)
        binary_preds = (preds > 0.5).astype(int)
        binary_labels = (labels > 0.5).astype(int)
        accuracy = accuracy_score(binary_labels, binary_preds)

        print(f"{folder.name} — Samples: {len(preds)}")
        print(f"   MSE: {mse:.6f}")
        print(f"   R² Score: {r2:.3f}")
        print(f"   Weighted MSE: {weighted_mse:.6f}")
        print(f"   Accuracy: {accuracy:.3f}")

        # === Save side-by-side grayscale image ===
        label_shape = np.load(folder / "x.npy").shape
        label_img = np.load(folder / "x.npy")
        pred_img = np.full(label_shape, np.nan)

        for (y, x), pred in zip(coords, preds):
            pred_img[y, x] = pred

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(label_img, cmap='gray')
        axs[0].set_title('Label (x.npy)')
        axs[0].axis('off')

        axs[1].imshow(pred_img, cmap='gray')
        axs[1].set_title('Model Prediction')
        axs[1].axis('off')

        plt.tight_layout()
        out_path = PREDICTIONS_DIR / f"{folder.name}_comparison.png"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved: {out_path.name}")

if __name__ == "__main__":
    evaluate_model_on_test_folders()
    print("Evaluation complete. Log written to:", LOG_FILE.name)
