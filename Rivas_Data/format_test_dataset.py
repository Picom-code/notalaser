# prep for rivas test data

import numpy as np
from pathlib import Path
from scipy.io import loadmat
import random

# === Config ===
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Test Dataset"
OUTPUT_DIR = BASE_DIR.parent / "Test_Dataset_Processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# === Normalization ===
def normalize_minmax(arr):
    arr = arr.astype(np.float32)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def normalize_fixed_range(arr, fixed_min=0, fixed_max=255):
    arr = arr.astype(np.float32)
    return np.clip((arr - fixed_min) / (fixed_max - fixed_min), 0, 1)

# === NaN Replacement ===
def replace_nans_band(data):
    height, width = data.shape
    result = data.copy()

    for y in range(height):
        for x in range(width):
            if np.isnan(data[y, x]):
                y_start = max(0, y - 5)
                y_end = min(height, y + 6)
                surrounding_vals = data[y_start:y_end, x]
                valid_vals = surrounding_vals[~np.isnan(surrounding_vals)]

                if len(valid_vals) > 0:
                    min_val, max_val = valid_vals.min(), valid_vals.max()
                    result[y, x] = np.random.uniform(min_val, max_val)
                else:
                    result[y, x] = 0
    return result

# === Target Keys ===
TARGET_KEYS = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
    'B13lo', 'B13hi', 'B14lo', 'B14hi', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
    'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30',
    'B31', 'B32', 'B33', 'B34', 'B35', 'B36'
]

# === Pass 1: Extract Bands and Save Normalized
for folder in DATASET_DIR.iterdir():
    if not folder.is_dir():
        continue

    mat_file = next((f for f in folder.iterdir() if f.name.endswith("full.mat")), None)
    x_file = next((f for f in folder.iterdir() if f.name.endswith("tmpclass.mat")), None)

    if not mat_file:
        print(f"No full.mat file found in: {folder.name}")
        continue

    print(f"\nLoading: {mat_file.name} from {folder.name}")
    data = loadmat(mat_file)
    folder_output = OUTPUT_DIR / folder.name
    folder_output.mkdir(exist_ok=True)

    for key in TARGET_KEYS:
        if key not in data:
            continue
        if key.lower().endswith(("hkm", "qkm")):
            continue

        value = data[key]
        if isinstance(value, np.ndarray) and 'Rad' in value.dtype.names:
            try:
                rad = value['Rad'][0][0]
                if isinstance(rad, np.ndarray) and rad.ndim >= 2:
                    norm_rad = normalize_minmax(rad)
                    np.save(folder_output / f"{key}.npy", norm_rad)
                    print(f"Saved: {key}.npy")
                else:
                    print(f"{key}.Rad is not a valid 2D array")
            except Exception as e:
                print(f"Could not process {key}.Rad: {e}")
        else:
            print(f"{key} does not have a Rad field or is not structured")

    # Save x.npy from tmpclass.mat if present
    if x_file:
        try:
            print(f"Also loading x from: {x_file.name}")
            x_data = loadmat(x_file).get("x", None)
            if x_data is not None:
                if isinstance(x_data, np.ndarray) and x_data.shape == (1, 1):
                    x_data = x_data[0][0]
                x_data = x_data.astype(np.float32)
                x_normalized = normalize_fixed_range(x_data)
                np.save(folder_output / "x.npy", x_normalized)
                print(f"Saved: x.npy — shape={x_normalized.shape}")
            else:
                print(f"Variable 'x' not found in {x_file.name}")
        except Exception as e:
            print(f"Could not process x: {e}")

# === Pass 2: Fix NaNs in Bands
print("\nPass 2: Replacing NaNs in processed bands...")
for folder in OUTPUT_DIR.iterdir():
    if not folder.is_dir():
        continue

    for band_file in folder.glob("B*.npy"):
        try:
            data = np.load(band_file)
            if np.isnan(data).any():
                print(f"Fixing NaNs in {band_file}")
                cleaned = replace_nans_band(data)
                np.save(band_file, cleaned)
        except Exception as e:
            print(f"Error processing {band_file}: {e}")

print("\nAll processing complete.")
