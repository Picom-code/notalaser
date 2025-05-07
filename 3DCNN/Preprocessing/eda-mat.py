# extract_bands_from_mat.py

import numpy as np
from pathlib import Path
from scipy.io import loadmat

# === Config ===
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Dataset"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# Target fields
TARGET_KEYS = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
    'B13lo', 'B13hi', 'B14lo', 'B14hi', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
    'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30',
    'B31', 'B32', 'B33', 'B34', 'B35', 'B36'
]

MAX_RUNS = 3
run_count = 0

def normalize_minmax(arr):
    arr = arr.astype(np.float32)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

# === Main loop
for folder in DATASET_DIR.iterdir():
    # if run_count >= MAX_RUNS:
    #     print(f"\nReached MAX_RUNS ({MAX_RUNS}). Stopping.")
    #     break

    if not folder.is_dir():
        continue

    mat_file = next((f for f in folder.iterdir() if f.name.endswith("full.mat")), None)
    if not mat_file:
        print(f"No .mat file found in: {folder.name}")
        continue

    print(f"\n🔍 Loading: {mat_file.name} from {folder.name}")
    data = loadmat(mat_file)
    folder_output = OUTPUT_DIR / folder.name
    folder_output.mkdir(exist_ok=True)

    for key in TARGET_KEYS:
        if key not in data:
            continue

        # Skip hkm/qkm just in case
        if key.lower().endswith("hkm") or key.lower().endswith("qkm"):
            continue

        value = data[key]

        # Special case: extract Ackerman from BenchMark
        if key == "BenchMark":
            try:
                ackerman = value['Ackerman'][0][0]
                np.save(folder_output / "x.npy", normalize_minmax(ackerman))
                print(f"Saved: x.npy (normalized Ackerman)")
            except Exception as e:
                print(f"Could not save BenchMark.Ackerman: {e}")
            continue

        # Extract and normalize Rad field
        if isinstance(value, np.ndarray) and 'Rad' in value.dtype.names:
            try:
                rad = value['Rad'][0][0]
                if isinstance(rad, np.ndarray) and rad.ndim >= 2:
                    norm_rad = normalize_minmax(rad)
                    np.save(folder_output / f"{key}.npy", norm_rad)
                else:
                    print(f"{key}.Rad is not a valid 2D array")
            except Exception as e:
                print(f"Could not process {key}.Rad: {e}")
        else:
            print(f"{key} does not have a Rad field or is not structured")

    print(f"[{run_count+1}] Finished: {folder.name}")
    run_count += 1
