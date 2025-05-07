import numpy as np
from pathlib import Path
import random

# Set your processed directory path
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

# Function to replace NaNs
def replace_nans_band(data):
    height, width = data.shape
    result = data.copy()

    for y in range(height):
        for x in range(width):
            if np.isnan(data[y, x]):
                # Get surrounding y-range, avoiding bounds
                y_start = max(0, y - 5)
                y_end = min(height, y + 6)

                # Get non-NaN values in that column within the y-range
                surrounding_vals = data[y_start:y_end, x]
                valid_vals = surrounding_vals[~np.isnan(surrounding_vals)]

                if len(valid_vals) > 0:
                    min_val, max_val = valid_vals.min(), valid_vals.max()
                    result[y, x] = np.random.uniform(min_val, max_val)
                else:
                    result[y, x] = 0  # fallback if no valid values found
    return result

# Process all folders and files
for folder in PROCESSED_DIR.iterdir():
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
