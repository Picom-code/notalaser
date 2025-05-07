# notalaser


# 3DCNN+

A modular pipeline for 3D cube regression using Box.com for data storage and PyTorch for model training.

## Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Setup](#setup)
* [Part 1: Data Download (box.py)](#part-1-data-download-boxpy)
* [Part 2: Model Training (cnnp.py)](#part-2-model-training-cncppy)
* [How It Works](#how-it-works)
* [Results](#results)

## Overview

**3DCNN+** is a two-part Python project:

1. **Data Download** (`box.py`): Fetches preprocessed NumPy data from Box.com using the Box Python SDK.
2. **Model Training** (`cnnp.py`): Trains a 3D convolutional neural network (`CubeRegressor`) on the downloaded data for voxel-based regression.

This README explains how to configure, run, and understand each script.

## Prerequisites

* Python 3.8 or higher
* Box SDK: `pip install boxsdk`
* Core libraries: `numpy`, `pandas`, `matplotlib`, `tqdm`, `torch`

Install everything with:

```bash
pip install -r requirements.txt
```

## Setup

1. **Environment Variables**: Create a `.env` or export before running `box.py`:

   ```bash
   export BOX_CLIENT_ID="<your_client_id>"
   export BOX_CLIENT_SECRET="<your_client_secret>"
   export BOX_DEVELOPER_TOKEN="<your_developer_token>"
   ```
2. **Project Folder ID**:

   * In `box.py`, set `project1_id` to your Box root folder ID.
   * The script auto-discovers subfolders named `code` and `processed`.

## Part 1: Data Download (box.py)

This script logs into Box via OAuth2 and downloads the `processed` folder (and all nested files/folders) into `./processed/`.

Run:

```bash
python box.py
```

Key functions:

* `get_folder_id_by_name(parent_folder_id, folder_name)`: Finds a subfolder ID by name.
* `download_folder(folder_id, local_path)`: Recursively downloads all files/subfolders.

After completion, check that `./processed/` contains `.npy` files for all bands and labels.

## Part 2: Model Training (cnnp.py)

This script loads the downloaded `.npy` data, builds a 3D CNN, and trains it.

Run:

```bash
python cnnp.py
```

### Configuration (top of `cnnp.py`)

* `BATCH_SIZE`: samples per batch (adjust for GPU RAM).
* `BIG_EPOCHS`: number of training epochs.
* `LEARNING_RATE`: initial learning rate.
* `NUM_SAMPLES`: patches sampled per epoch for mini-epochs.

### Dataset

* **`PatchDataset`** uses memory-mapped `.npy` arrays to avoid loading everything into RAM.
* It indexes all valid voxel coordinates and extracts 3D patches around each voxel.

### Model: `CubeRegressor`

* A simple 3D CNN with convolutional layers, pooling, adaptive pooling, and a fully connected head.

### Training Loop

1. **Compile** model with `torch.compile` for performance.
2. **Optimizer**: `Adam` with initial `LEARNING_RATE`.
3. **Scheduler**: `ReduceLROnPlateau` on validation loss.
4. **Mixed Precision**: `torch.amp.autocast` for faster, memory-efficient training.
5. **Checkpointing**: Saves weights (`.pth`) each epoch and a loss plot (`loss_plot.png`).

### Evaluation

* After training, computes MSE, MAE, and R² on the validation set and prints results.

## How It Works

1. **Authentication & Download**: `box.py` uses OAuth2 to fetch processed data from Box.com.
2. **Memory-Mapped Dataset**: Efficiently accesses large NumPy arrays without high RAM usage.
3. **Patch Extraction**: Returns 4D tensors `(batch, channels, D, H, W)` centered on voxels.
4. **3D CNN**: Processes each patch to output a single regression value.
5. **Training Strategy**: Large batch sizes, mini-epochs via sampling, mixed precision, and adaptive LR.

## Results

* **Output Directory**: `./checkpoints/`

  * Model weights: `epoch_<n>.pth`
  * Loss plot: `loss_plot.png`
* **Final Metrics** (printed):

  ```text
  Validation MSE: 0.001234, MAE: 0.025678, R2: 0.9123
  ```

---

You can adjust script parameters or swap in different models/datasets with minimal changes.
