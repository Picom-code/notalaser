# 3DCNN+ 🚀

> A modular pipeline for 3D cube regression using Box.com for data storage 📦 and PyTorch 🔥 for model training.

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
  
</div>

## Overview 🔍

**3DCNN+** is a two-part Python project designed for efficient 3D voxel-based regression:

1. **Data Download** (`box.py`): Securely fetches preprocessed NumPy data from Box.com using the Box Python SDK.
2. **Model Training** (`cnnp.py`): Trains a performant 3D convolutional neural network on the downloaded data.

## 📋 Table of Contents

- [Prerequisites](#prerequisites-)
- [Setup](#setup-%EF%B8%8F)
- [Part 1: Data Download](#part-1-data-download-boxpy-)
- [Part 2: Model Training](#part-2-model-training-cnnppy-%EF%B8%8F)
- [How It Works](#how-it-works-)
- [Results](#results-)

## Prerequisites ✅

```
Python 3.8+
PyTorch 2.0+
Box SDK
NumPy, Pandas, Matplotlib, tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup ⚙️

### 1. Environment Variables

Create a `.env` file or export these variables before running:

```bash
export BOX_CLIENT_ID="your_client_id"
export BOX_CLIENT_SECRET="your_client_secret"
export BOX_DEVELOPER_TOKEN="your_developer_token"
```

### 2. Project Folder ID

In `box.py`, set `project1_id` to your Box root folder ID.

## Part 1: Data Download (box.py) 📦

This script authenticates with Box.com via OAuth2 and downloads the `processed` folder containing all required data.

```bash
python box.py
```

### Key Functions

| Function | Description |
|----------|-------------|
| `get_folder_id_by_name()` | Locates a subfolder ID by name |
| `download_folder()` | Recursively downloads files and subfolders |

## Part 2: Model Training (cnnp.py) 🏋️‍♂️

This script handles data loading, model building, and training the 3D CNN.

```bash
python cnnp.py
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BATCH_SIZE` | Samples per batch | 64 |
| `BIG_EPOCHS` | Number of training epochs | 100 |
| `LEARNING_RATE` | Initial learning rate | 0.001 |
| `NUM_SAMPLES` | Patches per mini-epoch | 10000 |

### Model Architecture: `CubeRegressor`

```
Input → Conv3D → BatchNorm → ReLU → MaxPool3D → ... → AdaptiveAvgPool3D → FC → Output
```

### Training Features

- **Memory Efficiency**: Memory-mapped arrays for low RAM usage
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Speed**: Mixed precision training with torch.amp
- **Monitoring**: Automatic checkpointing and loss visualization

## How It Works 🔄

<div align="center">
  
```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Box.com  │ →  │ box.py    │ →  │ Local     │ →  │ cnnp.py   │
│  Storage  │    │ Download  │    │ Processed │    │ Training  │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
                                                        ↓
                                                  ┌───────────┐
                                                  │   Model   │
                                                  │  Outputs  │
                                                  └───────────┘
```
  
</div>

1. **Authentication**: Secure OAuth2 connection to Box.com
2. **Data Handling**: Memory-mapped NumPy arrays for efficient processing
3. **Patch Extraction**: 4D tensors `(batch, channels, D, H, W)` centered on voxels
4. **Training Strategy**: Large batches, sampling, mixed precision, adaptive learning rate

## Results 📊

### Output Files

- **Model Checkpoints**: `./checkpoints/epoch_<n>.pth`
- **Loss Visualization**: `./checkpoints/loss_plot.png`

### Performance Metrics

Example output:
```
Validation MSE: 0.014
Validation MAE: 0.1098
Validation R²:  -0.1282
```

## License

MIT License

---

<div align="center">
  
Made with ❤️ by your favorite ML group
  
</div