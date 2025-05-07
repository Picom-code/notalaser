# Presentation: https://www.canva.com/design/DAGmnsdCFRM/9IrlRw0kthWn_r9gtQrxjQ/edit?utm_content=DAGmnsdCFRM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton




# 3D CNN Base Model

![Neural Network](https://img.shields.io/badge/3D--CNN-Model-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)

A comprehensive implementation of a 3D Convolutional Neural Network (CNN) pipeline for volumetric data processing.

## 📋 Directory Structure

```
.
├── Preprocessing/
│   └── eda-mat.py            # Converts raw .mat data into model-compatible format
├── train-3dcnn-minibatches.py # Main training script
├── 3dcnn-summary.py          # Evaluation and visualization script
├── checkpoints/
│   └── big_epoch_3_finished.pth # Trained model weights after 3 epochs
└── Results/
    ├── 3DCNN_results.txt     # Full evaluation metrics and console output
    ├── loss_plot.png         # Training/validation loss curve
    └── [prediction images]   # Visualization of predictions vs ground truth
```

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- SciPy (for .mat file handling)

### Data Preparation

The `processed/` directory (not included in repository due to size constraints) should contain the preprocessed dataset in `.npy` format. To generate this data:

1. Place your raw `.mat` files in the appropriate input directory
2. Run the preprocessing script:
   ```
   python Preprocessing/eda-mat.py
   ```

### Training

To train the 3D CNN model:

```bash
python train-3dcnn-minibatches.py
```

The script will save model checkpoints in the `checkpoints/` directory.

### Evaluation

To evaluate model performance and generate visualizations:

```bash
python 3dcnn-summary.py
```

This will produce prediction images and performance metrics in the `Results/` directory.

## 📊 Results

Model performance can be assessed through:

- Side-by-side comparison images of predictions vs ground truth
- Complete metrics in `3DCNN_results.txt`
- Training/validation loss curves in `loss_plot.png`

## 📝 Notes

- The final model checkpoint (`big_epoch_3_finished.pth`) represents the model after 3 full training epochs
- For large datasets, ensure sufficient computational resources are available
- Hyperparameters can be adjusted in the respective training script

## 📬 Contact

For questions or issues, please open an issue in this repository.



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