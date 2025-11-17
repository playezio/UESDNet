# UESDNet: UAV Expert-Guided Prior and Segmentation-Detection Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: [中文版本](README.md) is also available.

## Project Overview

This is the official implementation of the paper "UESDNet: UAV Expert-Guided Prior and Segmentation-Detection Network". We've developed a multi-task perception framework specifically for UAV vision systems that simultaneously handles object detection and semantic segmentation.

Key innovations include:
- Fusing UAV attitude information (pitch, roll, yaw angles) to improve perception accuracy
- Creating a CTFMB cross-task feature mutual benefit structure allowing tasks to assist each other
- Developing the ExpertSync module to tackle small object detection and segmentation challenges caused by occlusion

## Key Features

Our system offers the following capabilities:
- Multi-task processing for simultaneous object detection and semantic segmentation
- Unique UAV expert-guided prior mechanism that dynamically adjusts processing based on drone attitude
- Cross-task feature sharing and mutual benefit with expert-guided synchronization
- C++ extensions for performance-critical real-time applications
- Clean, modular code structure that facilitates future development and extension

## Performance Metrics

Experimental results on STUDataset:

- Object Detection: 75.6% mAP@0.5:0.95
- Semantic Segmentation: 71.5% mIoU
- Inference speed up to 44.0 FPS on RTX 3090
- Computational complexity around 91.8 GFLOPs
- Model parameter count of 23.7 million

## Installation Steps

### Environment Requirements

Required environment:
- Operating System: Linux, macOS, or Windows
- Python 3.7 or newer
- CUDA 11.1+ recommended for GPU acceleration
- NVIDIA GPU with ≥8GB VRAM recommended (e.g., RTX 3090)

### Installation Steps

1. Clone the repository

```bash
git clone https://github.com/yourusername/UESDNet.git
cd UESDNet
```

2. Create and activate virtual environment

```bash
# Using conda
conda create -n uesdnet python=3.7
conda activate uesdnet

# Or using Python's built-in venv
python -m venv uesdnet
source uesdnet/bin/activate  # Linux/macOS
uesdnet\Scripts\activate  # Windows
```

3. Install required dependencies

```bash
pip install -r requirements.txt
```

4. Compile C extensions (optional, significantly improves performance)

```bash
cd uav_self
python setup.py build_ext --inplace
```

## Usage

### Data Preparation

We use STUDataset, available at: https://doi.org/10.57760/sciencedb.28912

### Train Model

Run phased training:

```bash
python src/training/phased_trainer.py --config configs/train_phase1.yaml
python src/training/phased_trainer.py --config configs/train_phase2.yaml
```

### Evaluate Model

Evaluate with trained model:

```bash
python src/experiments/eval_metrics.py --config configs/inference.yaml --checkpoint path/to/model.pth
```

### Using UAV Self-localization Module

Example usage of the UAV self-localization module:

```python
from uav_self import process_uav_image
import cv2

# Load image
img = cv2.imread('drone_image.jpg')

# Process to get UAV information
info = process_uav_image(img)
print(f"UAV position information: {info}")
```

## Project Structure

```
UESDNet/
├── src/            # Source code
│   ├── data/       # Data processing
│   ├── models/     # Model definitions
│   ├── training/   # Training code
│   ├── experiments/# Experiment scripts
│   └── config.py   # Configuration
├── configs/        # Config files
├── tests/          # Test cases
├── uav_self/       # UAV self-localization module
├── requirements.txt # Dependencies
├── LICENSE         # License
└── README.md       # Project docs
```

## Documentation

- See [uav_self/README.md](uav_self/README.md) for details on the UAV self-localization module

## Testing

Run tests to ensure everything works:

```bash
python -m pytest tests/ -v
```

## Contributing

We welcome contributions to this project!

Simple steps:
1. Fork this repository
2. Create your feature branch
3. Commit your code changes
4. Push to your fork
5. Submit a Pull Request to us

## License

MIT License. See [LICENSE](LICENSE) for details




## Contact

Questions or suggestions? Get in touch:
- GitHub Issues: Submit an issue directly in the repository
