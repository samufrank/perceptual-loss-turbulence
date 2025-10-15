# Perceptual Loss for Atmospheric Turbulence

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research on perceptual loss functions for atmospheric turbulence mitigation, combining multi-scale spatial features, frequency domain modeling, and contrastive learning.

## Overview

This project investigates perceptual losses using pre-trained CNNs (VGG-16) and develops novel turbulence-robust perceptual losses that better correlate with human perception compared to standard ℓp losses.

### Key Features

- Baseline ℓ1/ℓ2 loss analysis on translated images
- VGG-16 multi-layer feature extraction and comparison
- Noise robustness testing for perceptual losses
- Novel turbulence-robust perceptual loss combining:
  - Multi-scale spatial features
  - Frequency domain modeling
  - Contrastive learning

## Project Structure

```
perceptual-loss-turbulence/
├── config/              # Experiment configurations
├── data/               # Dataset (clean and turbulent images)
├── results/            # Experimental results and figures
├── scripts/            # Main implementation scripts
│   ├── part1_baseline_losses.py
│   ├── part2_vgg_features.py
│   ├── part3_noise_robustness.py
│   └── generate_turbulence.py
├── src/                # Core implementation (Part 4)
│   ├── losses.py       # Loss function implementations
│   ├── models.py       # Network architectures
│   ├── dataset.py      # Data loading utilities
│   ├── train.py        # Training script
│   └── utils.py        # Helper functions
└── tests/              # Unit tests
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory (recommended for Part 4)

### Setup

**Option 1: pip + venv (recommended)**

```bash
# Clone repository
git clone https://github.com/yourusername/perceptual-loss-turbulence.git
cd perceptual-loss-turbulence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Option 2: conda**

```bash
# Clone repository
git clone https://github.com/yourusername/perceptual-loss-turbulence.git
cd perceptual-loss-turbulence

# Create conda environment
conda create -n turbulence python=3.9
conda activate turbulence

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Baseline Analysis

```bash
python scripts/part1_baseline_losses.py --image data/test_images/sample.jpg --output results/part1
```

Demonstrates limitations of ℓ1 and ℓ2 losses on geometrically shifted images.

### Feature Extraction Analysis

```bash
python scripts/part2_vgg_features.py --image data/test_images/sample.jpg --output results/part2
```

Analyzes VGG-16 feature robustness across multiple layers to geometric transformations.

### Robustness Evaluation

```bash
python scripts/part3_noise_robustness.py --image data/test_images/sample.jpg --output results/part3 --layer relu3_3
```

Evaluates perceptual loss robustness to additive noise corruption.

### Turbulence-Robust Features

```bash
# Generate turbulence dataset
python scripts/generate_turbulence.py --input_dir data/clean/ --output_dir data/turbulent/ --num_variations 10

# Train model
python src/train.py --config config/experiments.yaml
```

## Methodology

### Perceptual Loss Background

Standard ℓp losses compare images pixel-by-pixel, which doesn't align with human perception. For example, a 1-pixel shift causes large ℓ2 loss despite images appearing nearly identical to humans.

**Perceptual losses** use neural network features (e.g., VGG-16) to compare images at multiple levels of abstraction:
- Early layers: edges, colors, textures
- Middle layers: patterns, object parts
- Deep layers: semantic content

### Novel Contributions

Our turbulence-robust perceptual loss combines:

1. **Multi-scale spatial features:** Capture distortions at different granularities
2. **Frequency domain loss:** Explicitly model blur component of turbulence
3. **Contrastive learning:** Learn features where clean/turbulent pairs are similar, different scenes are dissimilar

**Loss formulation:**
```
L_total = α·L_spatial + β·L_frequency + γ·L_contrastive
```

## Dataset

### Image Selection

We use complex scene images (not isolated objects) to better evaluate perceptual quality:
- Indoor scenes (living rooms, kitchens)
- Outdoor scenes (streets, parks)
- Multiple objects at different depths
- Rich textures and spatial relationships

### Turbulence Generation

Using DAATSim simulator to generate physics-based atmospheric turbulence:
- Geometric warping (spatially correlated distortions)
- Temporal blur
- Variable turbulence strength (weak, medium, strong)

## Results

### Baseline Analysis

| Metric | Value |
|--------|-------|
| ℓ1 Loss | X.XX |
| ℓ2 Loss | X.XX |

Despite minimal visual difference, pixel losses are non-trivial.

### Feature Extraction Analysis

Recommended layer for perceptual loss: **relu3_3**

Rationale: Optimal balance between robustness to geometric shifts and sensitivity to semantic changes.

### Robustness Evaluation

[Results pending]

### Turbulence-Robust Features

[Results pending]

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- scikit-image
- scikit-learn
- Pillow

## Citation

If you find this work useful, please cite:

```bibtex
@misc{perceptual_turbulence_2025,
  author = {Your Name},
  title = {Perceptual Loss for Atmospheric Turbulence Mitigation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/perceptual-loss-turbulence}
}
```

## Acknowledgments

- DAATSim atmospheric turbulence simulator
- VGG-16 pre-trained weights from torchvision
- PyTorch framework

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open a GitHub issue or contact sehrle@asu.edu