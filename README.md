# Turbulence-Robust Perceptual Loss

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-scale encoder that produces features invariant to atmospheric turbulence. On the OTIS benchmark, it reduces feature-space distance by 99.5% compared to VGG relu3_3 (23.16 → 0.12 L2).

## Problem

Standard perceptual losses using VGG features fail on atmospheric turbulence. While VGG relu3_3 is robust to additive noise, it cannot discriminate turbulence severity—feature distances vary only ~2% between weak and strong turbulence levels. This makes VGG unsuitable as a loss function for turbulence mitigation networks.

## Approach

Train a multi-scale encoder end-to-end on clean/turbulent image pairs with a dual-objective loss that enforces both magnitude proximity (L2) and directional invariance (cosine similarity):

```
L = (1 - w) · L_L2 + w · L_cosine
```

The cosine term is critical: L2-only models collapse directionally on real turbulence (cosine similarity drops from 0.95 on validation to 0.33 on OTIS benchmark), while the dual-objective model maintains 0.9999 alignment.

### Encoder Architecture

Four-stage hierarchical encoder with progressive downsampling:

| Stage | Resolution | Channels | Structure |
|-------|------------|----------|-----------|
| 1 | 256×256 | 64 | 2× (Conv3×3 → BN → ReLU) → MaxPool |
| 2 | 128×128 | 128 | 2× (Conv3×3 → BN → ReLU) → MaxPool |
| 3 | 64×64 | 256 | 2× (Conv3×3 → BN → ReLU) → MaxPool |
| 4 | 32×32 | 512 | 2× (Conv3×3 → BN → ReLU) → MaxPool |

Each stage has a projection head mapping to standardized dimensionality for multi-scale loss computation.

### Training

- Dataset: Places365 images paired with QuickTurbSim-generated turbulent variants (6 per clean image at medium/strong severity)
- Scale: 2,338 image pairs (large configuration)
- Setup: 50 epochs, batch 16, AdamW (lr=1e-4, weight decay=1e-5), cosine annealing
- Hardware: RTX 3070, FP16 mixed precision, 256×256 resolution

## Results

### OTIS Benchmark

The OTIS (Optical Turbulence Image Set) dataset contains test patterns imaged through a heated turbulence chamber at controlled severity levels. Lower L2 distance and higher cosine similarity indicate better invariance to turbulence.

| Model | L2 Distance ↓ | Cosine Similarity ↑ |
|-------|---------------|---------------------|
| VGG relu3_3 | 23.16 ± 3.27 | 0.736 ± 0.023 |
| VGG Multi-Layer | 15.86 ± 2.28 | 0.776 ± 0.017 |
| Ours (L2 only) | 0.35 ± 0.09 | 0.327 ± 0.079 |
| Ours (L2 + Cosine) | **0.12 ± 0.01** | **0.9999 ± 0.00001** |

### Cosine Weight Ablation

| Weight (w) | L2 Distance | Cosine Similarity |
|------------|-------------|-------------------|
| 0.0 (pure L2) | 2.21 | 0.37 |
| 0.5 (optimal) | 0.46 | 0.9998 |
| 1.0 (pure cosine) | ~1.0 | ~0.9999 |

Pure L2 training achieves low L2 but poor directional alignment. Adding the cosine constraint (w=0.5) preserves alignment without sacrificing L2 performance.

### Temporal Consistency (CLEAR Dataset)

Evaluated on video sequence "Tene_SD_001v2" (rotating windmill under atmospheric distortion) from the University of Bristol 2024 Turbulence Dataset:

- Frame-to-frame L2: 0.045 (vs 0.73 on static OTIS pairs)
- Frame-to-frame cosine: 1.0000

The 16× compression in temporal vs static L2 indicates the model learned to suppress turbulence variation while tracking real scene motion.

## Project Structure

```
├── models/
│   └── turbulence_encoder.py      # Multi-scale encoder architecture
├── losses/
│   └── turbulence_losses.py       # Dual-objective loss implementation
├── configs/                        # Experiment configurations (YAML)
├── scripts/
│   ├── baseline_pixel_losses.py   # L1/L2 baseline analysis
│   ├── vgg_feature_analysis.py    # VGG layer comparison
│   ├── noise_robustness.py        # Noise robustness evaluation
│   ├── turbulence_robustness.py   # Turbulence robustness evaluation
│   ├── evaluate_baselines.py      # OTIS benchmark evaluation
│   └── generate_quickturb_dataset.py
├── train_encoder.py               # Main training script
├── utils/
│   └── turbulence_dataset.py      # Data loading utilities
└── results/                        # Experiment outputs and visualizations
```

## Usage

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training

```bash
# Generate turbulent training pairs (requires QuickTurbSim)
python scripts/generate_quickturb_dataset.py --input data/clean_images --output data/training_pairs

# Train encoder
python train_encoder.py --config configs/exp6_full_model_large.yaml
```

### Evaluation

```bash
python scripts/evaluate_baselines.py
```

## Limitations

- Resolution: 256×256 (GPU memory constraint)—may not capture fine texture at higher resolution
- Synthetic training: ~4× performance gap between validation and OTIS suggests domain shift from simulated to real turbulence
- Benchmark scope: OTIS uses artificial test patterns; no evaluation on natural scenes
- No downstream tasks: Not yet tested as loss function for actual turbulence mitigation networks

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ VRAM recommended

See `requirements.txt` for full dependencies.

## References

- Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution," ECCV 2016
- Chimitt & Chan, "Simulating Anisoplanatic Turbulence by Sampling Correlated Zernike Coefficients," CVPR 2020

## Author

Sam Ehrle  
Arizona State University
