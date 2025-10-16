"""
Compare VGG activation maps for clean vs turbulent images
Updated to accept pre-generated turbulent images

Usage:
    # With real turbulent image
    python scripts/compare_turbulence_activations.py \
        --image data/test_images/boat.jpg \
        --turbulent-image data/turbulence_dataset/turbulent/boat_turb_001.png \
        --output results/turb_activations_real
    
    # With synthetic (fallback)
    python scripts/compare_turbulence_activations.py \
        --image data/test_images/boat.jpg \
        --turb-strength 20.0 \
        --output results/turb_activations_synthetic
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import argparse
from pathlib import Path


class VGGFeatureExtractor:
    """Extract features from VGG-16."""
    
    def __init__(self, device='cpu'):
        self.device = device
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features.to(device)
        self.features.eval()
        
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, img_array):
        """Convert numpy array to VGG input."""
        img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        img_tensor = self.transform(img_pil).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def extract_features(self, img_array, layer_name):
        """Extract features from specified layer."""
        img_tensor = self.preprocess(img_array)
        layer_idx = self.layer_indices[layer_name]
        
        with torch.no_grad():
            x = img_tensor
            for i in range(layer_idx + 1):
                x = self.features[i](x)
        
        return x


def add_synthetic_turbulence(img, warp_strength=20.0):
    """Apply synthetic turbulence distortion (geometric only, no blur)."""
    H, W = img.shape[:2]
    
    # Generate smooth random displacement
    dx = warp_strength * np.random.randn(H, W)
    dy = warp_strength * np.random.randn(H, W)
    
    # Smooth heavily to create spatially correlated warping
    dx = gaussian_filter(dx, sigma=H/20.0)
    dy = gaussian_filter(dy, sigma=H/20.0)
    
    # Apply warping
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    
    coords = np.array([yy + dy, xx + dx])
    warped = np.zeros_like(img, dtype=float)
    
    for c in range(img.shape[2]):
        warped[:,:,c] = map_coordinates(img[:,:,c], coords, order=3, mode='reflect')
    
    return np.clip(warped, 0, 255).astype(np.uint8)


def compare_activations_single_layer(clean_img, turb_img, layer_name, 
                                     extractor, save_dir, num_channels=16,
                                     select_top=False):
    """
    Compare activation maps for one layer: clean vs turbulent.
    
    Args:
        select_top: If True, show only top N most correlated channels.
                   If False, show evenly spaced channels (default).
    """
    # Extract features
    clean_features = extractor.extract_features(clean_img, layer_name)
    turb_features = extractor.extract_features(turb_img, layer_name)
    
    clean_features = clean_features.squeeze(0).cpu().numpy()  # [C, H, W]
    turb_features = turb_features.squeeze(0).cpu().numpy()
    
    n_channels_total = clean_features.shape[0]
    
    # Compute correlations for ALL channels first
    all_correlations = []
    for ch_idx in range(n_channels_total):
        clean_ch = clean_features[ch_idx].flatten()
        turb_ch = turb_features[ch_idx].flatten()
        
        if np.std(clean_ch) > 0 and np.std(turb_ch) > 0:
            corr = np.corrcoef(clean_ch, turb_ch)[0, 1]
        else:
            corr = 0.0
        all_correlations.append((ch_idx, corr))
    
    # Select channels to display
    if select_top:
        # Sort by absolute correlation and take top N
        sorted_corrs = sorted(all_correlations, key=lambda x: abs(x[1]), reverse=True)
        channel_indices = [c[0] for c in sorted_corrs[:num_channels]]
        correlations = [c[1] for c in sorted_corrs[:num_channels]]
    else:
        # Evenly spaced (original behavior)
        num_channels = min(num_channels, n_channels_total)
        channel_indices = np.linspace(0, n_channels_total-1, num_channels, dtype=int)
        correlations = [all_correlations[idx][1] for idx in channel_indices]
    
    mean_corr = np.mean([c[1] for c in all_correlations])
    
    # Create comparison grid (4x4 layout)
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    
    for idx, (ax, ch_idx) in enumerate(zip(axes.flat, channel_indices)):
        if idx >= len(channel_indices):
            ax.axis('off')
            continue
        
        # Get clean and turbulent feature maps
        clean_map = clean_features[ch_idx]
        turb_map = turb_features[ch_idx]
        
        # Normalize for visualization
        clean_map = (clean_map - clean_map.min()) / (clean_map.max() - clean_map.min() + 1e-8)
        turb_map = (turb_map - turb_map.min()) / (turb_map.max() - turb_map.min() + 1e-8)
        
        # Create side-by-side comparison with difference
        combined = np.hstack([clean_map, turb_map, np.abs(clean_map - turb_map)])
        
        ax.imshow(combined, cmap='viridis')
        ax.set_title(f'Ch {ch_idx} | Corr: {correlations[idx]:.3f}', 
                    fontsize=9, pad=15) 
        ax.axis('off')
        
        # Add labels for first row only
        if idx == 0:
            w = clean_map.shape[1]
            ax.text(w/2, -15, 'Clean', ha='center', fontsize=10, 
                   fontweight='bold', va='bottom')
            ax.text(w*1.5, -15, 'Turbulent', ha='center', fontsize=10, 
                   fontweight='bold', va='bottom')
            ax.text(w*2.5, -15, 'Difference', ha='center', fontsize=10, 
                   fontweight='bold', va='bottom')
    
    selection_mode = "Top Correlated" if select_top else "Evenly Spaced"
    plt.suptitle(f'{layer_name}: Clean vs Turbulent Activation Maps ({selection_mode})\n'
                f'Mean Channel Correlation: {mean_corr:.3f}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    
    suffix = "_top" if select_top else ""
    plt.savefig(os.path.join(save_dir, f'{layer_name}_turbulence_comparison{suffix}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  {layer_name}: Mean correlation = {mean_corr:.3f}")
    
    # Return all correlations for summary, not just displayed ones
    return mean_corr, [c[1] for c in all_correlations]


def create_correlation_summary(all_correlations, layers, save_dir):
    """Create summary plot of correlations across all layers."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot of channel correlations per layer
    positions = range(len(layers))
    bp = ax1.boxplot([all_correlations[layer] for layer in layers],
                      positions=positions,
                      tick_labels=layers,
                      patch_artist=True,
                      showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax1.axhline(y=0.9, color='green', linestyle='--', 
                label='High correlation (>0.9)', alpha=0.5)
    ax1.axhline(y=0.7, color='orange', linestyle='--',
                label='Moderate correlation (0.7)', alpha=0.5)
    ax1.set_ylabel('Channel Correlation (Clean vs Turbulent)', fontsize=12)
    ax1.set_xlabel('VGG Layer', fontsize=12)
    ax1.set_title('Turbulence Robustness: Activation Correlation by Layer', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Mean correlation bar chart
    mean_corrs = [np.mean(all_correlations[layer]) for layer in layers]
    bars = ax2.bar(layers, mean_corrs, color=['blue', 'green', 'orange', 'red', 'purple'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Mean Correlation', fontsize=12)
    ax2.set_xlabel('VGG Layer', fontsize=12)
    ax2.set_title('Mean Turbulence Robustness by Layer', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'turbulence_correlation_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("TURBULENCE ROBUSTNESS RANKING (by mean correlation)")
    print("="*60)
    for layer, mean_corr in sorted(zip(layers, mean_corrs), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {layer}: {mean_corr:.3f}")
    print("="*60)


def main(image_path, save_dir, turbulent_image_path=None, turb_strength=20.0, select_top=False):
    """Main execution."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("VGG Activation Comparison: Clean vs Turbulent")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load clean image
    print(f"Loading clean image from: {image_path}")
    clean_img = np.array(Image.open(image_path))

    # Load or generate turbulent image
    if turbulent_image_path:
        print(f"Loading turbulent image from: {turbulent_image_path}")
        turb_img = np.array(Image.open(turbulent_image_path))
        
        # Match turbulent to clean dimensions (same as part3)
        if turb_img.shape != clean_img.shape:
            print(f"Resizing turbulent from {turb_img.shape} to match clean {clean_img.shape}")
            turb_pil = Image.fromarray(turb_img)
            turb_resized = turb_pil.resize((clean_img.shape[1], clean_img.shape[0]), Image.LANCZOS)
            turb_img = np.array(turb_resized)
        
        turb_source = "QuickTurbSim (physics-based)"
    else:
        print(f"Generating synthetic turbulent version (strength={turb_strength})...")
        turb_img = add_synthetic_turbulence(clean_img, warp_strength=turb_strength)
        turb_source = "Synthetic (geometric warping)"
    
    # Save comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(clean_img)
    axes[0].set_title('Clean Image', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(turb_img)
    axes[1].set_title(f'Turbulent Image\n({turb_source})', fontsize=14)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'input_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Initialize extractor
    extractor = VGGFeatureExtractor(device=device)
    
    # Test all layers
    layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
    all_correlations = {}
    
    print("\nComparing activations across all layers...")
    for layer_name in layers:
        print(f"\n{layer_name}:")
        mean_corr, corrs = compare_activations_single_layer(
            clean_img, turb_img, layer_name, extractor, save_dir,
            select_top = select_top
        )
        all_correlations[layer_name] = corrs
    
    # Create summary
    print("\nGenerating correlation summary...")
    create_correlation_summary(all_correlations, layers, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    print("\nAnalysis complete!")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION FOR PART 4")
    print("="*60)
    mean_corrs = {layer: np.mean(all_correlations[layer]) for layer in layers}
    best_layer = max(mean_corrs, key=mean_corrs.get)
    print(f"\nMost turbulence-robust layer: {best_layer} (corr={mean_corrs[best_layer]:.3f})")
    print(f"\nFor perceptual loss baseline in Part 4, consider using:")
    print(f"  - {best_layer} (highest turbulence robustness)")
    
    # Find layer with good balance
    sorted_layers = sorted(mean_corrs.items(), key=lambda x: x[1], reverse=True)
    print(f"\nAlternatively, for multi-layer perceptual loss:")
    print(f"  - Combine {sorted_layers[0][0]} + {sorted_layers[1][0]}")
    print(f"  - Weights: 0.5 each or tune empirically")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare VGG activations: clean vs turbulent'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to clean input image')
    parser.add_argument('--turbulent-image', type=str, default=None,
                        help='Path to pre-generated turbulent image (overrides --turb-strength)')
    parser.add_argument('--output', type=str, 
                        default='results/turbulence_activations',
                        help='Directory to save results')
    parser.add_argument('--select-top', action='store_true',
                    help='Show top correlated channels instead of evenly spaced')
    parser.add_argument('--turb-strength', type=float, default=20.0,
                        help='Synthetic turbulence warp strength (only if no turbulent image provided)')
    
    args = parser.parse_args()
    main(args.image, args.output, args.turbulent_image, args.turb_strength, args.select_top)

