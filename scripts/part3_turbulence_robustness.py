"""
Turbulence Robustness Evaluation
Updated to accept pre-generated turbulent images from QuickTurbSim

Usage:
    # With pre-generated turbulent images
    python scripts/part3_turbulence_robustness.py \
        --image data/test_images/boat.jpg \
        --turbulent-images data/turbulence_dataset/turbulent/boat_turb_*.png \
        --output results/part3_turbulence_real
    
    # With synthetic turbulence (fallback)
    python scripts/part3_turbulence_robustness.py \
        --image data/test_images/boat.jpg \
        --output results/part3_turbulence_synthetic \
        --use-synthetic
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter, map_coordinates
import json
import os
import argparse
import glob
from pathlib import Path


class VGGFeatureExtractor:
    """Extract features from VGG-16 for perceptual loss."""
    
    def __init__(self, layer_name='relu3_3', device='cpu'):
        self.device = device
        self.layer_name = layer_name
        
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
        
        self.layer_idx = self.layer_indices[layer_name]
        
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
    
    def extract_features(self, img_array):
        """Extract features from specified layer."""
        img_tensor = self.preprocess(img_array)
        
        with torch.no_grad():
            x = img_tensor
            for i in range(self.layer_idx + 1):
                x = self.features[i](x)
        
        return x
    
    def compute_perceptual_loss(self, img1, img2):
        """Compute perceptual loss between two images."""
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)
        
        loss = torch.sqrt(torch.mean((feat1 - feat2) ** 2)).item()
        return loss


def add_gaussian_noise(img, sigma):
    """Add i.i.d. Gaussian noise to image."""
    noise = np.random.normal(0, sigma * 255, img.shape)
    noisy_img = img.astype(float) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def calculate_pixel_losses(img_original, img_distorted):
    """Calculate L1 and L2 pixel losses."""
    l1 = np.mean(np.abs(img_original.astype(float) - img_distorted.astype(float)))
    l2 = np.sqrt(np.mean((img_original.astype(float) - img_distorted.astype(float)) ** 2))
    return l1, l2

def load_turbulent_images(image_pattern, target_shape):
    """Load turbulent images from pattern or list and resize to target shape."""
    if isinstance(image_pattern, str):
        paths = sorted(glob.glob(image_pattern))
    else:
        paths = image_pattern
    
    # Sort by turbulence severity (weak < medium < strong)
    def turb_sort_key(path):
        stem = Path(path).stem.lower()
        if 'weak' in stem:
            return 0
        elif 'medium' in stem:
            return 1
        elif 'strong' in stem:
            return 2
        else:
            return 3
    
    paths = sorted(paths, key=turb_sort_key)
    
    images = []
    for path in paths:
        img = np.array(Image.open(path))
        # Resize to match original image dimensions
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
        images.append(np.array(img_resized))
    
    return images, paths


def visualize_comparison(img_original, turbulent_imgs, turbulent_labels,
                        noisy_imgs, noise_labels, save_dir):
    """Compare turbulence vs noise side-by-side."""
    n_turb = len(turbulent_imgs)
    n_noise = len(noisy_imgs)
    n_max = max(n_turb, n_noise)
    
    fig, axes = plt.subplots(3, n_max + 1, figsize=(4*(n_max+1), 12))
    
    # Original (first column)
    for row in range(3):
        axes[row, 0].imshow(img_original.astype(np.uint8))
        axes[row, 0].set_title('Original', fontsize=12)
        axes[row, 0].axis('off')
    
    # Turbulent images (row 0)
    for i in range(n_max):
        if i < n_turb:
            axes[0, i+1].imshow(turbulent_imgs[i])
            axes[0, i+1].set_title(f'Turbulent\n{turbulent_labels[i]}', fontsize=10)
        axes[0, i+1].axis('off')
    
    # Noisy images (row 1)
    for i in range(n_max):
        if i < n_noise:
            axes[1, i+1].imshow(noisy_imgs[i])
            axes[1, i+1].set_title(f'Noise\n{noise_labels[i]}', fontsize=10)
        axes[1, i+1].axis('off')
    
    # Difference images (row 2)
    for i in range(n_max):
        if i < n_turb:
            diff = np.abs(img_original.astype(float) - turbulent_imgs[i].astype(float))
            axes[2, i+1].imshow(np.mean(diff, axis=2), cmap='hot', vmin=0, vmax=50)
            axes[2, i+1].set_title(f'Diff {i+1}', fontsize=10)
        axes[2, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'turbulence_vs_noise_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: turbulence_vs_noise_comparison.png")


def plot_loss_curves(turbulent_labels, turb_l1, turb_l2, turb_perceptual,
                     noise_labels, noise_l1, noise_l2, noise_perceptual,
                     layer_name, save_dir):
    """Plot loss curves comparing turbulence vs noise."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create x-axis indices
    turb_x = range(len(turbulent_labels))
    noise_x = range(len(noise_labels))
    n_turb = len(turbulent_labels)
    n_noise = len(noise_labels)
    
    # L1 comparison
    axes[0, 0].plot(turb_x, turb_l1, 'o-', label='Turbulence', 
                   linewidth=2, markersize=8, color='blue')
    axes[0, 0].plot(noise_x, noise_l1, 's-', label='Noise', 
                   linewidth=2, markersize=8, color='red')
    axes[0, 0].set_xlabel('Distortion Level', fontsize=12)
    axes[0, 0].set_ylabel('L1 Loss', fontsize=12)
    axes[0, 0].set_title('L1 Loss: Turbulence vs Noise', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xticks(range(max(n_turb, n_noise)))
    axes[0, 0].set_xticklabels(['Weak', 'Medium', 'Strong'][:max(n_turb, n_noise)])

    # L2 comparison
    axes[0, 1].plot(turb_x, turb_l2, 'o-', label='Turbulence', 
                   linewidth=2, markersize=8, color='blue')
    axes[0, 1].plot(noise_x, noise_l2, 's-', label='Noise', 
                   linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Distortion Level', fontsize=12)
    axes[0, 1].set_ylabel('L2 Loss', fontsize=12)
    axes[0, 1].set_title('L2 Loss: Turbulence vs Noise', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xticks(range(max(n_turb, n_noise)))
    axes[0, 1].set_xticklabels(['Weak', 'Medium', 'Strong'][:max(n_turb, n_noise)])
    
    # Perceptual comparison
    axes[1, 0].plot(turb_x, turb_perceptual, 'o-', label='Turbulence', 
                   linewidth=2, markersize=8, color='blue')
    axes[1, 0].plot(noise_x, noise_perceptual, 's-', label='Noise', 
                   linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Distortion Level', fontsize=12)
    axes[1, 0].set_ylabel(f'Perceptual Loss ({layer_name})', fontsize=12)
    axes[1, 0].set_title('Perceptual Loss: Turbulence vs Noise', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xticks(range(max(n_turb, n_noise)))
    axes[1, 0].set_xticklabels(['Weak', 'Medium', 'Strong'][:max(n_turb, n_noise)])
    
    # Normalized comparison
    turb_l2_norm = np.array(turb_l2) / max(turb_l2)
    turb_perc_norm = np.array(turb_perceptual) / max(turb_perceptual)
    noise_l2_norm = np.array(noise_l2) / max(noise_l2)
    noise_perc_norm = np.array(noise_perceptual) / max(noise_perceptual)
    
    axes[1, 1].plot(turb_x, turb_perc_norm, 'o-', 
                   label='Perceptual (Turbulence)', linewidth=2, markersize=8, color='darkblue')
    axes[1, 1].plot(noise_x, noise_perc_norm, 's-', 
                   label='Perceptual (Noise)', linewidth=2, markersize=8, color='darkred')
    axes[1, 1].plot(turb_x, turb_l2_norm, 'o--', 
                   label='L2 (Turbulence)', linewidth=1.5, markersize=6, 
                   color='lightblue', alpha=0.7)
    axes[1, 1].plot(noise_x, noise_l2_norm, 's--', 
                   label='L2 (Noise)', linewidth=1.5, markersize=6, 
                   color='lightcoral', alpha=0.7)
    axes[1, 1].set_xlabel('Distortion Level', fontsize=12)
    axes[1, 1].set_ylabel('Normalized Loss', fontsize=12)
    axes[1, 1].set_title('Normalized Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10, loc='lower right')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xticks(range(max(n_turb, n_noise)))
    axes[1, 1].set_xticklabels(['Weak', 'Medium', 'Strong'][:max(n_turb, n_noise)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'turbulence_vs_noise_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: turbulence_vs_noise_curves.png")


def create_analysis_summary(turbulent_labels, turb_l1, turb_l2, turb_perceptual,
                            noise_labels, noise_l1, noise_l2, noise_perceptual,
                            layer_name, save_dir, turbulence_source):
    """Create detailed analysis summary."""
    
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axis('off')
    
    # Calculate statistics
    turb_l1_mean = np.mean(turb_l1)
    turb_l2_mean = np.mean(turb_l2)
    turb_perc_mean = np.mean(turb_perceptual)
    noise_l1_mean = np.mean(noise_l1)
    noise_l2_mean = np.mean(noise_l2)
    noise_perc_mean = np.mean(noise_perceptual)
    
    summary_text = f"""
Turbulence vs Noise Robustness Analysis

Perceptual Loss Layer: {layer_name}
Turbulence Source: {turbulence_source}
Number of Turbulent Images: {len(turbulent_labels)}
Number of Noise Levels: {len(noise_labels)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEAN RESULTS:
  Turbulence - L1: {turb_l1_mean:>8.4f}  L2: {turb_l2_mean:>8.4f}  Perceptual: {turb_perc_mean:>8.6f}
  Noise      - L1: {noise_l1_mean:>8.4f}  L2: {noise_l2_mean:>8.4f}  Perceptual: {noise_perc_mean:>8.6f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PIXEL LOSS COMPARISON:
  • Turbulence mean L2: {turb_l2_mean:.2f}
  • Noise mean L2: {noise_l2_mean:.2f}
  • Ratio (Turb/Noise): {turb_l2_mean/noise_l2_mean:.2f}x

PERCEPTUAL LOSS COMPARISON:
  • Turbulence mean perceptual: {turb_perc_mean:.4f}
  • Noise mean perceptual: {noise_perc_mean:.4f}
  • Ratio (Turb/Noise): {turb_perc_mean/noise_perc_mean:.2f}x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY OBSERVATIONS:

1. Turbulence Source: {turbulence_source}
   Real atmospheric turbulence simulation using QuickTurbSim

2. Perceptual Loss Sensitivity:
   VGG {layer_name} shows {turb_perc_mean/noise_perc_mean:.2f}x sensitivity ratio
   {"Higher" if turb_perc_mean > noise_perc_mean else "Lower"} for turbulence vs noise

3. Pixel Loss Comparison:
   L2 treats distortions with {turb_l2_mean/noise_l2_mean:.2f}x ratio
   {"Turbulence causes more" if turb_l2_mean > noise_l2_mean else "Noise causes more"} pixel-level degradation

4. Implications for Part 4:
   • VGG features {"more" if turb_perc_mean > noise_perc_mean else "less"} sensitive to turbulence
   • Custom loss must achieve <{min(turb_perceptual):.4f} to improve over VGG
   • {"Turbulence" if turb_perc_mean > noise_perc_mean else "Noise"} is the harder problem for VGG features
   • This {"validates" if turb_perc_mean > noise_perc_mean else "challenges"} using perceptual features for turbulence tasks

5. Why This Matters:
   Turbulence is spatially correlated (nearby pixels move together)
   VGG features encode spatial relationships (compositional patterns)
   When turbulence breaks these relationships, perceptual loss increases
   Custom features must be designed for geometric invariance
"""
    
    ax.text(0.5, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'turbulence_vs_noise_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: turbulence_vs_noise_analysis.png")


def main(image_path, save_dir, layer_name='relu3_3', 
         turbulent_images_pattern=None, use_synthetic=False):
    """Main execution function."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Turbulence vs Noise Robustness Evaluation")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    print(f"\nPerceptual loss layer: {layer_name}")
    
    # Load original image
    print(f"\nLoading image from: {image_path}")
    img_original = np.array(Image.open(image_path)) 

    # Load or generate turbulent images
    if turbulent_images_pattern and not use_synthetic:
        print(f"\nLoading pre-generated turbulent images: {turbulent_images_pattern}")
        turbulent_imgs, turb_paths = load_turbulent_images(
            turbulent_images_pattern, 
            img_original.shape  # Add this parameter
        )
        # Extract meaningful labels from sorted paths
        turbulent_labels = []
        for p in turb_paths:
            stem = Path(p).stem
            if 'weak' in stem:
                turbulent_labels.append('weak')
            elif 'medium' in stem:
                turbulent_labels.append('medium')
            elif 'strong' in stem:
                turbulent_labels.append('strong')
            else:
                # Fallback to original extraction
                turbulent_labels.append(stem.split('_')[-1])
        turbulence_source = "QuickTurbSim (physics-based)"
        print(f"Loaded {len(turbulent_imgs)} turbulent images: {turbulent_labels}")
    else:
        print("\nUsing synthetic turbulence (fallback)")
        # Fallback synthetic turbulence
        from scipy.ndimage import map_coordinates
        
        def add_synthetic_turbulence(img, warp_strength):
            H, W = img.shape[:2]
            dx = warp_strength * np.random.randn(H, W)
            dy = warp_strength * np.random.randn(H, W)
            dx = gaussian_filter(dx, sigma=H/20.0)
            dy = gaussian_filter(dy, sigma=H/20.0)
            
            x = np.arange(W)
            y = np.arange(H)
            xx, yy = np.meshgrid(x, y)
            coords = np.array([yy + dy, xx + dx])
            warped = np.zeros_like(img, dtype=float)
            
            for c in range(3):
                warped[:,:,c] = map_coordinates(img[:,:,c], coords, order=3, mode='reflect')
            
            return np.clip(warped, 0, 255).astype(np.uint8)
        
        strengths = [5.0, 10.0, 15.0, 20.0]
        turbulent_imgs = [add_synthetic_turbulence(img_original, s) for s in strengths]
        turbulent_labels = [f"synth_{s:.0f}" for s in strengths]
        turbulence_source = "Synthetic (geometric warping)"

    # Generate noisy images for comparison
    print("\nGenerating noisy images for comparison...")
    noise_levels = [0.01, 0.1, 0.2]
    noisy_imgs = [add_gaussian_noise(img_original, sigma) for sigma in noise_levels]
    noise_labels = [f"σ={sigma:.2f}" for sigma in noise_levels]
    
    # Initialize feature extractor
    extractor = VGGFeatureExtractor(layer_name=layer_name, device=device)
    
    # Compute losses for turbulent images
    print("\nComputing losses for turbulent images...")
    turb_l1, turb_l2, turb_perceptual = [], [], []

    for idx, turb_img in enumerate(turbulent_imgs):
        print(f"  Processing {turbulent_labels[idx]}:")
        print(f"    Turbulent shape: {turb_img.shape}, Original shape: {img_original.shape}")
        
        # Validate shapes match
        assert turb_img.shape == img_original.shape, \
            f"Shape mismatch: {turb_img.shape} vs {img_original.shape}"
        
        l1, l2 = calculate_pixel_losses(img_original, turb_img)
        perc = extractor.compute_perceptual_loss(img_original, turb_img)
        
        turb_l1.append(l1)
        turb_l2.append(l2)
        turb_perceptual.append(perc)
        
        print(f"    L1={l1:.4f}, L2={l2:.4f}, Perceptual={perc:.6f}")
    
    # Compute losses for noisy images
    print("\nComputing losses for noisy images...")
    noise_l1, noise_l2, noise_perceptual = [], [], []
    
    for idx, noisy_img in enumerate(noisy_imgs):
        l1, l2 = calculate_pixel_losses(img_original, noisy_img)
        perc = extractor.compute_perceptual_loss(img_original, noisy_img)
        
        noise_l1.append(l1)
        noise_l2.append(l2)
        noise_perceptual.append(perc)
        
        print(f"  {noise_labels[idx]}: L1={l1:.4f}, L2={l2:.4f}, Perceptual={perc:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_comparison(img_original, turbulent_imgs, turbulent_labels,
                        noisy_imgs, noise_labels, save_dir)
    plot_loss_curves(turbulent_labels, turb_l1, turb_l2, turb_perceptual,
                    noise_labels, noise_l1, noise_l2, noise_perceptual,
                    layer_name, save_dir)
    create_analysis_summary(turbulent_labels, turb_l1, turb_l2, turb_perceptual,
                           noise_labels, noise_l1, noise_l2, noise_perceptual,
                           layer_name, save_dir, turbulence_source)
    
    # Save results
    results = {
        'layer_name': layer_name,
        'turbulence_source': turbulence_source,
        'turbulent_labels': turbulent_labels,
        'noise_labels': noise_labels,
        'turbulence': {
            'l1_losses': turb_l1,
            'l2_losses': turb_l2,
            'perceptual_losses': turb_perceptual
        },
        'noise': {
            'l1_losses': noise_l1,
            'l2_losses': noise_l2,
            'perceptual_losses': noise_perceptual
        },
        'image_path': image_path
    }
    
    output_path = os.path.join(save_dir, 'part3_turbulence_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved: part3_turbulence_results.json")
    
    print(f"\nAll results saved to: {save_dir}")
    print("Turbulence robustness evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Turbulence vs Noise Robustness')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to clean input image')
    parser.add_argument('--output', type=str, default='results/part3_turbulence',
                        help='Directory to save results')
    parser.add_argument('--layer', type=str, default='relu3_3',
                        choices=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'],
                        help='VGG layer for perceptual loss')
    parser.add_argument('--turbulent-images', type=str, default=None,
                        help='Pattern for turbulent images (e.g., "data/turb/boat_*.png")')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic turbulence instead of loading real images')
    
    args = parser.parse_args()
    main(args.image, args.output, args.layer, args.turbulent_images, args.use_synthetic)

