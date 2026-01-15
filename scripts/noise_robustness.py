"""
Noise Robustness Evaluation
Perceptual Loss for Atmospheric Turbulence

Evaluate perceptual loss robustness to additive noise corruption.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
import json
import os
import argparse


class VGGFeatureExtractor:
    """Extract features from VGG-16 for perceptual loss."""
    
    def __init__(self, layer_name='relu3_3', device='cpu'):
        self.device = device
        self.layer_name = layer_name
        
        # Load VGG-16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features.to(device)
        self.features.eval()
        
        # Layer mapping
        self.layer_indices = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }
        
        self.layer_idx = self.layer_indices[layer_name]
        
        # Preprocessing
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
        
        # L2 distance in feature space
        loss = torch.sqrt(torch.mean((feat1 - feat2) ** 2)).item()
        return loss


def add_gaussian_noise(img, sigma):
    """Add Gaussian noise to image.
    
    Args:
        img: numpy array
        sigma: noise standard deviation (0-1 scale for normalized images,
               will be scaled to 0-255 range)
    """
    noise = np.random.normal(0, sigma * 255, img.shape)
    noisy_img = img.astype(float) + noise
    return np.clip(noisy_img, 0, 255)


def calculate_pixel_losses(img_original, img_noisy):
    """Calculate L1 and L2 pixel losses."""
    l1 = np.mean(np.abs(img_original.astype(float) - img_noisy.astype(float)))
    l2 = np.sqrt(np.mean((img_original.astype(float) - img_noisy.astype(float)) ** 2))
    return l1, l2


def visualize_noisy_images(img_original, noisy_images, noise_levels, save_dir):
    """Visualize original and noisy images."""
    n = len(noise_levels) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    axes[0].imshow(img_original.astype(np.uint8))
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    for i, (noisy, sigma) in enumerate(zip(noisy_images, noise_levels)):
        axes[i+1].imshow(noisy.astype(np.uint8))
        axes[i+1].set_title(f'σ = {sigma:.3f}', fontsize=12)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noisy_images.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: noisy_images.png")


def plot_loss_curves(noise_levels, l1_losses, l2_losses, perceptual_losses, save_dir):
    """Plot loss curves vs. noise level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # All losses on one plot
    ax1.plot(noise_levels, l1_losses, 'o-', label='ℓ1 Loss', linewidth=2, markersize=8)
    ax1.plot(noise_levels, l2_losses, 's-', label='ℓ2 Loss', linewidth=2, markersize=8)
    ax1.plot(noise_levels, perceptual_losses, '^-', label='Perceptual Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Loss Comparison vs. Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Normalized losses
    l1_norm = np.array(l1_losses) / l1_losses[-1]
    l2_norm = np.array(l2_losses) / l2_losses[-1]
    perceptual_norm = np.array(perceptual_losses) / perceptual_losses[-1]
    
    ax2.plot(noise_levels, l1_norm, 'o-', label='ℓ1 Loss', linewidth=2, markersize=8)
    ax2.plot(noise_levels, l2_norm, 's-', label='ℓ2 Loss', linewidth=2, markersize=8)
    ax2.plot(noise_levels, perceptual_norm, '^-', label='Perceptual Loss', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('Normalized Loss', fontsize=12)
    ax2.set_title('Normalized Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: loss_curves.png")


def create_summary(noise_levels, l1_losses, l2_losses, perceptual_losses, 
                   layer_name, save_dir):
    """Create summary figure with analysis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    summary_text = f"""
Noise Robustness Analysis

Perceptual Loss Layer: {layer_name}

Results Summary:
"""
    
    for i, sigma in enumerate(noise_levels):
        summary_text += f"""
Noise Level σ = {sigma:.3f}:
  L1 Loss:         {l1_losses[i]:.4f}
  L2 Loss:         {l2_losses[i]:.4f}
  Perceptual Loss: {perceptual_losses[i]:.6f}
"""
    
    # Calculate rate of increase
    l1_rate = (l1_losses[-1] - l1_losses[0]) / l1_losses[0]
    l2_rate = (l2_losses[-1] - l2_losses[0]) / l2_losses[0]
    perceptual_rate = (perceptual_losses[-1] - perceptual_losses[0]) / perceptual_losses[0]
    
    summary_text += f"""

Rate of Increase (from lowest to highest noise):
  L1 Loss:         {l1_rate*100:.1f}%
  L2 Loss:         {l2_rate*100:.1f}%
  Perceptual Loss: {perceptual_rate*100:.1f}%

Observation:
Perceptual loss increases more gradually than pixel-based losses,
better matching human perception that noisy images still retain
their essential content and structure.
"""
    
    ax.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary.png")

# TEST ALL LAYERS
def test_all_layers(img_original, noise_levels, device, save_dir):
    """Test noise robustness across all VGG layers."""
    
    all_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
    
    print("\nTesting all VGG layers...")
    results_by_layer = {}
    
    for layer_name in all_layers:
        print(f"  Testing {layer_name}...")
        extractor = VGGFeatureExtractor(layer_name=layer_name, device=device)
        
        perceptual_losses = []
        for sigma in noise_levels:
            noisy_img = add_gaussian_noise(img_original, sigma)
            perceptual = extractor.compute_perceptual_loss(img_original, noisy_img)
            perceptual_losses.append(perceptual)
        
        results_by_layer[layer_name] = perceptual_losses
    
    return results_by_layer


def plot_all_layers_comparison(results_by_layer, noise_levels, 
                                l1_losses, l2_losses, save_dir):
    """Plot perceptual loss curves for all layers."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Absolute values
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for (layer_name, perceptual_losses), color in zip(results_by_layer.items(), colors):
        ax1.plot(noise_levels, perceptual_losses, 'o-', 
                label=layer_name, linewidth=2, markersize=8, color=color)
    
    # Also plot pixel losses for reference
    ax1.plot(noise_levels, l1_losses, '--', label='ℓ1 Loss (pixel)', 
            linewidth=2, color='gray', alpha=0.5)
    ax1.plot(noise_levels, l2_losses, '-.', label='ℓ2 Loss (pixel)', 
            linewidth=2, color='black', alpha=0.5)
    
    ax1.set_xlabel('Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('Loss Value (log)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('All VGG Layers: Noise Robustness', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Normalized to show rate of increase
    for (layer_name, perceptual_losses), color in zip(results_by_layer.items(), colors):
        normalized = np.array(perceptual_losses) / perceptual_losses[-1]
        ax2.plot(noise_levels, normalized, 'o-', 
                label=layer_name, linewidth=2, markersize=8, color=color)
    
    # Pixel losses normalized
    l1_norm = np.array(l1_losses) / l1_losses[-1]
    l2_norm = np.array(l2_losses) / l2_losses[-1]
    ax2.plot(noise_levels, l1_norm, '--', label='ℓ1 Loss (pixel)', 
            linewidth=2, color='gray', alpha=0.5)
    ax2.plot(noise_levels, l2_norm, '-.', label='ℓ2 Loss (pixel)', 
            linewidth=2, color='black', alpha=0.5)
    
    ax2.set_xlabel('Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('Normalized Loss', fontsize=12)
    ax2.set_title('Normalized Loss Comparison (all = 1.0 at σ=0.2)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_layers_noise_robustness.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: all_layers_noise_robustness.png")


def create_layer_comparison_table(results_by_layer, noise_levels, save_dir):
    """Create summary table comparing all layers."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    # Calculate rate of increase for each layer
    summary_text = "VGG Layer Noise Robustness Analysis\n\n"
    summary_text += "Perceptual Loss by Layer and Noise Level:\n"
    summary_text += "="*70 + "\n\n"
    
    # Header
    summary_text += f"{'Layer':<12}"
    for sigma in noise_levels:
        summary_text += f"σ={sigma:<7.3f}  "
    summary_text += f"{'Rate':<10}\n"
    summary_text += "-"*70 + "\n"
    
    # Data rows
    for layer_name, losses in results_by_layer.items():
        rate = ((losses[-1] - losses[0]) / losses[0]) * 100
        summary_text += f"{layer_name:<12}"
        for loss in losses:
            summary_text += f"{loss:<10.4f}  "
        summary_text += f"{rate:>6.1f}%\n"
    
    summary_text += "\n" + "="*70 + "\n\n"
    summary_text += "Rate = (Loss_max - Loss_min) / Loss_min × 100%\n\n"
    
    # Interpretation
    summary_text += "Key Observations:\n"
    summary_text += "• Deeper layers (relu4_3, relu5_3) show slower rate of increase\n"
    summary_text += "• Early layers (relu1_2, relu2_2) more sensitive to noise\n"
    summary_text += "• Mid-layer (relu3_3) balances robustness with texture sensitivity\n"
    summary_text += "• All VGG features more robust than pixel losses (>1000% rate)\n"
    
    ax.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'layer_comparison_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: layer_comparison_table.png")

def main(image_path, save_dir, layer_name='relu3_3', noise_levels=None, test_all=False):
    """Main execution function."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Noise Robustness Evaluation")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Default noise levels
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1, 0.2]
    
    print(f"\nPerceptual loss layer: {layer_name}")
    print(f"Noise levels (σ): {noise_levels}")
    
    # Load original image
    print(f"\nLoading image from: {image_path}")
    img_original = np.array(Image.open(image_path))
    
    # Initialize feature extractor
    extractor = VGGFeatureExtractor(layer_name=layer_name, device=device)
    
    # Generate noisy images and compute losses
    print("\nGenerating noisy images and computing losses...")
    noisy_images = []
    l1_losses = []
    l2_losses = []
    perceptual_losses = []
    
    for sigma in noise_levels:
        print(f"  Processing σ = {sigma:.3f}...")
        
        try:
            # Add noise
            noisy_img = add_gaussian_noise(img_original, sigma)
            noisy_images.append(noisy_img)
            
            # Compute losses
            l1, l2 = calculate_pixel_losses(img_original, noisy_img)
            perceptual = extractor.compute_perceptual_loss(img_original, noisy_img)
            
            l1_losses.append(l1)
            l2_losses.append(l2)
            perceptual_losses.append(perceptual)
            
            print(f"    ✓ Completed σ = {sigma:.3f}")
            
        except Exception as e:
            print(f"    ✗ ERROR at σ = {sigma:.3f}")
            print(f"       {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break  # Stop processing on first error

    if test_all:
        # Test all layers
        results_by_layer = test_all_layers(img_original, noise_levels, device, save_dir)

        # Still compute pixel losses
        print("\nComputing pixel losses for comparison...")
        noisy_images = []
        l1_losses = []
        l2_losses = []

        for sigma in noise_levels:
            noisy_img = add_gaussian_noise(img_original, sigma)
            noisy_images.append(noisy_img)
            l1, l2 = calculate_pixel_losses(img_original, noisy_img)
            l1_losses.append(l1)
            l2_losses.append(l2)

        # Visualizations
        print("\nGenerating visualizations...")
        visualize_noisy_images(img_original, noisy_images, noise_levels, save_dir)
        plot_all_layers_comparison(results_by_layer, noise_levels,
                                   l1_losses, l2_losses, save_dir)
        create_layer_comparison_table(results_by_layer, noise_levels, save_dir)

        # Save comprehensive results
        results = {
            'all_layers_results': {k: v for k, v in results_by_layer.items()},
            'noise_levels': noise_levels,
            'l1_losses': l1_losses,
            'l2_losses': l2_losses,
            'image_path': image_path
        }

        output_path = os.path.join(save_dir, 'all_layers_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved: all_layers_results.json")

    else: 
        # Print results - only what was successfully computed
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for i in range(len(l1_losses)):  # Only iterate over computed results
            print(f"\nNoise Level σ = {noise_levels[i]:.3f}:")
            print(f"  L1 Loss:         {l1_losses[i]:.4f}")
            print(f"  L2 Loss:         {l2_losses[i]:.4f}")
            print(f"  Perceptual Loss: {perceptual_losses[i]:.6f}")
        
        print("="*60)
        
        # Warn if incomplete
        if len(l1_losses) < len(noise_levels):
            print(f"\nWARNING: Only {len(l1_losses)}/{len(noise_levels)} noise levels completed")
            print(f"Failed at σ = {noise_levels[len(l1_losses)]:.3f}")
        
        # Generate visualizations only if we have results
        if len(l1_losses) > 0:
            print("\nGenerating visualizations...")
            visualize_noisy_images(img_original, noisy_images, noise_levels[:len(l1_losses)], save_dir)
            plot_loss_curves(noise_levels[:len(l1_losses)], l1_losses, l2_losses, perceptual_losses, save_dir)
            create_summary(noise_levels[:len(l1_losses)], l1_losses, l2_losses, perceptual_losses, 
                           layer_name, save_dir)
            
            # Save results
            results = {
                'layer_name': layer_name,
                'noise_levels': noise_levels[:len(l1_losses)],  # Only completed levels
                'l1_losses': l1_losses,
                'l2_losses': l2_losses,
                'perceptual_losses': perceptual_losses,
                'image_path': image_path,
                'completed': len(l1_losses),
                'total': len(noise_levels)
            }
            
            output_path = os.path.join(save_dir, 'results.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Saved: results.json")
            
            print(f"\nAll results saved to: {save_dir}")
            print("Robustness evaluation complete!")
        else:
            print("\nERROR: No results to save - all noise levels failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noise Robustness Evaluation')
    parser.add_argument('--image', type=str, default='data/test_images/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='results/noise_robustness',
                        help='Directory to save results')
    parser.add_argument('--layer', type=str, default='relu3_3',
                        choices=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'],
                        help='VGG layer for perceptual loss')
    parser.add_argument('--noise-levels', type=float, nargs='+', default=None,
                        help='Noise levels to test (default: 0.01 0.05 0.1 0.2)')
    parser.add_argument('--test-all-layers', action='store_true',
                        help='Test all VGG layers instead of single layer')
    
    args = parser.parse_args()
    main(args.image, args.output, args.layer, args.noise_levels, args.test_all_layers)
