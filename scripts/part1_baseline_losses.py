"""
Baseline ℓp Losses Analysis
Perceptual Loss for Atmospheric Turbulence

Demonstrates limitations of ℓ1 and ℓ2 losses on geometrically shifted images.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import argparse


def load_image(path):
    """Load image and convert to numpy array."""
    img = Image.open(path)
    width, height = img.size
    print(f"Image size: {width} x {height}")
    
    if min(width, height) < 1080:
        print(f"Warning: Image resolution is below 1080p")
    
    return np.array(img)


def translate_left(img, pixels=1):
    """Translate image left by specified pixels using np.roll."""
    return np.roll(img, -pixels, axis=1)


def calculate_l1_loss(img1, img2):
    """Calculate ℓ1 loss between two images."""
    return np.mean(np.abs(img1.astype(float) - img2.astype(float)))


def calculate_l2_loss(img1, img2):
    """Calculate ℓ2 (RMSE) loss between two images."""
    return np.sqrt(np.mean((img1.astype(float) - img2.astype(float)) ** 2))


def visualize_translation(img_original, img_translated, save_dir):
    """Visualize original, translated, and difference images."""
    diff = np.abs(img_original.astype(float) - img_translated.astype(float))
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_translated)
    axes[1].set_title('Translated (1 pixel left)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(diff_amplified)
    axes[2].set_title('Difference (5x amplified)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'translation_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: translation_comparison.png")


def create_summary_figure(img_original, img_translated, l1_loss, l2_loss, save_dir):
    """Create comprehensive summary figure."""
    diff = np.abs(img_original.astype(float) - img_translated.astype(float))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Original
    axes[0, 0].imshow(img_original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Translated
    axes[0, 1].imshow(img_translated)
    axes[0, 1].set_title('Translated (1 pixel left)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference heatmap
    im = axes[1, 0].imshow(diff, cmap='hot')
    axes[1, 0].set_title('Absolute Difference Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Loss values text
    axes[1, 1].axis('off')
    loss_text = f"""
Loss Metrics for 1-Pixel Translation

l1 Loss: {l1_loss:.4f}
l2 Loss: {l2_loss:.4f}

Observation:
Despite the images appearing nearly 
identical to human perception, the 
pixel-wise losses are non-trivial.

This demonstrates the limitation of
lp losses for measuring perceptual
similarity.
"""
    axes[1, 1].text(0.1, 0.5, loss_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'part1_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: part1_summary.png")


def save_results(results, save_dir):
    """Save numerical results to JSON."""
    output_path = os.path.join(save_dir, 'part1_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved: part1_results.json")


def main(image_path, save_dir):
    """Main execution function."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Baseline ℓp Losses Analysis")
    print("="*60)
    
    # Load image
    print(f"\nLoading image from: {image_path}")
    img_original = load_image(image_path)
    print(f"Image shape: {img_original.shape}")
    print(f"Image dtype: {img_original.dtype}")
    print(f"Value range: [{img_original.min()}, {img_original.max()}]")
    
    # Save original image
    plt.figure(figsize=(12, 8))
    plt.imshow(img_original)
    plt.title('Original Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'original_image.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: original_image.png")
    
    # Translate image
    print("\nTranslating image 1 pixel to the left...")
    img_translated = translate_left(img_original, pixels=1)
    
    # Calculate losses
    print("Calculating losses...")
    l1_loss = calculate_l1_loss(img_original, img_translated)
    l2_loss = calculate_l2_loss(img_original, img_translated)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"l1 Loss: {l1_loss:.4f}")
    print(f"l2 Loss (RMSE): {l2_loss:.4f}")
    print("="*60)
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_translation(img_original, img_translated, save_dir)
    create_summary_figure(img_original, img_translated, l1_loss, l2_loss, save_dir)
    
    # Save results
    results = {
        'l1_loss': float(l1_loss),
        'l2_loss': float(l2_loss),
        'image_shape': list(img_original.shape),
        'translation_pixels': 1,
        'image_path': image_path
    }
    save_results(results, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    print("Baseline analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline lp Losses Analysis')
    parser.add_argument('--image', type=str, default='data/test_images/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='results/part1',
                        help='Directory to save results')
    
    args = parser.parse_args()
    main(args.image, args.output)

