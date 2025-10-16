"""
VGG-16 Feature Extraction and Analysis
Perceptual Loss for Atmospheric Turbulence

Extract features from VGG-16 layers and analyze robustness to geometric transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
import json
import os
import argparse


class VGGFeatureExtractor:
    """Extract features from specific VGG-16 layers."""
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"Initializing VGG-16 on device: {device}")
        
        # Load pre-trained VGG-16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features.to(device)
        self.features.eval()
        
        # Define layer indices (ReLU activation layers)
        self.layer_names = {
            'relu1_2': 3,   # Early: edges, colors
            'relu2_2': 8,   # Low-mid: simple textures
            'relu3_3': 15,  # Mid: complex patterns
            'relu4_3': 22,  # High: object parts
            'relu5_3': 29   # Very high: semantic content
        }
        
        # VGG preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, img_array):
        """Convert numpy array to VGG input format."""
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_tensor = self.transform(img_pil).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def extract_features(self, img_array, layer_names=None):
        """Extract features from specified layers."""
        if layer_names is None:
            layer_names = list(self.layer_names.keys())
        
        img_tensor = self.preprocess(img_array)
        features = {}
        
        with torch.no_grad():
            for name in layer_names:
                if name not in self.layer_names:
                    continue
                    
                layer_idx = self.layer_names[name]
                x = img_tensor
                
                # Forward pass up to this layer
                for i in range(layer_idx + 1):
                    x = self.features[i](x)
                
                features[name] = x.clone()
        
        return features
    
    def compute_feature_distance(self, features1, features2, metric='l2'):
        """Compute distance between two feature dictionaries."""
        distances = {}
        for layer_name in features1.keys():
            feat1 = features1[layer_name]
            feat2 = features2[layer_name]
            
            if metric == 'l1':
                dist = torch.mean(torch.abs(feat1 - feat2)).item()
            elif metric == 'l2':
                dist = torch.sqrt(torch.mean((feat1 - feat2) ** 2)).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            distances[layer_name] = dist
        
        return distances


def load_images(image_path):
    """Load original and translated images."""
    img_original = np.array(Image.open(image_path))
    img_translated = np.roll(img_original, -1, axis=1)
    return img_original, img_translated


def visualize_architecture(selected_layers, save_dir):
    """Create enhanced VGG-16 architecture diagram with dimensions."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    arch_text = """
VGG-16 Feature Extraction Architecture
(Assuming 224×224×3 input - typical ImageNet preprocessing)

Input Image (224 × 224 × 3)
    ↓
[Conv1_1] → [Conv1_2] → [ReLU1_2] ← Layer 1 (relu1_2)
    3×3×64      3×3×64      Output: 224×224×64
                            Features: Edges, colors, oriented gradients
    ↓
[MaxPool] (2×2, stride 2)
    ↓ Output: 112×112×64
    
[Conv2_1] → [Conv2_2] → [ReLU2_2] ← Layer 2 (relu2_2)
    3×3×128     3×3×128     Output: 112×112×128
                            Features: Simple textures, corner patterns
    ↓
[MaxPool] (2×2, stride 2)
    ↓ Output: 56×56×128
    
[Conv3_1] → [Conv3_2] → [Conv3_3] → [ReLU3_3] ← Layer 3 (relu3_3)
    3×3×256     3×3×256     3×3×256     Output: 56×56×256
                                        Features: Complex patterns, textures
                                        MOST COMMONLY USED FOR PERCEPTUAL LOSS
    ↓
[MaxPool] (2×2, stride 2)
    ↓ Output: 28×28×256
    
[Conv4_1] → [Conv4_2] → [Conv4_3] → [ReLU4_3] ← Layer 4 (relu4_3)
    3×3×512     3×3×512     3×3×512     Output: 28×28×512
                                        Features: Object parts, semantic content
                                        HIGHEST TRANSLATION ROBUSTNESS
    ↓
[MaxPool] (2×2, stride 2)
    ↓ Output: 14×14×512
    
[Conv5_1] → [Conv5_2] → [Conv5_3] → [ReLU5_3] ← Layer 5 (relu5_3)
    3×3×512     3×3×512     3×3×512     Output: 14×14×512
                                        Features: High-level semantics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Selected Layers for Analysis:
• relu1_2: Early features (edges, colors) - High spatial resolution
• relu2_2: Low-mid features (simple textures) - Moderate resolution  
• relu3_3: Mid features (complex patterns) - STANDARD PERCEPTUAL LOSS
• relu4_3: High features (object parts) - MAXIMUM ROBUSTNESS

Feature Map Sizes Decrease with Depth:
    relu1_2: 224×224 = 50,176 spatial locations
    relu2_2: 112×112 = 12,544 spatial locations
    relu3_3: 56×56   = 3,136 spatial locations
    relu4_3: 28×28   = 784 spatial locations

Receptive Field Sizes Increase with Depth:
    relu1_2: ~3×3 pixels
    relu2_2: ~10×10 pixels
    relu3_3: ~40×40 pixels
    relu4_3: ~92×92 pixels

Interpretation: Deeper layers integrate information over larger spatial regions,
making them more robust to local geometric distortions like pixel shifts.
"""
    
    ax.text(0.5, 0.5, arch_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                     edgecolor='darkblue', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vgg16_architecture_enhanced.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: vgg16_architecture_enhanced.png")


def plot_comparison(l1_distances, l2_distances, l1_pixel, l2_pixel, save_dir):
    """Plot feature distances vs pixel losses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    layers = list(l1_distances.keys())
    l1_vals = [l1_distances[layer] for layer in layers]
    l2_vals = [l2_distances[layer] for layer in layers]
    
    # ℓ1 comparison
    ax1.bar(range(len(layers)), l1_vals, color='steelblue', alpha=0.7, label='VGG Features')
    ax1.axhline(y=l1_pixel, color='red', linestyle='--', linewidth=2, 
                label=f'Pixel ℓ1: {l1_pixel:.2f}')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.set_ylabel('ℓ1 Distance (log)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('ℓ1 Feature Distances vs. Pixel Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ℓ2 comparison
    ax2.bar(range(len(layers)), l2_vals, color='darkgreen', alpha=0.7, label='VGG Features')
    ax2.axhline(y=l2_pixel, color='red', linestyle='--', linewidth=2, 
                label=f'Pixel ℓ2: {l2_pixel:.2f}')
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=45, ha='right')
    ax2.set_ylabel('ℓ2 Distance (log)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('ℓ2 Feature Distances vs. Pixel Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_vs_pixel_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: feature_vs_pixel_comparison.png")


def create_recommendation(l2_distances, l2_pixel, recommended_layer, save_dir):
    """Create recommendation summary figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    recommendation_text = f"""
RECOMMENDATION FOR PERCEPTUAL LOSS

Best Layer: {recommended_layer}

Rationale:
• {recommended_layer} shows the smallest feature distance for 1-pixel translation
• This indicates features at this layer are more robust to geometric shifts
• The layer captures mid-level features (patterns, structures)
• It balances:
  - Robustness to geometric distortions (low sensitivity to pixel shifts)
  - Sensitivity to semantic changes (would still detect meaningful differences)

Feature Distance Analysis (l2):
"""
    
    for layer, dist in l2_distances.items():
        norm_dist = dist / l2_pixel
        recommendation_text += f"\n• {layer}: {dist:.6f} ({norm_dist*100:.1f}% of pixel loss)"
    
    recommendation_text += f"""

Pixel l2 loss: {l2_pixel:.6f}

Conclusion:
Using {recommended_layer} features for perceptual loss would better correlate
with human perception compared to pixel-wise lp losses, as it focuses on
structural and semantic similarity rather than exact pixel alignment.
"""
    
    ax.text(0.5, 0.5, recommendation_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perceptual_loss_recommendation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: perceptual_loss_recommendation.png")


def main(image_path, save_dir, layers=None):
    """Main execution function."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("VGG-16 Feature Analysis")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load images
    print(f"\nLoading images from: {image_path}")
    img_original, img_translated = load_images(image_path)
    
    # Select layers
    if layers is None:
        selected_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
    else:
        selected_layers = layers
    
    print(f"\nSelected layers: {selected_layers}")
    
    # Initialize extractor and extract features
    extractor = VGGFeatureExtractor(device=device)
    
    print("\nExtracting VGG-16 features...")
    features_original = extractor.extract_features(img_original, selected_layers)
    features_translated = extractor.extract_features(img_translated, selected_layers)
    
    # Compute distances
    print("Computing feature distances...")
    l1_distances = extractor.compute_feature_distance(features_original, features_translated, metric='l1')
    l2_distances = extractor.compute_feature_distance(features_original, features_translated, metric='l2')
    
    # Calculate pixel losses for comparison
    l1_pixel = np.mean(np.abs(img_original.astype(float) - img_translated.astype(float)))
    l2_pixel = np.sqrt(np.mean((img_original.astype(float) - img_translated.astype(float)) ** 2))
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print("\nl1 Feature Distances:")
    for layer, dist in l1_distances.items():
        print(f"  {layer:12s}: {dist:.6f}")
    
    print(f"\nPixel l1 Loss: {l1_pixel:.6f}")
    
    print("\nl2 Feature Distances:")
    for layer, dist in l2_distances.items():
        print(f"  {layer:12s}: {dist:.6f}")
    
    print(f"\nPixel l2 Loss: {l2_pixel:.6f}")
    
    # Analyze robustness
    normalized_l2 = {layer: dist / l2_pixel for layer, dist in l2_distances.items()}
    recommended_layer = min(normalized_l2, key=normalized_l2.get)
    
    print("\n" + "="*60)
    print("ANALYSIS: Layer Robustness to Translation")
    print("="*60)
    print("\nNormalized l2 distances (relative to pixel ℓ2):")
    for layer, norm_dist in normalized_l2.items():
        print(f"  {layer:12s}: {norm_dist:.4f} ({norm_dist*100:.1f}% of pixel loss)")
    
    print(f"\nMost robust layer: {recommended_layer}")
    print("="*60)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_architecture(selected_layers, save_dir)
    plot_comparison(l1_distances, l2_distances, l1_pixel, l2_pixel, save_dir)
    create_recommendation(l2_distances, l2_pixel, recommended_layer, save_dir)
    
    # Save results
    results = {
        'selected_layers': selected_layers,
        'l1_distances': l1_distances,
        'l2_distances': l2_distances,
        'pixel_l1': float(l1_pixel),
        'pixel_l2': float(l2_pixel),
        'recommended_layer': recommended_layer,
        'normalized_distances': {k: float(v) for k, v in normalized_l2.items()},
        'image_path': image_path
    }
    
    output_path = os.path.join(save_dir, 'part2_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved: part2_results.json")
    
    print(f"\nAll results saved to: {save_dir}")
    print("Feature analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGG-16 Feature Analysis')
    parser.add_argument('--image', type=str, default='data/test_images/sample.jpg',
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='results/part2',
                        help='Directory to save results')
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Specific layers to analyze (default: relu1_2 relu2_2 relu3_3 relu4_3)')
    
    args = parser.parse_args()
    main(args.image, args.output, args.layers)

