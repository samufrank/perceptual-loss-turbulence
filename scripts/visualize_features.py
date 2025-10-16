"""
Feature Visualization for VGG Layers
Implements 3 visualization methods:
1. Activation maps (what each channel detects)
2. Saliency maps (where layer is looking)
3. Feature space PCA (clustering analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import argparse


class VGGFeatureExtractor:
    """Extract features from VGG-16 for visualization."""
    
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


# METHOD 1: Activation Maps
def visualize_activation_maps(img, layer_name, extractor, save_dir, num_channels=16):
    """Visualize what individual channels detect in a layer."""
    
    features = extractor.extract_features(img, layer_name)
    features = features.squeeze(0).cpu().numpy()  # [C, H, W]
    
    # Select channels to show (evenly spaced)
    num_channels = min(num_channels, features.shape[0])
    channel_indices = np.linspace(0, features.shape[0]-1, num_channels, dtype=int)
    
    # Create grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    
    for idx, (ax, channel_idx) in enumerate(zip(axes.flat, channel_indices)):
        feature_map = features[channel_idx]
        
        # Normalize for visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Channel {channel_idx}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f'{layer_name} Activation Maps (16 of {features.shape[0]} channels)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name}_activation_maps.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {layer_name}_activation_maps.png")


# METHOD 2: Saliency Maps
def compute_saliency_map(img, layer_name, extractor, save_dir):
    """Compute gradient-based saliency showing where layer focuses."""
    
    img_tensor = extractor.preprocess(img)
    img_tensor.requires_grad = True
    
    layer_idx = extractor.layer_indices[layer_name]
    
    # Forward pass
    x = img_tensor
    for i in range(layer_idx + 1):
        x = extractor.features[i](x)
    
    # Maximize mean activation
    loss = x.mean()
    loss.backward()
    
    # Get gradient (saliency)
    saliency = img_tensor.grad.abs().mean(dim=1).squeeze().cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Saliency map
    im = axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f'{layer_name} Saliency Map', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(saliency, cmap='hot', alpha=0.5)
    axes[2].set_title('Saliency Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f'{layer_name}: Where the Layer "Looks"', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name}_saliency_map.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {layer_name}_saliency_map.png")


# METHOD 3: Feature Space PCA/t-SNE
def visualize_feature_space(clean_img, distorted_imgs, distortion_labels, 
                            layer_name, extractor, save_dir, method='pca'):
    """Show feature space clustering using PCA or t-SNE."""
    
    all_features = []
    labels = []
    
    # Clean image features
    clean_feat = extractor.extract_features(clean_img, layer_name)
    clean_feat = clean_feat.flatten().cpu().numpy()
    all_features.append(clean_feat)
    labels.append('Clean')
    
    # Distorted image features
    for dist_img, label in zip(distorted_imgs, distortion_labels):
        dist_feat = extractor.extract_features(dist_img, layer_name)
        dist_feat = dist_feat.flatten().cpu().numpy()
        all_features.append(dist_feat)
        labels.append(label)
    
    # Dimensionality reduction
    features_array = np.array(all_features)
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        features_2d = reducer.fit_transform(features_array)
        var1, var2 = reducer.explained_variance_ratio_
        xlabel = f'PC1 ({var1:.1%} variance)'
        ylabel = f'PC2 ({var2:.1%} variance)'
        title_method = 'PCA'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features_array)
        xlabel = 't-SNE Dimension 1'
        ylabel = 't-SNE Dimension 2'
        title_method = 't-SNE'
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Clean image (green star)
    ax.scatter(features_2d[0, 0], features_2d[0, 1], 
              c='green', s=400, marker='*', 
              label='Clean', edgecolors='black', linewidths=2, zorder=10)
    
    # Distorted images (red circles)
    ax.scatter(features_2d[1:, 0], features_2d[1:, 1], 
              c='red', s=150, alpha=0.6, 
              label='Distorted', edgecolors='black', linewidths=1)
    
    # Annotate each point
    for i, label in enumerate(labels):
        ax.annotate(label, (features_2d[i, 0], features_2d[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    # Calculate mean distance from clean to distorted
    distances = np.sqrt(np.sum((features_2d[1:] - features_2d[0])**2, axis=1))
    mean_dist = np.mean(distances)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{layer_name} Feature Space ({title_method})\n'
                f'Mean Distance: {mean_dist:.3f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{layer_name}_feature_space_{method}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {layer_name}_feature_space_{method}.png")
    
    return mean_dist


def add_gaussian_noise(img, sigma):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma * 255, img.shape)
    noisy_img = img.astype(float) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def main(image_path, save_dir, layers=None, methods=None):
    """Main execution."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("VGG Feature Visualization")
    print("="*60)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}\n")
    
    # Load image
    print(f"Loading image from: {image_path}")
    img_original = np.array(Image.open(image_path))
    
    # Default layers
    if layers is None:
        layers = ['relu3_3', 'relu4_3']
    
    # Default methods
    if methods is None:
        methods = ['activation', 'saliency', 'pca']
    
    # Initialize extractor
    extractor = VGGFeatureExtractor(device=device)
    
    # Process each layer
    for layer_name in layers:
        print(f"\nVisualizing {layer_name}...")
        
        # Method 1: Activation maps
        if 'activation' in methods:
            print("  - Generating activation maps...")
            visualize_activation_maps(img_original, layer_name, extractor, save_dir)
        
        # Method 2: Saliency maps
        if 'saliency' in methods:
            print("  - Computing saliency map...")
            compute_saliency_map(img_original, layer_name, extractor, save_dir)
        
        # Method 3: Feature space visualization
        if 'pca' in methods or 'tsne' in methods:
            print("  - Generating feature space visualization...")
            
            # Create distorted versions for clustering
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            distorted_imgs = [add_gaussian_noise(img_original, sigma) for sigma in noise_levels]
            distortion_labels = [f'σ={sigma:.2f}' for sigma in noise_levels]
            
            if 'pca' in methods:
                dist = visualize_feature_space(
                    img_original, distorted_imgs, distortion_labels,
                    layer_name, extractor, save_dir, method='pca'
                )
                print(f"    PCA mean distance: {dist:.4f}")
            
            if 'tsne' in methods:
                dist = visualize_feature_space(
                    img_original, distorted_imgs, distortion_labels,
                    layer_name, extractor, save_dir, method='tsne'
                )
                print(f"    t-SNE mean distance: {dist:.4f}")
    
    print(f"\nAll visualizations saved to: {save_dir}")
    print("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGG Feature Visualization')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='results/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--layers', type=str, nargs='+', 
                        default=['relu3_3', 'relu4_3'],
                        help='VGG layers to visualize')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['activation', 'saliency', 'pca', 'tsne'],
                        default=['activation', 'saliency', 'pca'],
                        help='Visualization methods to use')
    
    args = parser.parse_args()
    main(args.image, args.output, args.layers, args.methods)
