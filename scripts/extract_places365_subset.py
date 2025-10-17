"""
Places365 Dataset Extraction for Part 4 Turbulence Dataset
Extracts high-quality subset from Places365-Standard for turbulence simulation.

This creates the base clean image set that will be used with generate_quickturb_dataset.py
to create the full Part 4 training/test dataset.

Prerequisites:
    1. Download Places365-Standard from:
       http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
    2. Extract in data/ to get: data/places365_standard/train/ directory structure
    
Usage:
    python scripts/extract_places365_subset.py \
        --places-root data/places365_standard/train/ \
        --output-dir data/clean_images/ \
        --images-per-category 10 \
        --total-target 150
        
    Then generate turbulence dataset:
    python scripts/generate_quickturb_dataset.py \
        --input-dir data/clean_images/ \
        --output-dir data/part4_dataset/ \
        --num-variations 5 \
        --presets medium strong
"""

import argparse
import shutil
import random
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


# Target categories: outdoor, edge-rich, texture-rich scenes
TARGET_CATEGORIES = [
    # Architecture with strong edges
    'bridge', 'tower', 'dam', 'castle', 'lighthouse', 'pavilion',
    
    # Natural scenes with texture
    'forest_path', 'mountain', 'canyon', 'cliff', 'forest_road',
    'mountain_snowy', 'tree_farm', 'valley',
    
    # Urban scenes
    'street', 'downtown', 'campus', 'plaza', 'crosswalk',
    'highway', 'parking_lot', 'railroad_track',
    
    # Infrastructure
    'industrial_area', 'construction_site', 'power_plant',
    'wind_farm', 'oil_refinery',
    
    # Waterfront (good horizon lines)
    'boardwalk', 'dock', 'pier', 'harbor', 'lighthouse'
]


def validate_image_quality(img_path, min_edge_density=0.05, min_dynamic_range=100):
    """
    Quality filter for turbulence suitability.
    
    Checks:
        - Edge content (turbulence most visible on edges)
        - Dynamic range (contrast)
        - Valid image format
    
    Args:
        img_path: Path to image file
        min_edge_density: Minimum fraction of edge pixels (0-1)
        min_dynamic_range: Minimum intensity range (0-255)
    
    Returns:
        bool: True if image passes quality checks
    """
    try:
        # Read as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return False
        
        # Resize to target resolution for consistent evaluation
        img = cv2.resize(img, (512, 512))
        
        # Edge density check
        edges = cv2.Canny(img, 50, 150)
        edge_ratio = edges.sum() / edges.size
        
        # Dynamic range check
        dynamic_range = img.max() - img.min()
        
        # Pass if both criteria met
        passed = (edge_ratio > min_edge_density) and (dynamic_range > min_dynamic_range)
        
        return passed
        
    except Exception as e:
        print(f"    Error validating {img_path.name}: {e}")
        return False


def upscale_image(img_path, output_path, target_size=512):
    """
    Upscale image to target size using high-quality resampling.
    Places365-Standard images are 256x256, need 512x512 for turbulence sim.
    
    Args:
        img_path: Input image path
        output_path: Output image path
        target_size: Target dimension (square)
    """
    try:
        img = Image.open(img_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Upscale with LANCZOS (high quality)
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        
        # Save with high quality
        img_resized.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"    Error upscaling {img_path.name}: {e}")
        return False


def extract_places365_subset(
    places_root,
    output_dir,
    categories=None,
    images_per_category=10,
    total_target=150,
    upscale=True,
    target_size=512,
    min_edge_density=0.05,
    min_dynamic_range=100,
    seed=42
):
    """
    Extract subset from Places365-Standard with quality filtering.
    
    Args:
        places_root: Path to places365_standard/train/
        output_dir: Output directory for selected images
        categories: List of category names (None = use defaults)
        images_per_category: Target per category
        total_target: Stop after this many total images
        upscale: Whether to upscale 256x256 -> 512x512
        target_size: Upscaling target size
        min_edge_density: Quality filter threshold
        min_dynamic_range: Quality filter threshold
        seed: Random seed for reproducibility
    
    Returns:
        dict: Metadata about extraction
    """
    random.seed(seed)
    
    if categories is None:
        categories = TARGET_CATEGORIES
    
    places_root = Path(places_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tracking
    metadata = {
        'source': 'Places365-Standard',
        'total_selected': 0,
        'categories': {},
        'upscaled': upscale,
        'target_size': target_size if upscale else 256,
        'quality_filters': {
            'min_edge_density': min_edge_density,
            'min_dynamic_range': min_dynamic_range
        }
    }
    
    selected_count = 0
    
    print("="*60)
    print("Places365 Dataset Extraction")
    print("="*60)
    print(f"Source: {places_root}")
    print(f"Output: {output_dir}")
    print(f"Target categories: {len(categories)}")
    print(f"Images per category: {images_per_category}")
    print(f"Total target: {total_target}")
    print(f"Upscale to {target_size}x{target_size}: {upscale}")
    print()
    
    # Process each category
    for category in tqdm(categories, desc="Categories"):
        if selected_count >= total_target:
            print(f"\nReached target of {total_target} images, stopping.")
            break
        
        cat_dir = places_root / category
        
        if not cat_dir.exists():
            print(f"  Warning: {category} not found, skipping")
            continue
        
        # Get all images in category
        images = list(cat_dir.glob('*.jpg'))
        images.extend(cat_dir.glob('*.JPG'))
        
        if not images:
            print(f"  Warning: No images in {category}, skipping")
            continue
        
        random.shuffle(images)
        
        # Select images with quality filter
        selected_in_category = 0
        checked = 0
        
        for img_path in images:
            if selected_count >= total_target:
                break
            if selected_in_category >= images_per_category:
                break
            
            checked += 1
            
            # Quality check
            if not validate_image_quality(
                img_path, 
                min_edge_density=min_edge_density,
                min_dynamic_range=min_dynamic_range
            ):
                continue
            
            # Generate output filename
            new_name = f"{category}_{selected_in_category:03d}.jpg"
            output_path = output_dir / new_name
            
            # Copy or upscale
            if upscale:
                success = upscale_image(img_path, output_path, target_size)
            else:
                try:
                    shutil.copy(img_path, output_path)
                    success = True
                except Exception as e:
                    print(f"  Error copying {img_path.name}: {e}")
                    success = False
            
            if success:
                selected_in_category += 1
                selected_count += 1
        
        # Record category stats
        metadata['categories'][category] = {
            'selected': selected_in_category,
            'checked': checked,
            'acceptance_rate': selected_in_category / checked if checked > 0 else 0
        }
        
        print(f"  {category}: {selected_in_category}/{images_per_category} " +
              f"(checked {checked}, rate: {selected_in_category/checked*100:.1f}%)")
    
    metadata['total_selected'] = selected_count
    
    # Save metadata
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Extraction Complete")
    print("="*60)
    print(f"Total images selected: {selected_count}")
    print(f"Categories used: {len([c for c in metadata['categories'] if metadata['categories'][c]['selected'] > 0])}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print()
    
    # Summary statistics
    acceptance_rates = [
        meta['acceptance_rate'] 
        for meta in metadata['categories'].values() 
        if meta['checked'] > 0
    ]
    
    if acceptance_rates:
        print(f"Quality filter acceptance rate: {np.mean(acceptance_rates)*100:.1f}% avg")
    
    return metadata


def visualize_sample(output_dir, num_samples=9):
    """
    Create a visualization grid of sample images.
    
    Args:
        output_dir: Directory containing extracted images
        num_samples: Number of samples to show (square root must be int)
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    images = list(output_dir.glob('*.jpg'))
    
    if not images:
        print("No images found to visualize")
        return
    
    # Random sample
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    # Grid dimensions
    grid_size = int(np.sqrt(num_samples))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_path.stem, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to: {output_dir / 'sample_visualization.png'}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract Places365 subset for turbulence dataset'
    )
    
    parser.add_argument('--places-root', type=str, required=True,
                       help='Path to places365_standard/train/ directory')
    parser.add_argument('--output-dir', type=str, default='data/clean_images/',
                       help='Output directory for selected images')
    parser.add_argument('--images-per-category', type=int, default=10,
                       help='Target images per category')
    parser.add_argument('--total-target', type=int, default=150,
                       help='Total target number of images')
    parser.add_argument('--no-upscale', action='store_true',
                       help='Disable upscaling (keep 256x256)')
    parser.add_argument('--target-size', type=int, default=512,
                       help='Target size for upscaling')
    parser.add_argument('--min-edge-density', type=float, default=0.05,
                       help='Minimum edge density threshold')
    parser.add_argument('--min-dynamic-range', type=int, default=100,
                       help='Minimum dynamic range threshold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                       help='Create sample visualization')
    
    args = parser.parse_args()
    
    # Run extraction
    metadata = extract_places365_subset(
        places_root=args.places_root,
        output_dir=args.output_dir,
        images_per_category=args.images_per_category,
        total_target=args.total_target,
        upscale=not args.no_upscale,
        target_size=args.target_size,
        min_edge_density=args.min_edge_density,
        min_dynamic_range=args.min_dynamic_range,
        seed=args.seed
    )
    
    # Optional visualization
    if args.visualize:
        visualize_sample(args.output_dir)

