#!/usr/bin/env python3
"""
Create train/test split for turbulence dataset.
Splits at clean image level to preserve pairing.

Usage:
    python scripts/create_train_test_split.py \
        --dataset-dir data/part4_dataset \
        --output-dir data/part4_dataset_split \
        --test-ratio 0.2 \
        --seed 42
"""

import json
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)

def load_metadata(metadata_path):
    """Load dataset metadata."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def create_split(dataset_dir, output_dir, test_ratio=0.2, seed=42):
    """
    Create train/test split preserving clean/turbulent pairing.
    
    Args:
        dataset_dir: Path to dataset with clean/ and turbulent/ subdirs
        output_dir: Path to output split dataset
        test_ratio: Fraction of data for test set (default 0.2)
        seed: Random seed for reproducibility
    """
    set_seed(seed)
    
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    metadata_path = dataset_dir / 'dataset_metadata.json'
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    
    # Group turbulent images by clean source
    clean_to_turbulent = defaultdict(list)
    for pair in metadata['turbulent_pairs']:
        clean_name = pair['clean_image']
        clean_to_turbulent[clean_name].append(pair)
    
    # Get all clean image names
    clean_images = sorted(clean_to_turbulent.keys())
    n_total = len(clean_images)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    
    print(f"\nDataset statistics:")
    print(f"  Total clean images: {n_total}")
    print(f"  Train images: {n_train}")
    print(f"  Test images: {n_test}")
    print(f"  Turbulent per clean: {len(clean_to_turbulent[clean_images[0]])}")
    print(f"  Total turbulent train: {n_train * len(clean_to_turbulent[clean_images[0]])}")
    print(f"  Total turbulent test: {n_test * len(clean_to_turbulent[clean_images[0]])}")
    
    # Shuffle and split
    random.shuffle(clean_images)
    train_images = clean_images[:n_train]
    test_images = clean_images[n_train:]
    
    # Create output directories
    for split in ['train', 'test']:
        (output_dir / split / 'clean').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'turbulent').mkdir(parents=True, exist_ok=True)
    
    # Copy files and create split metadata
    train_metadata = {'clean_images': [], 'turbulent_pairs': []}
    test_metadata = {'clean_images': [], 'turbulent_pairs': []}
    
    print("\nCopying files...")
    
    # Process train split
    for clean_name in train_images:
        # Copy clean image
        src = dataset_dir / 'clean' / clean_name
        dst = output_dir / 'train' / 'clean' / clean_name
        shutil.copy2(src, dst)
        
        train_metadata['clean_images'].append({
            'filename': clean_name,
            'path': str(dst)
        })
        
        # Copy all turbulent variations
        for pair in clean_to_turbulent[clean_name]:
            turb_name = pair['turbulent_image']
            src = dataset_dir / 'turbulent' / turb_name
            dst = output_dir / 'train' / 'turbulent' / turb_name
            shutil.copy2(src, dst)
            
            # Update paths in metadata
            pair_copy = pair.copy()
            pair_copy['clean_path'] = str(output_dir / 'train' / 'clean' / clean_name)
            pair_copy['turbulent_path'] = str(dst)
            train_metadata['turbulent_pairs'].append(pair_copy)
    
    # Process test split
    for clean_name in test_images:
        # Copy clean image
        src = dataset_dir / 'clean' / clean_name
        dst = output_dir / 'test' / 'clean' / clean_name
        shutil.copy2(src, dst)
        
        test_metadata['clean_images'].append({
            'filename': clean_name,
            'path': str(dst)
        })
        
        # Copy all turbulent variations
        for pair in clean_to_turbulent[clean_name]:
            turb_name = pair['turbulent_image']
            src = dataset_dir / 'turbulent' / turb_name
            dst = output_dir / 'test' / 'turbulent' / turb_name
            shutil.copy2(src, dst)
            
            # Update paths in metadata
            pair_copy = pair.copy()
            pair_copy['clean_path'] = str(output_dir / 'test' / 'clean' / clean_name)
            pair_copy['turbulent_path'] = str(dst)
            test_metadata['turbulent_pairs'].append(pair_copy)
    
    # Save split metadata
    train_metadata_path = output_dir / 'train' / 'metadata.json'
    test_metadata_path = output_dir / 'test' / 'metadata.json'
    
    with open(train_metadata_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(test_metadata_path, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    # Save split info
    split_info = {
        'seed': seed,
        'test_ratio': test_ratio,
        'n_train_clean': n_train,
        'n_test_clean': n_test,
        'train_images': train_images,
        'test_images': test_images
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit complete!")
    print(f"  Train metadata: {train_metadata_path}")
    print(f"  Test metadata: {test_metadata_path}")
    print(f"  Split info: {output_dir / 'split_info.json'}")
    print(f"\nVerification:")
    print(f"  Train clean: {len(train_metadata['clean_images'])}")
    print(f"  Train turbulent: {len(train_metadata['turbulent_pairs'])}")
    print(f"  Test clean: {len(test_metadata['clean_images'])}")
    print(f"  Test turbulent: {len(test_metadata['turbulent_pairs'])}")

def main():
    parser = argparse.ArgumentParser(
        description='Create train/test split for turbulence dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset-dir', type=str, 
                       default='data/part4_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/part4_dataset_split',
                       help='Output directory for split dataset')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    create_split(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()

