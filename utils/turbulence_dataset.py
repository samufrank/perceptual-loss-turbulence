"""
PyTorch Dataset for turbulence-robust feature learning.
Loads clean/turbulent image pairs for contrastive learning.
"""

import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TurbulenceDataset(Dataset):
    """
    Dataset for clean/turbulent image pairs.
    
    For contrastive learning, each batch should contain:
    - Positive pairs: (clean, turbulent) from same scene
    - Negative samples: clean images from different scenes
    """
    
    def __init__(self, data_dir, transform=None, return_paths=False):
        """
        Args:
            data_dir: Path to train/ or test/ directory with metadata.json
            transform: Optional torchvision transforms
            return_paths: If True, return image paths for debugging
        """
        self.data_dir = Path(data_dir)
        self.return_paths = return_paths
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.pairs = self.metadata['turbulent_pairs']
        self.clean_images = [c['filename'] for c in self.metadata['clean_images']]
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - clean: Clean image tensor
                - turbulent: Turbulent image tensor
                - clean_filename: Clean image filename (for pairing)
                - paths: Image paths (if return_paths=True)
        """
        pair = self.pairs[idx]
        
        # Load images
        clean_path = Path(pair['clean_path'])
        turb_path = Path(pair['turbulent_path'])
        
        clean_img = Image.open(clean_path).convert('RGB')
        turb_img = Image.open(turb_path).convert('RGB')
        
        # Apply transforms
        clean_tensor = self.transform(clean_img)
        turb_tensor = self.transform(turb_img)
        
        sample = {
            'clean': clean_tensor,
            'turbulent': turb_tensor,
            'clean_filename': pair['clean_image']
        }
        
        if self.return_paths:
            sample['paths'] = {
                'clean': str(clean_path),
                'turbulent': str(turb_path)
            }
        
        return sample


def get_contrastive_batch_sampler(dataset, batch_size):
    """
    Create batches ensuring diversity for contrastive learning.
    Each batch should have images from different scenes for negative sampling.
    
    Args:
        dataset: TurbulenceDataset instance
        batch_size: Batch size
        
    Returns:
        List of batch indices
    """
    # Group indices by clean image
    from collections import defaultdict
    clean_to_indices = defaultdict(list)
    
    for idx, pair in enumerate(dataset.pairs):
        clean_name = pair['clean_image']
        clean_to_indices[clean_name].append(idx)
    
    # Create batches with diverse scenes
    batches = []
    clean_names = list(clean_to_indices.keys())
    
    # Shuffle clean names
    import random
    random.shuffle(clean_names)
    
    current_batch = []
    used_cleans = set()
    
    for clean_name in clean_names:
        if clean_name in used_cleans:
            continue
            
        # Add one sample from this clean image
        idx = random.choice(clean_to_indices[clean_name])
        current_batch.append(idx)
        used_cleans.add(clean_name)
        
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []
            used_cleans = set()
    
    # Handle remaining samples
    if len(current_batch) > 0:
        batches.append(current_batch)
    
    return batches


def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=4,
                   transform=None, contrastive_sampling=False):
    """
    Create DataLoader for training or testing.
    
    Args:
        data_dir: Path to train/ or test/ directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transforms
        contrastive_sampling: Use contrastive batch sampling
        
    Returns:
        DataLoader instance
    """
    dataset = TurbulenceDataset(data_dir, transform=transform)
    
    if contrastive_sampling:
        # Use custom batch sampler for contrastive learning
        from torch.utils.data import BatchSampler, SequentialSampler
        batch_indices = get_contrastive_batch_sampler(dataset, batch_size)
        
        # Flatten batch indices for sampler
        all_indices = [idx for batch in batch_indices for idx in batch]
        sampler = SequentialSampler(all_indices)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # Standard DataLoader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loader


def get_augmentation_transform(resolution=512):
    """
    Get training augmentation transforms.
    Moderate augmentation to improve generalization without changing turbulence characteristics.
    
    Args:
        resolution: Target image resolution
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((int(resolution * 0.9), int(resolution * 0.9))),
        transforms.Resize((resolution, resolution)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_test_transform(resolution=512):
    """
    Get test transforms (no augmentation).
    
    Args:
        resolution: Target image resolution
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    # Test dataset loading
    print("Testing TurbulenceDataset...")
    
    data_dir = 'data/part4_dataset_split/train'
    dataset = TurbulenceDataset(data_dir, return_paths=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of unique clean images: {len(dataset.clean_images)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Clean shape: {sample['clean'].shape}")
    print(f"  Turbulent shape: {sample['turbulent'].shape}")
    print(f"  Clean filename: {sample['clean_filename']}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    loader = get_dataloader(data_dir, batch_size=4, shuffle=True, num_workers=0)
    
    batch = next(iter(loader))
    print(f"Batch clean shape: {batch['clean'].shape}")
    print(f"Batch turbulent shape: {batch['turbulent'].shape}")
    print(f"Batch clean filenames: {batch['clean_filename']}")
    
    print("\nDataset test complete!")

