#!/usr/bin/env python3
"""
Training script for turbulence-robust feature encoder.

Usage:
    python scripts/train_encoder.py --config configs/exp3_spatial_only.yaml
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.turbulence_encoder import get_model
from losses.turbulence_losses import TurbulenceRobustLoss, FeatureDistanceMetric
from utils.turbulence_dataset import get_dataloader, get_augmentation_transform, get_test_transform


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    """Setup logging to file and console."""
    log_file = output_dir / 'training.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, metrics, checkpoint_dir, 
                    filename='checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})


def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {
        'total': [],
        'spatial': [],
        'frequency': [],
        'contrastive': []
    }
    
    for batch_idx, batch in enumerate(train_loader):
        clean = batch['clean'].to(device)
        turb = batch['turbulent'].to(device)
        filenames = batch['clean_filename']
        
        # Forward pass
        optimizer.zero_grad()
        
        clean_out = model(clean)
        turb_out = model(turb)
        
        clean_feat = clean_out['bottleneck']
        turb_feat = turb_out['bottleneck']
        
        # Compute loss
        losses = criterion(clean_feat, turb_feat, filenames)
        
        # Backward pass
        losses['total'].backward()
        optimizer.step()
        
        # Record losses
        for key in epoch_losses.keys():
            epoch_losses[key].append(losses[key].item())
        
        # Log batch
        if batch_idx % 10 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(train_loader)}: "
                f"Loss={losses['total'].item():.4f}"
            )
    
    # Compute epoch averages
    avg_losses = {key: np.mean(vals) for key, vals in epoch_losses.items()}
    
    return avg_losses


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    
    val_losses = {
        'total': [],
        'spatial': [],
        'frequency': [],
        'contrastive': []
    }
    
    with torch.no_grad():
        for batch in val_loader:
            clean = batch['clean'].to(device)
            turb = batch['turbulent'].to(device)
            filenames = batch['clean_filename']
            
            # Forward pass
            clean_out = model(clean)
            turb_out = model(turb)
            
            clean_feat = clean_out['bottleneck']
            turb_feat = turb_out['bottleneck']
            
            # Compute loss
            losses = criterion(clean_feat, turb_feat, filenames)
            
            # Record losses
            for key in val_losses.keys():
                val_losses[key].append(losses[key].item())
    
    # Compute averages
    avg_losses = {key: np.mean(vals) for key, vals in val_losses.items()}
    
    # Compute feature distance metrics
    metrics = FeatureDistanceMetric.mean_feature_distance(model, val_loader, device)
    avg_losses.update(metrics)
    
    return avg_losses


def train(config):
    """Main training function."""
    
    # Setup
    output_dir = Path(config['output']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("Starting training")
    logger.info("="*60)
    logger.info(f"Config: {config}")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Data loaders
    logger.info("Setting up data loaders...")
    
    train_transform = get_augmentation_transform(config['data']['resolution'])
    test_transform = get_test_transform(config['data']['resolution'])
    
    train_loader = get_dataloader(
        config['data']['train_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        transform=train_transform
    )
    
    test_loader = get_dataloader(
        config['data']['test_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        transform=test_transform
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Model
    logger.info("Creating model...")
    model = get_model(**config['model'])
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss
    criterion = TurbulenceRobustLoss(**config['loss'])
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    
    # Scheduler
    if config['training'].get('use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Training loop
    logger.info("Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_losses = validate(model, test_loader, criterion, device)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config['training']['learning_rate']
        
        # Log
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_losses['total']:.4f}")
        logger.info(f"  Val Loss: {val_losses['total']:.4f}")
        logger.info(f"  Val L2 Distance: {val_losses['l2_distance']:.4f}")
        logger.info(f"  Val Cosine Sim: {val_losses['cosine_similarity']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_losses['total'], epoch)
        writer.add_scalar('Loss/val', val_losses['total'], epoch)
        writer.add_scalar('Metrics/l2_distance', val_losses['l2_distance'], epoch)
        writer.add_scalar('Metrics/cosine_similarity', val_losses['cosine_similarity'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_every', 5) == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_losses['total'], val_losses,
                output_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, val_losses['total'], val_losses,
                output_dir, 'best_model.pth'
            )
            logger.info(f"New best model! Val loss: {best_val_loss:.4f}")
    
    # Final save
    final_path = save_checkpoint(
        model, optimizer, config['training']['epochs']-1,
        val_losses['total'], val_losses,
        output_dir, 'final_model.pth'
    )
    logger.info(f"Training complete! Final model: {final_path}")
    
    writer.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train turbulence-robust encoder')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    model = train(config)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

