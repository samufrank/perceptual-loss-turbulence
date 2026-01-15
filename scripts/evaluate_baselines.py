#!/usr/bin/env python3
"""
Evaluate VGG baselines on turbulence dataset.

Experiment 1: VGG relu3_3 (single layer)
Experiment 2: VGG multi-layer (relu2_2 + relu3_3 + relu4_3)

Usage:
    python scripts/evaluate_baselines.py \
        --test-dir data/part4_dataset_split/test \
        --output-dir results/encoder_training/baselines
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.turbulence_encoder import VGGFeatureExtractor, VGGMultiLayerExtractor
from losses.turbulence_losses import FeatureDistanceMetric
from utils.turbulence_dataset import get_dataloader, get_test_transform

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def setup_logging(output_dir):
    """Setup logging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'baseline_evaluation.log'
    
    # Console handler - cleaner format without timestamps
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # File handler - keep timestamps for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate model on test set.
    
    Returns:
        dict with metrics
    """
    model.eval()
    model = model.to(device)
    
    all_l2_distances = []
    all_cosine_sims = []
    
    logger.info(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            clean = batch['clean'].to(device)
            turb = batch['turbulent'].to(device)
            
            # Extract features
            clean_out = model(clean)
            turb_out = model(turb)
            
            # Get bottleneck features
            clean_feat = clean_out['bottleneck']
            turb_feat = turb_out['bottleneck']
            
            # Compute metrics
            l2_dist = FeatureDistanceMetric.l2_distance(clean_feat, turb_feat)
            cosine_sim = FeatureDistanceMetric.cosine_similarity(clean_feat, turb_feat)
            
            all_l2_distances.append(l2_dist.cpu())
            all_cosine_sims.append(cosine_sim.cpu())
    
    # Concatenate all batches
    all_l2_distances = torch.cat(all_l2_distances)
    all_cosine_sims = torch.cat(all_cosine_sims)
    
    # Compute statistics
    metrics = {
        'l2_distance_mean': all_l2_distances.mean().item(),
        'l2_distance_std': all_l2_distances.std().item(),
        'l2_distance_median': all_l2_distances.median().item(),
        'l2_distance_min': all_l2_distances.min().item(),
        'l2_distance_max': all_l2_distances.max().item(),
        'cosine_similarity_mean': all_cosine_sims.mean().item(),
        'cosine_similarity_std': all_cosine_sims.std().item(),
        'cosine_similarity_median': all_cosine_sims.median().item(),
        'n_samples': len(all_l2_distances)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VGG baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test-dir', type=str,
                       default='data/part4_dataset_split/test',
                       help='Path to test directory')
    parser.add_argument('--output-dir', type=str,
                       default='results/encoder_training/baselines',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--resolution', type=int, default=512,
                       help='Image resolution')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info("VGG Baseline Evaluation")
    logger.info("="*60)
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Data loader
    logger.info("\nSetting up data loader...")
    test_transform = get_test_transform(args.resolution)
    test_loader = get_dataloader(
        args.test_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=test_transform
    )
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Results dictionary
    all_results = {}
    
    # ========================================
    # Experiment 1: VGG relu3_3 (single layer)
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Experiment 1: VGG relu3_3 (Single Layer)")
    logger.info("="*60)
    
    model_relu3 = VGGFeatureExtractor(layer_names=['relu3_3'], use_bottleneck=True, bottleneck_dim=512)
    metrics_relu3 = evaluate_model(model_relu3, test_loader, device, logger)
    
    logger.info("\nResults:")
    logger.info(f"  L2 Distance: {metrics_relu3['l2_distance_mean']:.4f} ± {metrics_relu3['l2_distance_std']:.4f}")
    logger.info(f"  L2 Median: {metrics_relu3['l2_distance_median']:.4f}")
    logger.info(f"  L2 Range: [{metrics_relu3['l2_distance_min']:.4f}, {metrics_relu3['l2_distance_max']:.4f}]")
    logger.info(f"  Cosine Similarity: {metrics_relu3['cosine_similarity_mean']:.4f} ± {metrics_relu3['cosine_similarity_std']:.4f}")
    
    all_results['exp1_vgg_relu3_3'] = metrics_relu3
    
    # ========================================
    # Experiment 2: VGG multi-layer
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Experiment 2: VGG Multi-Layer (relu2_2 + relu3_3 + relu4_3)")
    logger.info("="*60)
    
    model_multi = VGGMultiLayerExtractor(
        layer_names=['relu2_2', 'relu3_3', 'relu4_3'],
        layer_weights=[0.2, 0.5, 0.3]
    )
    metrics_multi = evaluate_model(model_multi, test_loader, device, logger)
    
    logger.info("\nResults:")
    logger.info(f"  L2 Distance: {metrics_multi['l2_distance_mean']:.4f} ± {metrics_multi['l2_distance_std']:.4f}")
    logger.info(f"  L2 Median: {metrics_multi['l2_distance_median']:.4f}")
    logger.info(f"  L2 Range: [{metrics_multi['l2_distance_min']:.4f}, {metrics_multi['l2_distance_max']:.4f}]")
    logger.info(f"  Cosine Similarity: {metrics_multi['cosine_similarity_mean']:.4f} ± {metrics_multi['cosine_similarity_std']:.4f}")
    
    all_results['exp2_vgg_multilayer'] = metrics_multi
    
    # ========================================
    # Comparison
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("Comparison")
    logger.info("="*60)
    
    # Compare L2 distances
    l2_relu3 = metrics_relu3['l2_distance_mean']
    l2_multi = metrics_multi['l2_distance_mean']
    l2_improvement = ((l2_relu3 - l2_multi) / l2_relu3) * 100
    
    logger.info(f"\nL2 Distance (lower is better):")
    logger.info(f"  VGG relu3_3:     {l2_relu3:.4f}")
    logger.info(f"  VGG multi-layer: {l2_multi:.4f}")
    if l2_improvement > 0:
        logger.info(f"  Improvement: {l2_improvement:.2f}% (multi-layer better)")
    else:
        logger.info(f"  Change: {l2_improvement:.2f}% (single layer better)")
    
    # Compare cosine similarities
    cos_relu3 = metrics_relu3['cosine_similarity_mean']
    cos_multi = metrics_multi['cosine_similarity_mean']
    cos_improvement = ((cos_multi - cos_relu3) / (1 - cos_relu3)) * 100
    
    logger.info(f"\nCosine Similarity (higher is better):")
    logger.info(f"  VGG relu3_3:     {cos_relu3:.4f}")
    logger.info(f"  VGG multi-layer: {cos_multi:.4f}")
    if cos_improvement > 0:
        logger.info(f"  Improvement: {cos_improvement:.2f}% (multi-layer better)")
    else:
        logger.info(f"  Change: {cos_improvement:.2f}% (single layer better)")
    
    # ========================================
    # Save results
    # ========================================
    results_file = output_dir / 'baseline_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Create comparison table
    comparison_file = output_dir / 'baseline_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write("VGG Baseline Comparison\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Experiment':<30} {'L2 Distance':<15} {'Cosine Sim':<15}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'VGG relu3_3 (Exp 1)':<30} {l2_relu3:<15.4f} {cos_relu3:<15.4f}\n")
        f.write(f"{'VGG multi-layer (Exp 2)':<30} {l2_multi:<15.4f} {cos_multi:<15.4f}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Improvement (%)':<30} {l2_improvement:<15.2f} {cos_improvement:<15.2f}\n")
    
    logger.info(f"Comparison table saved to: {comparison_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Baseline evaluation complete!")
    logger.info("="*60)
    
    # Print target for custom encoders
    logger.info("\nTarget for custom encoders:")
    logger.info(f"  Beat VGG relu3_3: L2 < {l2_relu3:.4f}, Cosine > {cos_relu3:.4f}")
    if l2_multi < l2_relu3:
        logger.info(f"  Beat VGG multi-layer: L2 < {l2_multi:.4f}, Cosine > {cos_multi:.4f}")


if __name__ == '__main__':
    main()

