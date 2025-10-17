#!/usr/bin/env python3
"""
Export all TensorBoard data to CSV and generate high-quality plots.

Usage:
    python scripts/export_tensorboard_data.py \
        --logdir results/part4 \
        --output-dir results/part4/analysis
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Set matplotlib to use high-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_tensorboard_data(log_dir):
    """Load all scalar data from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        scalars = ea.Scalars(tag)
        data[tag] = {
            'steps': [s.step for s in scalars],
            'values': [s.value for s in scalars],
            'wall_time': [s.wall_time for s in scalars]
        }
    
    return data


def export_to_csv(data, output_file, experiment_name):
    """Export scalar data to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        tags = list(data.keys())
        writer.writerow(['step'] + tags)
        
        # Get max steps
        max_steps = max(len(data[tag]['steps']) for tag in tags)
        
        # Write data
        for i in range(max_steps):
            row = [i]
            for tag in tags:
                if i < len(data[tag]['steps']):
                    row.append(data[tag]['values'][i])
                else:
                    row.append('')
            writer.writerow(row)
    
    print(f"Exported {experiment_name} to {output_file}")


def plot_loss_comparison(all_data, output_file):
    """Plot training and validation loss side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training loss (log scale)
    for exp_name, data in all_data.items():
        if 'Loss/train' in data:
            steps = data['Loss/train']['steps']
            values = data['Loss/train']['values']
            axes[0].plot(steps, values, label=exp_name, 
                        marker='o', markersize=3, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (log)', fontweight='bold')
    axes[0].set_title('Training Loss Comparison', fontweight='bold', fontsize=14)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss (log scale)
    for exp_name, data in all_data.items():
        if 'Loss/val' in data:
            steps = data['Loss/val']['steps']
            values = data['Loss/val']['values']
            axes[1].plot(steps, values, label=exp_name, 
                        marker='o', markersize=3, linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Loss (log)', fontweight='bold')
    axes[1].set_title('Validation Loss Comparison', fontweight='bold', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")


def plot_metrics_comparison(all_data, output_file):
    """Plot L2 distance and cosine similarity side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # L2 Distance
    for exp_name, data in all_data.items():
        if 'Metrics/l2_distance' in data:
            steps = data['Metrics/l2_distance']['steps']
            values = data['Metrics/l2_distance']['values']
            axes[0].plot(steps, values, label=exp_name, 
                        marker='o', markersize=3, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('L2 Distance (log)', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].set_title('Feature Distance Comparison (Lower = better)', fontweight='bold', fontsize=14)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Cosine Similarity
    for exp_name, data in all_data.items():
        if 'Metrics/cosine_similarity' in data:
            steps = data['Metrics/cosine_similarity']['steps']
            values = data['Metrics/cosine_similarity']['values']
            axes[1].plot(steps, values, label=exp_name, 
                        marker='o', markersize=3, linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Cosine Similarity', fontweight='bold')
    axes[1].set_title('Cosine Similarity Comparison (Higher = better)', fontweight='bold', fontsize=14)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")

def create_comparison_table(all_data, output_file):
    """Create comparison table of final metrics."""
    with open(output_file, 'w') as f:
        f.write("Experiment Comparison - Final Epoch Metrics\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Experiment':<30} {'Train Loss':<15} {'Val Loss':<15} "
                f"{'L2 Distance':<15} {'Cosine Sim':<15}\n")
        f.write("-" * 80 + "\n")
        
        for exp_name, data in all_data.items():
            # Get final values
            train_loss = data.get('Loss/train', {}).get('values', [None])[-1]
            val_loss = data.get('Loss/val', {}).get('values', [None])[-1]
            l2_dist = data.get('Metrics/l2_distance', {}).get('values', [None])[-1]
            cosine = data.get('Metrics/cosine_similarity', {}).get('values', [None])[-1]
            
            f.write(f"{exp_name:<30} "
                   f"{train_loss if train_loss else 'N/A':<15.6f} "
                   f"{val_loss if val_loss else 'N/A':<15.6f} "
                   f"{l2_dist if l2_dist else 'N/A':<15.6f} "
                   f"{cosine if cosine else 'N/A':<15.6f}\n")
    
    print(f"Saved comparison table: {output_file}")


def load_from_csv(csv_file):
    """Load data from CSV file."""
    import csv
    
    data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Get all column names except 'step'
        tags = [col for col in rows[0].keys() if col != 'step']
        
        for tag in tags:
            steps = []
            values = []
            for row in rows:
                if row[tag]:  # Not empty
                    steps.append(int(row['step']))
                    values.append(float(row[tag]))
            
            if steps:
                data[tag] = {
                    'steps': steps,
                    'values': values,
                    'wall_time': []
                }
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Export TensorBoard data and create plots'
    )
    
    parser.add_argument('--logdir', type=str, default='results/part4',
                       help='Base directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='results/part4/analysis',
                       help='Output directory for exported data and plots')
    parser.add_argument('--experiments', type=str, nargs='+',
                       default=['exp3_spatial_only', 'exp4_spatial_frequency',
                               'exp5_spatial_contrastive', 'exp6_full_model'],
                       help='List of experiment names to process')
    parser.add_argument('--from-csv', action='store_true',
                       help='Load data from existing CSV files instead of TensorBoard')
    
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    csv_dir = output_dir / 'csv'
    plots_dir = output_dir / 'plots'
    csv_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Exporting TensorBoard Data")
    print("=" * 60)
    
    # Load all experiment data
    all_data = {}
    
    if args.from_csv:
        print("\nLoading from CSV files...")
        for exp_name in args.experiments:
            csv_file = csv_dir / f'{exp_name}.csv'
            
            if not csv_file.exists():
                print(f"Warning: {csv_file} not found, skipping {exp_name}")
                continue
            
            print(f"Loading {exp_name} from CSV...")
            data = load_from_csv(csv_file)
            all_data[exp_name] = data
    else:
        print("\nLoading from TensorBoard...")
        for exp_name in args.experiments:
            exp_dir = logdir / exp_name / 'tensorboard'
            
            if not exp_dir.exists():
                print(f"Warning: {exp_dir} not found, skipping {exp_name}")
                continue
            
            print(f"Loading {exp_name}...")
            data = load_tensorboard_data(exp_dir)
            all_data[exp_name] = data
            
            # Export to CSV (only when loading from TensorBoard)
            csv_file = csv_dir / f'{exp_name}.csv'
            export_to_csv(data, csv_file, exp_name)
    
    if not all_data:
        print("\nNo data found! Check --logdir path.")
        return
    
    # Create comparison plots
    print("\n" + "=" * 60)
    print("Creating Comparison Plots")
    print("=" * 60)
    
    # Loss comparison (train + val side by side)
    plot_loss_comparison(all_data, plots_dir / 'loss_comparison.png')
    
    # Metrics comparison (L2 + cosine side by side)
    plot_metrics_comparison(all_data, plots_dir / 'metrics_comparison.png')
    
    # Create comparison table
    print("\n" + "=" * 60)
    print("Creating Comparison Table")
    print("=" * 60)
    
    table_file = output_dir / 'comparison_table.txt'
    create_comparison_table(all_data, table_file)
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  CSV files: {csv_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Table: {table_file}")


if __name__ == '__main__':
    main()
