"""
Simple QuickTurbSim Dataset Generator

Generates turbulence datasets with configurable parameters and metadata tracking.

Usage:
    # Generate dataset with multiple variations per image
    python scripts/generate_quickturb_dataset.py \
        --input-dir data/clean_images/ \
        --output-dir data/turbulence_dataset/ \
        --num-variations 5 \
        --presets weak medium strong
        
    # Generate with random seeds for diversity
    python scripts/generate_quickturb_dataset.py \
        --input-dir data/clean_images/ \
        --output-dir data/turbulence_dataset/ \
        --num-variations 5 \
        --presets medium \
        --use-random-seeds
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
import re
import random

# QuickTurbSim path
QUICKTURB_PATH = Path(__file__).parent.parent / 'QuickTurbSim'


def modify_quickturb_script(input_image_name, turb_strength, sigma, turb_scale, num_frames):
    """
    Modify QuickTurbSim's RunQuickTurbSim.py config dictionary.
    
    Returns the modified content as a string.
    """
    script_path = QUICKTURB_PATH / 'RunQuickTurbSim.py'
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace parameters in config dictionary
    replacements = {
        'input_img': f"'input/{input_image_name}'",
        'TurbulenceStrength': str(turb_strength),
        'sigma': str(sigma),
        'TurbScale': turb_scale,  # Already formatted as string like "1 / 64"
        'num_frames_in_video': str(num_frames),
        'num_videos': '1',
    }
    
    # Apply replacements - look for 'key': value patterns in config dict
    lines = content.split('\n')
    for i, line in enumerate(lines):
        for param, value in replacements.items():
            # Match patterns like: 'TurbulenceStrength': 10,
            # Capture the key, colon, and trailing comma/whitespace
            pattern = rf"^(\s*['\"]?{param}['\"]?\s*:\s*)([^,\n]+)(,?.*)$"
            match = re.match(pattern, line)
            if match:
                # Keep the indentation, key, colon, and trailing comma
                lines[i] = f"{match.group(1)}{value}{match.group(3)}"
                break
    
    return '\n'.join(lines)


def run_quickturb(input_image_path, turb_strength, sigma, turb_scale, num_frames=1, clear_output=True):
    """
    Run QuickTurbSim on a single image.
    
    Args:
        turb_scale: Turbulence scale as string (e.g., "1 / 64")
        clear_output: If True, clear output directory before running
    
    Returns list of generated frame paths.
    """
    input_path = Path(input_image_path)
    
    # Copy image to QuickTurbSim/input/
    input_dir = QUICKTURB_PATH / 'input'
    input_dir.mkdir(exist_ok=True)
    shutil.copy(input_path, input_dir / input_path.name)
    
    # Clear previous outputs only if requested
    output_dir = QUICKTURB_PATH / 'Simulated_Images'
    if clear_output:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
    
    # Backup and modify script
    script_path = QUICKTURB_PATH / 'RunQuickTurbSim.py'
    backup_path = QUICKTURB_PATH / 'RunQuickTurbSim_backup.py'
    
    with open(script_path, 'r') as f:
        original_content = f.read()
    
    with open(backup_path, 'w') as f:
        f.write(original_content)
    
    try:
        # Write modified script
        modified_content = modify_quickturb_script(
            input_path.name, turb_strength, sigma, turb_scale, num_frames
        )
        
        with open(script_path, 'w') as f:
            f.write(modified_content)
        
        # Run QuickTurbSim
        result = subprocess.run(
            [sys.executable, 'RunQuickTurbSim.py'],
            cwd=str(QUICKTURB_PATH),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"    QuickTurbSim error: {result.stderr[:200]}")
            return []
        
        # Find generated frames
        frames = list(output_dir.glob('frame_*.png'))
        return frames
        
    finally:
        # Restore original script
        with open(backup_path, 'r') as f:
            original = f.read()
        with open(script_path, 'w') as f:
            f.write(original)
        backup_path.unlink()


def generate_dataset(input_dir, output_dir, num_variations, presets, use_random_seeds=False, custom_configs=None):
    """
    Generate full turbulence dataset.
    
    Args:
        custom_configs: Optional list of dicts with custom turbulence parameters
                       Format: [{'name': 'extreme', 'turb_strength': 15, 'sigma': 20, 'turb_scale': '1 / 32'}, ...]
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    clean_dir = output_dir / 'clean'
    turbulent_dir = output_dir / 'turbulent'
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    turbulent_dir.mkdir(parents=True, exist_ok=True)
    
    # Turbulence presets (from QuickTurbSim README examples)
    turbulence_presets = {
        'weak': {
            'turb_strength': 3,
            'sigma': 5,
            'turb_scale': '1 / 64'
        },
        'medium': {
            'turb_strength': 7,
            'sigma': 10,
            'turb_scale': '1 / 96'
        },
        'strong': {
            'turb_strength': 10,
            'sigma': 15,
            'turb_scale': '1 / 128'
        }
    }
    
    # Add custom configs if provided
    if custom_configs:
        for config in custom_configs:
            turbulence_presets[config['name']] = {
                'turb_strength': config['turb_strength'],
                'sigma': config['sigma'],
                'turb_scale': config['turb_scale']
            }
        presets.extend([c['name'] for c in custom_configs])
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
        image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    print(f"Generating {num_variations} variations at {len(presets)} turbulence levels")
    print(f"Total: {len(image_paths) * num_variations * len(presets)} turbulent images")
    if use_random_seeds:
        print("Using random seeds for variation diversity")
    print()
    
    metadata = {
        'simulation_method': 'QuickTurbSim',
        'clean_images': [],
        'turbulent_pairs': [],
        'turbulence_presets': turbulence_presets
    }
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        print(f"\nProcessing: {img_path.name}")
        
        # Save clean image
        clean_save_path = clean_dir / img_path.name
        shutil.copy(img_path, clean_save_path)
        
        metadata['clean_images'].append({
            'filename': img_path.name,
            'path': str(clean_save_path)
        })
        
        # Process each turbulence level
        for preset_name in presets:
            preset = turbulence_presets[preset_name]
            
            print(f"  Level: {preset_name} (strength={preset['turb_strength']}, sigma={preset['sigma']}, scale={preset['turb_scale']})")
            
            # Clear output directory once per preset level
            output_dir_sim = QUICKTURB_PATH / 'Simulated_Images'
            if output_dir_sim.exists():
                shutil.rmtree(output_dir_sim)
            output_dir_sim.mkdir()
            
            # Optionally vary turb_scale slightly for each variation
            base_turb_scale = preset['turb_scale']
            
            # Generate all variations for this level
            for var_idx in range(num_variations):
                # Optional: add small random variation to parameters
                if use_random_seeds:
                    # Parse turb_scale (e.g., "1 / 64" -> 64)
                    denominator = int(base_turb_scale.split('/')[-1].strip())
                    # Vary by ±10%
                    varied_denom = int(denominator * random.uniform(0.9, 1.1))
                    turb_scale = f"1 / {varied_denom}"
                    
                    # Vary sigma slightly
                    sigma = int(preset['sigma'] * random.uniform(0.9, 1.1))
                else:
                    turb_scale = base_turb_scale
                    sigma = preset['sigma']
                
                frames = run_quickturb(
                    img_path,
                    preset['turb_strength'],
                    sigma,
                    turb_scale,
                    num_frames=1,
                    clear_output=False  # Don't clear between variations
                )
                
                if not frames:
                    print(f"    Variation {var_idx}: No frames generated")
                    continue
                
                # Find the frame for this variation (frames accumulate)
                if var_idx < len(frames):
                    frame_to_use = frames[var_idx]
                else:
                    frame_to_use = frames[-1]
                
                # Copy to dataset
                turb_filename = f"{img_path.stem}_turb_{preset_name}_{var_idx}.png"
                turb_save_path = turbulent_dir / turb_filename
                shutil.copy(frame_to_use, turb_save_path)
                
                # Record detailed metadata
                pair_metadata = {
                    'clean_image': img_path.name,
                    'turbulent_image': turb_filename,
                    'preset': preset_name,
                    'variation_idx': var_idx,
                    'turb_strength': preset['turb_strength'],
                    'sigma': sigma,
                    'turb_scale': turb_scale,
                    'clean_path': str(clean_save_path),
                    'turbulent_path': str(turb_save_path)
                }
                
                if use_random_seeds:
                    pair_metadata['random_seed'] = True
                
                metadata['turbulent_pairs'].append(pair_metadata)
                
                print(f"    Variation {var_idx}: OK")
    
    # Save metadata
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"Clean images: {len(metadata['clean_images'])}")
    print(f"Turbulent images: {len(metadata['turbulent_pairs'])}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_path}")
    
    # Save metadata
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"Clean images: {len(metadata['clean_images'])}")
    print(f"Turbulent images: {len(metadata['turbulent_pairs'])}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate turbulence dataset using QuickTurbSim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: 200 images × 5 variations × 2 levels = 2000 pairs
  python scripts/generate_quickturb_dataset.py \\
      --input-dir data/clean_images/ \\
      --output-dir data/turbulence_dataset/ \\
      --num-variations 5 \\
      --presets medium strong
  
  # With random parameter variation
  python scripts/generate_quickturb_dataset.py \\
      --input-dir data/clean_images/ \\
      --output-dir data/turbulence_dataset/ \\
      --num-variations 5 \\
      --presets medium \\
      --use-random-seeds
  
  # Single image test
  python scripts/generate_quickturb_dataset.py \\
      --input-image data/test_images/boat.jpg \\
      --output-dir data/test_output/ \\
      --num-variations 3 \\
      --presets weak medium strong
        """
    )
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing clean images')
    parser.add_argument('--input-image', type=str, default=None,
                       help='Single image to process')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--num-variations', type=int, default=5,
                       help='Number of turbulent variations per image per level (default: 5)')
    parser.add_argument('--presets', type=str, nargs='+',
                       choices=['weak', 'medium', 'strong'],
                       default=['medium', 'strong'],
                       help='Turbulence levels to generate')
    parser.add_argument('--use-random-seeds', action='store_true',
                       help='Add random variation to turbulence parameters for diversity')
    
    args = parser.parse_args()
    
    # Validation
    if not args.input_dir and not args.input_image:
        parser.error("Must specify either --input-dir or --input-image")
    if args.input_dir and args.input_image:
        parser.error("Cannot specify both --input-dir and --input-image")
    
    # Verify QuickTurbSim exists
    if not QUICKTURB_PATH.exists():
        print(f"ERROR: QuickTurbSim not found at {QUICKTURB_PATH}")
        print("Run: git clone https://github.com/Riponcs/QuickTurbSim.git")
        sys.exit(1)
    
    print("="*60)
    print("QuickTurbSim Dataset Generator")
    print("="*60)
    print()
    
    # If single image, create temp directory with just that image
    if args.input_image:
        input_path = Path(args.input_image)
        temp_dir = Path(args.output_dir) / 'temp_input'
        temp_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, temp_dir / input_path.name)
        
        generate_dataset(
            temp_dir,
            args.output_dir,
            args.num_variations,
            args.presets,
            use_random_seeds=args.use_random_seeds
        )
        
        # Clean up temp
        shutil.rmtree(temp_dir)
    else:
        generate_dataset(
            args.input_dir,
            args.output_dir,
            args.num_variations,
            args.presets,
            use_random_seeds=args.use_random_seeds
        )


if __name__ == "__main__":
    main()

