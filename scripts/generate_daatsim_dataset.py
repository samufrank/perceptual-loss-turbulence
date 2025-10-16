"""
DAATSim Dataset Generator (Robust Version)
Generates turbulence-corrupted image pairs using DAATSim simulator

Usage:
    python scripts/generate_daatsim_dataset.py \
        --input-dir data/clean_images/ \
        --output-dir data/turbulence_dataset/ \
        --num-variations 10 \
        --presets medium strong
        
    OR use simplified turbulence simulation:
    
    python scripts/generate_daatsim_dataset.py \
        --input-dir data/clean_images/ \
        --output-dir data/turbulence_dataset/ \
        --num-variations 10 \
        --use-simple-sim
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
import cv2
from scipy.ndimage import gaussian_filter

# DAATSim path
DAATSIM_PATH = Path(__file__).parent.parent / 'DAATSim'


def simple_turbulence_simulation(image, strength='medium', seed=None):
    """
    Simplified turbulence simulation: geometric warping + blur.
    
    Approximates atmospheric turbulence effects without full physics simulation.
    Faster and more reliable than DAATSim for proof-of-concept work.
    
    Args:
        image: numpy array (H, W, 3)
        strength: 'weak', 'medium', or 'strong'
        seed: random seed for reproducibility
        
    Returns:
        Turbulent image as numpy array
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image.shape[:2]
    
    # Turbulence parameters
    params = {
        'weak': {'displacement_scale': 2.0, 'blur_sigma': 1.0, 'displacement_smoothness': 25},
        'medium': {'displacement_scale': 5.0, 'blur_sigma': 2.0, 'displacement_smoothness': 20},
        'strong': {'displacement_scale': 10.0, 'blur_sigma': 3.5, 'displacement_smoothness': 15}
    }
    
    p = params[strength]
    
    # Generate smooth random displacement field (simulates refractive index variations)
    dx = gaussian_filter(np.random.randn(h, w), sigma=p['displacement_smoothness']) * p['displacement_scale']
    dy = gaussian_filter(np.random.randn(h, w), sigma=p['displacement_smoothness']) * p['displacement_scale']
    
    # Create meshgrid and apply displacement
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    
    # Remap image (geometric warping)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Apply blur (simulates temporal averaging of turbulence)
    blurred = cv2.GaussianBlur(warped, (0, 0), p['blur_sigma'])
    
    return blurred.astype(np.uint8)


class DAATSimGenerator:
    """Generate turbulence dataset using DAATSim or simplified simulation."""
    
    def __init__(self, output_dir, use_simple_sim=False):
        self.output_dir = Path(output_dir)
        self.clean_dir = self.output_dir / 'clean'
        self.turbulent_dir = self.output_dir / 'turbulent'
        self.use_simple_sim = use_simple_sim
        
        # Create directories
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.turbulent_dir.mkdir(parents=True, exist_ok=True)
        
        # Turbulence parameter presets
        self.turbulence_presets = {
            'weak': {'r0': 0.1, 'strength': 'weak'},
            'medium': {'r0': 0.05, 'strength': 'medium'},
            'strong': {'r0': 0.02, 'strength': 'strong'}
        }
        
        self.metadata = {
            'simulation_method': 'simple' if use_simple_sim else 'daatsim',
            'clean_images': [],
            'turbulent_pairs': [],
            'turbulence_presets': self.turbulence_presets
        }
        
        # Verify DAATSim exists if needed
        if not use_simple_sim and not DAATSIM_PATH.exists():
            raise FileNotFoundError(
                f"DAATSim not found at {DAATSIM_PATH}\n"
                f"Run: git clone https://github.com/Riponcs/DAATSim.git\n"
                f"Or use --use-simple-sim flag for simplified turbulence"
            )
    
    def run_simple_simulation(self, input_image_path, level_name, strength, var_idx):
        """
        Run simplified turbulence simulation.
        
        Args:
            input_image_path: Path to clean image
            level_name: Name for this turbulence level
            strength: 'weak', 'medium', or 'strong'
            var_idx: Variation index
            
        Returns:
            Path to generated turbulent image
        """
        # Load image
        img = cv2.imread(str(input_image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {input_image_path}")
        
        # Generate turbulent version with reproducible seed
        seed = hash(f"{input_image_path.name}_{level_name}_{var_idx}") % (2**32)
        turb_img = simple_turbulence_simulation(img, strength, seed=seed)
        
        # Save
        turb_filename = f"{input_image_path.stem}_turb_{level_name}_{var_idx}.png"
        turb_save_path = self.turbulent_dir / turb_filename
        cv2.imwrite(str(turb_save_path), turb_img)
        
        return turb_save_path
    
    def create_daatsim_config(self, input_image_name, output_dir, r0_value, num_frames=1):
        """
        Create complete DAATSim config content.
        
        CRITICAL: Sets USE_DEPTH_PARAM = False to avoid mmcv dependency.
        Depth models (Depth Anything V2, Metric3D) require mmcv which is
        difficult to install. Disabling depth makes turbulence uniform
        across image, which is acceptable for this assignment.
        
        Returns config as string to be written to config.py.
        """
        config_content = f'''# Temporary config for dataset generation
import torch

# Device
DEVICE_OVERRIDE = None

# Output
BASE_OUTPUT_DIR = "{output_dir}"

# Image parameters
SIZE_D_PARAM = 512
NUM_ZERN_PARAM = 100
D_APERTURE_PARAM = 0.2
L_PROP_PARAM = 7000.0
WAVELENGTH_PARAM = 0.525e-6
OBJ_SIZE_PARAM = 2.06

# Turbulence parameters
R0_VALUES_TO_TEST_PARAM = [{r0_value}]
L0_PARAM = float('inf')  # Kolmogorov spectrum (simpler than Von Karman)
l0_PARAM = 0.0           # Inner scale (0.0 for Kolmogorov)

# Tilt parameters
FOCAL_LENGTH_PARAM = 0.3
PHASE_SCREEN_SCALE_FACTOR_PARAM = 4e7
TILT_SCALE_FACTOR_PARAM = 1e4

# PSF parameters
NN_BASE_PARAM = 48
MIN_BLUR_REL_STRENGTH_PARAM = 0.0025
MAX_BLUR_REL_STRENGTH_PARAM = 0.025
NUM_PSF_LEVELS_HIGH_RES = 20
NUM_PSF_LEVELS_LOW_RES = 5

# Zernike
USE_CORRELATED_COEFFS_PARAM = True

# CRITICAL: Disable depth to avoid mmcv/depth model dependencies
# This makes turbulence uniform across the image rather than depth-varying
USE_DEPTH_PARAM = False

# Temporal consistency (AR(1) model for frame-to-frame evolution)
ALPHA_BLUR_PARAM = 0.7
ALPHA_TILT_PARAM = 0.7

# Generation parameters
NUM_IMAGES_TEST_PARAM = {num_frames}
INPUT_IMAGE_PATH_PARAM = "inputImages/{input_image_name}"

# Output format
SAVE_VIDEO_PARAM = False
SHOW_SAMPLE_PLOT_PARAM = False
'''
        return config_content
    
    def run_daatsim(self, input_image_path, r0_value, num_frames=1):
        """
        Run DAATSim on a single image.
        
        DAATSim ignores BASE_OUTPUT_DIR and always writes to its own
        output_simulations/ directory. We read from there.
        
        Args:
            input_image_path: Path to clean image
            r0_value: Fried parameter (meters)
            num_frames: Number of frames to generate
        
        Returns:
            Path to DAATSim's output directory with generated frames, or None on failure
        """
        input_path = Path(input_image_path).resolve()
        
        # Copy image to DAATSim/inputImages/
        daatsim_input_dir = DAATSIM_PATH / 'inputImages'
        daatsim_input_dir.mkdir(exist_ok=True)
        daatsim_input_path = daatsim_input_dir / input_path.name
        shutil.copy(input_path, daatsim_input_path)
        
        # Backup original config
        config_path = DAATSIM_PATH / 'config.py'
        backup_path = DAATSIM_PATH / 'config_backup.py'
        
        config_existed = config_path.exists()
        if config_existed:
            shutil.copy(config_path, backup_path)
        
        try:
            # Write new config
            config_content = self.create_daatsim_config(
                input_path.name,
                "output_simulations",  # DAATSim ignores this anyway
                r0_value,
                num_frames
            )
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Clear previous outputs to avoid confusion
            daatsim_output_dir = DAATSIM_PATH / 'output_simulations'
            if daatsim_output_dir.exists():
                shutil.rmtree(daatsim_output_dir)
            
            # Run DAATSim from its directory
            print(f"      Running DAATSim (r0={r0_value:.3f}m)...")
            
            result = subprocess.run(
                [sys.executable, 'main.py'],
                cwd=str(DAATSIM_PATH),
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            
            # Print stdout for debugging (first run only)
            if result.stdout:
                print(f"      DAATSim stdout (first 500 chars):")
                print("      " + result.stdout[:500].replace("\n", "\n      "))
            
            # Check for actual errors (ignore warnings about slow processors, etc.)
            if result.returncode != 0:
                # Only fail if stderr contains actual error indicators
                stderr_lower = result.stderr.lower()
                error_indicators = ['error:', 'traceback', 'exception', 'failed', 'fatal']
                has_real_error = any(indicator in stderr_lower for indicator in error_indicators)
                
                # Also check if output was actually generated
                output_exists = daatsim_output_dir.exists() and any(daatsim_output_dir.iterdir())
                
                if has_real_error and not output_exists:
                    print("      Status: FAILED")
                    print(f"      Error: {result.stderr[:300]}")
                    return None
                else:
                    # Warnings only, but generation succeeded
                    print("      Status: OK (with warnings)")
            else:
                print("      Status: OK")
            
            return daatsim_output_dir
            
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            return None
            
        except Exception as e:
            print(f"ERROR: {e}")
            return None
            
        finally:
            # Restore original config
            if config_existed and backup_path.exists():
                shutil.copy(backup_path, config_path)
                backup_path.unlink()
            elif not config_existed and config_path.exists():
                # Remove config if it didn't exist before
                config_path.unlink()
    
    def find_turbulent_frames(self, output_dir, debug=False):
        """
        Find generated turbulent frames in DAATSim output.
        
        Enhanced with debug mode to inspect directory structure.
        
        Args:
            output_dir: DAATSim output directory
            debug: If True, print full directory tree
        
        Returns:
            List of paths to turbulent frames
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return []
        
        if debug:
            print(f"\n      DEBUG: Inspecting {output_path}")
            for root, dirs, files in os.walk(output_path):
                level = root.replace(str(output_path), '').count(os.sep)
                indent = '  ' * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = '  ' * (level + 1)
                for file in files[:10]:  # Limit to first 10 files
                    print(f'{subindent}{file}')
                if len(files) > 10:
                    print(f'{subindent}... and {len(files) - 10} more files')
        
        # Search patterns (DAATSim may use various naming conventions)
        patterns = [
            '**/*.png',
            '**/*.jpg',
            '**/*.jpeg',
            '**/frame_*.png',
            '**/turbulent_*.png',
            '**/*_turb*.png',
            '**/*distorted*.png',
            '**/SampleOutput/**/*.png',  # Common DAATSim structure
            '**/SampleOutput/**/*.jpg',
            '**/output/**/*.png',
            '**/results/**/*.png'
        ]
        
        frames = []
        for pattern in patterns:
            found = list(output_path.glob(pattern))
            frames.extend(found)
        
        # Remove duplicates
        frames = list(set(frames))
        
        # Exclude input/reference images
        exclude_keywords = ['input', 'reference', 'original', 'clean', 'ground_truth']
        frames = [f for f in frames if not any(kw in f.name.lower() for kw in exclude_keywords)]
        
        # Sort by name
        frames = sorted(frames)
        
        if debug:
            if frames:
                print(f"      Found {len(frames)} frames:")
                for f in frames[:5]:
                    print(f"        {f.name}")
                if len(frames) > 5:
                    print(f"        ... and {len(frames) - 5} more")
            else:
                print(f"      WARNING: No frames found with patterns: {patterns}")
                print(f"      Check the directory structure above")
        
        return frames
    
    def process_image(self, image_path, num_variations, r0_values):
        """
        Generate multiple turbulent variations of a single image.
        
        Args:
            image_path: Path to clean image
            num_variations: Number of variations per turbulence level
            r0_values: List of r0 values (or preset names)
        
        Returns:
            List of (clean_path, turbulent_path, metadata) tuples
        """
        image_path = Path(image_path)
        
        # Save clean image
        clean_img = Image.open(image_path)
        clean_save_path = self.clean_dir / image_path.name
        clean_img.save(clean_save_path)
        
        self.metadata['clean_images'].append({
            'filename': image_path.name,
            'path': str(clean_save_path),
            'shape': list(np.array(clean_img).shape)
        })
        
        pairs = []
        
        # Process each turbulence level
        for r0_spec in r0_values:
            # Convert preset name to parameters
            if isinstance(r0_spec, str) and r0_spec in self.turbulence_presets:
                preset = self.turbulence_presets[r0_spec]
                r0_value = preset['r0']
                strength = preset['strength']
                level_name = r0_spec
            else:
                r0_value = float(r0_spec)
                strength = 'medium'  # Default for custom r0
                level_name = f"r0_{r0_value:.3f}"
            
            print(f"  Level: {level_name} (r0={r0_value:.3f}m, strength={strength})")
            
            # Generate variations
            for var_idx in range(num_variations):
                if self.use_simple_sim:
                    # Simple simulation
                    try:
                        turb_save_path = self.run_simple_simulation(
                            image_path, level_name, strength, var_idx
                        )
                        print(f"      Variation {var_idx}: OK")
                        
                    except Exception as e:
                        print(f"      Variation {var_idx}: FAILED ({e})")
                        continue
                
                else:
                    # DAATSim simulation
                    daatsim_output = self.run_daatsim(
                        image_path,
                        r0_value,
                        num_frames=1
                    )
                    
                    if daatsim_output is None:
                        print(f"      Variation {var_idx}: Failed to generate")
                        continue
                    
                    # Find generated frames
                    frames = self.find_turbulent_frames(daatsim_output, debug=(var_idx == 0))
                    
                    if not frames:
                        print(f"      Variation {var_idx}: No frames found")
                        continue
                    
                    # Use first frame
                    turbulent_frame_path = frames[0]
                    
                    # Copy to dataset directory
                    turb_filename = f"{image_path.stem}_turb_{level_name}_{var_idx}.png"
                    turb_save_path = self.turbulent_dir / turb_filename
                    
                    turb_img = Image.open(turbulent_frame_path)
                    turb_img.save(turb_save_path)
                    
                    print(f"      Variation {var_idx}: OK")
                
                # Record pair
                pair_metadata = {
                    'clean_image': image_path.name,
                    'turbulent_image': turb_save_path.name,
                    'r0_value': r0_value,
                    'turbulence_level': level_name,
                    'strength': strength,
                    'variation_idx': var_idx,
                    'clean_path': str(clean_save_path),
                    'turbulent_path': str(turb_save_path)
                }
                
                self.metadata['turbulent_pairs'].append(pair_metadata)
                pairs.append((clean_save_path, turb_save_path, pair_metadata))
        
        return pairs
    
    def generate_dataset(self, input_dir, num_variations=10, r0_values=None):
        """
        Generate full dataset from directory of clean images.
        
        Args:
            input_dir: Directory containing clean images
            num_variations: Number of turbulent versions per image per level
            r0_values: List of r0 values or preset names
        """
        if r0_values is None:
            r0_values = ['medium', 'strong']
        
        input_dir = Path(input_dir)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f'*{ext}'))
            image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(set(image_paths))
        
        if not image_paths:
            print(f"No images found in {input_dir}")
            return []
        
        print(f"Found {len(image_paths)} images in {input_dir}")
        print(f"Method: {'Simplified simulation' if self.use_simple_sim else 'DAATSim'}")
        print(f"Generating {num_variations} variations at {len(r0_values)} turbulence levels")
        print(f"Total expected: {len(image_paths) * num_variations * len(r0_values)} turbulent images")
        print()
        
        # Process each image with progress bar
        all_pairs = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            print(f"\nProcessing: {img_path.name}")
            pairs = self.process_image(img_path, num_variations, r0_values)
            all_pairs.extend(pairs)
        
        # Save metadata
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print("\n" + "="*60)
        print("Dataset generation complete!")
        print("="*60)
        print(f"Clean images: {len(self.metadata['clean_images'])}")
        print(f"Turbulent images: {len(self.metadata['turbulent_pairs'])}")
        print(f"Metadata: {metadata_path}")
        print(f"Dataset: {self.output_dir}")
        
        return all_pairs
    
    def visualize_samples(self, num_samples=5):
        """Create visualization of sample pairs."""
        import matplotlib.pyplot as plt
        
        if not self.metadata['turbulent_pairs']:
            print("No turbulent pairs to visualize")
            return
        
        # Sample diverse pairs (different turbulence levels if possible)
        pairs = self.metadata['turbulent_pairs']
        
        # Try to get one from each turbulence level
        sampled = []
        levels_seen = set()
        for pair in pairs:
            level = pair['turbulence_level']
            if level not in levels_seen:
                sampled.append(pair)
                levels_seen.add(level)
            if len(sampled) >= num_samples:
                break
        
        # Fill remaining with random samples
        while len(sampled) < num_samples and len(sampled) < len(pairs):
            sampled.append(pairs[len(sampled)])
        
        # Create visualization
        fig, axes = plt.subplots(len(sampled), 2, figsize=(12, 4*len(sampled)))
        if len(sampled) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, pair in enumerate(sampled):
            clean_img = Image.open(pair['clean_path'])
            turb_img = Image.open(pair['turbulent_path'])
            
            axes[idx, 0].imshow(clean_img)
            axes[idx, 0].set_title('Clean', fontsize=12)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(turb_img)
            title = f"Turbulent ({pair['turbulence_level']}, r0={pair['r0_value']:.3f}m)"
            axes[idx, 1].set_title(title, fontsize=12)
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        viz_path = self.output_dir / 'sample_pairs.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sample visualization: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate turbulence dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with simplified simulation
  python scripts/generate_daatsim_dataset.py \\
      --input-image data/test_images/boat.jpg \\
      --output-dir data/daatsim_test/ \\
      --num-variations 10 \\
      --use-simple-sim \\
      --visualize
  
  # Single image with DAATSim
  python scripts/generate_daatsim_dataset.py \\
      --input-image data/test_images/boat.jpg \\
      --output-dir data/daatsim_test/ \\
      --num-variations 5 \\
      --presets medium strong
  
  # Directory with simplified simulation
  python scripts/generate_daatsim_dataset.py \\
      --input-dir data/clean_images/ \\
      --output-dir data/turbulence_dataset/ \\
      --num-variations 10 \\
      --use-simple-sim
  
  # Custom r0 values
  python scripts/generate_daatsim_dataset.py \\
      --input-dir data/clean_images/ \\
      --output-dir data/turbulence_dataset/ \\
      --num-variations 10 \\
      --r0-values 0.03 0.06 0.1
        """
    )
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing clean images')
    parser.add_argument('--input-image', type=str, default=None,
                       help='Process a single image instead of a directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--num-variations', type=int, default=10,
                       help='Number of turbulent variations per image per level')
    parser.add_argument('--r0-values', type=float, nargs='+',
                       default=None,
                       help='Custom r0 values in meters (e.g., 0.1 0.05 0.02)')
    parser.add_argument('--presets', type=str, nargs='+',
                       choices=['weak', 'medium', 'strong'],
                       default=['medium', 'strong'],
                       help='Use preset turbulence levels (ignored if --r0-values given)')
    parser.add_argument('--use-simple-sim', action='store_true',
                       help='Use simplified turbulence simulation instead of DAATSim')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of sample pairs')
    
    args = parser.parse_args()
    
    # Validation
    if not args.input_dir and not args.input_image:
        parser.error("Must specify either --input-dir or --input-image")
    if args.input_dir and args.input_image:
        parser.error("Cannot specify both --input-dir and --input-image")
    
    # Determine r0 values
    if args.r0_values:
        r0_values = args.r0_values
    else:
        r0_values = args.presets
    
    # Generate dataset
    print("="*60)
    print("Turbulence Dataset Generator")
    print("="*60)
    print()
    
    if args.use_simple_sim:
        print("Using simplified turbulence simulation")
        print("Note: This is faster but less physically accurate than DAATSim")
        print("Suitable for proof-of-concept and rapid prototyping")
    else:
        print("Using DAATSim physics-based simulation")
        print("Note: This is slower but more accurate")
        print("Note: Ensure USE_DEPTH_PARAM = False in DAATSim/config.py to avoid mmcv")
    print()
    
    generator = DAATSimGenerator(args.output_dir, use_simple_sim=args.use_simple_sim)
    
    if args.input_image:
        # Single image mode
        print(f"Processing single image: {args.input_image}")
        generator.process_image(
            args.input_image,
            num_variations=args.num_variations,
            r0_values=r0_values
        )
        # Save metadata
        metadata_path = Path(args.output_dir) / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(generator.metadata, f, indent=2)
        print(f"\nMetadata saved to: {metadata_path}")
    else:
        # Directory mode
        generator.generate_dataset(
            args.input_dir,
            num_variations=args.num_variations,
            r0_values=r0_values
        )
    
    # Optional visualization
    if args.visualize:
        print("\nGenerating visualization...")
        generator.visualize_samples()


if __name__ == "__main__":
    main()

