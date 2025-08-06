import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec

def load_depth_map(path):
    """Load depth map from PNG file."""
    img = Image.open(path)
    depth_array = np.array(img).astype(np.float32)
    
    # Convert to grayscale if needed
    if len(depth_array.shape) == 3:
        depth_array = np.mean(depth_array[:, :, :3], axis=2)
    
    return depth_array

def load_rgb_image(path):
    """Load RGB image."""
    return np.array(Image.open(path))

def normalize_depth_maps(depth1, depth2):
    """Normalize two depth maps to the same scale."""
    # Resize to match dimensions if needed
    if depth1.shape != depth2.shape:
        h, w = depth2.shape
        depth1_pil = Image.fromarray(depth1)
        depth1 = np.array(depth1_pil.resize((w, h), Image.BILINEAR))
    
    # Normalize to 0-1 range
    depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
    depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
    
    return depth1_norm, depth2_norm

def compute_metrics(depth_map, reference_map):
    """Compute difference metrics between depth map and reference."""
    # Normalize both maps
    depth_norm, ref_norm = normalize_depth_maps(depth_map, reference_map)
    
    # Compute difference
    diff_map = np.abs(depth_norm - ref_norm)
    
    # Compute metrics
    mae = np.mean(diff_map)
    rmse = np.sqrt(np.mean(diff_map ** 2))
    correlation = np.corrcoef(depth_norm.flatten(), ref_norm.flatten())[0, 1]
    
    return {
        'diff_map': diff_map,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation
    }

def create_visualization(rgb_image, depth_maps, depth_names, metrics, output_path):
    """Create comparison visualization."""
    # Create figure: 5 columns (RGB + 4 depth sources), 2 rows
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.15, hspace=0.2)
    
    # Plot RGB image (spans both rows)
    ax_rgb = fig.add_subplot(gs[:, 0])
    ax_rgb.imshow(rgb_image)
    ax_rgb.set_title('RGB Image', fontsize=14, fontweight='bold')
    ax_rgb.axis('off')
    
    # Plot each depth map and its difference
    for i, (depth_map, name, metric) in enumerate(zip(depth_maps, depth_names, metrics)):
        col = i + 1
        
        # Top row: Original depth map
        ax_depth = fig.add_subplot(gs[0, col])
        im_depth = ax_depth.imshow(depth_map, cmap='viridis')
        ax_depth.set_title(name, fontsize=12, fontweight='bold')
        ax_depth.axis('off')
        
        # Bottom row: Difference map with metrics
        ax_diff = fig.add_subplot(gs[1, col])
        im_diff = ax_diff.imshow(metric['diff_map'], cmap='hot', vmin=0, vmax=0.5)
        
        # Add metrics in upper left
        metrics_text = f"MAE: {metric['mae']:.3f}\nRMSE: {metric['rmse']:.3f}\nCorr: {metric['correlation']:.3f}"
        ax_diff.text(0.05, 0.95, metrics_text, transform=ax_diff.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
        
        ax_diff.set_title('Difference from GT', fontsize=10)
        ax_diff.axis('off')
        
        # Add colorbars to last column
        if col == 4:
            plt.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04)
            cbar = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
            cbar.set_label('Absolute Difference', fontsize=8)
    
    plt.suptitle('Depth Map Comparison with Ground Truth', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_image(dataset_name, image_name, base_dir='EOGS_dataset', output_dir='comparisons'):
    """Process a single image comparison."""
    # Build paths
    base_path = Path(base_dir)
    
    # Ground truth uses dataset name
    gt_path = base_path / 'gt_eogs' / dataset_name / f'{dataset_name}_DSM.png'
    
    # RGB image
    rgb_path = base_path / 'images' / dataset_name / f'{image_name}_RGB.png'
    
    # Depth sources
    depth_paths = {
        'altitude_eogs': base_path / 'altitude_eogs' / dataset_name / f'{image_name}.png',
        'dsm_eogs': base_path / 'dsm_eogs' / dataset_name / f'{image_name}.png',
        'depths_vggt': base_path / 'depths_vggt' / dataset_name / f'{image_name}_RGB.png',
        'depths_frames': base_path / 'depths_frames' / dataset_name / f'{image_name}_RGB.png'
    }
    
    # Check files exist
    if not rgb_path.exists():
        print(f"RGB image not found: {rgb_path}")
        return
    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        return
    
    missing = [name for name, path in depth_paths.items() if not path.exists()]
    if missing:
        print(f"Missing depth maps: {missing}")
        return
    
    # Load images
    print(f"Processing {dataset_name}/{image_name}")
    rgb_image = load_rgb_image(rgb_path)
    gt_depth = load_depth_map(gt_path)
    
    # Load and compute metrics for each depth source
    depth_maps = []
    depth_names = []
    metrics = []
    
    for name, path in depth_paths.items():
        depth_map = load_depth_map(path)
        metric = compute_metrics(depth_map, gt_depth)
        
        depth_maps.append(depth_map)
        depth_names.append(name)
        metrics.append(metric)
        
        print(f"  {name}: MAE={metric['mae']:.4f}, RMSE={metric['rmse']:.4f}, Corr={metric['correlation']:.4f}")
    
    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    viz_path = output_path / f'{image_name}_comparison.png'
    create_visualization(rgb_image, depth_maps, depth_names, metrics, viz_path)
    print(f"  Saved: {viz_path}\n")

def main():
    """Process specific datasets in EOGS_dataset."""
    base_dir = 'EOGS_dataset'
    
    # Hard-code which datasets to process
    datasets_to_process = ['JAX_004', 'JAX_068', 'JAX_214', 'JAX_260']  # Modify this list as needed

    rgb_base = Path(base_dir) / 'images'
    
    if not rgb_base.exists():
        print(f"Error: {rgb_base} not found")
        return
    
    print(f"Processing {len(datasets_to_process)} selected datasets\n")
    
    # Process each selected dataset
    for dataset_name in datasets_to_process:
        dataset_path = rgb_base / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found, skipping...")
            continue
            
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Find all RGB images
        rgb_files = sorted(dataset_path.glob('*_RGB.png'))
        
        if not rgb_files:
            print(f"No RGB images found in {dataset_name}\n")
            continue
            
        print(f"Found {len(rgb_files)} images to process\n")
        
        for rgb_file in rgb_files:
            # Extract base image name (remove _RGB suffix)
            image_name = rgb_file.stem.replace('_RGB', '')
            
            try:
                process_image(dataset_name, image_name)
            except Exception as e:
                print(f"Error processing {image_name}: {e}\n")

if __name__ == "__main__":
    main()