import torch
import numpy as np
import os
import json
from PIL import Image
import argparse
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import matplotlib.pyplot as plt

def normalize_depth_for_visualization(depth_map):
    """Normalize depth map to 0-65535 for visualization and saving as PNG."""
    if len(depth_map.shape) > 2:
        depth_map = np.squeeze(depth_map)
    
    depth_map = np.nan_to_num(depth_map)
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    
    if max_val - min_val < 1e-8:
        return np.zeros_like(depth_map, dtype=np.uint16)
    
    normalized = ((depth_map - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
    return normalized

def create_depth_visualization(depth_map, output_path, colormap='viridis'):
    """Create and save a colorized depth visualization."""
    if len(depth_map.shape) > 2:
        depth_map = np.squeeze(depth_map)
    
    depth_map = np.nan_to_num(depth_map)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='Depth Value')
    plt.title('Depth Map Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_batch(image_paths, output_dir, model, device, dtype, vis_dir=None):
    """Process a batch of images and return camera parameters."""
    # Load and preprocess images
    images = load_and_preprocess_images(image_paths).to(device)
    
    # Run inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            # Predict camera parameters
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            
            # Predict depth
            depth_maps, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    # Convert to numpy and process each frame
    depth_maps = depth_maps.squeeze(0).cpu().numpy()
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    
    # Clear GPU memory after inference
    torch.cuda.empty_cache()
    
    camera_params = {}
    for i, image_path in enumerate(image_paths):
        frame_name = Path(image_path).stem
        
        # Extract depth map
        depth_map = depth_maps[i]
        if len(depth_map.shape) > 2:
            depth_map = np.squeeze(depth_map)
        
        # Save depth map
        depth_norm = normalize_depth_for_visualization(depth_map)
        depth_path = os.path.join(output_dir, f"{frame_name}.png")
        Image.fromarray(depth_norm).save(depth_path)
        
        # Save visualization if requested
        if vis_dir:
            vis_path = os.path.join(vis_dir, f"{frame_name}.png")
            create_depth_visualization(depth_map, vis_path)
        
        # Store camera parameters
        camera_params[frame_name] = {
            "extrinsic": extrinsic[i].tolist(),
            "intrinsic": intrinsic[i].tolist()
        }
    
    return camera_params

def main():
    parser = argparse.ArgumentParser(description="VGGT batch depth and camera prediction")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to image file, multiple files, or directory")
    parser.add_argument("--output", type=str, default="vggt_output", 
                        help="Output directory (default: vggt_output)")
    parser.add_argument("--vis", type=str, default=None, 
                        help="Path to save visualizations (default: None)")
    parser.add_argument("--batch_size", type=int, default=10, 
                        help="Number of frames to process together (default: 10)")
    parser.add_argument("--cache_dir", type=str, default="model", 
                        help="Directory to cache VGGT model")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    if args.vis:
        os.makedirs(args.vis, exist_ok=True)
    
    # Determine input type and collect image files
    input_path = Path(args.input)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    if input_path.is_file():
        # Single file
        image_files = [str(input_path)]
        print(f"Processing single image: {input_path.name}")
    elif input_path.is_dir():
        # Directory
        image_files = sorted([
            str(f) for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        print(f"Found {len(image_files)} images in {input_path}")
        if not image_files:
            print(f"Error: No images found in {input_path}")
            return
    else:
        # Check if it's a glob pattern or multiple files
        if '*' in args.input:
            from glob import glob
            image_files = sorted(glob(args.input))
        else:
            # Assume it's a single non-existent file
            print(f"Error: {input_path} not found")
            return
    
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("Loading VGGT model...")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir=args.cache_dir).to(device)
    model.eval()
    
    # Process images in batches
    all_camera_params = {}
    total_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(0, len(image_files), args.batch_size):
        batch_end = min(batch_idx + args.batch_size, len(image_files))
        batch_files = image_files[batch_idx:batch_end]
        
        current_batch = (batch_idx // args.batch_size) + 1
        print(f"Processing batch [{current_batch}/{total_batches}]: {len(batch_files)} images")
        
        try:
            camera_params = process_batch(
                batch_files, args.output, model, device, dtype, args.vis
            )
            all_camera_params.update(camera_params)
        except torch.cuda.OutOfMemoryError:
            print(f"GPU out of memory. Trying to process batch with reduced size...")
            torch.cuda.empty_cache()
            
            # Process one by one if batch fails
            for img_file in batch_files:
                try:
                    camera_params = process_batch(
                        [img_file], args.output, model, device, dtype, args.vis
                    )
                    all_camera_params.update(camera_params)
                except Exception as e:
                    print(f"Failed to process {Path(img_file).name}: {e}")
    
    # Save camera parameters
    camera_params_path = os.path.join(args.output, "cam_params.json")
    with open(camera_params_path, 'w') as f:
        json.dump(all_camera_params, f, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(all_camera_params)} images")
    print(f"Depth maps saved to: {args.output}")
    print(f"Camera parameters saved to: {camera_params_path}")
    if args.vis:
        print(f"Visualizations saved to: {args.vis}")

if __name__ == "__main__":
    main()