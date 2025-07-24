import torch
import numpy as np
import cv2
import os
import json
from PIL import Image
import argparse
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def normalize_depth_for_visualization(depth_map):
    """
    Normalize depth map to 0-65535 for visualization and saving as PNG.
    
    Args:
        depth_map: Numpy array of depth values
    
    Returns:
        Normalized depth map as uint16 (0-65535)
    """
    # Ensure depth_map is 2D by squeezing any extra dimensions
    if len(depth_map.shape) > 2:
        depth_map = np.squeeze(depth_map)
    
    # Ensure no NaN or inf values
    depth_map = np.nan_to_num(depth_map)
    
    # Get min and max, avoiding division by zero
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    
    if max_val - min_val < 1e-8:
        # Return zeros if the depth map is flat (avoid division by zero)
        return np.zeros_like(depth_map, dtype=np.uint16)
    
    # Normalize to 0-65535 for 16-bit PNG
    normalized = ((depth_map - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
    
    return normalized

def create_depth_visualization(depth_map, frame_name, colormap='viridis'):
    """
    Create a colorized depth visualization with colorbar, displaying actual depth values.
    Reference: vis.py visualization approach
    
    Args:
        depth_map: Numpy array of depth values
        frame_name: Name to display as title
        colormap: Matplotlib colormap name (default: 'viridis')
    
    Returns:
        RGB visualization image as uint8 array
    """
    # Ensure depth_map is 2D
    if len(depth_map.shape) > 2:
        depth_map = np.squeeze(depth_map)
    
    # Handle NaN or inf values
    depth_map = np.nan_to_num(depth_map)
    
    # Check if this might be inverse depth (values close to 1)
    mean_val = np.mean(depth_map)
    if 0.8 < mean_val < 1.2 and np.std(depth_map) < 0.1:
        print("Detected possible inverse depth. Converting to depth...")
        # Convert from inverse depth to depth
        # Avoid division by zero
        depth_map = np.where(depth_map > 1e-8, 1.0 / depth_map, 0)
    
    # Get file information (similar to vis.py)
    print(f"数据形状: {depth_map.shape}")
    print(f"数据类型: {depth_map.dtype}")
    print(f"数值范围: {depth_map.min():.3f} - {depth_map.max():.3f}")
    
    # Create figure (following vis.py style)
    plt.figure(figsize=(12, 8))
    
    # Display depth map with viridis colormap
    plt.imshow(depth_map, cmap=colormap)
    plt.colorbar(label='Value')
    plt.title(f'{frame_name} - Single Channel')
    plt.axis('off')
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    
    # Convert RGBA to RGB
    img_rgb = img_array[:, :, :3]
    
    # Close figure to free memory
    plt.close()
    
    return img_rgb

def predict_frame(frame_path, output_dir, model=None, device=None, dtype=None, all_camera_params=None, cache_dir="./vggt_models"):
    """
    Use VGGT to predict depth map and camera parameters for a single frame.
    
    Args:
        frame_path: Path to the frame
        output_dir: Directory to save outputs
        model: Pre-loaded VGGT model (optional)
        device: Device to use (optional)
        dtype: Data type for computation (optional)
        all_camera_params: Dictionary to collect all camera parameters (optional)
        cache_dir: Directory to cache/load VGGT model
    
    Returns:
        Updated all_camera_params dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize all_camera_params if not provided
    if all_camera_params is None:
        all_camera_params = {}
    
    # Set device and datatype based on hardware capabilities if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Initialize VGGT model if not provided
    if model is None:
        print("Initializing VGGT model...")
        model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir=cache_dir).to(device)
        model.eval()
    
    # Load the single frame
    print(f"Processing frame: {os.path.basename(frame_path)}")
    frame_paths = [frame_path]
    images = load_and_preprocess_images(frame_paths).to(device)
    
    # Get VGGT predictions
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Add batch dimension for scene
            images_batch = images[None]  # [1, 1, 3, H, W]
            
            # Get aggregated tokens
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            # Predict camera parameters
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Get extrinsic and intrinsic matrices
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            
            # Predict depth maps
            depth_maps, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    # Convert outputs to numpy and remove batch dimension
    depth_maps = depth_maps.squeeze(0).cpu().numpy()  # Should be [1, H, W]
    extrinsic = extrinsic.squeeze(0).cpu().numpy()    # [1, 4, 4]
    intrinsic = intrinsic.squeeze(0).cpu().numpy()    # [1, 3, 3]
    
    # Get the frame filename without extension
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    
    # Extract the depth map for this frame, ensuring it's 2D
    depth_map = depth_maps[0]
    if len(depth_map.shape) > 2:
        depth_map = np.squeeze(depth_map)
    
    # Save normalized depth as PNG (original functionality)
    depth_norm = normalize_depth_for_visualization(depth_map)
    depth_img = Image.fromarray(depth_norm)
    depth_path = os.path.join(output_dir, f"{frame_name}_depth.png")
    depth_img.save(depth_path)
    print(f"Saved depth map to {depth_path}")
    
    # Create and save depth visualization with colorbar
    # Use a suitable frame identifier
    display_name = "00000"  # Or extract from frame_name if needed
    depth_vis = create_depth_visualization(depth_map, display_name, colormap='viridis')
    depth_vis_img = Image.fromarray(depth_vis)
    depth_vis_path = os.path.join(output_dir, f"{frame_name}_depth_vis.png")
    depth_vis_img.save(depth_vis_path)
    print(f"Saved depth visualization to {depth_vis_path}")
    
    # Collect camera parameters for this frame
    all_camera_params[frame_name] = {
        "extrinsic": extrinsic[0].tolist(),
        "intrinsic": intrinsic[0].tolist()
    }
    
    return all_camera_params

def process_folder(input_folder, output_folder, cache_dir="./vggt_models"):
    """
    Process all images in a folder individually.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save outputs
        cache_dir: Directory to cache/load VGGT model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all image files in the folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in sorted(os.listdir(input_folder)):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(os.path.join(input_folder, file))
    
    if len(image_files) < 1:
        print(f"Error: Found no images in {input_folder}.")
        return
    
    # Sort image files to ensure sequential processing
    image_files.sort()
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    # Dictionary to collect all camera parameters
    all_camera_params = {}
    
    # Set device and datatype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Initialize VGGT model once
    print("Initializing VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B", cache_dir=cache_dir).to(device)
    model.eval()
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        all_camera_params = predict_frame(image_path, output_folder, model, device, dtype, all_camera_params, cache_dir)
    
    # Save all camera parameters to a single JSON file
    camera_params_path = os.path.join(output_folder, "cam_params.json")
    with open(camera_params_path, 'w') as f:
        json.dump(all_camera_params, f, indent=4)
    print(f"\nSaved all camera parameters to {camera_params_path}")
    
    print(f"\nProcessing complete! All results saved to {output_folder}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict depth and camera parameters using VGGT")
    parser.add_argument("--mode", type=str, choices=["single", "folder"], required=True, help="Processing mode: 'single' for one image, 'folder' for all images in a folder")
    parser.add_argument("--image", type=str, help="Path to image (for single mode)")
    parser.add_argument("--input_folder", type=str, help="Path to folder containing images (for folder mode)")
    parser.add_argument("--output", type=str, default="vggt_output", help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="./vggt_models", help="Directory to cache/load VGGT model")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.image:
            print("Error: --image is required for single mode")
            parser.print_help()
            exit(1)
        
        # Run prediction on a single image and save camera parameters
        all_camera_params = predict_frame(args.image, args.output, cache_dir=args.cache_dir)
        
        # Save camera parameters to a single JSON file
        camera_params_path = os.path.join(args.output, "cam_params.json")
        with open(camera_params_path, 'w') as f:
            json.dump(all_camera_params, f, indent=4)
        print(f"Saved all camera parameters to {camera_params_path}")
    else:  # folder mode
        if not args.input_folder:
            print("Error: --input_folder is required for folder mode")
            parser.print_help()
            exit(1)
        
        # Process entire folder
        process_folder(args.input_folder, args.output, args.cache_dir)