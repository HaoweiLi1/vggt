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
    """
    Normalize depth map to 0-65535 for visualization and saving as PNG.
    """
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
    """
    Create and save a colorized depth visualization.
    """
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

def process_image(image_path, output_dir, model, device, dtype, vis_dir=None):
    """
    Process a single image and return camera parameters.
    """
    frame_name = Path(image_path).stem
    
    # Load and preprocess image
    images = load_and_preprocess_images([image_path]).to(device)
    
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
    
    # Convert to numpy
    depth_map = depth_maps.squeeze().cpu().numpy()
    extrinsic = extrinsic.squeeze().cpu().numpy()
    intrinsic = intrinsic.squeeze().cpu().numpy()
    
    # Save depth map
    depth_norm = normalize_depth_for_visualization(depth_map)
    depth_path = os.path.join(output_dir, f"{frame_name}_depth.png")
    Image.fromarray(depth_norm).save(depth_path)
    
    # Save visualization if requested
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{frame_name}_depth_vis.png")
        create_depth_visualization(depth_map, vis_path)
    
    # Return camera parameters
    return {
        frame_name: {
            "extrinsic": extrinsic.tolist(),
            "intrinsic": intrinsic.tolist()
        }
    }

def load_model_from_pt(model_path, device):
    """
    Load VGGT model from a local .pt file.
    
    Args:
        model_path: Path to the .pt model file
        device: Device to load the model on
    
    Returns:
        Loaded VGGT model
    """
    print(f"Loading model from: {model_path}")
    
    # Initialize the model architecture
    model = VGGT()
    
    # Load the state dict from the .pt file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load state dict with map_location for proper device handling
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model

def main():
    parser = argparse.ArgumentParser(description="VGGT depth and camera prediction")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to image file or directory containing images")
    parser.add_argument("--output", type=str, default="vggt_output1", 
                        help="Output directory")
    parser.add_argument("--vis", type=str, default=None, 
                        help="Path to save visualizations (default: None)")
    parser.add_argument("--model_path", type=str, default="model/vggt_1B.pt", 
                        help="Path to the VGGT model .pt file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine input type
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image mode
        image_files = [str(input_path)]
        print(f"Processing single image: {input_path.name}")
    elif input_path.is_dir():
        # Folder mode
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_files = sorted([
            str(f) for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        print(f"Found {len(image_files)} images in {input_path}")
        if not image_files:
            print(f"Error: No images found in {input_path}")
            return
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return
    
    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load model from local .pt file
    try:
        model = load_model_from_pt(args.model_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the model file exists at: {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process images
    all_camera_params = {}
    for i, image_path in enumerate(image_files):
        print(f"Processing [{i+1}/{len(image_files)}]: {Path(image_path).name}")
        camera_params = process_image(
            image_path, args.output, model, device, dtype, args.vis
        )
        all_camera_params.update(camera_params)
    
    # Save camera parameters
    camera_params_path = os.path.join(args.output, "cam_params.json")
    with open(camera_params_path, 'w') as f:
        json.dump(all_camera_params, f, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"Depth maps saved to: {args.output}")
    print(f"Camera parameters saved to: {camera_params_path}")
    if args.vis:
        print(f"Visualizations saved to: {args.vis}")

if __name__ == "__main__":
    main()