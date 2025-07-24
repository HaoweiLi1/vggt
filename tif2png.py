import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

def check_tif_properties(tif_path):
    """Check properties of TIF file to understand its format."""
    try:
        # Try with PIL first
        with Image.open(tif_path) as img:
            print(f"\nPIL Info for {os.path.basename(tif_path)}:")
            print(f"  Mode: {img.mode}")
            print(f"  Size: {img.size}")
            print(f"  Format: {img.format}")
            if hasattr(img, 'bits'):
                print(f"  Bits: {img.bits}")
    except Exception as e:
        print(f"PIL failed to read {tif_path}: {e}")
    
    # Try with OpenCV
    img_cv = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img_cv is not None:
        print(f"\nOpenCV Info for {os.path.basename(tif_path)}:")
        print(f"  Shape: {img_cv.shape}")
        print(f"  Dtype: {img_cv.dtype}")
        print(f"  Min/Max values: {img_cv.min()}, {img_cv.max()}")
        if len(img_cv.shape) == 3:
            for i in range(img_cv.shape[2]):
                print(f"  Channel {i} - Min/Max: {img_cv[:,:,i].min()}, {img_cv[:,:,i].max()}")
    else:
        print(f"OpenCV also failed to read {tif_path}")
    
    return img_cv is not None

def robust_normalize(img, percentile_clip=0.1):
    """
    Robustly normalize image data to 0-255 range using percentile clipping.
    This handles outliers better than simple min-max normalization.
    """
    # Handle each channel separately for better color preservation
    if len(img.shape) == 3:
        normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            channel = img[:,:,i]
            # Use percentiles to handle outliers
            p_low = np.percentile(channel, percentile_clip)
            p_high = np.percentile(channel, 100 - percentile_clip)
            
            if p_high - p_low > 1e-6:
                # Clip and normalize
                channel_norm = np.clip(channel, p_low, p_high)
                channel_norm = (channel_norm - p_low) / (p_high - p_low)
                normalized[:,:,i] = channel_norm
            else:
                # Channel is constant
                normalized[:,:,i] = 0
        return (normalized * 255).astype(np.uint8)
    else:
        # Single channel
        p_low = np.percentile(img, percentile_clip)
        p_high = np.percentile(img, 100 - percentile_clip)
        
        if p_high - p_low > 1e-6:
            img_norm = np.clip(img, p_low, p_high)
            img_norm = (img_norm - p_low) / (p_high - p_low)
            return (img_norm * 255).astype(np.uint8)
        else:
            return np.zeros_like(img, dtype=np.uint8)

def convert_tif_to_rgb(tif_path, output_path, format='png', quality=95, debug=False):
    """
    Convert TIF file to RGB format suitable for VGGT.
    
    Args:
        tif_path: Path to input TIF file
        output_path: Path for output file
        format: Output format ('png' or 'jpg')
        quality: JPEG quality (1-100), only used for JPEG
        debug: If True, save intermediate visualization steps
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First try with OpenCV (often handles difficult TIFs better)
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            # Fallback to PIL
            try:
                pil_img = Image.open(tif_path)
                img = np.array(pil_img)
            except Exception as e:
                print(f"Error: Cannot read {tif_path} with either OpenCV or PIL: {e}")
                return False
        
        # Handle different image formats
        original_dtype = img.dtype
        original_shape = img.shape
        
        print(f"\nProcessing: {os.path.basename(tif_path)}")
        print(f"  Input shape: {original_shape}, dtype: {original_dtype}")
        
        # Convert different data types to uint8
        if img.dtype == np.uint16:
            # Check the actual range of values
            actual_max = img.max()
            if actual_max <= 255:
                # Values are already in 8-bit range
                img = img.astype(np.uint8)
            elif actual_max <= 4095:
                # 12-bit data
                img = (img / 16).astype(np.uint8)
            else:
                # Full 16-bit
                img = (img / 256).astype(np.uint8)
                
        elif img.dtype == np.float32 or img.dtype == np.float64:
            # For float images, we need to check the actual range
            img_min, img_max = img.min(), img.max()
            print(f"  Float range: [{img_min:.4f}, {img_max:.4f}]")
            
            if img_min >= 0 and img_max <= 1.0:
                # Already normalized to [0, 1]
                img = (img * 255).astype(np.uint8)
            elif img_min >= 0 and img_max <= 255:
                # Float values in byte range
                img = img.astype(np.uint8)
            else:
                # Use robust normalization for arbitrary float ranges
                # This handles outliers better
                img = robust_normalize(img, percentile_clip=0.5)
                
        elif img.dtype != np.uint8:
            # For other types, use robust normalization
            img = robust_normalize(img, percentile_clip=0.5)
        
        # Handle different channel configurations
        if len(img.shape) == 2:
            # Grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                # Single channel to RGB
                img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                # BGR to RGB (OpenCV uses BGR by default)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                # BGRA/RGBA to RGB
                # Check if alpha channel has any transparency
                if img[:,:,3].min() < 255:
                    # Handle transparency by blending with white background
                    alpha = img[:,:,3:4] / 255.0
                    rgb = img[:,:,:3]
                    white_bg = np.ones_like(rgb) * 255
                    img = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # No transparency, just drop alpha
                    img = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
            else:
                # Multi-channel image (e.g., multispectral)
                print(f"Warning: {tif_path} has {img.shape[2]} channels. Using first 3.")
                if img.shape[2] >= 3:
                    img = img[:,:,:3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    # Less than 3 channels, replicate to make RGB
                    img = np.stack([img[:,:,0]] * 3, axis=2)
        
        # Apply histogram equalization if image looks too dark or too bright
        mean_val = img.mean()
        if mean_val < 30 or mean_val > 225:
            print(f"  Applying histogram equalization (mean: {mean_val:.1f})")
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Ensure we have a valid RGB image
        assert img.shape[2] == 3, f"Expected 3 channels, got {img.shape[2]}"
        assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
        
        # Save the image
        pil_img = Image.fromarray(img)
        
        if format.lower() == 'png':
            pil_img.save(output_path, 'PNG', optimize=True)
        elif format.lower() in ['jpg', 'jpeg']:
            pil_img.save(output_path, 'JPEG', quality=quality, optimize=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Log conversion info
        print(f"✓ Converted successfully")
        print(f"  Output: {img.shape}, {img.dtype}")
        print(f"  Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {tif_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_folder(input_folder, output_folder, format='png', quality=95, check_only=False):
    """
    Convert all TIF files in a folder to RGB format.
    
    Args:
        input_folder: Path to folder containing TIF files
        output_folder: Path to save converted images
        format: Output format ('png' or 'jpg')
        quality: JPEG quality (1-100)
        check_only: If True, only check TIF properties without converting
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all TIF files
    tif_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    tif_files = []
    for pattern in tif_patterns:
        tif_files.extend(Path(input_folder).glob(pattern))
    
    if not tif_files:
        print(f"No TIF files found in {input_folder}")
        return
    
    print(f"Found {len(tif_files)} TIF files")
    
    if check_only:
        # Just check properties
        for tif_path in tif_files:
            check_tif_properties(str(tif_path))
        return
    
    # Convert files
    successful = 0
    failed = 0
    
    for tif_path in tqdm(tif_files, desc="Converting TIF files"):
        # Generate output filename
        output_name = tif_path.stem + f'.{format.lower()}'
        output_path = os.path.join(output_folder, output_name)
        
        if convert_tif_to_rgb(str(tif_path), output_path, format, quality):
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files might need special handling.")
        print("Try running with --check flag to inspect their properties.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF files to RGB format suitable for VGGT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file to PNG
  python tif_convert.py --input test.tif --output test.png
  
  # Convert all TIF files in folder to PNG
  python tif_convert.py --input_folder ./tif_images --output_folder ./png_images
  
  # Convert to JPEG with custom quality
  python tif_convert.py --input_folder ./tif_images --output_folder ./jpg_images --format jpg --quality 90
  
  # Check TIF properties without converting
  python tif_convert.py --input test.tif --check
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to single TIF file')
    input_group.add_argument('--input_folder', type=str, help='Path to folder containing TIF files')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output path for single file conversion')
    parser.add_argument('--output_folder', type=str, help='Output folder for batch conversion')
    
    # Format options
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'jpeg'],
                        help='Output format (default: png)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality 1-100 (default: 95, only for JPEG format)')
    
    # Other options
    parser.add_argument('--check', action='store_true',
                        help='Check TIF properties without converting')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.check:
        if not args.output:
            # Auto-generate output filename
            input_path = Path(args.input)
            args.output = str(input_path.with_suffix(f'.{args.format.lower()}'))
            print(f"Output path not specified, using: {args.output}")
    
    if args.input_folder and not args.check:
        if not args.output_folder:
            args.output_folder = args.input_folder + '_converted'
            print(f"Output folder not specified, using: {args.output_folder}")
    
    # Process based on mode
    if args.input:
        # Single file mode
        if args.check:
            check_tif_properties(args.input)
        else:
            success = convert_tif_to_rgb(args.input, args.output, args.format, args.quality)
            if success:
                print(f"Successfully converted to: {args.output}")
            else:
                print("Conversion failed!")
    else:
        # Folder mode
        convert_folder(args.input_folder, args.output_folder, args.format, args.quality, args.check)

if __name__ == "__main__":
    main()