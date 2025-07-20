from osgeo import gdal
import numpy as np
from PIL import Image
import os
import json


def enhance_elevation_contrast(data, valid_mask, enhancement_factor=2.0):
    """
    Enhance contrast for elevation data visualization
    """
    valid_data = data[valid_mask]
    
    # Use a tighter percentile range for better contrast
    pmin = np.percentile(valid_data, 5)
    pmax = np.percentile(valid_data, 95)
    
    # Apply contrast enhancement
    center = (pmin + pmax) / 2
    enhanced = center + (data - center) * enhancement_factor
    
    # Clip to valid range
    enhanced = np.clip(enhanced, pmin, pmax)
    
    return enhanced, pmin, pmax


def geotiff_to_png_gdal(input_path, output_path=None, force_grayscale=False, 
                       enhance_depth=True, save_lossless=True):
    """
    Convert GeoTIFF to PNG with improved depth visualization
    
    Args:
        input_path: Path to GeoTIFF file
        output_path: Output PNG path (optional)
        force_grayscale: Force grayscale output
        enhance_depth: Apply contrast enhancement to depth data
        save_lossless: Also save 16-bit version for depth data
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.png'
    
    # Open the dataset
    dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    if dataset is None:
        print(f"Error: Could not open {input_path}")
        return
    
    # Get dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    
    print(f"\nProcessing: {os.path.basename(input_path)}")
    print(f"Dimensions: {width}x{height}, {bands} bands")
    
    # Check if this is a depth/elevation file
    is_depth = any(keyword in input_path.lower() for keyword in 
                  ['depth', 'elevation', '3dep', 'dem', 'dtm', 'dsm'])
    
    # Handle multi-band depth files (DEPTH_MULTI)
    if is_depth and bands > 1:
        print(f"\nDetected multi-band depth file with {bands} bands")
        print("Processing elevation band (band 1) for primary output")
        
        # Save each band separately for analysis
        band_names = ['elevation', 'slope', 'aspect', 'hillshade']
        for i in range(min(bands, len(band_names))):
            band_data = dataset.GetRasterBand(i + 1).ReadAsArray()
            band_output = os.path.splitext(output_path)[0] + f'_{band_names[i]}.png'
            
            # Process each band
            process_single_band(band_data, band_output, band_names[i], 
                              dataset.GetRasterBand(i + 1).GetNoDataValue(),
                              enhance_depth and i == 0)  # Only enhance elevation
            
        # Use elevation (band 1) for main output
        band_data = dataset.GetRasterBand(1).ReadAsArray()
        nodata = dataset.GetRasterBand(1).GetNoDataValue()
        
    elif bands >= 3 and not (is_depth or force_grayscale):
        # RGB image processing (unchanged)
        r = dataset.GetRasterBand(1).ReadAsArray()
        g = dataset.GetRasterBand(2).ReadAsArray()
        b = dataset.GetRasterBand(3).ReadAsArray()
        
        # Stack bands
        rgb = np.dstack((r, g, b))
        
        # Normalize each band to 0-255
        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
        
        for i in range(3):
            band = rgb[:, :, i]
            # Handle NaN and invalid values
            valid_mask = ~np.isnan(band) & (band != dataset.GetRasterBand(i+1).GetNoDataValue())
            valid_data = band[valid_mask]
            
            if len(valid_data) > 0:
                # Use full range for RGB to maintain quality
                pmin = np.min(valid_data)
                pmax = np.max(valid_data)
                
                if pmax > pmin:
                    # Normalize and clip
                    band_norm = (band - pmin) / (pmax - pmin)
                    band_norm = np.clip(band_norm, 0, 1)
                    rgb_normalized[:, :, i] = (band_norm * 255).astype(np.uint8)
        
        # Save as RGB
        Image.fromarray(rgb_normalized, 'RGB').save(output_path, 'PNG')
        print(f"Saved RGB image: {output_path}")
        dataset = None
        return
        
    else:
        # Single band processing
        band_data = dataset.GetRasterBand(1).ReadAsArray()
        nodata = dataset.GetRasterBand(1).GetNoDataValue()
    
    # Process single band data
    dataset = None
    process_single_band(band_data, output_path, 'depth' if is_depth else 'grayscale', 
                       nodata, enhance_depth and is_depth, save_lossless and is_depth)


def process_single_band(band_data, output_path, data_type, nodata, 
                       enhance=True, save_lossless=True):
    """
    Process single band data with enhancement options
    """
    # Create mask for valid data
    if nodata is not None:
        valid_mask = (band_data != nodata) & ~np.isnan(band_data)
    else:
        valid_mask = ~np.isnan(band_data)
    
    valid_data = band_data[valid_mask]
    
    if len(valid_data) == 0:
        print("Error: No valid data found")
        return
    
    # Calculate statistics
    data_min = float(np.min(valid_data))
    data_max = float(np.max(valid_data))
    data_mean = float(np.mean(valid_data))
    data_std = float(np.std(valid_data))
    
    print(f"\n{data_type} statistics:")
    print(f"  Min: {data_min:.2f}")
    print(f"  Max: {data_max:.2f}")
    print(f"  Mean: {data_mean:.2f}")
    print(f"  Std: {data_std:.2f}")
    
    # Save 16-bit lossless version first (if requested)
    if save_lossless and data_type in ['depth', 'elevation']:
        lossless_path = os.path.splitext(output_path)[0] + '_16bit.png'
        save_16bit_lossless(band_data, valid_mask, data_min, data_max, lossless_path)
    
    # Create enhanced 8-bit visualization
    if enhance and data_type in ['depth', 'elevation']:
        print("\nApplying contrast enhancement for better visualization...")
        
        # Method 1: Histogram equalization
        enhanced_data = histogram_equalize(band_data, valid_mask)
        
        # Method 2: Alternative - use standard deviation stretch
        # enhanced_data = std_stretch(band_data, valid_mask, data_mean, data_std)
        
    else:
        # Standard normalization
        pmin = np.percentile(valid_data, 2)
        pmax = np.percentile(valid_data, 98)
        
        if pmax > pmin:
            enhanced_data = (band_data - pmin) / (pmax - pmin)
            enhanced_data = np.clip(enhanced_data, 0, 1) * 255
        else:
            enhanced_data = np.zeros_like(band_data)
    
    # Convert to uint8 and set invalid pixels
    output_8bit = enhanced_data.astype(np.uint8)
    output_8bit[~valid_mask] = 0
    
    # Save 8-bit visualization
    Image.fromarray(output_8bit, 'L').save(output_path)
    print(f"Saved {data_type} visualization: {output_path}")


def histogram_equalize(data, valid_mask):
    """
    Apply histogram equalization for better contrast
    """
    valid_data = data[valid_mask]
    
    # Create histogram
    hist, bins = np.histogram(valid_data, bins=256)
    
    # Calculate CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255.0 / cdf[-1]
    
    # Interpolate
    output = np.zeros_like(data)
    valid_min = valid_data.min()
    valid_max = valid_data.max()
    
    if valid_max > valid_min:
        # Map values to bins
        normalized = (data - valid_min) / (valid_max - valid_min)
        indices = np.clip((normalized * 255).astype(int), 0, 255)
        output = cdf_normalized[indices]
    
    return output


def std_stretch(data, valid_mask, mean, std, n_std=2.5):
    """
    Stretch data based on standard deviation
    """
    lower = mean - n_std * std
    upper = mean + n_std * std
    
    stretched = (data - lower) / (upper - lower)
    stretched = np.clip(stretched, 0, 1) * 255
    
    return stretched


def save_16bit_lossless(data, valid_mask, data_min, data_max, output_path):
    """
    Save as 16-bit PNG with metadata for lossless storage
    """
    if data_max > data_min:
        scale_factor = 65535.0 / (data_max - data_min)
        normalized = (data - data_min) * scale_factor
        normalized[~valid_mask] = 0
        
        # Convert to 16-bit
        data_16bit = np.clip(normalized, 0, 65535).astype(np.uint16)
        
        # Save as 16-bit PNG
        Image.fromarray(data_16bit, mode='I;16').save(output_path)
        
        # Save metadata
        metadata = {
            'original_min': data_min,
            'original_max': data_max,
            'scale_factor': float(scale_factor),
            'reconstruction_formula': 'original_value = (png_value / scale_factor) + original_min'
        }
        
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved 16-bit lossless version: {output_path}")


if __name__ == "__main__":
    # Process your files
    # This will create multiple outputs for better analysis
    geotiff_to_png_gdal('output/washington_dc_1m_3DEP_ELEVATION_1m_20250719_231011.tif')
    geotiff_to_png_gdal('output/washington_dc_1m_3DEP_DEPTH_MULTI_1m_20250719_231012.tif')