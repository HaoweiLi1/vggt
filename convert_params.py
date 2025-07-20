#!/usr/bin/env python3
"""
Convert GeoTIFF files to camera parameters using multiple methods
Usage: python convert_params.py <input.tif> [--output cam_params.json] [--method auto]
"""

import numpy as np
import json
import os
import argparse
from osgeo import gdal, osr
import warnings
warnings.filterwarnings('ignore')


def orthographic_virtual_camera(geotiff_path, virtual_altitude=1000):
    """
    Method 1: Create virtual orthographic camera parameters
    Most suitable for orthorectified imagery
    """
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    # Get dimensions and geotransform
    width, height = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    
    # Get coordinate system
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    
    # Calculate scene bounds
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + width * gt[1]
    miny = maxy + height * gt[5]
    
    # Scene center in projected coordinates
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # For orthographic projection, approximate with large focal length
    scale_x = width / (maxx - minx)  # pixels per meter
    scale_y = height / (maxy - miny)
    
    # Virtual focal length (approximate infinity with large value)
    virtual_focal = max(width, height) * virtual_altitude / max(maxx - minx, maxy - miny)
    
    intrinsic = np.array([
        [virtual_focal, 0, width/2],
        [0, virtual_focal, height/2],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix: camera looking straight down
    extrinsic = np.eye(4)
    extrinsic[0, 3] = center_x
    extrinsic[1, 3] = center_y
    extrinsic[2, 3] = virtual_altitude
    
    # Rotation for downward-looking camera (OpenCV convention)
    R = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    extrinsic[:3, :3] = R
    
    return {
        'intrinsic': intrinsic.tolist(),
        'extrinsic': extrinsic.tolist(),
        'method': 'orthographic_virtual',
        'parameters': {
            'virtual_altitude_m': virtual_altitude,
            'scene_width_m': maxx - minx,
            'scene_height_m': maxy - miny,
            'gsd_x': abs(gt[1]),
            'gsd_y': abs(gt[5])
        }
    }


def pinhole_camera_from_gsd(geotiff_path, flight_height=3000, sensor_width_mm=60):
    """
    Method 2: Estimate pinhole camera parameters from Ground Sample Distance
    More realistic for aerial photography
    """
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    width, height = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()
    
    # Ground Sample Distance (meters per pixel)
    gsd_x = abs(gt[1])
    gsd_y = abs(gt[5])
    
    # Calculate scene bounds
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + width * gt[1]
    miny = maxy + height * gt[5]
    
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # Estimate focal length from GSD
    # GSD = (pixel_size_on_sensor * flight_height) / focal_length
    # focal_length_mm = (sensor_width_mm * flight_height) / (image_width_px * gsd)
    focal_mm = (sensor_width_mm * flight_height) / (width * gsd_x)
    
    # Convert to pixels
    pixel_size_mm = sensor_width_mm / width
    focal_px_x = focal_mm / pixel_size_mm
    focal_px_y = focal_px_x * (gsd_x / gsd_y)  # Account for non-square pixels
    
    # Intrinsic matrix
    intrinsic = np.array([
        [focal_px_x, 0, width/2],
        [0, focal_px_y, height/2],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[0, 3] = center_x
    extrinsic[1, 3] = center_y
    extrinsic[2, 3] = flight_height
    
    # Rotation for downward-looking camera
    R = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    extrinsic[:3, :3] = R
    
    return {
        'intrinsic': intrinsic.tolist(),
        'extrinsic': extrinsic.tolist(),
        'method': 'pinhole_from_gsd',
        'parameters': {
            'estimated_focal_length_mm': focal_mm,
            'assumed_flight_height_m': flight_height,
            'assumed_sensor_width_mm': sensor_width_mm,
            'gsd_x': gsd_x,
            'gsd_y': gsd_y
        }
    }


def rpc_to_camera_params(geotiff_path):
    """
    Method 3: Convert Rational Polynomial Coefficients to camera model
    Only works if GeoTIFF contains RPC metadata
    """
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    # Check for RPC metadata
    rpc_md = ds.GetMetadata('RPC')
    
    if not rpc_md:
        return None
    
    width, height = ds.RasterXSize, ds.RasterYSize
    
    # Extract RPC parameters
    params = {}
    for key in ['LINE_OFF', 'SAMP_OFF', 'LAT_OFF', 'LONG_OFF', 'HEIGHT_OFF',
                'LINE_SCALE', 'SAMP_SCALE', 'LAT_SCALE', 'LONG_SCALE', 'HEIGHT_SCALE']:
        params[key] = float(rpc_md.get(key, 0))
    
    # Approximate camera center from RPC offsets
    center_lat = params['LAT_OFF']
    center_lon = params['LONG_OFF']
    center_height = params['HEIGHT_OFF']
    
    # Image center
    center_x = params['SAMP_OFF']
    center_y = params['LINE_OFF']
    
    # Approximate focal length from scales
    # This is a simplification - full RPC model is much more complex
    approx_focal = max(width, height) * params['HEIGHT_SCALE'] / max(params['LAT_SCALE'], params['LONG_SCALE'])
    
    # Build approximate intrinsic matrix
    intrinsic = np.array([
        [approx_focal, 0, center_x],
        [0, approx_focal, center_y],
        [0, 0, 1]
    ])
    
    # Approximate extrinsic (would need full RPC evaluation for accuracy)
    extrinsic = np.eye(4)
    # Convert lat/lon to projected coordinates if needed
    # This is simplified - proper implementation would project coordinates
    extrinsic[0, 3] = center_lon * 111000  # Rough meters per degree
    extrinsic[1, 3] = center_lat * 111000
    extrinsic[2, 3] = center_height
    
    return {
        'intrinsic': intrinsic.tolist(),
        'extrinsic': extrinsic.tolist(),
        'method': 'rpc_approximation',
        'parameters': {
            'rpc_metadata': params,
            'warning': 'This is a simplified RPC conversion - full accuracy requires complete RPC evaluation'
        }
    }


def combined_method(geotiff_path, elevation_path=None):
    """
    Method 4: Combined approach using available metadata and elevation
    """
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    result = {}
    
    # Try to extract any existing metadata
    metadata = ds.GetMetadata()
    exif = ds.GetMetadata('EXIF')
    
    # Default parameters
    flight_height = 3000  # Default NAIP altitude
    sensor_width = 60     # Medium format sensor
    
    # Try to get better estimates from metadata
    if exif:
        if 'EXIF_FocalLength' in exif:
            focal_mm = float(exif['EXIF_FocalLength'].split('/')[0])
            result['detected_focal_length_mm'] = focal_mm
        
        if 'EXIF_GPSAltitude' in exif:
            alt_str = exif['EXIF_GPSAltitude']
            if '/' in alt_str:
                num, den = map(float, alt_str.split('/'))
                flight_height = num / den
            else:
                flight_height = float(alt_str)
            result['detected_altitude_m'] = flight_height
    
    # Check for RPC first
    if ds.GetMetadata('RPC'):
        print("Found RPC metadata - using RPC method")
        params = rpc_to_camera_params(geotiff_path)
        if params:
            return params
    
    # Use elevation to refine altitude if available
    if elevation_path and os.path.exists(elevation_path):
        elev_ds = gdal.Open(elevation_path)
        if elev_ds:
            elev_data = elev_ds.GetRasterBand(1).ReadAsArray()
            valid_elev = elev_data[~np.isnan(elev_data)]
            if len(valid_elev) > 0:
                mean_elevation = np.mean(valid_elev)
                result['mean_ground_elevation_m'] = float(mean_elevation)
                # Adjust flight height to above ground level
                flight_height_agl = flight_height - mean_elevation
                result['flight_height_agl_m'] = flight_height_agl
    
    # Use pinhole model with best available parameters
    params = pinhole_camera_from_gsd(geotiff_path, flight_height, sensor_width)
    params['additional_info'] = result
    
    return params


def auto_select_method(geotiff_path, elevation_path=None):
    """
    Automatically select the best method based on available data
    """
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    print(f"Analyzing {os.path.basename(geotiff_path)}...")
    
    # Check what metadata is available
    has_rpc = bool(ds.GetMetadata('RPC'))
    has_exif = bool(ds.GetMetadata('EXIF'))
    
    print(f"  RPC metadata: {'Found' if has_rpc else 'Not found'}")
    print(f"  EXIF metadata: {'Found' if has_exif else 'Not found'}")
    
    if has_rpc:
        print("  → Using RPC method")
        result = rpc_to_camera_params(geotiff_path)
        if result:
            return result
    
    if has_exif or elevation_path:
        print("  → Using combined method with available metadata")
        return combined_method(geotiff_path, elevation_path)
    
    # Check image type from filename
    filename = os.path.basename(geotiff_path).lower()
    if 'naip' in filename or 'rgb' in filename:
        print("  → Detected aerial imagery - using pinhole model")
        return pinhole_camera_from_gsd(geotiff_path)
    else:
        print("  → Using orthographic virtual camera")
        return orthographic_virtual_camera(geotiff_path)


def main():
    parser = argparse.ArgumentParser(description='Convert GeoTIFF to camera parameters')
    parser.add_argument('input', help='Input GeoTIFF file')
    parser.add_argument('--output', default='cam_params.json', help='Output JSON file')
    parser.add_argument('--method', choices=['auto', 'orthographic', 'pinhole', 'rpc', 'combined'], 
                        default='auto', help='Conversion method')
    parser.add_argument('--elevation', help='Elevation GeoTIFF for refined altitude')
    parser.add_argument('--altitude', type=float, help='Override flight altitude (meters)')
    parser.add_argument('--sensor-width', type=float, default=60, help='Sensor width in mm')
    
    args = parser.parse_args()
    
    # Select and run method
    if args.method == 'auto':
        params = auto_select_method(args.input, args.elevation)
    elif args.method == 'orthographic':
        altitude = args.altitude or 1000
        params = orthographic_virtual_camera(args.input, altitude)
    elif args.method == 'pinhole':
        altitude = args.altitude or 3000
        params = pinhole_camera_from_gsd(args.input, altitude, args.sensor_width)
    elif args.method == 'rpc':
        params = rpc_to_camera_params(args.input)
        if not params:
            print("Error: No RPC metadata found")
            return
    elif args.method == 'combined':
        params = combined_method(args.input, args.elevation)
    
    # Add source file info
    params['source_file'] = os.path.abspath(args.input)
    params['image_dimensions'] = {
        'width': gdal.Open(args.input).RasterXSize,
        'height': gdal.Open(args.input).RasterYSize
    }
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"\nCamera parameters saved to: {args.output}")
    print(f"Method used: {params['method']}")


if __name__ == '__main__':
    main()