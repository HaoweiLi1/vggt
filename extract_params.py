#!/usr/bin/env python3
"""
Extract all shooting-related parameters from GeoTIFF files
Usage: python extract_params.py <input.tif> [--output metadata.json]
"""

import json
import os
import argparse
import numpy as np
from osgeo import gdal, osr
from datetime import datetime
import re


def safe_float(value):
    """Safely convert value to float"""
    if value is None:
        return None
    if isinstance(value, str) and '/' in value:
        # Handle rational numbers (e.g., "72/1")
        try:
            num, den = map(float, value.split('/'))
            return num / den if den != 0 else None
        except:
            return None
    try:
        return float(value)
    except:
        return None


def parse_exif_gps(exif_data):
    """Parse GPS data from EXIF"""
    gps_data = {}
    
    # GPS tags mapping
    gps_tags = {
        'EXIF_GPSLatitude': 'latitude',
        'EXIF_GPSLongitude': 'longitude',
        'EXIF_GPSAltitude': 'altitude',
        'EXIF_GPSLatitudeRef': 'lat_ref',
        'EXIF_GPSLongitudeRef': 'lon_ref',
        'EXIF_GPSAltitudeRef': 'alt_ref',
        'EXIF_GPSDateStamp': 'date',
        'EXIF_GPSTimeStamp': 'time',
        'EXIF_GPSImgDirection': 'image_direction',
        'EXIF_GPSImgDirectionRef': 'direction_ref'
    }
    
    for exif_key, gps_key in gps_tags.items():
        if exif_key in exif_data:
            value = exif_data[exif_key]
            if 'Latitude' in exif_key or 'Longitude' in exif_key:
                # Parse DMS format
                if value and ',' in value:
                    try:
                        parts = value.split(',')
                        degrees = safe_float(parts[0])
                        minutes = safe_float(parts[1]) if len(parts) > 1 else 0
                        seconds = safe_float(parts[2]) if len(parts) > 2 else 0
                        decimal = degrees + minutes/60 + seconds/3600
                        gps_data[gps_key] = decimal
                    except:
                        gps_data[gps_key] = value
                else:
                    gps_data[gps_key] = safe_float(value)
            else:
                gps_data[gps_key] = safe_float(value) or value
    
    return gps_data


def extract_geotiff_metadata(geotiff_path):
    """Extract comprehensive metadata from GeoTIFF"""
    ds = gdal.Open(geotiff_path)
    if not ds:
        raise ValueError(f"Cannot open {geotiff_path}")
    
    metadata = {
        'filename': os.path.basename(geotiff_path),
        'file_path': os.path.abspath(geotiff_path),
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # 1. Basic image properties
    metadata['image_properties'] = {
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'driver': ds.GetDriver().ShortName
    }
    
    # 2. Georeferencing information
    gt = ds.GetGeoTransform()
    metadata['georeferencing'] = {
        'geotransform': list(gt),
        'origin_x': gt[0],
        'origin_y': gt[3],
        'pixel_width': gt[1],
        'pixel_height': gt[5],
        'rotation_x': gt[2],
        'rotation_y': gt[4],
        'ground_sample_distance': {
            'x_meters': abs(gt[1]),
            'y_meters': abs(gt[5])
        }
    }
    
    # 3. Coordinate system
    srs = osr.SpatialReference()
    projection = ds.GetProjection()
    if projection:
        srs.ImportFromWkt(projection)
        metadata['coordinate_system'] = {
            'wkt': projection,
            'projection': srs.GetAttrValue('PROJECTION'),
            'datum': srs.GetAttrValue('DATUM'),
            'units': srs.GetLinearUnitsName(),
            'epsg': srs.GetAuthorityCode(None)
        }
        
        # Try to get geographic bounds
        try:
            # Create transformer to WGS84
            wgs84 = osr.SpatialReference()
            wgs84.ImportFromEPSG(4326)
            transform = osr.CoordinateTransformation(srs, wgs84)
            
            # Transform corners
            corners = [
                (gt[0], gt[3]),  # Top-left
                (gt[0] + ds.RasterXSize * gt[1], gt[3]),  # Top-right
                (gt[0], gt[3] + ds.RasterYSize * gt[5]),  # Bottom-left
                (gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5])  # Bottom-right
            ]
            
            geo_corners = []
            for x, y in corners:
                lon, lat, _ = transform.TransformPoint(x, y)
                geo_corners.append({'longitude': lon, 'latitude': lat})
            
            metadata['geographic_bounds'] = {
                'corners': geo_corners,
                'center': {
                    'longitude': sum(c['longitude'] for c in geo_corners) / 4,
                    'latitude': sum(c['latitude'] for c in geo_corners) / 4
                }
            }
        except:
            pass
    
    # 4. General metadata
    general_md = ds.GetMetadata()
    if general_md:
        metadata['general_metadata'] = general_md
        
        # Parse acquisition date if present
        for key in ['TIFFTAG_DATETIME', 'ACQUISITIONDATETIME', 'ACQUISITION_DATE']:
            if key in general_md:
                metadata['acquisition_datetime'] = general_md[key]
                break
    
    # 5. EXIF metadata (camera parameters)
    exif_md = ds.GetMetadata('EXIF')
    if exif_md:
        metadata['exif_metadata'] = exif_md
        
        # Extract camera-specific parameters
        camera_params = {}
        camera_tags = {
            'EXIF_Make': 'camera_make',
            'EXIF_Model': 'camera_model',
            'EXIF_FocalLength': 'focal_length_mm',
            'EXIF_FocalLengthIn35mmFilm': 'focal_length_35mm_equivalent',
            'EXIF_FNumber': 'f_number',
            'EXIF_ExposureTime': 'exposure_time',
            'EXIF_ISOSpeedRatings': 'iso',
            'EXIF_LensModel': 'lens_model'
        }
        
        for exif_key, param_key in camera_tags.items():
            if exif_key in exif_md:
                camera_params[param_key] = safe_float(exif_md[exif_key]) or exif_md[exif_key]
        
        if camera_params:
            metadata['camera_parameters'] = camera_params
        
        # Extract GPS data
        gps_data = parse_exif_gps(exif_md)
        if gps_data:
            metadata['gps_data'] = gps_data
    
    # 6. RPC metadata (for satellite imagery)
    rpc_md = ds.GetMetadata('RPC')
    if rpc_md:
        metadata['rpc_metadata'] = {}
        for key, value in rpc_md.items():
            metadata['rpc_metadata'][key] = safe_float(value) or value
    
    # 7. Band-specific information
    band_info = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        band_data = {
            'band_number': i,
            'data_type': gdal.GetDataTypeName(band.DataType),
            'no_data_value': band.GetNoDataValue(),
            'color_interpretation': gdal.GetColorInterpretationName(band.GetColorInterpretation())
        }
        
        # Get statistics
        stats = band.GetStatistics(False, True)
        if stats:
            band_data['statistics'] = {
                'min': stats[0],
                'max': stats[1],
                'mean': stats[2],
                'std_dev': stats[3]
            }
        
        # Band metadata
        band_md = band.GetMetadata()
        if band_md:
            band_data['metadata'] = band_md
        
        band_info.append(band_data)
    
    metadata['band_information'] = band_info
    
    # 8. Try to infer shooting parameters from filename and metadata
    inferred = infer_shooting_parameters(geotiff_path, metadata)
    if inferred:
        metadata['inferred_parameters'] = inferred
    
    ds = None
    return metadata


def infer_shooting_parameters(filepath, metadata):
    """Infer shooting parameters from filename and metadata"""
    inferred = {}
    filename = os.path.basename(filepath).lower()
    
    # 1. Infer data type
    if 'rgb' in filename:
        inferred['data_type'] = 'RGB imagery'
        inferred['likely_platform'] = 'Aerial (NAIP)' if 'naip' in filename else 'Unknown'
    elif 'elevation' in filename or '3dep' in filename:
        inferred['data_type'] = 'Elevation model'
        inferred['likely_source'] = 'LiDAR' if '1m' in filename else 'Photogrammetry or IFSAR'
    elif 'depth' in filename:
        inferred['data_type'] = 'Depth/terrain derivatives'
    
    # 2. Extract date from filename
    date_patterns = [
        r'(\d{8})',  # YYYYMMDD
        r'(\d{4}[-_]\d{2}[-_]\d{2})',  # YYYY-MM-DD or YYYY_MM_DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            inferred['filename_date'] = match.group(1)
            break
    
    # 3. Estimate platform altitude from GSD
    if 'georeferencing' in metadata:
        gsd = metadata['georeferencing']['ground_sample_distance']['x_meters']
        
        # Rough estimates based on typical platforms
        if gsd <= 0.15:
            inferred['likely_platform'] = 'UAV/Drone'
            inferred['estimated_altitude_m'] = '50-150'
        elif gsd <= 0.5:
            inferred['likely_platform'] = 'Low-altitude aerial'
            inferred['estimated_altitude_m'] = '300-1000'
        elif gsd <= 2.0:
            inferred['likely_platform'] = 'High-altitude aerial (NAIP typical)'
            inferred['estimated_altitude_m'] = '3000-6000'
        else:
            inferred['likely_platform'] = 'Satellite or high-altitude aerial'
            inferred['estimated_altitude_m'] = '>10000'
    
    # 4. Platform-specific parameters
    if 'naip' in filename:
        inferred['platform_details'] = {
            'program': 'National Agriculture Imagery Program',
            'typical_altitude_m': 3000,
            'typical_focal_length_mm': '150-210',
            'typical_sensor': 'Medium format digital',
            'typical_overlap': '60% forward, 30% side'
        }
    
    return inferred


def process_multiple_files(file_list, output_prefix='metadata'):
    """Process multiple GeoTIFF files and extract metadata"""
    all_metadata = {}
    
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        print(f"Processing: {filepath}")
        try:
            metadata = extract_geotiff_metadata(filepath)
            
            # Determine file type
            filename = os.path.basename(filepath).lower()
            if 'rgb' in filename and 'rgbn' not in filename:
                file_type = 'RGB'
            elif 'rgbn' in filename:
                file_type = 'RGBN'
            elif 'elevation' in filename:
                file_type = 'ELEVATION'
            elif 'depth' in filename:
                file_type = 'DEPTH_MULTI'
            else:
                file_type = 'UNKNOWN'
            
            all_metadata[file_type] = metadata
            
            # Save individual file metadata
            output_file = f"{output_prefix}_{file_type}.json"
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"  → Saved to: {output_file}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save combined metadata
    combined_output = f"{output_prefix}_combined.json"
    with open(combined_output, 'w') as f:
        json.dump(all_metadata, f, indent=4)
    print(f"\nCombined metadata saved to: {combined_output}")
    
    return all_metadata


def main():
    parser = argparse.ArgumentParser(description='Extract shooting parameters from GeoTIFF files')
    parser.add_argument('input', nargs='+', help='Input GeoTIFF file(s)')
    parser.add_argument('--output', help='Output JSON file (default: auto-generated)')
    parser.add_argument('--prefix', default='metadata', help='Prefix for output files when processing multiple')
    
    args = parser.parse_args()
    
    if len(args.input) == 1:
        # Single file processing
        metadata = extract_geotiff_metadata(args.input[0])
        
        output_file = args.output
        if not output_file:
            base = os.path.splitext(os.path.basename(args.input[0]))[0]
            output_file = f"{base}_metadata.json"
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\nMetadata extracted to: {output_file}")
        
        # Print summary
        print("\nSummary of shooting-related parameters:")
        if 'camera_parameters' in metadata:
            print("  Camera parameters found:")
            for key, value in metadata['camera_parameters'].items():
                print(f"    {key}: {value}")
        
        if 'gps_data' in metadata:
            print("  GPS data found:")
            for key, value in metadata['gps_data'].items():
                print(f"    {key}: {value}")
        
        if 'inferred_parameters' in metadata:
            print("  Inferred parameters:")
            for key, value in metadata['inferred_parameters'].items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"    {key}: {value}")
    else:
        # Multiple file processing
        process_multiple_files(args.input, args.prefix)


if __name__ == '__main__':
    main()