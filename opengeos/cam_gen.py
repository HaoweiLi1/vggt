#!/usr/bin/env python3
"""
cam_gen.py - Generate camera intrinsic and extrinsic parameters from NAIP metadata

This script calculates pinhole camera model parameters from NAIP imagery metadata,
suitable for use with VGGT or other computer vision applications.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from pyproj import Transformer, CRS
from scipy.spatial.transform import Rotation
import rasterio
from rasterio.transform import Affine


class NAIPCameraCalculator:
    """Calculate camera parameters from NAIP metadata"""
    
    def __init__(self, metadata_path, hints_path, image_path=None):
        """
        Initialize calculator with metadata files
        
        Args:
            metadata_path: Path to camera_metadata.json
            hints_path: Path to camera_hints.json
            image_path: Optional path to actual GeoTIFF for validation
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        with open(hints_path, 'r') as f:
            self.hints = json.load(f)
        
        self.image_path = image_path
        
        # Extract key parameters
        self.parse_metadata()
    
    def parse_metadata(self):
        """Extract and validate key parameters from metadata"""
        # Image dimensions
        band_info = self.metadata['bands_info'][0]
        self.image_width = band_info['dimensions'][0]
        self.image_height = band_info['dimensions'][1]
        
        # CRS and transform
        self.crs = CRS.from_string(band_info['crs'])
        transform_params = band_info['crs_transform']
        self.transform = Affine(
            transform_params[0], transform_params[1], transform_params[2],
            transform_params[3], transform_params[4], transform_params[5]
        )
        
        # GSD and flight parameters
        self.gsd = abs(transform_params[0])  # Ground sample distance in meters
        self.flight_altitude = self.hints['estimated_flight_altitude_meters']
        self.focal_length_mm = self.hints['estimated_focal_length_mm']
        self.sensor_width_mm = self.hints['estimated_sensor_width_mm']
        
        # Geographic bounds
        self.bounds = self.metadata['geometry']['coordinates'][0]
        
        print(f"Parsed metadata:")
        print(f"  Image: {self.image_width} × {self.image_height} pixels")
        print(f"  GSD: {self.gsd} meters/pixel")
        print(f"  CRS: {self.crs}")
        print(f"  Flight altitude: {self.flight_altitude} meters")
    
    def calculate_intrinsic_matrix(self):
        """
        Calculate camera intrinsic matrix K
        
        The intrinsic matrix relates 3D camera coordinates to 2D image coordinates:
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        # Calculate focal length in pixels
        # Using the relationship: focal_length_pixels = (focal_length_mm * image_width) / sensor_width_mm
        focal_length_pixels = (self.focal_length_mm * self.image_width) / self.sensor_width_mm
        
        # For aerial imagery, we can also verify using GSD:
        # focal_length_pixels = (flight_altitude * image_width) / (GSD * image_width_meters)
        image_width_meters = self.gsd * self.image_width
        focal_length_from_gsd = (self.flight_altitude * self.image_width) / image_width_meters
        
        print(f"\nFocal length calculations:")
        print(f"  From sensor specs: {focal_length_pixels:.2f} pixels")
        print(f"  From GSD relation: {focal_length_from_gsd:.2f} pixels")
        
        # Use average for robustness
        focal_length_final = (focal_length_pixels + focal_length_from_gsd) / 2
        
        # Principal point (image center for orthorectified imagery)
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        
        # Build intrinsic matrix
        K = np.array([
            [focal_length_final, 0, cx],
            [0, focal_length_final, cy],
            [0, 0, 1]
        ])
        
        self.intrinsic_matrix = K
        self.focal_length_pixels = focal_length_final
        
        print(f"\nIntrinsic matrix K:")
        print(K)
        
        return K
    
    def calculate_extrinsic_matrix(self, elevation=None):
        """
        Calculate camera extrinsic matrix [R|t]
        
        For NAIP nadir imagery, we assume:
        - Camera looking straight down (nadir view)
        - North-aligned flight lines
        - Camera at flight altitude above ground
        
        Args:
            elevation: Ground elevation at image center (meters). If None, uses 0.
        
        Returns:
            4x4 extrinsic matrix (world to camera transformation)
        """
        # Get image center in projected coordinates
        center_x = self.transform.c + (self.image_width / 2) * self.transform.a
        center_y = self.transform.f + (self.image_height / 2) * self.transform.e
        
        # Camera position in world coordinates (UTM)
        if elevation is None:
            elevation = 0  # Sea level if not provided
        
        camera_x = center_x
        camera_y = center_y
        camera_z = elevation + self.flight_altitude
        
        print(f"\nCamera position (UTM):")
        print(f"  X (Easting): {camera_x:.2f} meters")
        print(f"  Y (Northing): {camera_y:.2f} meters")
        print(f"  Z (Altitude): {camera_z:.2f} meters")
        
        # For nadir imagery, rotation matrix is often identity or has small tilts
        # We'll assume perfect nadir for orthorectified NAIP
        # Camera axes: X=East, Y=North, Z=Up (looking down)
        
        # World axes to camera axes transformation
        # For nadir view: camera Z axis points down (-Z world)
        # Camera Y axis points north (+Y world)
        # Camera X axis points east (+X world)
        
        R = np.array([
            [1,  0,  0],   # Camera X = World X (East)
            [0,  1,  0],   # Camera Y = World Y (North)  
            [0,  0, -1]    # Camera Z = -World Z (Down)
        ])
        
        # Translation vector (world origin to camera position)
        t = np.array([camera_x, camera_y, camera_z])
        
        # Build 4x4 extrinsic matrix [R|t]
        # This transforms from world coordinates to camera coordinates
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = -R @ t  # Negative because we want world-to-camera
        
        self.extrinsic_matrix = extrinsic
        self.camera_position = t
        self.rotation_matrix = R
        
        print(f"\nExtrinsic matrix [R|t]:")
        print(extrinsic)
        
        return extrinsic
    
    def calculate_projection_matrix(self):
        """
        Calculate full projection matrix P = K[R|t]
        
        This matrix projects 3D world points to 2D image coordinates
        """
        # Extract 3x4 part of extrinsic matrix
        RT = self.extrinsic_matrix[:3, :]
        
        # Projection matrix
        P = self.intrinsic_matrix @ RT
        
        self.projection_matrix = P
        
        print(f"\nProjection matrix P:")
        print(P)
        
        return P
    
    def convert_to_geographic(self):
        """Convert camera position to geographic coordinates (lat/lon)"""
        # Create transformer from UTM to WGS84
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        
        # Transform camera position
        lon, lat = transformer.transform(self.camera_position[0], self.camera_position[1])
        
        print(f"\nCamera position (Geographic):")
        print(f"  Longitude: {lon:.6f}°")
        print(f"  Latitude: {lat:.6f}°")
        print(f"  Altitude: {self.camera_position[2]:.2f} meters")
        
        return lon, lat, self.camera_position[2]
    
    def validate_with_image(self):
        """Validate calculations using actual GeoTIFF if provided"""
        if self.image_path and Path(self.image_path).exists():
            print(f"\nValidating with image: {self.image_path}")
            
            with rasterio.open(self.image_path) as src:
                # Check dimensions
                assert src.width == self.image_width, f"Width mismatch: {src.width} vs {self.image_width}"
                assert src.height == self.image_height, f"Height mismatch: {src.height} vs {self.image_height}"
                
                # Check transform
                img_transform = src.transform
                print(f"  Image transform matches: {np.allclose(img_transform, self.transform)}")
                
                # Check CRS
                print(f"  CRS matches: {src.crs == self.crs}")
    
    def save_parameters(self, output_path):
        """
        Save camera parameters in multiple formats
        
        Args:
            output_path: Base path for output files (without extension)
        """
        output_path = Path(output_path)
        
        # 1. VGGT format (simplified)
        vggt_params = {
            output_path.stem: {
                "extrinsic": self.extrinsic_matrix.tolist(),
                "intrinsic": self.intrinsic_matrix.tolist()
            }
        }
        
        with open(f"{output_path}_vggt.json", 'w') as f:
            json.dump(vggt_params, f, indent=4)
        
        # 2. Detailed format with all parameters
        detailed_params = {
            "intrinsic": {
                "matrix": self.intrinsic_matrix.tolist(),
                "focal_length_pixels": self.focal_length_pixels,
                "focal_length_mm": self.focal_length_mm,
                "principal_point": [self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]],
                "sensor_width_mm": self.sensor_width_mm,
                "image_dimensions": [self.image_width, self.image_height]
            },
            "extrinsic": {
                "matrix": self.extrinsic_matrix.tolist(),
                "rotation_matrix": self.rotation_matrix.tolist(),
                "camera_position_utm": {
                    "x": self.camera_position[0],
                    "y": self.camera_position[1],
                    "z": self.camera_position[2],
                    "crs": str(self.crs)
                },
                "camera_position_geographic": {
                    "longitude": self.convert_to_geographic()[0],
                    "latitude": self.convert_to_geographic()[1],
                    "altitude": self.convert_to_geographic()[2]
                }
            },
            "projection": {
                "matrix": self.projection_matrix.tolist(),
                "gsd": self.gsd,
                "image_to_world_transform": list(self.transform)
            },
            "metadata": {
                "source": "NAIP",
                "image_date": self.hints['image_date'],
                "flight_altitude": self.flight_altitude,
                "processing_date": self.metadata['extracted_date']
            }
        }
        
        with open(f"{output_path}_detailed.json", 'w') as f:
            json.dump(detailed_params, f, indent=4)
        
        # 3. COLMAP format (cameras.txt style)
        with open(f"{output_path}_colmap.txt", 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {self.image_width} {self.image_height} ")
            f.write(f"{self.focal_length_pixels} {self.focal_length_pixels} ")
            f.write(f"{self.intrinsic_matrix[0, 2]} {self.intrinsic_matrix[1, 2]}\n")
        
        print(f"\nParameters saved to:")
        print(f"  - {output_path}_vggt.json (VGGT format)")
        print(f"  - {output_path}_detailed.json (Detailed format)")
        print(f"  - {output_path}_colmap.txt (COLMAP format)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate camera parameters from NAIP metadata"
    )
    parser.add_argument(
        "metadata_json",
        nargs='?',
        default='output/washington_dc_1m_camera_metadata.json',
        help="Path to camera_metadata.json file"
    )
    parser.add_argument(
        "hints_json",
        nargs='?',
        default='output/washington_dc_1m_camera_hints.json',
        help="Path to camera_hints.json file"
    )
    parser.add_argument(
        "-i", "--image",
        # default='output/washington_dc_1m_3DEP_ELEVATION_1m_20250720_051107.tif',
        help="Optional path to GeoTIFF for validation"
    )
    parser.add_argument(
        "-e", "--elevation",
        type=float,
        default=None,
        help="Ground elevation in meters (default: 0)"
    )
    parser.add_argument(
        "-o", "--output",
        default="camera_params",
        help="Output base filename (default: camera_params)"
    )
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = NAIPCameraCalculator(
        args.metadata_json,
        args.hints_json,
        args.image
    )
    
    # Calculate parameters
    print("="*60)
    print("NAIP Camera Parameter Calculator")
    print("="*60)
    
    # Calculate intrinsic matrix
    K = calculator.calculate_intrinsic_matrix()
    
    # Calculate extrinsic matrix
    RT = calculator.calculate_extrinsic_matrix(args.elevation)
    
    # Calculate projection matrix
    P = calculator.calculate_projection_matrix()
    
    # Convert to geographic coordinates
    calculator.convert_to_geographic()
    
    # Validate if image provided
    if args.image:
        calculator.validate_with_image()
    
    # Save results
    calculator.save_parameters(args.output)
    
    print("\nCalculation complete!")


if __name__ == "__main__":
    main()