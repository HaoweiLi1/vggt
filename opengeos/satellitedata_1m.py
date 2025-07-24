import ee
import geemap
import os
import time
from datetime import datetime

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='map-api-465501')


def check_data_availability(bounds, year_range):
    """
    Check if both NAIP and USGS 3DEP 1m data are available for the given bounds
    
    Returns:
        tuple: (naip_available, dep3_available, naip_count, dep3_count)
    """
    region = ee.Geometry.Rectangle(bounds)
    
    # Check NAIP availability
    naip_collection = ee.ImageCollection('USDA/NAIP/DOQQ') \
        .filterBounds(region) \
        .filterDate(f'{year_range[0]}-01-01', f'{year_range[1]}-12-31') \
        .select(['R', 'G', 'B', 'N'])
    
    naip_count = naip_collection.size().getInfo()
    naip_available = naip_count > 0
    
    # Check USGS 3DEP 1m availability
    dep3_collection = ee.ImageCollection('USGS/3DEP/1m') \
        .filterBounds(region)
    
    dep3_count = dep3_collection.size().getInfo()
    dep3_available = dep3_count > 0
    
    return naip_available, dep3_available, naip_count, dep3_count


def extract_naip_3dep_1m_data(region_name, bounds, year_range=[2020, 2023], output_dir='./output'):
    """
    Extract high-precision RGB (NAIP 1m) and elevation (USGS 3DEP 1m) data
    Only processes areas with both datasets available
    
    Args:
        region_name: Name for the export files
        bounds: [west, south, east, north]
        year_range: [start_year, end_year] for NAIP imagery
        output_dir: Output directory for local files
    """
    os.makedirs(output_dir, exist_ok=True)
    region = ee.Geometry.Rectangle(bounds)
    
    print(f"\nExtracting high-resolution data for {region_name}")
    print("="*50)
    
    # Check data availability
    print("\nChecking data availability...")
    naip_available, dep3_available, naip_count, dep3_count = check_data_availability(bounds, year_range)
    
    print(f"NAIP 1m RGB: {'Available' if naip_available else 'NOT AVAILABLE'} ({naip_count} images)")
    print(f"USGS 3DEP 1m: {'Available' if dep3_available else 'NOT AVAILABLE'} ({dep3_count} tiles)")
    
    if not naip_available or not dep3_available:
        print("\n❌ ERROR: This area does not have both NAIP 1m and USGS 3DEP 1m coverage!")
        print("Please select a different area with both datasets available.")
        print("\nSuggested areas with both datasets:")
        print("- Major urban areas (Chicago, Washington DC, Denver)")
        print("- Yosemite Valley")
        print("- Check https://apps.nationalmap.gov/lidar-explorer/ for 3DEP 1m coverage")
        return None
    
    print("\n✓ Both datasets available. Proceeding with extraction...")
    
    # 1. NAIP RGB Imagery (1m resolution)
    print("\n1. Acquiring NAIP imagery (1m resolution)...")
    
    naip_collection = ee.ImageCollection('USDA/NAIP/DOQQ') \
        .filterBounds(region) \
        .filterDate(f'{year_range[0]}-01-01', f'{year_range[1]}-12-31') \
        .select(['R', 'G', 'B', 'N'])
    
    # Get the most recent NAIP image
    naip_image = naip_collection.sort('system:time_start', False).first().clip(region)
    
    # Get image date
    image_date = ee.Date(naip_image.get('system:time_start')).format('YYYY-MM-dd')
    print(f"   Using NAIP image from: {image_date.getInfo()}")
    
    # Normalize NAIP data (0-255 to 0-1)
    naip_normalized = naip_image.divide(255.0).toFloat()
    
    # 2. USGS 3DEP 1m Elevation Data
    print("\n2. Acquiring USGS 3DEP 1m elevation data...")
    
    # Get 3DEP 1m DEM - mosaic all available tiles for the region
    dem_1m = ee.ImageCollection('USGS/3DEP/1m') \
        .filterBounds(region) \
        .mosaic() \
        .clip(region)
    
    elevation_source = dem_1m
    elevation_resolution = 1
    elevation_name = "USGS 3DEP 1m"
    
    print(f"   Using {elevation_name} ({elevation_resolution}m resolution)")
    
    # 3. Create enhanced 3D representation
    print("\n3. Creating enhanced 3D representation...")
    
    # Calculate terrain derivatives
    terrain = ee.Terrain.products(elevation_source)
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    hillshade = terrain.select('hillshade')
    
    # Create multi-band depth image
    depth_multispectral = elevation_source.addBands([slope, aspect, hillshade]).toFloat()
    
    # 4. Local Export Function
    def export_and_download(image, description, scale, bands=None):
        """Export to Earth Engine asset and download locally"""
        
        # Generate filename
        filename = f"{description}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # First export to Drive (required for download)
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder='naip_3dep_exports',
            fileNamePrefix=filename,
            scale=scale,
            region=region,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True},
            maxPixels=1e10
        )
        
        task.start()
        print(f"\n   Export started: {filename}")
        print(f"   Task ID: {task.id}")
        
        # Also prepare for local download using geemap
        local_path = os.path.join(output_dir, f"{filename}.tif")
        
        # Use geemap to download directly
        try:
            geemap.ee_export_image(
                image,
                filename=local_path,
                scale=scale,
                region=region,
                file_per_band=False
            )
            print(f"   Downloaded locally: {local_path}")
        except Exception as e:
            print(f"   Local download pending. Check Google Drive after export completes.")
            
        return task, local_path
    
    # 5. Export all data at 1m resolution
    print("\n4. Exporting data at 1m resolution...")
    
    # Export NAIP RGB at 1m resolution
    rgb_task, rgb_path = export_and_download(
        naip_normalized.select(['R', 'G', 'B']),
        f'{region_name}_NAIP_RGB_1m',
        1  # 1 meter resolution
    )
    
    # Export NAIP with NIR band (useful for vegetation analysis)
    rgbn_task, rgbn_path = export_and_download(
        naip_normalized,
        f'{region_name}_NAIP_RGBN_1m',
        1  # 1 meter resolution
    )
    
    # Export elevation at 1m resolution
    elevation_task, elevation_path = export_and_download(
        elevation_source,
        f'{region_name}_3DEP_ELEVATION_1m',
        1  # 1 meter resolution
    )
    
    # Export enhanced depth (multispectral) at 1m resolution
    depth_task, depth_path = export_and_download(
        depth_multispectral,
        f'{region_name}_3DEP_DEPTH_MULTI_1m',
        1  # 1 meter resolution
    )
    
    # 6. Create visualization
    print("\n5. Creating visualizations...")
    
    Map = geemap.Map()
    Map.centerObject(region, zoom=15)
    
    # Add layers
    Map.addLayer(naip_normalized.select(['R', 'G', 'B']), 
                 {'min': 0, 'max': 1}, 
                 'NAIP RGB (1m)')
    
    Map.addLayer(naip_normalized.select(['N', 'R', 'G']), 
                 {'min': 0, 'max': 1}, 
                 'NAIP False Color (NIR)')
    
    Map.addLayer(elevation_source, 
                 {'min': 0, 'max': 300, 'palette': ['blue', 'green', 'yellow', 'red', 'white']}, 
                 f'3DEP Elevation (1m)')
    
    Map.addLayer(hillshade, 
                 {'min': 0, 'max': 255}, 
                 'Hillshade')
    
    # Save map
    map_path = os.path.join(output_dir, f'{region_name}_map.html')
    Map.save(map_path)
    
    # 7. Generate metadata
    metadata_path = os.path.join(output_dir, f'{region_name}_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Region: {region_name}\n")
        f.write(f"Bounds: {bounds}\n")
        f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nData Sources:\n")
        f.write(f"- RGB: NAIP {image_date.getInfo()} (1m resolution)\n")
        f.write(f"- Elevation: USGS 3DEP (1m resolution)\n")
        f.write(f"\nExported Files (all at 1m resolution):\n")
        f.write(f"- RGB (3 bands): {rgb_path}\n")
        f.write(f"- RGBN (4 bands): {rgbn_path}\n")
        f.write(f"- Elevation: {elevation_path}\n")
        f.write(f"- Depth Multi (elevation, slope, aspect, hillshade): {depth_path}\n")
        f.write(f"\nVisualization:\n")
        f.write(f"- Interactive map: {map_path}\n")
    
    print(f"\n6. Export Summary:")
    print(f"   ✓ All data exported at 1m resolution")
    print(f"   - Metadata saved: {metadata_path}")
    print(f"   - Interactive map: {map_path}")
    print(f"   - Files will be available in: {output_dir}")
    print(f"   - Google Drive folder: 'naip_3dep_exports'")
    
    return {
        'tasks': {
            'rgb': rgb_task,
            'rgbn': rgbn_task,
            'elevation': elevation_task,
            'depth': depth_task
        },
        'paths': {
            'rgb': rgb_path,
            'rgbn': rgbn_path,
            'elevation': elevation_path,
            'depth': depth_path,
            'map': map_path,
            'metadata': metadata_path
        },
        'map': Map
    }


def wait_for_export(task_id, output_path):
    """
    Wait for export to complete and provide status updates
    """
    print(f"\nMonitoring export task...")
    
    while True:
        task_status = ee.data.getTaskStatus(task_id)[0]
        state = task_status['state']
        
        if state == 'COMPLETED':
            print(f"✓ Export completed successfully!")
            break
        elif state == 'FAILED':
            print(f"✗ Export failed: {task_status.get('error_message', 'Unknown error')}")
            return False
        else:
            print(f"   Export {state}... waiting 30 seconds")
            time.sleep(30)
    
    print(f"Download the file from Google Drive: 'naip_3dep_exports' folder")
    print(f"Then move it to: {output_path}")
    return True


# Example usage with areas known to have both NAIP 1m and USGS 3DEP 1m coverage
if __name__ == "__main__":
    
    # Example 1: Washington DC area (confirmed 3DEP 1m coverage)
    print("\n" + "="*60)
    print("Extracting data for Washington DC area...")
    print("="*60)
    dc_result = extract_naip_3dep_1m_data(
        region_name='washington_dc_1m',
        bounds=[-77.05, 38.88, -77.00, 38.93],  # Small area around DC
        year_range=[2020, 2023]
    )
    
    # Example 2: Yosemite Valley (confirmed 3DEP 1m coverage)
    print("\n\n" + "="*60)
    print("Extracting data for Yosemite Valley...")
    print("="*60)
    yosemite_result = extract_naip_3dep_1m_data(
        region_name='yosemite_valley_1m',
        bounds=[-119.65, 37.72, -119.55, 37.77],
        year_range=[2020, 2023]
    )
    
    # Example 3: Chicago area (confirmed 3DEP 1m coverage)
    print("\n\n" + "="*60)
    print("Extracting data for Chicago area...")
    print("="*60)
    chicago_result = extract_naip_3dep_1m_data(
        region_name='chicago_1m',
        bounds=[-87.65, 41.85, -87.60, 41.90],  # Downtown Chicago area
        year_range=[2020, 2023]
    )
    
    # Example 4: Test an area that might not have 3DEP 1m coverage
    print("\n\n" + "="*60)
    print("Testing area that might lack 3DEP 1m coverage...")
    print("="*60)
    rural_result = extract_naip_3dep_1m_data(
        region_name='rural_test',
        bounds=[-100.5, 35.5, -100.4, 35.6],  # Random rural area
        year_range=[2020, 2023]
    )
    
    print("\n" + "="*60)
    print("All processing initiated!")
    print("="*60)
    print("\nIMPORTANT NOTES:")
    print("- Only areas with BOTH NAIP 1m AND USGS 3DEP 1m coverage will be processed")
    print("- All outputs are at TRUE 1m resolution (no interpolation)")
    print("- Check https://apps.nationalmap.gov/lidar-explorer/ to verify 3DEP 1m coverage")
    print("- Major urban areas typically have the best coverage")