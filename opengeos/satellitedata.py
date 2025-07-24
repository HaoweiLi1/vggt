import ee
import geemap
import os
from datetime import datetime

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='map-api-465501')

def extract_satellite_3d_data(region_name, bounds, date_range=['2023-01-01', '2023-12-31'], output_dir='./output'):
    """
    Extract high-precision RGB and depth data for 3D modeling
    
    Args:
        region_name: Name for the export files
        bounds: [west, south, east, north]
        date_range: [start_date, end_date] in 'YYYY-MM-DD' format
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    region = ee.Geometry.Rectangle(bounds)
    
    # High-resolution RGB from Sentinel-2 (10m)
    def mask_clouds(image):
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(mask).divide(10000)
    
    rgb = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(region) \
        .filterDate(date_range[0], date_range[1]) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .map(mask_clouds) \
        .select(['B4', 'B3', 'B2']) \
        .median() \
        .clip(region)
    
    # High-precision elevation data - Copernicus DEM GLO-30 (30m)
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30') \
        .select('DEM') \
        .mosaic() \
        .clip(region)
    
    # Enhanced 3D representation with terrain derivatives
    terrain = ee.Terrain.products(dem)
    depth_3d = dem.addBands([
        terrain.select('slope'),
        terrain.select('aspect'), 
        terrain.select('hillshade')
    ]).float()
    
    # Create visualization map and save as PNG
    Map = geemap.Map()
    Map.centerObject(region, zoom=12)
    Map.addLayer(rgb, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'RGB')
    Map.addLayer(dem, {'min': 0, 'max': 2000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}, 'Elevation')
    
    # Save map visualization as HTML and PNG
    map_html = os.path.join(output_dir, f'{region_name}_map.html')
    Map.save(map_html)
    
    # Export map as PNG image
    map_png = os.path.join(output_dir, f'{region_name}_visualization.png')
    Map.to_image(filename=map_png, monitor=1)
    
    # Export high-precision GeoTIFF files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    rgb_export = ee.batch.Export.image.toDrive(
        image=rgb.toFloat(),
        description=f'{region_name}_RGB_{timestamp}',
        folder='satellite_3d_exports',
        fileNamePrefix=f'{region_name}_RGB_{timestamp}',
        scale=10,
        region=region,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True},
        maxPixels=1e10
    )
    
    depth_export = ee.batch.Export.image.toDrive(
        image=depth_3d,
        description=f'{region_name}_DEPTH_{timestamp}',
        folder='satellite_3d_exports',
        fileNamePrefix=f'{region_name}_DEPTH_{timestamp}',
        scale=10,  # Resample to match RGB resolution
        region=region,
        fileFormat='GeoTIFF',
        formatOptions={'cloudOptimized': True},
        maxPixels=1e10
    )
    
    # Start exports
    rgb_export.start()
    depth_export.start()
    
    print(f"\nExport Status:")
    print(f"RGB GeoTIFF: {rgb_export.id}")
    print(f"Depth GeoTIFF: {depth_export.id}")
    print(f"\nLocal visualizations saved:")
    print(f"- Interactive map: {map_html}")
    print(f"- PNG screenshot: {map_png}")
    print(f"\nHigh-precision GeoTIFF files will be exported to Google Drive folder: 'satellite_3d_exports'")
    print(f"Monitor progress at: https://code.earthengine.google.com/tasks")
    
    return {
        'exports': {
            'rgb_geotiff': rgb_export.id,
            'depth_geotiff': depth_export.id
        },
        'map': Map,
        'files': {
            'map_html': map_html,
            'map_png': map_png
        }
    }


# Extract high-precision satellite data
if __name__ == "__main__":
    # Example: San Francisco area
    result = extract_satellite_3d_data(
        region_name='san_francisco_3d',
        bounds=[-122.5, 37.7, -122.4, 37.8],
        date_range=['2023-06-01', '2023-09-30']
    )