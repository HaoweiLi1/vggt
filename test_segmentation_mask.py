#!/usr/bin/env python3
"""
Test script to verify that segmentation masks are correctly applied
to remove sky regions from depth maps in the AerialMegaDepth dataset.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import os.path as osp

# Test paths
scene = "0001"
img_name = "0001_001.jpeg"

# Paths
depth_root = "/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped"
seg_root = "/home/haowei/Documents/vggt/training/dataset_aerialmd/cropped_seg"

depth_path = osp.join(depth_root, scene, img_name + '.exr')
seg_path = osp.join(seg_root, scene, img_name + '.png')

print("=" * 60)
print("Testing Segmentation Mask Application")
print("=" * 60)

# Load depth map
print(f"\n1. Loading depth map from: {depth_path}")
depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if depth_map is None:
    print("   ❌ Failed to load depth map!")
    exit(1)

if len(depth_map.shape) > 2:
    depth_map = depth_map[:, :, 0]

print(f"   ✓ Depth map shape: {depth_map.shape}")
print(f"   ✓ Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
print(f"   ✓ Valid depth pixels: {(depth_map > 0).sum()} / {depth_map.size}")

# Load segmentation mask
print(f"\n2. Loading segmentation mask from: {seg_path}")
if not osp.exists(seg_path):
    print(f"   ❌ Segmentation mask not found!")
    exit(1)

segmap = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
if segmap is None:
    print("   ❌ Failed to load segmentation mask!")
    exit(1)

print(f"   ✓ Segmentation mask shape: {segmap.shape}")
print(f"   ✓ Unique labels: {np.unique(segmap)}")
print(f"   ✓ Sky pixels (label=2): {(segmap == 2).sum()} / {segmap.size}")

# Apply mask (simulate what the dataloader does)
print(f"\n3. Applying segmentation mask to remove sky...")
depth_before = depth_map.copy()
sky_mask = (segmap == 2)
sky_depth_pixels = (sky_mask & (depth_map > 0)).sum()

print(f"   ✓ Sky pixels with valid depth (before): {sky_depth_pixels}")

# Apply the mask
depth_map[sky_mask] = 0

sky_depth_pixels_after = (sky_mask & (depth_map > 0)).sum()
print(f"   ✓ Sky pixels with valid depth (after): {sky_depth_pixels_after}")

# Statistics
print(f"\n4. Results:")
print(f"   ✓ Valid depth pixels before: {(depth_before > 0).sum()}")
print(f"   ✓ Valid depth pixels after:  {(depth_map > 0).sum()}")
print(f"   ✓ Pixels removed: {(depth_before > 0).sum() - (depth_map > 0).sum()}")

if sky_depth_pixels > 0 and sky_depth_pixels_after == 0:
    print(f"\n✅ SUCCESS: Segmentation mask correctly removes sky regions!")
elif sky_depth_pixels == 0:
    print(f"\n⚠️  WARNING: No sky pixels had valid depth in the first place")
else:
    print(f"\n❌ FAILED: Sky pixels still have depth values after masking!")

# Additional check: visualize percentage
total_pixels = depth_map.size
sky_percentage = (sky_mask.sum() / total_pixels) * 100
print(f"\n5. Sky coverage: {sky_percentage:.1f}% of image")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
