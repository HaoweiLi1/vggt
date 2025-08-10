#!/usr/bin/env python3
"""
Script to inspect the contents of train.npz and understand the data format
"""
import numpy as np
import os
import os.path as osp

def inspect_npz(npz_path, output_dir):
    """Inspect the contents of train.npz file"""
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {npz_path}")
    print(f"{'='*60}\n")
    
    # Load the NPZ file
    with np.load(npz_path, allow_pickle=True) as data:
        # Print all keys
        print("Keys in NPZ file:")
        for key in data.keys():
            print(f"  - {key}")
        
        # Load arrays
        scenes = data['scenes']
        images = data['images']
        pairs = data['pairs']
        images_scene_name = data.get('images_scene_name', None)
        
        print(f"\n{'='*60}")
        print("Data shapes and types:")
        print(f"  - scenes: shape={scenes.shape}, dtype={scenes.dtype}")
        print(f"  - images: shape={images.shape}, dtype={images.dtype}")
        print(f"  - pairs: shape={pairs.shape}, dtype={pairs.dtype}")
        if images_scene_name is not None:
            print(f"  - images_scene_name: shape={images_scene_name.shape}, dtype={images_scene_name.dtype}")
        
        print(f"\n{'='*60}")
        print("Scene information:")
        print(f"  Total scenes: {len(scenes)}")
        for i, scene in enumerate(scenes[:5]):  # Show first 5
            print(f"  Scene {i}: {scene}")
        
        print(f"\n{'='*60}")
        print("Image name samples (first 20):")
        for i, img in enumerate(images[:20]):
            if img is not None:
                scene_name = images_scene_name[i] if images_scene_name is not None else "Unknown"
                print(f"  Image {i:4d}: scene={scene_name}, name={img}")
        
        print(f"\n{'='*60}")
        print("Image name patterns analysis:")
        
        # Analyze image name patterns
        jpeg_count = 0
        jpg_count = 0
        jpg_jpg_count = 0
        other_count = 0
        
        for img in images:
            if img is not None:
                img_str = str(img)
                if img_str.endswith('.jpeg'):
                    jpeg_count += 1
                elif img_str.endswith('.jpg.jpg'):
                    jpg_jpg_count += 1
                elif img_str.endswith('.jpg'):
                    jpg_count += 1
                else:
                    other_count += 1
        
        print(f"  - Files ending with .jpeg: {jpeg_count}")
        print(f"  - Files ending with .jpg: {jpg_count}")
        print(f"  - Files ending with .jpg.jpg: {jpg_jpg_count}")
        print(f"  - Other patterns: {other_count}")
        
        print(f"\n{'='*60}")
        print("Checking actual files on disk (first 10 images):")
        
        # Check what actually exists on disk
        for i, img in enumerate(images[:10]):
            if img is not None:
                scene_name = images_scene_name[i] if images_scene_name is not None else scenes[0]
                scene_path = osp.join(output_dir, str(scene_name))
                
                print(f"\n  Checking image {i}: {img}")
                print(f"    Scene: {scene_name}")
                
                # Try different possible paths
                possible_paths = [
                    osp.join(scene_path, str(img)),
                    osp.join(scene_path, str(img) + '.jpg'),
                    osp.join(scene_path, str(img) + '.jpeg'),
                ]
                
                for path in possible_paths:
                    exists = osp.exists(path)
                    print(f"    {path}: {'EXISTS' if exists else 'NOT FOUND'}")
                
                # Also check for .exr and .npz files
                base_names = [
                    str(img),
                    str(img)[:-4] if str(img).endswith('.jpg') else str(img),
                    str(img)[:-5] if str(img).endswith('.jpeg') else str(img),
                ]
                
                for base in base_names:
                    exr_path = osp.join(scene_path, base + '.exr')
                    npz_path = osp.join(scene_path, base + '.npz')
                    if osp.exists(exr_path):
                        print(f"    Found EXR: {exr_path}")
                    if osp.exists(npz_path):
                        print(f"    Found NPZ: {npz_path}")
        
        print(f"\n{'='*60}")
        print("Pair information (first 5 pairs):")
        for i, pair in enumerate(pairs[:5]):
            scene_id, im1_id, im2_id = pair[0], pair[1], pair[2]
            score = pair[3] if len(pair) > 3 else "N/A"
            print(f"  Pair {i}: scene_id={scene_id}, im1={images[im1_id]}, im2={images[im2_id]}, score={score}")

if __name__ == "__main__":
    # Adjust these paths to match your setup
    npz_path = "/home/haowei/Documents/vggt/training/dataset_aerialmegadepth/train.npz"
    output_dir = "/home/haowei/Documents/vggt/training/dataset_aerialmegadepth"
    
    inspect_npz(npz_path, output_dir)