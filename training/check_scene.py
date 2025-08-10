#!/usr/bin/env python3
"""
Script to verify scene-image mapping in train.npz
"""
import numpy as np
import os.path as osp

def check_scene_mapping(npz_path):
    print(f"\nChecking scene-image mapping in: {npz_path}\n")
    
    with np.load(npz_path, allow_pickle=True) as data:
        scenes = data['scenes']
        images = data['images']
        pairs = data['pairs']
        images_scene_name = data['images_scene_name']
        
        print(f"Scenes: {scenes}")
        print(f"\nChecking for mismatched scene assignments:\n")
        
        mismatches = []
        for i, img in enumerate(images[:100]):  # Check first 100
            if img is not None:
                img_str = str(img)
                scene_assigned = images_scene_name[i]
                
                # Extract the scene number from the image name
                if img_str.startswith('0000_'):
                    expected_scene = '0000'
                elif img_str.startswith('0001_'):
                    expected_scene = '0001'
                else:
                    expected_scene = scene_assigned  # Can't determine, assume correct
                
                if expected_scene != scene_assigned:
                    mismatches.append((i, img_str, scene_assigned, expected_scene))
                    print(f"  MISMATCH: Image {i}: {img_str} assigned to scene {scene_assigned}, expected {expected_scene}")
        
        if not mismatches:
            print("  No mismatches found in first 100 images")
        else:
            print(f"\n  Found {len(mismatches)} mismatches")
        
        # Check the specific pair that's failing
        print(f"\nChecking specific failing pairs:")
        for i, pair in enumerate(pairs[:20]):
            scene_id, im1_id, im2_id = pair[0], pair[1], pair[2]
            scene = scenes[scene_id]
            im1 = images[im1_id]
            im2 = images[im2_id]
            im1_scene = images_scene_name[im1_id]
            im2_scene = images_scene_name[im2_id]
            
            if '0001_409' in str(im1) or '0001_196' in str(im1) or '0001_409' in str(im2) or '0001_196' in str(im2):
                print(f"  Pair {i}: scene={scene}, im1={im1} (scene: {im1_scene}), im2={im2} (scene: {im2_scene})")

if __name__ == "__main__":
    npz_path = "/home/haowei/Documents/vggt/training/dataset_aerialmegadepth/train.npz"
    check_scene_mapping(npz_path)