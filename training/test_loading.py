#!/usr/bin/env python3
# test_loading.py
import sys
import os
sys.path.append('/home/haowei/Documents/vggt/training')

import numpy as np
from data.datasets.megadepth_aerial import MegaDepthAerialDataset
from dataclasses import dataclass

# Create a mock common_conf
@dataclass
class CommonConf:
    img_size: int = 384
    patch_size: int = 14
    debug: bool = True
    training: bool = True
    get_nearby: bool = False
    load_depth: bool = True
    inside_random: bool = False  # IMPORTANT: Set to False for testing
    allow_duplicate_img: bool = True
    load_track: bool = False
    track_num: int = 512
    rescale: bool = True
    rescale_aug: bool = False
    landscape_check: bool = False
    class augs:
        scales: list = None
        def __init__(self):
            self.scales = [0.8, 1.2]

common_conf = CommonConf()
common_conf.augs = CommonConf.augs()

# Create dataset
dataset = MegaDepthAerialDataset(
    common_conf=common_conf,
    split="train",
    ROOT="/home/haowei/Documents/vggt/training/dataset_aerialmegadepth",
    split_file="train.npz",
    len_train=10000,
)

print(f"Dataset initialized with {len(dataset.pairs)} pairs")
print(f"Dataset length: {len(dataset)}")
print(f"Valid scenes: {dataset.valid_scenes}")

# Test loading specific samples
for i in range(10):
    try:
        print(f"\n--- Testing sample {i} ---")
        batch = dataset.get_data(seq_index=i, img_per_seq=4, aspect_ratio=1.0)
        
        if batch is None:
            print(f"Sample {i}: Returned None")
        else:
            print(f"Sample {i}: Successfully loaded")
            print(f"  - seq_name: {batch['seq_name']}")
            print(f"  - num_images: {len(batch['images'])}")
            print(f"  - image_shapes: {[img.shape for img in batch['images'][:2]]}")
            print(f"  - has_depths: {batch['depths'] is not None and len(batch['depths']) > 0}")
            print(f"  - frame_num: {batch['frame_num']}")
    except Exception as e:
        print(f"Sample {i}: ERROR - {e}")
        import traceback
        traceback.print_exc()

# Test with __getitem__ if it exists
print("\n--- Testing __getitem__ ---")
try:
    sample = dataset[0]  # Simple index
    print(f"Simple index [0]: {type(sample)}")
    
    sample = dataset[(5, 3, 0.8)]  # Tuple index
    print(f"Tuple index [(5, 3, 0.8)]: num_images={len(sample['images'])}")
except Exception as e:
    print(f"__getitem__ test failed: {e}")