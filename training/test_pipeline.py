# Create a test script to check the full pipeline:
# test_full_pipeline.py

import sys
sys.path.append('/home/haowei/Documents/vggt/training')

from hydra import initialize, compose
from hydra.utils import instantiate
import torch

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")

# Instantiate the full data pipeline
train_data_cfg = cfg.data.train
train_dataset = instantiate(train_data_cfg.dataset, common_config=train_data_cfg.common_config)

print(f"Dataset type: {type(train_dataset)}")
print(f"Dataset length: {len(train_dataset)}")

# Test getting a single sample
print("\n--- Testing single sample ---")
sample = train_dataset[(0, 3, 1.0)]
if sample is not None:
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"Images dtype: {sample['images'].dtype}")
else:
    print("ERROR: Sample is None!")

# Test the dataloader
print("\n--- Testing DataLoader ---")
from torch.utils.data import DataLoader

# Simple dataloader without the dynamic sampler
simple_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=None  # Use default collate
)

for i, batch in enumerate(simple_loader):
    if i >= 3:
        break
    print(f"Batch {i}:")
    if batch is None:
        print("  ERROR: Batch is None!")
    elif isinstance(batch, dict):
        print(f"  Images shape: {batch['images'].shape if 'images' in batch else 'NO IMAGES'}")
    else:
        print(f"  Batch type: {type(batch)}")