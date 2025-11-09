#!/usr/bin/env python3
"""
测试模型加载是否正确
"""
import torch
import sys
sys.path.insert(0, 'training')

from vggt.models.vggt import VGGT

def test_model_loading():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 测试 1: 加载 baseline 模型
    print("\n" + "="*60)
    print("Test 1: Loading Baseline Model")
    print("="*60)
    
    try:
        model = VGGT(
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_track=False,
            use_vit_features=True
        )
        
        checkpoint = torch.load("model/vggt_1B_commercial.pt", map_location=device)
        
        # Baseline 模型直接是 state_dict
        if isinstance(checkpoint, dict) and 'model' not in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Baseline model loaded")
        print(f"   Missing keys: {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # 测试 2: 加载 fine-tuned 模型
    print("\n" + "="*60)
    print("Test 2: Loading Fine-tuned Model")
    print("="*60)
    
    try:
        model = VGGT(
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_track=False,
            use_vit_features=True
        )
        
        checkpoint = torch.load("training/logs/single_pair_test/ckpts/checkpoint.pt", 
                               map_location=device)
        
        # Fine-tuned checkpoint 包含训练状态
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"   Loaded from epoch: {checkpoint.get('prev_epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        # 移除 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Fine-tuned model loaded")
        print(f"   Missing keys: {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
        
        if unexpected:
            print(f"\n   First 5 unexpected keys:")
            for key in list(unexpected)[:5]:
                print(f"     - {key}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_model_loading()
