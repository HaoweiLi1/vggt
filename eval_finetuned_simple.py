#!/usr/bin/env python3
"""
Simplified evaluation script that evaluates models sequentially to save memory.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import OpenEXR
import Imath

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_depth_map(depth_path):
    exr_file = OpenEXR.InputFile(str(depth_path))
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('Y', FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape(height, width)
    
    return depth


def load_gt_pose(pose_path):
    data = np.load(pose_path)
    return data['cam2world']


def align_depth_scale(pred_depth, gt_depth):
    valid_mask = gt_depth > 0
    if valid_mask.sum() == 0:
        return pred_depth
    
    valid_pred = pred_depth[valid_mask]
    valid_gt = gt_depth[valid_mask]
    
    gt_min, gt_max = np.min(valid_gt), np.max(valid_gt)
    pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)
    
    if pred_max - pred_min < 1e-8:
        return np.ones_like(pred_depth) * gt_min
    
    normalized_pred = (pred_depth - pred_min) / (pred_max - pred_min)
    return normalized_pred * (gt_max - gt_min) + gt_min


def calculate_depth_metrics(pred_depth, gt_depth):
    valid_mask = gt_depth > 0
    if valid_mask.sum() == 0:
        return None
    
    aligned_pred = align_depth_scale(pred_depth, gt_depth)
    valid_pred = aligned_pred[valid_mask]
    valid_gt = gt_depth[valid_mask]
    
    rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    mae = np.mean(np.abs(valid_pred - valid_gt))
    abs_rel = np.mean(np.abs(valid_pred - valid_gt) / np.maximum(valid_gt, 1e-8))
    
    return {'rmse': float(rmse), 'mae': float(mae), 'abs_rel': float(abs_rel)}


def calculate_pose_metrics(pred_pose, gt_pose):
    pred_R, pred_t = pred_pose[:3, :3], pred_pose[:3, 3]
    gt_R, gt_t = gt_pose[:3, :3], gt_pose[:3, 3]
    
    R_diff = pred_R @ gt_R.T
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1, 3)
    rotation_error = np.degrees(np.arccos((trace - 1) / 2))
    
    translation_error = np.linalg.norm(pred_t - gt_t)
    
    return {
        'rotation_error': float(rotation_error),
        'translation_error': float(translation_error)
    }


def predict_scene(model, image_paths, device, dtype):
    images = load_and_preprocess_images(image_paths).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
            
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
            
            depth_maps, _ = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
    
    depth_maps = depth_maps.squeeze(0).cpu().numpy()
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    
    torch.cuda.empty_cache()
    
    return depth_maps, extrinsic


def evaluate_scene(scene_dir, model, device, dtype, batch_size=20):
    scene_dir = Path(scene_dir)
    
    # Get all image files (*.jpg)
    image_files = sorted(scene_dir.glob('*.jpg'))
    
    depth_metrics = []
    pose_metrics = []
    
    # Process in batches to avoid OOM
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        pred_depths, pred_poses = predict_scene(
            model, 
            [str(f) for f in batch_files], 
            device, 
            dtype
        )
        
        for i, img_file in enumerate(batch_files):
            # Get corresponding depth and pose files
            base_name = img_file.stem  # e.g., "0000_000.jpeg"
            gt_depth_path = scene_dir / f'{base_name}.exr'
            gt_pose_path = scene_dir / f'{base_name}.npz'
            
            if gt_depth_path.exists():
                gt_depth = load_depth_map(gt_depth_path)
                pred_depth = pred_depths[i]
                
                # Squeeze any extra dimensions
                if pred_depth.ndim > 2:
                    pred_depth = pred_depth.squeeze()
                
                if pred_depth.shape != gt_depth.shape:
                    import cv2
                    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                metrics = calculate_depth_metrics(pred_depth, gt_depth)
                if metrics:
                    depth_metrics.append(metrics)
            
            if gt_pose_path.exists():
                gt_pose = load_gt_pose(gt_pose_path)
                pred_pose = pred_poses[i]
                
                metrics = calculate_pose_metrics(pred_pose, gt_pose)
                pose_metrics.append(metrics)
    
    return {
        'depth_metrics': depth_metrics,
        'pose_metrics': pose_metrics,
        'num_frames': len(image_files)
    }


def aggregate_metrics(scene_results):
    all_depth_rmse = []
    all_depth_mae = []
    all_depth_abs_rel = []
    all_rotation_error = []
    all_translation_error = []
    
    for result in scene_results.values():
        for m in result['depth_metrics']:
            all_depth_rmse.append(m['rmse'])
            all_depth_mae.append(m['mae'])
            all_depth_abs_rel.append(m['abs_rel'])
        
        for m in result['pose_metrics']:
            all_rotation_error.append(m['rotation_error'])
            all_translation_error.append(m['translation_error'])
    
    return {
        'depth': {
            'rmse': {'mean': np.mean(all_depth_rmse), 'std': np.std(all_depth_rmse), 'values': all_depth_rmse},
            'mae': {'mean': np.mean(all_depth_mae), 'std': np.std(all_depth_mae), 'values': all_depth_mae},
            'abs_rel': {'mean': np.mean(all_depth_abs_rel), 'std': np.std(all_depth_abs_rel), 'values': all_depth_abs_rel}
        },
        'pose': {
            'rotation_error': {'mean': np.mean(all_rotation_error), 'std': np.std(all_rotation_error), 'values': all_rotation_error},
            'translation_error': {'mean': np.mean(all_translation_error), 'std': np.std(all_translation_error), 'values': all_translation_error}
        }
    }


def visualize_comparison(finetuned_metrics, baseline_metrics, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    depth_metrics = ['rmse', 'mae', 'abs_rel']
    pose_metrics = ['rotation_error', 'translation_error']
    
    for idx, metric in enumerate(depth_metrics):
        ax = axes[0, idx]
        data = [
            baseline_metrics['depth'][metric]['values'],
            finetuned_metrics['depth'][metric]['values']
        ]
        bp = ax.boxplot(data, tick_labels=['Baseline', 'Fine-tuned'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Depth {metric.upper()}')
        ax.grid(True, alpha=0.3)
    
    for idx, metric in enumerate(pose_metrics):
        ax = axes[1, idx]
        data = [
            baseline_metrics['pose'][metric]['values'],
            finetuned_metrics['pose'][metric]['values']
        ]
        bp = ax.boxplot(data, tick_labels=['Baseline', 'Fine-tuned'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ylabel = 'Rotation Error (deg)' if metric == 'rotation_error' else 'Translation Error'
        ax.set_ylabel(ylabel)
        ax.set_title(f'Pose {ylabel}')
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    
    summary_text = f"Baseline vs Fine-tuned Model Comparison\n\n"
    summary_text += f"Depth RMSE: {baseline_metrics['depth']['rmse']['mean']:.4f} → {finetuned_metrics['depth']['rmse']['mean']:.4f}\n"
    summary_text += f"Depth MAE: {baseline_metrics['depth']['mae']['mean']:.4f} → {finetuned_metrics['depth']['mae']['mean']:.4f}\n"
    summary_text += f"Rotation Error: {baseline_metrics['pose']['rotation_error']['mean']:.4f}° → {finetuned_metrics['pose']['rotation_error']['mean']:.4f}°\n"
    summary_text += f"Translation Error: {baseline_metrics['pose']['translation_error']['mean']:.4f} → {finetuned_metrics['pose']['translation_error']['mean']:.4f}"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuned_model', type=str, required=True)
    parser.add_argument('--baseline_model', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'])
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(args.data_root)
    
    # Map split names to actual scene directories
    split_mapping = {
        'train': ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009',],
        'test': ['0015']
    }
    
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Evaluating {split} split")
        print(f"{'='*60}")
        
        scene_names = split_mapping.get(split, [])
        scene_dirs = [data_root / name for name in scene_names if (data_root / name).exists()]
        
        # Evaluate fine-tuned model
        print("\nLoading fine-tuned model...")
        finetuned_model = VGGT()
        checkpoint = torch.load(args.finetuned_model, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        finetuned_model.load_state_dict(state_dict, strict=False)
        del checkpoint, state_dict
        finetuned_model = finetuned_model.to(device)
        finetuned_model.eval()
        
        finetuned_results = {}
        for scene_dir in tqdm(scene_dirs, desc=f"Fine-tuned model on {split}"):
            scene_name = scene_dir.name
            finetuned_results[scene_name] = evaluate_scene(scene_dir, finetuned_model, device, dtype)
        
        # Free memory
        del finetuned_model
        torch.cuda.empty_cache()
        
        # Evaluate baseline model
        print("\nLoading baseline model...")
        baseline_model = VGGT()
        baseline_checkpoint = torch.load(args.baseline_model, map_location='cpu')
        if 'model' in baseline_checkpoint:
            baseline_state_dict = baseline_checkpoint['model']
        else:
            baseline_state_dict = baseline_checkpoint
        baseline_model.load_state_dict(baseline_state_dict, strict=False)
        del baseline_checkpoint, baseline_state_dict
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        
        baseline_results = {}
        for scene_dir in tqdm(scene_dirs, desc=f"Baseline model on {split}"):
            scene_name = scene_dir.name
            baseline_results[scene_name] = evaluate_scene(scene_dir, baseline_model, device, dtype)
        
        # Free memory
        del baseline_model
        torch.cuda.empty_cache()
        
        # Aggregate and save results
        finetuned_agg = aggregate_metrics(finetuned_results)
        baseline_agg = aggregate_metrics(baseline_results)
        
        split_output = output_dir / split
        split_output.mkdir(exist_ok=True)
        
        with open(split_output / 'finetuned_results.json', 'w') as f:
            json.dump({
                'per_scene': finetuned_results,
                'aggregated': {k: {m: {s: v for s, v in mv.items() if s != 'values'} 
                                   for m, mv in cat.items()} 
                               for k, cat in finetuned_agg.items()}
            }, f, indent=2)
        
        with open(split_output / 'baseline_results.json', 'w') as f:
            json.dump({
                'per_scene': baseline_results,
                'aggregated': {k: {m: {s: v for s, v in mv.items() if s != 'values'} 
                                   for m, mv in cat.items()} 
                               for k, cat in baseline_agg.items()}
            }, f, indent=2)
        
        visualize_comparison(
            finetuned_agg,
            baseline_agg,
            split_output / 'comparison.png'
        )
        
        print(f"\n{split.upper()} Split Results:")
        print(f"Fine-tuned - Depth RMSE: {finetuned_agg['depth']['rmse']['mean']:.4f} ± {finetuned_agg['depth']['rmse']['std']:.4f}")
        print(f"Baseline   - Depth RMSE: {baseline_agg['depth']['rmse']['mean']:.4f} ± {baseline_agg['depth']['rmse']['std']:.4f}")
        print(f"Fine-tuned - Rotation Error: {finetuned_agg['pose']['rotation_error']['mean']:.4f}° ± {finetuned_agg['pose']['rotation_error']['std']:.4f}°")
        print(f"Baseline   - Rotation Error: {baseline_agg['pose']['rotation_error']['mean']:.4f}° ± {baseline_agg['pose']['rotation_error']['std']:.4f}°")
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
