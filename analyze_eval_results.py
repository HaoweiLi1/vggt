#!/usr/bin/env python3
"""
Comprehensive analysis script for VGGT evaluation results.
Provides detailed statistics, visualizations, and insights.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load evaluation results from JSON files."""
    results_dir = Path(results_dir)
    
    with open(results_dir / 'finetuned_results.json', 'r') as f:
        finetuned = json.load(f)
    
    with open(results_dir / 'baseline_results.json', 'r') as f:
        baseline = json.load(f)
    
    return finetuned, baseline


def calculate_improvements(baseline_agg, finetuned_agg):
    """Calculate improvement percentages for all metrics."""
    improvements = {}
    
    # Depth metrics
    for metric in ['rmse', 'mae', 'abs_rel']:
        bl_val = baseline_agg['depth'][metric]['mean']
        ft_val = finetuned_agg['depth'][metric]['mean']
        improvement = ((bl_val - ft_val) / bl_val) * 100
        improvements[f'depth_{metric}'] = {
            'baseline': bl_val,
            'finetuned': ft_val,
            'improvement_pct': improvement,
            'absolute_diff': bl_val - ft_val
        }
    
    # Pose metrics
    for metric in ['rotation_error', 'translation_error']:
        bl_val = baseline_agg['pose'][metric]['mean']
        ft_val = finetuned_agg['pose'][metric]['mean']
        improvement = ((bl_val - ft_val) / bl_val) * 100
        improvements[f'pose_{metric}'] = {
            'baseline': bl_val,
            'finetuned': ft_val,
            'improvement_pct': improvement,
            'absolute_diff': bl_val - ft_val
        }
    
    return improvements


def print_detailed_analysis(split_name, finetuned, baseline):
    """Print detailed analysis of evaluation results."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS - {split_name.upper()} SPLIT")
    print(f"{'='*80}\n")
    
    # Per-scene analysis
    print("Per-Scene Statistics:")
    print("-" * 80)
    
    for scene_name in finetuned['per_scene'].keys():
        ft_scene = finetuned['per_scene'][scene_name]
        bl_scene = baseline['per_scene'][scene_name]
        
        print(f"\nScene: {scene_name}")
        print(f"  Number of frames: {ft_scene['num_frames']}")
        
        if ft_scene['depth_metrics']:
            ft_rmse = np.mean([m['rmse'] for m in ft_scene['depth_metrics']])
            bl_rmse = np.mean([m['rmse'] for m in bl_scene['depth_metrics']])
            improvement = ((bl_rmse - ft_rmse) / bl_rmse) * 100
            
            print(f"  Depth RMSE:")
            print(f"    Baseline:    {bl_rmse:8.2f}")
            print(f"    Fine-tuned:  {ft_rmse:8.2f}")
            print(f"    Improvement: {improvement:7.2f}%")
        
        if ft_scene['pose_metrics']:
            ft_rot = np.mean([m['rotation_error'] for m in ft_scene['pose_metrics']])
            bl_rot = np.mean([m['rotation_error'] for m in bl_scene['pose_metrics']])
            
            print(f"  Rotation Error:")
            print(f"    Baseline:    {bl_rot:8.2f}°")
            print(f"    Fine-tuned:  {ft_rot:8.2f}°")
    
    # Aggregated analysis
    print(f"\n{'='*80}")
    print("AGGREGATED STATISTICS")
    print(f"{'='*80}\n")
    
    improvements = calculate_improvements(baseline['aggregated'], finetuned['aggregated'])
    
    print("Depth Metrics:")
    print("-" * 80)
    for metric in ['rmse', 'mae', 'abs_rel']:
        key = f'depth_{metric}'
        imp = improvements[key]
        print(f"\n{metric.upper()}:")
        print(f"  Baseline:    {imp['baseline']:10.4f} ± {baseline['aggregated']['depth'][metric]['std']:.4f}")
        print(f"  Fine-tuned:  {imp['finetuned']:10.4f} ± {finetuned['aggregated']['depth'][metric]['std']:.4f}")
        print(f"  Improvement: {imp['improvement_pct']:9.2f}% ({imp['absolute_diff']:+.4f})")
    
    print(f"\n{'='*80}")
    print("Pose Metrics:")
    print("-" * 80)
    for metric in ['rotation_error', 'translation_error']:
        key = f'pose_{metric}'
        imp = improvements[key]
        metric_name = metric.replace('_', ' ').title()
        unit = '°' if 'rotation' in metric else ''
        print(f"\n{metric_name}:")
        print(f"  Baseline:    {imp['baseline']:10.4f}{unit} ± {baseline['aggregated']['pose'][metric]['std']:.4f}")
        print(f"  Fine-tuned:  {imp['finetuned']:10.4f}{unit} ± {finetuned['aggregated']['pose'][metric]['std']:.4f}")
        print(f"  Improvement: {imp['improvement_pct']:9.2f}% ({imp['absolute_diff']:+.4f})")
    
    return improvements


def create_detailed_plots(split_name, finetuned, baseline, output_path):
    """Create detailed visualization plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Extract data
    ft_agg = finetuned['aggregated']
    bl_agg = baseline['aggregated']
    
    # 1. Depth RMSE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['RMSE', 'MAE', 'Abs Rel']
    ft_vals = [ft_agg['depth']['rmse']['mean'], ft_agg['depth']['mae']['mean'], ft_agg['depth']['abs_rel']['mean']]
    bl_vals = [bl_agg['depth']['rmse']['mean'], bl_agg['depth']['mae']['mean'], bl_agg['depth']['abs_rel']['mean']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, bl_vals, width, label='Baseline', color='lightblue', alpha=0.8)
    ax1.bar(x + width/2, ft_vals, width, label='Fine-tuned', color='lightcoral', alpha=0.8)
    ax1.set_ylabel('Value')
    ax1.set_title('Depth Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pose metrics comparison
    ax2 = fig.add_subplot(gs[0, 1])
    pose_metrics = ['Rotation\nError (°)', 'Translation\nError']
    ft_pose = [ft_agg['pose']['rotation_error']['mean'], ft_agg['pose']['translation_error']['mean']]
    bl_pose = [bl_agg['pose']['rotation_error']['mean'], bl_agg['pose']['translation_error']['mean']]
    
    x = np.arange(len(pose_metrics))
    ax2.bar(x - width/2, bl_pose, width, label='Baseline', color='lightblue', alpha=0.8)
    ax2.bar(x + width/2, ft_pose, width, label='Fine-tuned', color='lightcoral', alpha=0.8)
    ax2.set_ylabel('Value')
    ax2.set_title('Pose Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pose_metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Improvement percentages
    ax3 = fig.add_subplot(gs[0, 2:])
    improvements = calculate_improvements(bl_agg, ft_agg)
    imp_names = list(improvements.keys())
    imp_values = [improvements[k]['improvement_pct'] for k in imp_names]
    
    colors = ['green' if v > 0 else 'red' for v in imp_values]
    bars = ax3.barh(imp_names, imp_values, color=colors, alpha=0.6)
    ax3.set_xlabel('Improvement (%)')
    ax3.set_title('Performance Improvement by Metric')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, imp_values)):
        ax3.text(val, i, f' {val:.2f}%', va='center', fontsize=9)
    
    # 4-6. Distribution plots for depth metrics
    depth_metrics_list = ['rmse', 'mae', 'abs_rel']
    for idx, metric in enumerate(depth_metrics_list):
        ax = fig.add_subplot(gs[1, idx])
        
        # Collect per-frame values
        ft_values = []
        bl_values = []
        for scene_data in finetuned['per_scene'].values():
            ft_values.extend([m[metric] for m in scene_data['depth_metrics']])
        for scene_data in baseline['per_scene'].values():
            bl_values.extend([m[metric] for m in scene_data['depth_metrics']])
        
        ax.hist([bl_values, ft_values], bins=30, label=['Baseline', 'Fine-tuned'], 
                color=['lightblue', 'lightcoral'], alpha=0.6)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {metric.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Per-scene RMSE comparison
    ax7 = fig.add_subplot(gs[1, 3])
    scene_names = list(finetuned['per_scene'].keys())
    ft_scene_rmse = []
    bl_scene_rmse = []
    
    for scene in scene_names:
        ft_metrics = finetuned['per_scene'][scene]['depth_metrics']
        bl_metrics = baseline['per_scene'][scene]['depth_metrics']
        
        if ft_metrics:
            ft_scene_rmse.append(np.mean([m['rmse'] for m in ft_metrics]))
            bl_scene_rmse.append(np.mean([m['rmse'] for m in bl_metrics]))
    
    x = np.arange(len(scene_names))
    ax7.bar(x - width/2, bl_scene_rmse, width, label='Baseline', color='lightblue', alpha=0.8)
    ax7.bar(x + width/2, ft_scene_rmse, width, label='Fine-tuned', color='lightcoral', alpha=0.8)
    ax7.set_ylabel('RMSE')
    ax7.set_title('Per-Scene Depth RMSE')
    ax7.set_xticks(x)
    ax7.set_xticklabels(scene_names, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8-9. Pose distribution plots
    pose_metrics_list = ['rotation_error', 'translation_error']
    for idx, metric in enumerate(pose_metrics_list):
        ax = fig.add_subplot(gs[2, idx])
        
        ft_values = []
        bl_values = []
        for scene_data in finetuned['per_scene'].values():
            ft_values.extend([m[metric] for m in scene_data['pose_metrics']])
        for scene_data in baseline['per_scene'].values():
            bl_values.extend([m[metric] for m in scene_data['pose_metrics']])
        
        ax.hist([bl_values, ft_values], bins=30, label=['Baseline', 'Fine-tuned'],
                color=['lightblue', 'lightcoral'], alpha=0.6)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 10. Summary statistics table
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    summary_text = f"EVALUATION SUMMARY - {split_name.upper()} SPLIT\n\n"
    summary_text += f"Total Scenes: {len(finetuned['per_scene'])}\n"
    
    total_frames = sum(s['num_frames'] for s in finetuned['per_scene'].values())
    summary_text += f"Total Frames: {total_frames}\n\n"
    
    summary_text += "KEY IMPROVEMENTS:\n"
    improvements = calculate_improvements(bl_agg, ft_agg)
    summary_text += f"• Depth RMSE: {improvements['depth_rmse']['improvement_pct']:.2f}%\n"
    summary_text += f"• Depth MAE: {improvements['depth_mae']['improvement_pct']:.2f}%\n"
    summary_text += f"• Abs Rel: {improvements['depth_abs_rel']['improvement_pct']:.2f}%\n"
    summary_text += f"• Rotation: {improvements['pose_rotation_error']['improvement_pct']:.2f}%\n"
    summary_text += f"• Translation: {improvements['pose_translation_error']['improvement_pct']:.2f}%\n"
    
    ax10.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
              family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Comprehensive Evaluation Analysis - {split_name.upper()} Split', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nDetailed visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze VGGT evaluation results')
    parser.add_argument('--results_dir', type=str, default='eval_results',
                        help='Directory containing evaluation results')
    parser.add_argument('--splits', type=str, nargs='+', default=['test', 'train'],
                        help='Splits to analyze')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    for split in args.splits:
        split_dir = results_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        print(f"\nAnalyzing {split} split...")
        finetuned, baseline = load_results(split_dir)
        
        # Print detailed analysis
        improvements = print_detailed_analysis(split, finetuned, baseline)
        
        # Create detailed plots
        output_plot = split_dir / 'detailed_analysis.png'
        create_detailed_plots(split, finetuned, baseline, output_plot)
        
        # Save improvements to JSON
        output_json = split_dir / 'improvements.json'
        with open(output_json, 'w') as f:
            json.dump(improvements, f, indent=2)
        print(f"Improvements saved to: {output_json}")


if __name__ == '__main__':
    main()
