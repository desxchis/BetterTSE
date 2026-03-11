"""
LLM-TEdit Pipeline V8 - Soft-Boundary Temporal Injection Validation

This script validates the Soft-Boundary Temporal Injection innovation:
- Compares hard boundary (legacy) vs soft boundary (new) methods
- Visualizes the elimination of "cliff effect" at region boundaries
- Demonstrates training-free attention region injection

Key Innovation:
- Traditional: result[start:end] = edited_region (hard splicing)
- New method: latent blending with soft mask at each diffusion step

Configuration:
- Uses unified config.py for API credentials and model paths
- DeepSeek API with model: deepseek-chat
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_environment, get_api_config, get_model_config, get_openai_client
from openai import OpenAI
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series
from tool.ts_editors import (
    hybrid_up, hybrid_up_soft,
    hybrid_down, hybrid_down_soft,
    ensemble_smooth, ensemble_smooth_soft
)

setup_environment()

print("=" * 80)
print("Pipeline V8: Soft-Boundary Temporal Injection Validation")
print("=" * 80)

# =====================================================================
# [Step 1] 初始化环境
# =====================================================================
print("\n[Step 1] Initializing Environment...")

api_config = get_api_config()
print(f"[Config] API: {api_config['base_url']}")
print(f"[Config] Model: {api_config['model_name']}")

client = get_openai_client()
MODEL_NAME = api_config["model_name"]

model_config = get_model_config("synthetic")
tedit_instance = get_tedit_instance(
    model_config["model_path"],
    model_config["config_path"],
    device="cuda:0"
)
print(f"[Init] TEdit Model loaded from: {model_config['model_path']}")

ts, components = synthesize_time_series(length=120, noise_params={"std": 2.5}, seed=42)
base_ts = ts[:100].astype(np.float32)

# =====================================================================
# [Step 2] 定义测试场景
# =====================================================================
test_configs = [
    {
        "name": "Hybrid Up (Surge)",
        "start_idx": 20,
        "end_idx": 60,
        "math_shift": 15.0,
        "hard_fn": hybrid_up,
        "soft_fn": hybrid_up_soft,
    },
    {
        "name": "Hybrid Down (Drop)",
        "start_idx": 30,
        "end_idx": 70,
        "math_shift": -15.0,
        "hard_fn": hybrid_down,
        "soft_fn": hybrid_down_soft,
    },
]

# =====================================================================
# [Step 3] 执行对比测试
# =====================================================================
results = []

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"[Test] {config['name']}")
    print(f"  Region: [{config['start_idx']}, {config['end_idx']})")
    print(f"  Math Shift: {config['math_shift']}")
    print(f"{'='*60}")
    
    print("  Running hard boundary method...")
    edited_hard = config['hard_fn'](
        ts=base_ts.copy(),
        start_idx=config['start_idx'],
        end_idx=config['end_idx'],
        math_shift=config['math_shift'],
        tedit=tedit_instance,
    )
    
    print("  Running soft boundary method...")
    edited_soft = config['soft_fn'](
        ts=base_ts.copy(),
        start_idx=config['start_idx'],
        end_idx=config['end_idx'],
        math_shift=config['math_shift'],
        tedit=tedit_instance,
        smooth_radius=5.0,
    )
    
    start, end = config['start_idx'], config['end_idx']
    
    def compute_gradient(ts):
        return np.abs(np.diff(ts))
    
    grad_hard = compute_gradient(edited_hard)
    grad_soft = compute_gradient(edited_soft)
    
    boundary_region = 5
    left_start = max(0, start - boundary_region)
    left_end = min(len(grad_hard), start + boundary_region)
    right_start = max(0, end - boundary_region)
    right_end = min(len(grad_hard), end + boundary_region)
    
    if left_end > left_start:
        cliff_hard_left = np.max(grad_hard[left_start:left_end])
        cliff_soft_left = np.max(grad_soft[left_start:left_end])
    else:
        cliff_hard_left = 0.0
        cliff_soft_left = 0.0
    
    if right_end > right_start:
        cliff_hard_right = np.max(grad_hard[right_start:right_end])
        cliff_soft_right = np.max(grad_soft[right_start:right_end])
    else:
        cliff_hard_right = 0.0
        cliff_soft_right = 0.0
    
    print(f"\n  [Boundary Analysis]")
    print(f"    Left boundary cliff (hard): {cliff_hard_left:.4f}")
    print(f"    Left boundary cliff (soft): {cliff_soft_left:.4f}")
    if cliff_hard_left > 0:
        print(f"    Reduction: {(1 - cliff_soft_left/cliff_hard_left)*100:.1f}%")
        reduction_left = (1 - cliff_soft_left/cliff_hard_left)*100
    else:
        print(f"    Reduction: N/A")
        reduction_left = 0.0
    print(f"    Right boundary cliff (hard): {cliff_hard_right:.4f}")
    print(f"    Right boundary cliff (soft): {cliff_soft_right:.4f}")
    if cliff_hard_right > 0:
        print(f"    Reduction: {(1 - cliff_soft_right/cliff_hard_right)*100:.1f}%")
        reduction_right = (1 - cliff_soft_right/cliff_hard_right)*100
    else:
        print(f"    Reduction: N/A")
        reduction_right = 0.0
    
    results.append({
        "name": config['name'],
        "start_idx": start,
        "end_idx": end,
        "edited_hard": edited_hard,
        "edited_soft": edited_soft,
        "grad_hard": grad_hard,
        "grad_soft": grad_soft,
        "cliff_reduction_left": reduction_left,
        "cliff_reduction_right": reduction_right,
    })

# =====================================================================
# [Step 4] 可视化对比
# =====================================================================
print(f"\n{'='*60}")
print("[Step 4] Generating Visualization...")
print(f"{'='*60}")

n_results = len(results)
fig, axes = plt.subplots(n_results, 3, figsize=(18, 5 * n_results))

if n_results == 1:
    axes = axes.reshape(1, -1)

for i, result in enumerate(results):
    start, end = result['start_idx'], result['end_idx']
    
    ax1 = axes[i, 0]
    ax1.plot(base_ts, 'k--', alpha=0.5, linewidth=1.5, label='Original')
    ax1.plot(result['edited_hard'], 'r-', linewidth=2, label='Hard Boundary')
    ax1.plot(result['edited_soft'], 'b-', linewidth=2, label='Soft Boundary')
    ax1.axvspan(start, end, alpha=0.15, color='yellow', label='Edit Region')
    ax1.axvline(start, color='red', linestyle=':', alpha=0.7)
    ax1.axvline(end, color='red', linestyle=':', alpha=0.7)
    ax1.set_title(f"{result['name']}\nTime Series Comparison", fontsize=12)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    ax2 = axes[i, 1]
    ax2.plot(result['grad_hard'], 'r-', linewidth=1.5, label='Hard Boundary', alpha=0.8)
    ax2.plot(result['grad_soft'], 'b-', linewidth=1.5, label='Soft Boundary', alpha=0.8)
    ax2.axvspan(start, end, alpha=0.15, color='yellow')
    ax2.axvline(start, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(end, color='red', linestyle=':', alpha=0.7)
    ax2.set_title(f"Gradient Analysis\n|dx/dt| Comparison", fontsize=12)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('|Gradient|')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 99)
    
    ax3 = axes[i, 2]
    boundary_width = 10
    left_start = max(0, start - boundary_width)
    left_end = min(len(result['grad_hard']), start + boundary_width)
    right_start = max(0, end - boundary_width)
    right_end = min(len(result['grad_hard']), end + boundary_width)
    
    x_left = np.arange(left_start, left_end)
    
    if left_end > left_start:
        ax3.plot(x_left, result['grad_hard'][left_start:left_end], 'r-o', markersize=4, 
                 label=f'Hard (Left)', alpha=0.8)
        ax3.plot(x_left, result['grad_soft'][left_start:left_end], 'b-o', markersize=4, 
                 label=f'Soft (Left)', alpha=0.8)
    ax3.axvline(start, color='red', linestyle='--', alpha=0.7, label='Boundary')
    
    ax3.set_title(f"Boundary Zoom-In\nCliff Reduction: L={result['cliff_reduction_left']:.1f}%, R={result['cliff_reduction_right']:.1f}%", 
                  fontsize=11)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('|Gradient|')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/llm_tedit_pipeline_v8_soft_boundary.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Visualization Saved to: {output_path}")

# =====================================================================
# [Step 5] 打印汇总报告
# =====================================================================
print("\n" + "=" * 80)
print("📊 Soft-Boundary Temporal Injection Summary")
print("=" * 80)

print("\n🔬 Key Innovation:")
print("  - Traditional: result[start:end] = edited_region (hard splicing)")
print("  - New method: latent blending with soft mask at each diffusion step")
print("  - Eliminates 'cliff effect' at region boundaries")

print("\n📈 Results Summary:")
for result in results:
    print(f"\n  [{result['name']}]")
    print(f"    Left boundary cliff reduction: {result['cliff_reduction_left']:.1f}%")
    print(f"    Right boundary cliff reduction: {result['cliff_reduction_right']:.1f}%")

avg_reduction = np.mean([r['cliff_reduction_left'] + r['cliff_reduction_right'] for r in results]) / 2
print(f"\n  Average cliff reduction: {avg_reduction:.1f}%")

print("\n✅ Pipeline V8 Complete!")
print("=" * 80)
