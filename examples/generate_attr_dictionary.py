"""
TEdit Synthetic Attributes - Visual Dictionary Generator
通过 Ensemble 平均去除噪声，反向可视化 32 种属性组合的真实数学形态。
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series

print("=" * 70)
print("Generating Visual Dictionary for All 32 Attribute Combinations")
print("=" * 70)

# 1. Create base data
print("\n[Step 1] Creating base time series...")
history_ts, _ = synthesize_time_series(
    length=120,
    trend_params={"slope": 0.1, "intercept": 20, "trend_type": "linear"},
    seasonality_params={"period": 24, "amplitude": 5, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 1.0},
    seed=42
)
forecast_ts = history_ts[:100].astype(np.float32)
print(f"  Base series: Length={len(forecast_ts)}, Std={np.std(forecast_ts):.2f}")

# 2. Load model
print("\n[Step 2] Loading Synthetic TEdit model...")
model_path = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth")
config_path = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml")
tedit = get_tedit_instance(
    model_path=model_path,
    config_path=config_path,
    device="cuda:0"
)
tedit.set_edit_steps(20)  # Reduce steps for faster generation
print("  Model loaded. Edit steps set to 20.")

start, end = 20, 80  # Larger editing region to see patterns
n_ensemble = 8       # Average 8 samples to remove noise

# 3. Prepare figure (4x8 grid)
print("\n[Step 3] Rendering 32 combinations...")
print("-" * 70)

fig, axes = plt.subplots(4, 8, figsize=(24, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
axes = axes.flatten()

# Store results for summary
results_summary = []

idx = 0
for t_type in range(4):
    for t_dir in range(2):
        for s_cycle in range(4):
            tgt_attrs = [t_type, t_dir, s_cycle]
            print(f"  [{idx+1:2d}/32] Attrs {tgt_attrs}...", end="", flush=True)
            
            ensemble_samples = []
            for k in range(n_ensemble):
                # Inject randomness
                seed = np.random.randint(0, 100000)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                res = tedit.edit_region(
                    forecast_ts, start, end,
                    src_attrs=[0, 0, 0],
                    tgt_attrs=tgt_attrs,
                    n_samples=1,
                    sampler="ddim"
                )
                ensemble_samples.append(res[start:end].copy())
            
            # Average to extract skeleton
            avg_curve = np.mean(ensemble_samples, axis=0)
            
            # Calculate statistics
            slope = (avg_curve[-1] - avg_curve[0]) / len(avg_curve)
            ptp = np.ptp(avg_curve)  # Peak-to-peak range
            std = np.std(avg_curve)
            mean_val = np.mean(avg_curve)
            
            results_summary.append({
                'attrs': tgt_attrs,
                'slope': slope,
                'range': ptp,
                'std': std,
                'mean': mean_val
            })
            
            print(f" Slope={slope:+.2f}, Range={ptp:.1f}, Std={std:.2f}")
            
            # Plot
            ax = axes[idx]
            ax.plot(avg_curve, 'r-', linewidth=2)
            ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5)
            
            # Color code based on characteristics
            if abs(slope) < 0.02 and ptp < 3:
                title_color = 'green'  # Likely flat/smooth
                label = "FLAT?"
            elif abs(slope) > 0.1:
                title_color = 'blue'   # Strong trend
                label = "TREND"
            elif ptp > 5:
                title_color = 'orange' # Oscillating
                label = "WAVE"
            else:
                title_color = 'black'
                label = ""
            
            ax.set_title(f"Attr: {tgt_attrs}\nSlope: {slope:+.2f}, Range: {ptp:.1f}\n{label}", 
                        fontsize=9, color=title_color)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            
            idx += 1

# Save figure
output_dir = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "synthetic_visual_dictionary.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[Step 4] Visual Dictionary saved: {output_path}")

plt.close()

# Print summary
print("\n" + "=" * 70)
print("SUMMARY: Finding the Best Attributes")
print("=" * 70)

# Sort by different criteria
print("\n[Most Flat (Lowest |Slope|)]")
results_summary.sort(key=lambda x: abs(x['slope']))
for i, r in enumerate(results_summary[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']}, Slope: {r['slope']:+.3f}, Range: {r['range']:.2f}, Std: {r['std']:.2f}")

print("\n[Most Smooth (Lowest Range)]")
results_summary.sort(key=lambda x: x['range'])
for i, r in enumerate(results_summary[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']}, Range: {r['range']:.2f}, Slope: {r['slope']:+.3f}, Std: {r['std']:.2f}")

print("\n[Most Stable (Lowest Std)]")
results_summary.sort(key=lambda x: x['std'])
for i, r in enumerate(results_summary[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']}, Std: {r['std']:.2f}, Range: {r['range']:.2f}, Slope: {r['slope']:+.3f}")

print("\n[Strongest Upward Trend]")
results_summary.sort(key=lambda x: -x['slope'])
for i, r in enumerate(results_summary[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']}, Slope: {r['slope']:+.3f}, Range: {r['range']:.2f}")

print("\n[Strongest Downward Trend]")
results_summary.sort(key=lambda x: x['slope'])
for i, r in enumerate(results_summary[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']}, Slope: {r['slope']:+.3f}, Range: {r['range']:.2f}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
# Find best smoothing attributes
results_summary.sort(key=lambda x: abs(x['slope']) + x['range'] * 0.1)
best_smooth = results_summary[0]
print(f"Best for SMOOTHING: {best_smooth['attrs']}")
print(f"  - Slope: {best_smooth['slope']:+.3f}")
print(f"  - Range: {best_smooth['range']:.2f}")
print(f"  - Std: {best_smooth['std']:.2f}")

# Find best upward trend
results_summary.sort(key=lambda x: -x['slope'])
best_up = results_summary[0]
print(f"\nBest for UPWARD TREND: {best_up['attrs']}")
print(f"  - Slope: {best_up['slope']:+.3f}")

print("\n" + "=" * 70)
print("Please check the visual dictionary image to verify these findings!")
print("=" * 70)
