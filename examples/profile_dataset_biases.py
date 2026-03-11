"""
TEdit Dataset Bias Profiler (Model Bias Profiling)
自动读取配置，遍历属性空间，揭示模型在不同数据集上的趋势倾向。
"""

import sys
import os
import numpy as np
import torch
import yaml
import itertools

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series

# =====================================================================
# Configuration: Modify here for different datasets
# =====================================================================
# Option 1: Air dataset
DATASET_NAME = "air"
MODEL_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml")

# Option 2: Synthetic dataset
# DATASET_NAME = "synthetic"
# MODEL_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth")
# CONFIG_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml")

# Option 3: Motor dataset
# DATASET_NAME = "motor"
# MODEL_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/motor/pretrain_multi_weaver/0/ckpts/model_best.pth")
# CONFIG_PATH = os.path.join(PROJECT_ROOT, "TEdit-main/save/motor/pretrain_multi_weaver/0/model_configs.yaml")
# =====================================================================

print("=" * 70)
print(f"Dataset Bias Profiler for: [{DATASET_NAME.upper()}]")
print("=" * 70)

# 1. Dynamically read attribute dimensions
if not os.path.exists(CONFIG_PATH):
    print(f"ERROR: Config file not found: {CONFIG_PATH}")
    print("Please check the path and try again!")
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

num_attr_ops = config['attrs']['num_attr_ops']
n_attrs = len(num_attr_ops)
total_combinations = int(np.prod(num_attr_ops))

print(f"\n[Config] Attribute dimensions: {num_attr_ops}")
print(f"[Config] Number of attributes: {n_attrs}")
print(f"[Config] Total combinations: {total_combinations}")

# 2. Create base time series
print("\n[Step 1] Creating base time series...")
history_ts, _ = synthesize_time_series(
    length=120,
    trend_params={"slope": 0.0, "intercept": 20, "trend_type": "linear"},
    seasonality_params={"period": 24, "amplitude": 0, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 1.0},
    seed=42
)
forecast_ts = history_ts[:100].astype(np.float32)
print(f"  Base series: Length={len(forecast_ts)}, Mean={np.mean(forecast_ts):.2f}")

start, end = 0, 100

# 3. Load model
print("\n[Step 2] Loading model...")
tedit = get_tedit_instance(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    device="cuda:0",
    force_reload=True
)
print("  Model loaded successfully.")

# 4. Grid search through all combinations
print(f"\n[Step 3] Profiling {total_combinations} attribute combinations...")
print("-" * 70)

all_results = []
ranges = [range(dim) for dim in num_attr_ops]
combinations = list(itertools.product(*ranges))

for idx, tgt_attrs in enumerate(combinations):
    tgt_attrs = list(tgt_attrs)
    src_attrs = [0] * n_attrs
    
    # Lock seed for fair comparison
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        res = tedit.edit_region(
            forecast_ts, start, end,
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            n_samples=1,
            sampler="ddim"
        )
        
        curve = res[start:end]
        
        # Calculate core metrics
        slope = (curve[-1] - curve[0]) / len(curve)
        ptp = np.ptp(curve)  # Peak-to-peak range
        std = np.std(curve)
        mean_val = np.mean(curve)
        
        all_results.append({
            "attrs": tgt_attrs,
            "slope": slope,
            "range": ptp,
            "std": std,
            "mean": mean_val
        })
        
        # Print progress with slope indicator
        if slope > 0.05:
            indicator = "↗ UP"
        elif slope < -0.05:
            indicator = "↘ DOWN"
        elif ptp > 5:
            indicator = "〰 WAVE"
        else:
            indicator = "→ FLAT"
        
        print(f"  [{idx+1:3d}/{total_combinations}] Attrs {tgt_attrs} -> Slope: {slope:+.3f} | Std: {std:.2f} | {indicator}")
        
    except Exception as e:
        print(f"  [{idx+1:3d}/{total_combinations}] Attrs {tgt_attrs} -> ERROR: {e}")

# 5. Generate final report
print("\n" + "=" * 70)
print(f"[{DATASET_NAME.upper()}] DATASET PROFILING REPORT")
print("=" * 70)

if not all_results:
    print("No results to analyze!")
    sys.exit(1)

# Find flattest (lowest absolute slope)
all_results.sort(key=lambda x: abs(x["slope"]))
print("\n[Most Flat] - Best for SMOOTHING")
print("-" * 50)
for i, r in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']} | Slope: {r['slope']:+.4f} | Range: {r['range']:.2f} | Std: {r['std']:.2f}")

# Find strongest upward trend
all_results.sort(key=lambda x: x["slope"], reverse=True)
print("\n[Strongest Upward] - Best for TREND UP")
print("-" * 50)
for i, r in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']} | Slope: {r['slope']:+.4f} | Range: {r['range']:.2f} | Std: {r['std']:.2f}")

# Find strongest downward trend
all_results.sort(key=lambda x: x["slope"])
print("\n[Strongest Downward] - Best for TREND DOWN")
print("-" * 50)
for i, r in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']} | Slope: {r['slope']:+.4f} | Range: {r['range']:.2f} | Std: {r['std']:.2f}")

# Find most stable (lowest std)
all_results.sort(key=lambda x: x["std"])
print("\n[Most Stable] - Lowest variance")
print("-" * 50)
for i, r in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']} | Std: {r['std']:.2f} | Slope: {r['slope']:+.4f} | Range: {r['range']:.2f}")

# Find most volatile (highest range)
all_results.sort(key=lambda x: x["range"], reverse=True)
print("\n[Most Volatile] - Highest oscillation")
print("-" * 50)
for i, r in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {r['attrs']} | Range: {r['range']:.2f} | Std: {r['std']:.2f} | Slope: {r['slope']:+.4f}")

# Summary recommendations
print("\n" + "=" * 70)
print("RECOMMENDED ATTRIBUTE CODES")
print("=" * 70)

# Best for smoothing
all_results.sort(key=lambda x: abs(x["slope"]) + x["std"] * 0.1)
best_smooth = all_results[0]
print(f"\nBest for SMOOTHING: {best_smooth['attrs']}")
print(f"  Slope: {best_smooth['slope']:+.4f}, Std: {best_smooth['std']:.2f}")

# Best for upward trend
all_results.sort(key=lambda x: x["slope"], reverse=True)
best_up = all_results[0]
print(f"\nBest for UPWARD TREND: {best_up['attrs']}")
print(f"  Slope: {best_up['slope']:+.4f}")

# Best for downward trend
all_results.sort(key=lambda x: x["slope"])
best_down = all_results[0]
print(f"\nBest for DOWNWARD TREND: {best_down['attrs']}")
print(f"  Slope: {best_down['slope']:+.4f}")

print("\n" + "=" * 70)
print("Copy these attribute codes to use in your editing scripts!")
print("=" * 70)
