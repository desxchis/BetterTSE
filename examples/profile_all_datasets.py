"""
TEdit Multi-Dataset Bias Profiler
批量测试 Air, Synthetic, Motor 三个数据集的属性偏见。
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
# Dataset configurations
# =====================================================================
DATASETS = {
    "air": {
        "model_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth"),
        "config_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml"),
    },
    "synthetic": {
        "model_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"),
        "config_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"),
    },
    "motor": {
        "model_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/motor/pretrain_multi_weaver/0/ckpts/model_best.pth"),
        "config_path": os.path.join(PROJECT_ROOT, "TEdit-main/save/motor/pretrain_multi_weaver/0/model_configs.yaml"),
    },
}

# Store results for all datasets
all_dataset_results = {}

print("=" * 80)
print("TEdit Multi-Dataset Bias Profiler")
print("Testing: Air, Synthetic, Motor")
print("=" * 80)

# Create base time series (same for all datasets)
print("\n[Preparing] Creating base time series...")
history_ts, _ = synthesize_time_series(
    length=120,
    trend_params={"slope": 0.0, "intercept": 20, "trend_type": "linear"},
    seasonality_params={"period": 24, "amplitude": 0, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 1.0},
    seed=42
)
forecast_ts = history_ts[:100].astype(np.float32)
start, end = 0, 100
print(f"  Base series: Length={len(forecast_ts)}, Mean={np.mean(forecast_ts):.2f}")

# =====================================================================
# Test each dataset
# =====================================================================
for dataset_name, dataset_config in DATASETS.items():
    print("\n" + "=" * 80)
    print(f"[{dataset_name.upper()}] Starting Profile...")
    print("=" * 80)
    
    model_path = dataset_config["model_path"]
    config_path = dataset_config["config_path"]
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"  [SKIP] Config not found: {config_path}")
        continue
    
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        continue
    
    # Read attribute dimensions
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    num_attr_ops = config['attrs']['num_attr_ops']
    n_attrs = len(num_attr_ops)
    total_combinations = int(np.prod(num_attr_ops))
    
    print(f"  Attribute dimensions: {num_attr_ops}")
    print(f"  Total combinations: {total_combinations}")
    
    # Load model
    print(f"  Loading model...")
    try:
        tedit = get_tedit_instance(
            model_path=model_path,
            config_path=config_path,
            device="cuda:0",
            force_reload=True
        )
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        continue
    
    # Grid search
    print(f"  Profiling {total_combinations} combinations...")
    print("-" * 80)
    
    results = []
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
            slope = (curve[-1] - curve[0]) / len(curve)
            ptp = np.ptp(curve)
            std = np.std(curve)
            
            results.append({
                "attrs": tgt_attrs,
                "slope": slope,
                "range": ptp,
                "std": std
            })
            
            # Progress indicator
            if slope > 0.05:
                indicator = "↗"
            elif slope < -0.05:
                indicator = "↘"
            else:
                indicator = "→"
            
            print(f"    [{idx+1:3d}/{total_combinations}] {tgt_attrs} -> Slope: {slope:+.3f} {indicator}")
            
        except Exception as e:
            print(f"    [{idx+1:3d}/{total_combinations}] {tgt_attrs} -> ERROR: {str(e)[:30]}")
    
    all_dataset_results[dataset_name] = {
        "num_attr_ops": num_attr_ops,
        "results": results
    }
    
    # Print summary for this dataset
    if results:
        print(f"\n  [{dataset_name.upper()}] Summary:")
        
        # Best for smoothing
        results.sort(key=lambda x: abs(x["slope"]) + x["std"] * 0.1)
        best_smooth = results[0]
        print(f"    Best SMOOTHING: {best_smooth['attrs']} (slope: {best_smooth['slope']:+.3f})")
        
        # Best for upward
        results.sort(key=lambda x: x["slope"], reverse=True)
        best_up = results[0]
        print(f"    Best UPWARD:    {best_up['attrs']} (slope: {best_up['slope']:+.3f})")
        
        # Best for downward
        results.sort(key=lambda x: x["slope"])
        best_down = results[0]
        print(f"    Best DOWNWARD:  {best_down['attrs']} (slope: {best_down['slope']:+.3f})")

# =====================================================================
# Final Comparison Report
# =====================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON REPORT")
print("=" * 80)

for dataset_name, data in all_dataset_results.items():
    if not data["results"]:
        continue
    
    print(f"\n[{dataset_name.upper()}] (attrs: {data['num_attr_ops']})")
    print("-" * 60)
    
    results = data["results"]
    
    # Smoothing
    results.sort(key=lambda x: abs(x["slope"]) + x["std"] * 0.1)
    print(f"  SMOOTHING:  {results[0]['attrs']} | Slope: {results[0]['slope']:+.4f}")
    
    # Upward
    results.sort(key=lambda x: x["slope"], reverse=True)
    print(f"  UPWARD:     {results[0]['attrs']} | Slope: {results[0]['slope']:+.4f}")
    
    # Downward
    results.sort(key=lambda x: x["slope"])
    print(f"  DOWNWARD:   {results[0]['attrs']} | Slope: {results[0]['slope']:+.4f}")

print("\n" + "=" * 80)
print("ATTRIBUTE CODE SUMMARY")
print("=" * 80)
print("\nCopy these codes to your V4 script:\n")

for dataset_name, data in all_dataset_results.items():
    if not data["results"]:
        continue
    
    results = data["results"]
    
    results.sort(key=lambda x: abs(x["slope"]) + x["std"] * 0.1)
    smooth = results[0]['attrs']
    
    results.sort(key=lambda x: x["slope"], reverse=True)
    up = results[0]['attrs']
    
    results.sort(key=lambda x: x["slope"])
    down = results[0]['attrs']
    
    print(f"# {dataset_name.upper()}")
    print(f"SMOOTHING = {smooth}")
    print(f"UPWARD    = {up}")
    print(f"DOWNWARD  = {down}")
    print()

print("=" * 80)
print("Done!")
