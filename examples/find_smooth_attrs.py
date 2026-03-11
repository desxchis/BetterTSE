"""
TEdit Synthetic Attributes Cracker
Grid Search through 32 attribute combinations to find the "absolute smoothing" password.
"""

import sys
import os
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables with absolute path
from dotenv import load_dotenv
env_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_path)

# Fallback: set environment variables directly if not loaded
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-30cb957d01eb473aac1cb85fdee68352"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series

print("=" * 70)
print("TEdit Synthetic Attributes Grid Search - Finding the Smoothing Password")
print("=" * 70)

# 1. Create test data with high-frequency noise
print("\n[Step 1] Creating high-noise test data...")
history_ts, _ = synthesize_time_series(
    length=120,
    trend_params={"slope": 0.1, "intercept": 20, "trend_type": "linear"},
    seasonality_params={"period": 24, "amplitude": 8, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 2.0},
    seed=42
)
forecast_ts = history_ts[:100].astype(np.float32)
print(f"  Test data created: Length={len(forecast_ts)}, Std={np.std(forecast_ts):.2f}")

# 2. Load Synthetic model
print("\n[Step 2] Loading Synthetic TEdit model...")
model_path = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth")
config_path = os.path.join(PROJECT_ROOT, "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml")
tedit = get_tedit_instance(
    model_path=model_path,
    config_path=config_path,
    device="cuda:0"
)
# Reduce steps for faster search
tedit.set_edit_steps(20)
print("  Model loaded. Edit steps set to 20 for faster search.")

start, end = 30, 70
best_std = float('inf')
best_attrs = None
all_results = []

print("\n[Step 3] Grid Search: Testing 32 combinations (4 x 2 x 4)...")
print("-" * 70)

# 3. Iterate through all combinations
for t_type in range(4):
    for t_dir in range(2):
        for s_cycle in range(4):
            tgt_attrs = [t_type, t_dir, s_cycle]
            
            # Generate with this attribute combination
            edited_ts = tedit.edit_region(
                forecast_ts, start, end,
                src_attrs=[0, 0, 0],
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler="ddim"
            )
            
            # Calculate std of edited region (lower = smoother)
            region_std = np.std(edited_ts[start:end])
            region_mean = np.mean(edited_ts[start:end])
            all_results.append((region_std, region_mean, tgt_attrs))
            
            # Print current result
            print(f"  Attrs {tgt_attrs} -> Std: {region_std:.3f}, Mean: {region_mean:.2f}")
            
            if region_std < best_std:
                best_std = region_std
                best_attrs = tgt_attrs

print("\n" + "=" * 70)
print("SEARCH COMPLETE!")
print("=" * 70)

# Sort by std
all_results.sort(key=lambda x: x[0])

print("\n[Top 5 Smoothest Combinations]")
for i, (std, mean, attrs) in enumerate(all_results[:5]):
    print(f"  {i+1}. Attrs: {attrs}, Std: {std:.3f}, Mean: {mean:.2f}")

print("\n[Top 5 Most Volatile Combinations]")
for i, (std, mean, attrs) in enumerate(all_results[-5:][::-1]):
    print(f"  {i+1}. Attrs: {attrs}, Std: {std:.3f}, Mean: {mean:.2f}")

print("\n" + "=" * 70)
print("RESULT: The Smoothing Password")
print("=" * 70)
print(f"  Best smoothing attributes: {best_attrs}")
print(f"  Lowest std achieved: {best_std:.3f}")
print(f"  Original std: {np.std(forecast_ts[start:end]):.3f}")
print(f"  Reduction: {(1 - best_std/np.std(forecast_ts[start:end]))*100:.1f}%")
print("=" * 70)

print("\n[Usage in V3 Script]")
print(f"  For Test 2 (Deep Smoothing), use:")
print(f"    tgt_attrs = {best_attrs}")
