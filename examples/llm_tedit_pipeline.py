"""End-to-end pipeline: LLM decomposition + TEdit editing.

This script demonstrates the complete workflow:
1. User inputs natural language instruction
2. LLM decomposes the instruction into structured plan
3. LLM selects appropriate region
4. TEdit performs the editing
5. Results are visualized
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
import matplotlib.pyplot as plt
from agent.llm_instruction_decomposer import get_llm_decomposer
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series

print("=" * 70)
print("LLM + TEdit End-to-End Pipeline")
print("=" * 70)

# Step 1: Prepare data
print("\n[Step 1] Preparing time series data...")
history_ts, _ = synthesize_time_series(
    length=100,
    trend_params={"slope": 0.5, "intercept": 10, "trend_type": "linear"},
    seasonality_params={"period": 12, "amplitude": 5, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 0.5},
    seed=42
)
forecast_ts = np.mean(history_ts) + np.random.randn(50) * 0.5
print(f"  Forecast length: {len(forecast_ts)}")

# Step 2: Initialize components
print("\n[Step 2] Initializing LLM decomposer and TEdit...")
try:
    decomposer = get_llm_decomposer()
    print("  LLM decomposer: OK")
except Exception as e:
    print(f"  Error initializing LLM decomposer: {e}")
    sys.exit(1)

tedit = get_tedit_instance(
    model_path="TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth",
    config_path="TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml",
    device="cuda:0"
)
print("  TEdit model: OK")

# Step 3: Get user instruction and decompose
print("\n[Step 3] Processing user instruction...")

# Example instructions to test
instructions = [
    "Make the first half grow faster",
    "Reduce volatility in the middle section",
    "Increase the trend in the last 20 points",
]

results = []

for idx, instruction in enumerate(instructions):
    print(f"\n  Test {idx+1}: \"{instruction}\"")
    print("  " + "-" * 50)
    
    # Decompose instruction
    decomposition = decomposer.decompose(
        instruction,
        ts_length=len(forecast_ts),
        ts_values=forecast_ts
    )
    
    print(f"    Intent: {decomposition['intent']}")
    print(f"    Two-stage: {decomposition['is_two_stage']}")
    
    region = decomposition['region_selection']
    print(f"    Region: [{region['start_idx']}, {region['end_idx']})")
    print(f"    Reasoning: {region['reasoning']}")
    
    # Execute editing based on decomposition
    start_idx = region['start_idx']
    end_idx = region['end_idx']
    
    # Map intent to TEdit attributes
    src_attrs = np.array([0, 0], dtype=np.int64)
    
    if decomposition['intent'] == 'trend':
        tgt_attrs = np.array([1, 0], dtype=np.int64)
    elif decomposition['intent'] == 'seasonality':
        tgt_attrs = np.array([0, 1], dtype=np.int64)
    elif decomposition['intent'] == 'volatility':
        # For volatility reduction, we use seasonality attribute
        tgt_attrs = np.array([0, 1], dtype=np.int64)
    else:
        tgt_attrs = np.array([1, 1], dtype=np.int64)
    
    # Perform editing
    if start_idx == 0 and end_idx == len(forecast_ts):
        # Full series editing
        edited_result = tedit.edit_time_series(
            forecast_ts.astype(np.float32),
            src_attrs,
            tgt_attrs,
            n_samples=1,
            sampler="ddim"
        )
        edited = edited_result[0].flatten() if edited_result.ndim > 1 else edited_result.flatten()
    else:
        # Region editing
        edited = tedit.edit_region(
            forecast_ts.astype(np.float32),
            start_idx,
            end_idx,
            src_attrs=src_attrs.tolist(),
            tgt_attrs=tgt_attrs.tolist()
        )
    
    print(f"    Editing completed")
    print(f"    Mean change: {np.mean(edited) - np.mean(forecast_ts):.2f}")
    
    results.append({
        'instruction': instruction,
        'intent': decomposition['intent'],
        'region': (start_idx, end_idx),
        'edited': edited,
        'reasoning': region['reasoning']
    })

# Step 4: Visualize results
print("\n[Step 4] Generating visualization...")

n_results = len(results)
fig, axes = plt.subplots(n_results, 2, figsize=(14, 4*n_results))
if n_results == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('LLM-Guided TEdit Editing Results', fontsize=14, fontweight='bold')

for i, result in enumerate(results):
    start_idx, end_idx = result['region']
    
    # Left plot: Full view
    ax_left = axes[i, 0]
    ax_left.plot(forecast_ts, 'b-', label='Original', linewidth=2)
    ax_left.plot(result['edited'], 'r-', label='Edited', linewidth=2)
    if start_idx != 0 or end_idx != len(forecast_ts):
        ax_left.axvspan(start_idx, end_idx, alpha=0.2, color='yellow', label='Edited Region')
    ax_left.set_title(f'Test {i+1}: {result["instruction"][:40]}...')
    ax_left.set_xlabel('Time Step')
    ax_left.set_ylabel('Value')
    ax_left.legend(loc='best')
    ax_left.grid(True, alpha=0.3)
    
    # Right plot: Zoomed view of edited region
    ax_right = axes[i, 1]
    
    # Show context around edited region
    context_start = max(0, start_idx - 5)
    context_end = min(len(forecast_ts), end_idx + 5)
    
    x_range = range(context_start, context_end)
    ax_right.plot(x_range, forecast_ts[context_start:context_end], 'b-', 
                  label='Original', linewidth=2, marker='o', markersize=4)
    ax_right.plot(x_range, result['edited'][context_start:context_end], 'r-', 
                  label='Edited', linewidth=2, marker='s', markersize=4)
    ax_right.axvspan(start_idx, end_idx, alpha=0.3, color='yellow', label='Edited Region')
    ax_right.set_title(f'Zoomed View [{start_idx}, {end_idx})')
    ax_right.set_xlabel('Time Step')
    ax_right.set_ylabel('Value')
    ax_right.legend(loc='best')
    ax_right.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "llm_tedit_pipeline.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualization saved: {output_path}")

plt.close()

# Summary
print("\n" + "=" * 70)
print("Pipeline Execution Summary")
print("=" * 70)

for i, result in enumerate(results):
    print(f"\nTest {i+1}: {result['instruction']}")
    print(f"  Intent: {result['intent']}")
    print(f"  Region: [{result['region'][0]}, {result['region'][1]})")
    print(f"  Mean: {np.mean(forecast_ts):.2f} -> {np.mean(result['edited']):.2f}")
    print(f"  Reasoning: {result['reasoning'][:60]}...")

print("\n" + "=" * 70)
print("Key Features Demonstrated:")
print("  1. Natural language instruction understanding (LLM)")
print("  2. Intelligent region selection (LLM)")
print("  3. Semantic time series editing (TEdit)")
print("  4. Two-stage editing workflow")
print("=" * 70)
