"""
LLM-Guided TEdit Pipeline V2: Enhanced Demonstration
More typical experiment demo: includes high-noise data smoothing, hybrid control, and detailed comparison.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(".env")

from agent.llm_instruction_decomposer import get_llm_decomposer
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series

print("=" * 80)
print("LLM + TEdit Pipeline V2: Advanced Editing & Hybrid Control")
print("=" * 80)

# -----------------------------------------------------------------------------
# Step 1: Prepare High-Contrast Data
# -----------------------------------------------------------------------------
print("\n[Step 1] Preparing High-Contrast Data...")
history_ts, _ = synthesize_time_series(
    length=120,
    trend_params={"slope": 0.1, "intercept": 20, "trend_type": "linear"},
    seasonality_params={"period": 24, "amplitude": 8, "seasonality_type": "sine"},
    noise_params={"noise_type": "gaussian", "std": 1.5},
    seed=123
)
forecast_ts = history_ts[:100]
print(f"  Series created: Length={len(forecast_ts)}, Std={np.std(forecast_ts):.2f} (High Noise)")

# -----------------------------------------------------------------------------
# Step 2: Initialize Components
# -----------------------------------------------------------------------------
print("\n[Step 2] Initializing Components...")
try:
    decomposer = get_llm_decomposer()
    print("  LLM Decomposer: Ready")
except Exception as e:
    print(f"  [Error] LLM init failed: {e}")
    sys.exit(1)

tedit = get_tedit_instance(
    model_path="TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth",
    config_path="TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml",
    device="cuda:0"
)
print("  TEdit Model: Ready")

# -----------------------------------------------------------------------------
# Step 3: Define Challenging Test Cases
# -----------------------------------------------------------------------------
test_cases = [
    {
        "desc": "Standard Trend Editing",
        "instruction": "Make the last 30 points drop significantly",
        "mode": "standard"
    },
    {
        "desc": "Volatility Suppression",
        "instruction": "Smooth out the fluctuations in the middle section (indices 30 to 70)",
        "mode": "standard"
    },
    {
        "desc": "Hybrid Control (Math + AI)",
        "instruction": "Increase the trend in the first half",
        "mode": "hybrid"
    }
]

results = []

for i, case in enumerate(test_cases):
    print(f"\n>>> Test {i+1}: {case['desc']}")
    print(f"    Instruction: \"{case['instruction']}\"")
    
    # 1. LLM Decomposition
    try:
        decomposition = decomposer.decompose(
            case['instruction'],
            ts_length=len(forecast_ts),
            ts_values=forecast_ts
        )
    except Exception as e:
        print(f"    LLM Parsing failed, using mock for demo. Error: {e}")
        if i == 0:
            decomposition = {'intent': 'trend', 'region_selection': {'start_idx': 70, 'end_idx': 100, 'reasoning': 'Last 30 points'}}
        elif i == 1:
            decomposition = {'intent': 'volatility', 'region_selection': {'start_idx': 30, 'end_idx': 70, 'reasoning': 'Middle section'}}
        else:
            decomposition = {'intent': 'trend', 'region_selection': {'start_idx': 0, 'end_idx': 50, 'reasoning': 'First half'}}

    start = decomposition['region_selection']['start_idx']
    end = decomposition['region_selection']['end_idx']
    intent = decomposition['intent']
    print(f"    [Plan] Region: {start}-{end} | Intent: {intent}")

    # 2. Prepare editing data
    current_ts = forecast_ts.copy().astype(np.float32)
    intermediate_ts = None
    
    # === HYBRID MODE LOGIC ===
    if case['mode'] == "hybrid":
        print("    [Hybrid] Applying coarse linear guidance first...")
        slope = np.linspace(0, 10, end - start)
        current_ts[start:end] += slope
        intermediate_ts = current_ts.copy()
        print("    [Hybrid] Linear trend applied, now refining with TEdit...")
    
    # 3. Attribute Mapping and Editing Strategy
    src_attrs = np.array([0, 0], dtype=np.int64)
    
    if intent in ['volatility', 'smoothing']:
        # Use traditional smoothing for volatility reduction
        # TEdit attributes don't directly control volatility
        print("    [Config] Using Moving Average Smoothing (TEdit doesn't support direct volatility control)")
        window_size = 15  # Larger window for more aggressive smoothing
        region = current_ts[start:end].copy()
        # Use 'same' mode with padding for edge handling
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(region, kernel, mode='same')
        edited_full = current_ts.copy()
        edited_full[start:end] = smoothed
        print(f"    [Config] Applied {window_size}-point moving average")
    elif intent == 'trend':
        tgt_attrs = np.array([1, 0], dtype=np.int64)
        print("    [Config] Target: Trend Enabled")
        edited_full = tedit.edit_region(
            current_ts,
            start,
            end,
            src_attrs=src_attrs.tolist(),
            tgt_attrs=tgt_attrs.tolist(),
            n_samples=1,
            sampler="ddim"
        )
    else:
        tgt_attrs = np.array([1, 1], dtype=np.int64)
        print("    [Config] Target: Trend + Seasonality")
        edited_full = tedit.edit_region(
            current_ts,
            start,
            end,
            src_attrs=src_attrs.tolist(),
            tgt_attrs=tgt_attrs.tolist(),
            n_samples=1,
            sampler="ddim"
        )
    
    # 5. Record results
    results.append({
        "case": case,
        "region": (start, end),
        "original": forecast_ts.copy(),
        "intermediate": intermediate_ts,
        "edited": edited_full,
        "intent": intent,
        "mode": case['mode']
    })
    
    # Print statistics
    orig_region = forecast_ts[start:end]
    edit_region = edited_full[start:end]
    print(f"    [Result] Original std: {np.std(orig_region):.2f} -> Edited std: {np.std(edit_region):.2f}")
    print(f"    [Result] Original mean: {np.mean(orig_region):.2f} -> Edited mean: {np.mean(edit_region):.2f}")

# -----------------------------------------------------------------------------
# Step 4: Enhanced Visualization
# -----------------------------------------------------------------------------
print("\n[Step 4] Generating Enhanced Visualization...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle('LLM + TEdit Pipeline V2: Advanced Editing & Hybrid Control', fontsize=14, fontweight='bold')

for i, result in enumerate(results):
    start, end = result['region']
    mode = result['mode']
    
    # Row 1: Full view
    ax1 = fig.add_subplot(3, 2, i*2 + 1)
    ax1.plot(result['original'], 'b-', label='Original', linewidth=1.5, alpha=0.7)
    
    if result['intermediate'] is not None:
        ax1.plot(result['intermediate'], 'g--', label='Intermediate (Linear)', linewidth=1.5, alpha=0.7)
    
    ax1.plot(result['edited'], 'r-', label='Final Edited', linewidth=2)
    ax1.axvspan(start, end, alpha=0.2, color='yellow', label='Edited Region')
    ax1.set_title(f"Test {i+1}: {result['case']['desc']}")
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Row 2: Zoomed comparison
    ax2 = fig.add_subplot(3, 2, i*2 + 2)
    context_start = max(0, start - 10)
    context_end = min(len(forecast_ts), end + 10)
    x_range = range(context_start, context_end)
    
    ax2.plot(x_range, result['original'][context_start:context_end], 
             'b-', label='Original', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax2.plot(x_range, result['edited'][context_start:context_end], 
             'r-', label='Edited', linewidth=2, marker='s', markersize=4)
    ax2.axvspan(start, end, alpha=0.3, color='yellow', label='Edited Region')
    ax2.set_title(f"Zoomed View: Region [{start}, {end})")
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = "results/figures/examples"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "llm_tedit_pipeline_v2.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Visualization saved: {output_path}")

plt.close()

# -----------------------------------------------------------------------------
# Step 5: Summary Report
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Pipeline V2 Execution Summary")
print("=" * 80)

for i, result in enumerate(results):
    start, end = result['region']
    print(f"\nTest {i+1}: {result['case']['desc']}")
    print(f"  Mode: {result['mode']}")
    print(f"  Instruction: {result['case']['instruction']}")
    print(f"  Region: [{start}, {end})")
    print(f"  Intent: {result['intent']}")
    
    orig = result['original'][start:end]
    edit = result['edited'][start:end]
    print(f"  Statistics:")
    print(f"    Mean: {np.mean(orig):.2f} -> {np.mean(edit):.2f} (Δ={np.mean(edit)-np.mean(orig):.2f})")
    print(f"    Std:  {np.std(orig):.2f} -> {np.std(edit):.2f} (Δ={np.std(edit)-np.std(orig):.2f})")

print("\n" + "=" * 80)
print("Key Features in V2:")
print("  1. High-noise data for visible editing effects")
print("  2. Hybrid control mode (Math + AI) for mean drift correction")
print("  3. Traditional smoothing for volatility reduction (TEdit limitation)")
print("  4. LLM-guided region selection and intent detection")
print("=" * 80)
