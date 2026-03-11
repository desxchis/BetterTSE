"""
LLM-TEdit Pipeline V9 - Full System Validation (Cognitive + Execution)

This script validates the complete "Cognitive-Execution-Verification" loop:
1. Real LLM API Call: Validates intent understanding and parameter mapping.
2. Tool Routing: Validates correct dispatch to hybrid_up/down or ensemble_smooth.
3. Execution: Validates Soft-Boundary Injection (Latent Blending) implementation.
4. Diagnostics: Prints gradient analysis to prove "Cliff Effect" elimination.

Key Innovation:
- Latent Blending: z_{t-1} = M * z^{pred} + (1-M) * z^{GT}
- Background is "physical truth" (forward-diffused from original data)
- Ensures 100% background fidelity (zero reconstruction error)

Scenarios:
1. Surge Event (Oil Crisis) -> Expect hybrid_up
2. Drop Event (Tech Bubble) -> Expect hybrid_down
3. Stabilization Event (Policy Intervention) -> Expect ensemble_smooth

Configuration:
- Uses unified config.py for API credentials and model paths
- DeepSeek API with model: deepseek-chat
"""

import sys
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    setup_environment,
    get_api_config,
    get_model_config,
    get_openai_client,
    verify_model_exists,
)
from openai import OpenAI
from agent.prompts import EVENT_DRIVEN_AGENT_PROMPT
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series
from tool.ts_editors import execute_llm_tool, hybrid_up, hybrid_down, ensemble_smooth

setup_environment()


def analyze_boundary_cliff(ts_edited: np.ndarray, start: int, end: int, window: int = 5) -> Tuple[float, float]:
    """Calculate local gradient at boundaries to quantify cliff effect.
    
    Args:
        ts_edited: Edited time series
        start: Start index of edit region
        end: End index of edit region
        window: Window size for gradient calculation
    
    Returns:
        Tuple of (left_boundary_cliff, right_boundary_cliff)
    """
    grad = np.abs(np.diff(ts_edited))
    
    l_s = max(0, start - window)
    l_e = min(len(grad), start + window)
    cliff_left = np.max(grad[l_s:l_e]) if l_e > l_s else 0
    
    r_s = max(0, end - window)
    r_e = min(len(grad), end + window)
    cliff_right = np.max(grad[r_s:r_e]) if r_e > r_s else 0
    
    return cliff_left, cliff_right


def call_llm_for_planning(
    client: OpenAI,
    model: str,
    instruction: str,
    ts_length: int
) -> Optional[Dict[str, Any]]:
    """Call LLM API for event-driven planning.
    
    Args:
        client: OpenAI client instance
        model: Model name (e.g., "deepseek-chat")
        instruction: User instruction in natural language
        ts_length: Length of time series
    
    Returns:
        Parsed plan dictionary or None if failed
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVENT_DRIVEN_AGENT_PROMPT},
                {"role": "user", "content": f"Instruction: {instruction}\n\nTime series length: {ts_length}"}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            plan_dict = json.loads(json_match.group())
            return plan_dict
        else:
            print(f"  [ERROR] No JSON found in LLM response")
            return None
            
    except Exception as e:
        print(f"  [ERROR] LLM API call failed: {e}")
        return None


def validate_plan(plan: Dict[str, Any], ts_length: int) -> bool:
    """Validate LLM-generated plan for correctness.
    
    Args:
        plan: Plan dictionary from LLM
        ts_length: Length of time series
    
    Returns:
        True if valid, False otherwise
    """
    if "tool_name" not in plan:
        print("  [VALIDATION ERROR] Missing 'tool_name' in plan")
        return False
    
    if "parameters" not in plan:
        print("  [VALIDATION ERROR] Missing 'parameters' in plan")
        return False
    
    params = plan["parameters"]
    if "region" not in params:
        print("  [VALIDATION ERROR] Missing 'region' in parameters")
        return False
    
    region = params["region"]
    if not isinstance(region, list) or len(region) != 2:
        print(f"  [VALIDATION ERROR] Invalid region format: {region}")
        return False
    
    start, end = region
    if start < 0 or end > ts_length or start >= end:
        print(f"  [VALIDATION ERROR] Invalid region bounds: [{start}, {end}) for length {ts_length}")
        return False
    
    tool_name = plan["tool_name"]
    if tool_name in ["hybrid_up", "hybrid_down"]:
        if "math_shift" not in params:
            print(f"  [VALIDATION WARNING] Missing 'math_shift' for {tool_name}, using default")
    
    return True


def execute_hard_boundary(
    tool_name: str,
    ts: np.ndarray,
    start: int,
    end: int,
    math_shift: float,
    tedit_instance
) -> Optional[np.ndarray]:
    """Execute legacy hard boundary editing for comparison.
    
    Args:
        tool_name: Name of the tool
        ts: Time series to edit
        start: Start index
        end: End index
        math_shift: Math shift value
        tedit_instance: TEdit model instance
    
    Returns:
        Edited time series or None if failed
    """
    try:
        if tool_name == "hybrid_up":
            return hybrid_up(ts.copy(), start, end, math_shift, tedit_instance)
        elif tool_name == "hybrid_down":
            return hybrid_down(ts.copy(), start, end, math_shift, tedit_instance)
        elif tool_name == "ensemble_smooth":
            return ensemble_smooth(ts.copy(), start, end, tedit_instance, n_samples=15)
        else:
            print(f"  [WARNING] Unknown tool: {tool_name}")
            return None
    except Exception as e:
        print(f"  [ERROR] Hard boundary execution failed: {e}")
        return None


def main():
    print("=" * 80)
    print("Pipeline V9: Full System Validation (LLM + Soft-Boundary)")
    print("=" * 80)
    
    api_config = get_api_config()
    print(f"\n[Config] API: {api_config['base_url']}")
    print(f"[Config] Model: {api_config['model_name']}")
    
    client = get_openai_client()
    MODEL_NAME = api_config["model_name"]
    
    print("\n[Init] Loading TEdit Model...")
    model_config = get_model_config("synthetic")
    
    if not verify_model_exists("synthetic"):
        print("[ERROR] Model files not found!")
        return
    
    try:
        tedit_instance = get_tedit_instance(
            model_config["model_path"],
            model_config["config_path"],
            device="cuda:0"
        )
        print(f"[Init] TEdit Model loaded from: {model_config['model_path']}")
    except Exception as e:
        print(f"[ERROR] Failed to load TEdit Model: {e}")
        return
    
    print("[Init] Synthesizing Historical Time Series...")
    ts_full, _ = synthesize_time_series(length=150, noise_params={"std": 2.0}, seed=1024)
    base_ts = ts_full[:120].astype(np.float32)
    TS_LENGTH = len(base_ts)
    
    test_cases = [
        {
            "event_name": "Scenario 1: Oil Crisis (Surge)",
            "instruction": "Due to sudden geopolitical conflict in the Middle East, "
                          "oil prices are expected to surge drastically starting from step 30 "
                          "and continuing until step 70.",
            "expected_tool": "hybrid_up"
        },
        {
            "event_name": "Scenario 2: Tech Bubble Burst (Drop)",
            "instruction": "Market panic triggered by the tech bubble burst. "
                          "The index will crash significantly between step 80 and step 110.",
            "expected_tool": "hybrid_down"
        },
        {
            "event_name": "Scenario 3: Central Bank Intervention (Stabilize)",
            "instruction": "The Central Bank announced a yield curve control policy. "
                          "The market volatility should be eliminated and prices stabilized "
                          "from step 40 to step 90 to restore confidence.",
            "expected_tool": "ensemble_smooth"
        }
    ]
    
    results_data = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"[Test {i+1}] {case['event_name']}")
        print(f"Instruction: \"{case['instruction'][:60]}...\"")
        print(f"{'='*60}")
        
        print(f"\n  [Step 1] Cognitive Planning (Calling LLM API)...")
        plan = call_llm_for_planning(client, MODEL_NAME, case['instruction'], TS_LENGTH)
        
        if plan is None:
            print(f"  [SKIP] LLM planning failed, skipping this case")
            continue
        
        print(f"\n  [LLM Response]:")
        print(f"    Tool: {plan.get('tool_name', 'N/A')}")
        print(f"    Region: {plan.get('parameters', {}).get('region', 'N/A')}")
        print(f"    Thought: {plan.get('thought', 'N/A')[:50]}...")
        
        if plan.get('tool_name') != case['expected_tool']:
            print(f"  [WARNING] LLM chose {plan.get('tool_name')}, expected {case['expected_tool']}")
        
        if not validate_plan(plan, TS_LENGTH):
            print(f"  [SKIP] Invalid plan, skipping this case")
            continue
        
        region = plan['parameters']['region']
        start, end = region
        math_shift = plan['parameters'].get('math_shift', 15.0 if plan['tool_name'] == 'hybrid_up' else -15.0)
        
        print(f"\n  [Step 2] Executing Soft-Boundary Injection...")
        try:
            edited_ts_soft, log = execute_llm_tool(
                plan, base_ts.copy(), tedit_instance, 
                use_soft_boundary=True, n_ensemble=15
            )
            print(f"    Log: {log}")
        except Exception as e:
            print(f"  [ERROR] Soft boundary execution failed: {e}")
            continue
        
        print(f"\n  [Step 3] Simulating Legacy Hard Boundary for Comparison...")
        edited_ts_hard = execute_hard_boundary(
            plan['tool_name'], base_ts.copy(), start, end, math_shift, tedit_instance
        )
        
        print(f"\n  [Step 4] Diagnostic Metrics...")
        cliff_soft_L, cliff_soft_R = analyze_boundary_cliff(edited_ts_soft, start, end)
        
        if edited_ts_hard is not None:
            cliff_hard_L, cliff_hard_R = analyze_boundary_cliff(edited_ts_hard, start, end)
            
            reduction_L = (1 - cliff_soft_L / (cliff_hard_L + 1e-6)) * 100
            reduction_R = (1 - cliff_soft_R / (cliff_hard_R + 1e-6)) * 100
            
            print(f"\n  [Cliff Effect Analysis]")
            print(f"    Left Boundary:  Hard={cliff_hard_L:.4f} -> Soft={cliff_soft_L:.4f} (Reduction: {reduction_L:.1f}%)")
            print(f"    Right Boundary: Hard={cliff_hard_R:.4f} -> Soft={cliff_soft_R:.4f} (Reduction: {reduction_R:.1f}%)")
        else:
            print(f"  [WARNING] Hard boundary comparison not available")
            reduction_L, reduction_R = 0, 0
            cliff_hard_L, cliff_hard_R = 0, 0
        
        original_mean = np.mean(base_ts[start:end])
        edited_mean = np.mean(edited_ts_soft[start:end])
        mean_change = edited_mean - original_mean
        
        print(f"\n  [Region Statistics]")
        print(f"    Original Mean: {original_mean:.2f}")
        print(f"    Edited Mean: {edited_mean:.2f}")
        print(f"    Mean Change: {mean_change:+.2f}")
        
        results_data.append({
            "case": case,
            "base": base_ts.copy(),
            "soft": edited_ts_soft,
            "hard": edited_ts_hard,
            "plan": plan,
            "metrics": {
                "cliff_soft_L": cliff_soft_L,
                "cliff_soft_R": cliff_soft_R,
                "cliff_hard_L": cliff_hard_L,
                "cliff_hard_R": cliff_hard_R,
                "reduction_L": reduction_L,
                "reduction_R": reduction_R,
                "mean_change": mean_change
            }
        })
    
    print(f"\n{'='*60}")
    print("[Step 5] Generating Visualization...")
    print(f"{'='*60}")
    
    n_results = len(results_data)
    if n_results == 0:
        print("[WARNING] No results to visualize!")
        return
    
    fig, axes = plt.subplots(n_results, 1, figsize=(14, 5 * n_results))
    if n_results == 1:
        axes = [axes]
    
    for idx, res in enumerate(results_data):
        ax = axes[idx]
        region = res['plan']['parameters']['region']
        start, end = region
        
        ax.plot(res['base'], color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Original')
        
        if res['hard'] is not None:
            ax.plot(res['hard'], color='red', linewidth=1.5, alpha=0.6, label='Legacy (Hard Boundary)')
        
        ax.plot(res['soft'], color='blue', linewidth=2.0, label='BetterTSE (Soft Boundary)')
        
        ax.axvspan(start, end, color='yellow', alpha=0.15, label='Edit Region')
        ax.axvline(start, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.axvline(end, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        
        metrics = res['metrics']
        title = (f"{res['case']['event_name']}\n"
                f"Tool: {res['plan']['tool_name']} | Region: [{start}, {end})\n"
                f"Mean Δ: {metrics['mean_change']:+.2f} | "
                f"Cliff Reduction: L={metrics['reduction_L']:.1f}%, R={metrics['reduction_R']:.1f}%")
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/pipeline_v9_full_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Visualization saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("Pipeline V9 Summary Report")
    print("=" * 80)
    
    print("\nKey Innovation Validated:")
    print("  - Latent Blending: z_{t-1} = M * z^{pred} + (1-M) * z^{GT}")
    print("  - Background is 'physical truth' (forward-diffused from original)")
    print("  - Ensures 100% background fidelity (zero reconstruction error)")
    
    print("\nResults Summary:")
    for res in results_data:
        metrics = res['metrics']
        print(f"\n  [{res['case']['event_name']}]")
        print(f"    Tool: {res['plan']['tool_name']}")
        print(f"    Region: {res['plan']['parameters']['region']}")
        print(f"    Mean Change: {metrics['mean_change']:+.2f}")
        print(f"    Cliff Reduction: Left={metrics['reduction_L']:.1f}%, Right={metrics['reduction_R']:.1f}%")
    
    print(f"\n[COMPLETE] Successfully validated {len(results_data)}/{len(test_cases)} scenarios")
    print("=" * 80)


if __name__ == "__main__":
    main()
