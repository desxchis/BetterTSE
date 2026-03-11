"""
LLM-TEdit Pipeline V9 - Complete Integration Test

This script integrates:
1. LLM API (DeepSeek) for event-driven planning
2. Soft-Boundary Temporal Injection (Latent Blending)
3. Ground-Truth Preservation Editing

Key Innovation:
- Latent Blending: z_{t-1} = M * z^{pred} + (1-M) * z^{GT}
- Background is "physical truth" (forward-diffused from original data)
- Ensures 100% background fidelity (zero reconstruction error)

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
from agent.prompts import EVENT_DRIVEN_AGENT_PROMPT
from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series
from tool.ts_editors import execute_llm_tool

setup_environment()

print("=" * 80)
print("Pipeline V9: LLM API + Soft-Boundary Temporal Injection")
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
# [Step 2] 定义事件场景
# =====================================================================
test_scenarios = [
    {
        "name": "Geopolitical Oil Shock",
        "news": "BREAKING: Trump announces the takeover of Venezuelan oil facilities, causing massive global supply panic.",
        "instruction": "Oil prices will surge aggressively in the first 30 days."
    },
    {
        "name": "Tech Product Recall",
        "news": "BREAKING: Major smartphone manufacturer announces recall of 5 million devices due to battery safety concerns.",
        "instruction": "Stock price will drop significantly between days 20-60."
    },
    {
        "name": "Central Bank Intervention",
        "news": "BREAKING: Federal Reserve announces emergency quantitative easing measures to stabilize financial markets.",
        "instruction": "Market volatility will decrease and stabilize from day 40 to day 80."
    }
]

# =====================================================================
# [Step 3] 调用 LLM API 并执行编辑
# =====================================================================
results = []

for i, scenario in enumerate(test_scenarios):
    print(f"\n{'='*60}")
    print(f"[Test {i+1}] {scenario['name']}")
    print(f"{'='*60}")
    
    print(f"\n  [Input News]: {scenario['news']}")
    print(f"  [Instruction]: {scenario['instruction']}")
    print("  Calling DeepSeek API...")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": EVENT_DRIVEN_AGENT_PROMPT},
                {"role": "user", "content": f"News: {scenario['news']}\n\nInstruction: {scenario['instruction']}"}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            plan_dict = json.loads(json_match.group())
        else:
            print(f"  ❌ No JSON found in LLM response")
            continue
            
    except Exception as e:
        print(f"  ❌ LLM API call failed: {e}")
        continue
    
    print("\n" + "-"*50)
    print(f"🧠 [LLM Response]")
    print(json.dumps(plan_dict, indent=4, ensure_ascii=False))
    print("-"*50)
    
    print("\n  Executing Soft-Boundary Editing...")
    edited_ts, log = execute_llm_tool(
        plan=plan_dict,
        ts=base_ts.copy(),
        tedit=tedit_instance,
        n_ensemble=15,
        use_soft_boundary=True,
    )
    print(f"  [Execution Log]: {log}")
    
    start, end = plan_dict["parameters"]["region"]
    original_mean = np.mean(base_ts[start:end])
    edited_mean = np.mean(edited_ts[start:end])
    print(f"  [Metrics] Mean changed from {original_mean:.2f} -> {edited_mean:.2f}")
    
    def compute_gradient(ts):
        return np.abs(np.diff(ts))
    
    grad_original = compute_gradient(base_ts)
    grad_edited = compute_gradient(edited_ts)
    
    boundary_region = 5
    left_start = max(0, start - boundary_region)
    left_end = min(len(grad_edited), start + boundary_region)
    
    if left_end > left_start:
        cliff_original = np.max(grad_original[left_start:left_end])
        cliff_edited = np.max(grad_edited[left_start:left_end])
        cliff_reduction = (1 - cliff_edited / cliff_original) * 100 if cliff_original > 0 else 0
    else:
        cliff_reduction = 0
    
    print(f"  [Boundary] Cliff reduction: {cliff_reduction:.1f}%")
    
    results.append({
        "name": scenario['name'],
        "news": scenario['news'],
        "plan": plan_dict,
        "edited_ts": edited_ts,
        "region": (start, end),
        "mean_change": edited_mean - original_mean,
        "cliff_reduction": cliff_reduction,
    })

# =====================================================================
# [Step 4] 可视化
# =====================================================================
print(f"\n{'='*60}")
print("[Step 4] Generating Visualization...")
print(f"{'='*60}")

n_results = len(results)
fig, axes = plt.subplots(n_results, 1, figsize=(14, 5 * n_results))

if n_results == 1:
    axes = [axes]

for i, result in enumerate(results):
    ax = axes[i]
    
    ax.plot(base_ts, 'k--', alpha=0.5, linewidth=1.5, label='Original Baseline')
    ax.plot(result['edited_ts'], 'b-', linewidth=2.5, 
            label=f"Edited ({result['plan']['tool_name']})")
    
    start, end = result['region']
    ax.axvspan(start, end, alpha=0.15, color='yellow', label='Edit Region')
    ax.axvline(start, color='red', linestyle=':', alpha=0.7)
    ax.axvline(end, color='red', linestyle=':', alpha=0.7)
    
    tool_name = result['plan']['tool_name']
    ax.set_title(
        f"[Test {i+1}] {result['name']}\n"
        f"Tool: {tool_name} | Mean Δ: {result['mean_change']:+.2f} | "
        f"Cliff ↓: {result['cliff_reduction']:.1f}%", 
        fontsize=11
    )
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/llm_tedit_pipeline_v9.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Visualization Saved to: {output_path}")

# =====================================================================
# [Step 5] 汇总报告
# =====================================================================
print("\n" + "=" * 80)
print("📊 Pipeline V9 Summary Report")
print("=" * 80)

print("\n🔬 Key Innovation:")
print("  - Noise Blending: Blend at NOISE level, not LATENT level")
print("  - Variance Preserved: Var[noise_blend] = Var[noise]")
print("  - Smooth Boundaries: Soft mask eliminates 'cliff effect'")

print("\n📈 Results Summary:")
for result in results:
    print(f"\n  [{result['name']}]")
    print(f"    Tool: {result['plan']['tool_name']}")
    print(f"    Region: {result['region']}")
    print(f"    Mean Change: {result['mean_change']:+.2f}")
    print(f"    Cliff Reduction: {result['cliff_reduction']:.1f}%")

print(f"\n✅ Successfully tested {len(results)}/{len(test_scenarios)} scenarios")
print("=" * 80)
