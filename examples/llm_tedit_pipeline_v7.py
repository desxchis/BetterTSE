"""
LLM-Guided TEdit Pipeline V7 (Real LLM API Test)
测试解耦后的 Agent 架构：基于真实 LLM 的事件驱动时序校正
批量测试多个事件场景
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool.tedit_wrapper import get_tedit_instance
from tool.ts_synthesizer import synthesize_time_series
from modules.llm import get_event_driven_plan
from agent.nodes import execute_llm_tool

print("=" * 80)
print("Pipeline V7: Testing Decoupled Real LLM Agent Architecture")
print("=" * 80)

# =====================================================================
# [Step 1] 初始化环境、LLM 客户端与底层 TEdit 模型
# =====================================================================
print("\n[Step 1] Initializing Environment...")

API_KEY = os.environ.get("OPENAI_API_KEY", "sk-30cb957d01eb473aac1cb85fdee68352")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
MODEL_NAME = "deepseek-chat"

model_path = "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"
config_path = "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tedit_instance = get_tedit_instance(
    os.path.join(project_root, model_path),
    os.path.join(project_root, config_path),
    device="cuda:0"
)

ts, components = synthesize_time_series(length=120, noise_params={"std": 2.5}, seed=42)
base_ts = ts[:100].astype(np.float32)

# =====================================================================
# [Step 2] 定义多个事件场景
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
# [Step 3] 批量测试每个场景
# =====================================================================
results = []

for i, scenario in enumerate(test_scenarios):
    print(f"\n{'='*60}")
    print(f"[Test {i+1}] {scenario['name']}")
    print(f"{'='*60}")
    
    print(f"\n  [Input News]: {scenario['news']}")
    print(f"  [Instruction]: {scenario['instruction']}")
    print("  Waiting for LLM reasoning and routing decision...")
    
    try:
        plan_dict = get_event_driven_plan(
            scenario['news'], 
            scenario['instruction'], 
            client, 
            model=MODEL_NAME
        )
    except Exception as e:
        print(f"\n❌ LLM API 请求失败: {e}")
        continue
    
    print("\n" + "-"*50)
    print(f"🧠 [LLM Agent JSON Output]")
    print(json.dumps(plan_dict, indent=4, ensure_ascii=False))
    print("-"*50)
    
    edited_ts = execute_llm_tool(base_ts.copy(), plan_dict, tedit_instance)
    
    start, end = plan_dict["parameters"]["region"]
    original_mean = np.mean(base_ts[start:end])
    edited_mean = np.mean(edited_ts[start:end])
    
    print(f"\n  [Metrics] Mean changed from {original_mean:.2f} -> {edited_mean:.2f}")
    
    results.append({
        "name": scenario['name'],
        "news": scenario['news'],
        "plan": plan_dict,
        "edited_ts": edited_ts,
        "region": (start, end)
    })

# =====================================================================
# [Step 4] 可视化所有结果
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
    
    ax.plot(base_ts, color='lightgray', linestyle='--', linewidth=2, label="Original Baseline")
    ax.plot(result['edited_ts'], color='red', linewidth=2.5, 
            label=f"Edited ({result['plan']['tool_name']})")
    
    start, end = result['region']
    ax.axvspan(start, end, color='red', alpha=0.1, label='Edit Region')
    
    tool_name = result['plan']['tool_name']
    ax.set_title(f"[Test {i+1}] {result['name']}\nTool: {tool_name}", fontsize=12)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/llm_tedit_pipeline_v7.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Pipeline V7 Complete! Visualization Saved to: {output_path}")
print(f"✅ Successfully tested {len(results)}/{len(test_scenarios)} scenarios")
print("=" * 80)

# =====================================================================
# [Step 5] 打印汇总报告
# =====================================================================
print("\n📊 Summary Report:")
print("-" * 60)
for i, result in enumerate(results):
    print(f"\n[Test {i+1}] {result['name']}")
    print(f"  Tool: {result['plan']['tool_name']}")
    print(f"  Region: {result['region']}")
    if 'math_shift' in result['plan']['parameters']:
        print(f"  Math Shift: {result['plan']['parameters']['math_shift']}")
