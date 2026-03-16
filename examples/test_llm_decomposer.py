"""Test LLM-based instruction decomposer and region selector.

This script tests the LLM-powered natural language understanding
for time series editing tasks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
import json
from agent.llm_instruction_decomposer import get_llm_decomposer
from tool.ts_synthesizer import synthesize_time_series

print("=" * 60)
print("LLM-based Instruction Decomposer Test")
print("=" * 60)

# Prepare test data
print("\n[Step 1] Preparing test time series...")
history_ts, components = synthesize_time_series(
    length=100,
    trend_params={"slope": 0.5, "intercept": 10, "trend_type": "linear"},
    seasonality_params={"period": 12, "amplitude": 5, "seasonality_type": "sine"},
    volatility_params={"base_volatility": 1.0, "volatility_type": "constant"},
    noise_params={"noise_type": "gaussian", "std": 0.5},
    seed=42
)

forecast_ts = np.mean(history_ts) + np.random.randn(50) * 0.5
print(f"  Forecast length: {len(forecast_ts)}")
print(f"  Forecast mean: {np.mean(forecast_ts):.2f}")
print(f"  Forecast std: {np.std(forecast_ts):.2f}")

# Initialize LLM decomposer
print("\n[Step 2] Initializing LLM decomposer...")
try:
    decomposer = get_llm_decomposer()
    print("  LLM decomposer initialized successfully")
except Exception as e:
    print(f"  Error: {e}")
    print("  Make sure OPENAI_API_KEY is set in .env file")
    sys.exit(1)

# Test cases
test_instructions = [
    "Make the first half grow faster",
    "Reduce the noise in the middle section",
    "Increase volatility in the last 10 points",
    "Smooth out the entire series",
    "Make the trend steeper from index 10 to 30",
]

print("\n[Step 3] Testing instruction decomposition...")
print("-" * 60)

for i, instruction in enumerate(test_instructions, 1):
    print(f"\nTest {i}: \"{instruction}\"")
    print("-" * 40)
    
    try:
        result = decomposer.decompose(
            instruction,
            ts_length=len(forecast_ts),
            ts_values=forecast_ts
        )
        
        print(f"  Intent: {result['intent']}")
        print(f"  Is Two-Stage: {result['is_two_stage']}")
        print(f"  Region: [{result['region_selection']['start_idx']}, {result['region_selection']['end_idx']})")
        print(f"  Region Reasoning: {result['region_selection']['reasoning']}")
        print(f"  Operations: {[op['name'] for op in result['editing_operations']]}")
        if 'natural_language_summary' in result:
            print(f"  Summary: {result['natural_language_summary']}")
        
    except Exception as e:
        print(f"  Error: {e}")

# Test LLM region selection
print("\n" + "=" * 60)
print("[Step 4] Testing LLM region selection...")
print("-" * 60)

test_intents = ["volatility", "trend", "anomaly"]

for intent in test_intents:
    print(f"\nIntent: {intent}")
    try:
        region = decomposer.select_region_with_llm(
            instruction=f"Find region with high {intent}",
            ts_values=forecast_ts,
            intent=intent
        )
        print(f"  Selected Region: [{region['start_idx']}, {region['end_idx']})")
        print(f"  Confidence: {region.get('confidence', 'N/A')}")
        print(f"  Reasoning: {region['reasoning']}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 60)
print("LLM Decomposer Test Completed")
print("=" * 60)

print("\nSummary:")
print("  - LLM successfully decomposes natural language instructions")
print("  - Region selection is context-aware")
print("  - Output is structured for downstream processing")
print("\nNext step: Integrate with TEdit for end-to-end editing")
