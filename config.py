"""Unified configuration for BetterTSE project.

This module provides centralized configuration for:
- DeepSeek API credentials
- TEdit model paths
- Default parameters

Usage:
    from config import get_api_config, get_model_config, setup_environment
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

DEFAULT_MODEL_PATH = "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth"
DEFAULT_CONFIG_PATH = "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml"

MODEL_PATHS = {
    "synthetic": {
        "model": "TEdit-main/save/synthetic/pretrain_multi_weaver/0/ckpts/model_best.pth",
        "config": "TEdit-main/save/synthetic/pretrain_multi_weaver/0/model_configs.yaml",
    },
    "air": {
        "model": "TEdit-main/save/air/pretrain_multi_weaver/0/ckpts/model_best.pth",
        "config": "TEdit-main/save/air/pretrain_multi_weaver/0/model_configs.yaml",
    },
    "motor": {
        "model": "TEdit-main/save/motor/pretrain_multi_weaver/0/ckpts/model_best.pth",
        "config": "TEdit-main/save/motor/pretrain_multi_weaver/0/model_configs.yaml",
    },
}


def get_api_config() -> Dict[str, str]:
    """Get API configuration with fallback to hardcoded values.
    
    Returns:
        Dictionary with api_key, base_url, and model_name
    """
    api_key = os.environ.get("OPENAI_API_KEY") or DEEPSEEK_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL") or DEEPSEEK_BASE_URL
    model_name = os.environ.get("MODEL_NAME") or DEEPSEEK_MODEL
    
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }


def get_model_config(dataset: str = "synthetic") -> Dict[str, str]:
    """Get model path configuration for specified dataset.
    
    Args:
        dataset: Dataset name ('synthetic', 'air', 'motor')
    
    Returns:
        Dictionary with model_path and config_path (absolute paths)
    """
    if dataset not in MODEL_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(MODEL_PATHS.keys())}")
    
    paths = MODEL_PATHS[dataset]
    return {
        "model_path": str(PROJECT_ROOT / paths["model"]),
        "config_path": str(PROJECT_ROOT / paths["config"]),
    }


def setup_environment() -> None:
    """Setup environment variables for DeepSeek API.
    
    This function ensures that OPENAI_API_KEY and OPENAI_BASE_URL
    are set in the environment, using hardcoded fallbacks if necessary.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = DEEPSEEK_BASE_URL
    
    if not os.environ.get("MODEL_NAME"):
        os.environ["MODEL_NAME"] = DEEPSEEK_MODEL


def get_openai_client():
    """Get configured OpenAI client for DeepSeek API.
    
    Returns:
        OpenAI client instance
    """
    from openai import OpenAI
    
    config = get_api_config()
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )


def verify_model_exists(dataset: str = "synthetic") -> bool:
    """Verify that model files exist for the specified dataset.
    
    Args:
        dataset: Dataset name
    
    Returns:
        True if both model and config files exist
    """
    config = get_model_config(dataset)
    model_exists = Path(config["model_path"]).exists()
    config_exists = Path(config["config_path"]).exists()
    
    if not model_exists:
        print(f"[WARNING] Model file not found: {config['model_path']}")
    if not config_exists:
        print(f"[WARNING] Config file not found: {config['config_path']}")
    
    return model_exists and config_exists


if __name__ == "__main__":
    print("=" * 60)
    print("BetterTSE Configuration Check")
    print("=" * 60)
    
    print("\n[API Configuration]")
    api_config = get_api_config()
    print(f"  API Key: {api_config['api_key'][:10]}...{api_config['api_key'][-4:]}")
    print(f"  Base URL: {api_config['base_url']}")
    print(f"  Model: {api_config['model_name']}")
    
    print("\n[Model Paths]")
    for dataset in MODEL_PATHS:
        exists = verify_model_exists(dataset)
        status = "✅" if exists else "❌"
        print(f"  {status} {dataset}")
        if exists:
            config = get_model_config(dataset)
            print(f"      Model: {config['model_path']}")
            print(f"      Config: {config['config_path']}")
    
    print("\n" + "=" * 60)
