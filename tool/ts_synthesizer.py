import numpy as np
from typing import Optional, Tuple, Union, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_trend(
    length: int,
    slope: float = 1.0,
    intercept: float = 0.0,
    trend_type: str = "linear",
    degree: int = 2
) -> np.ndarray:
    x = np.arange(length, dtype=float)
    
    if trend_type == "linear":
        trend = slope * x + intercept
    elif trend_type == "polynomial":
        coeffs = [intercept] + [slope] * degree
        trend = np.polyval(coeffs, x)
    elif trend_type == "exponential":
        trend = intercept * np.exp(slope * x / length)
    elif trend_type == "logarithmic":
        trend = slope * np.log(x + 1) + intercept
    else:
        raise ValueError(f"Unknown trend_type: {trend_type}")
    
    return trend


def generate_seasonality(
    length: int,
    period: int = 12,
    amplitude: float = 1.0,
    phase: float = 0.0,
    seasonality_type: str = "sine",
    harmonics: int = 1
) -> np.ndarray:
    x = np.arange(length, dtype=float)
    seasonality = np.zeros(length)
    
    if seasonality_type == "sine":
        for h in range(1, harmonics + 1):
            seasonality += amplitude * np.sin(2 * np.pi * h * x / period + phase) / h
    elif seasonality_type == "cosine":
        for h in range(1, harmonics + 1):
            seasonality += amplitude * np.cos(2 * np.pi * h * x / period + phase) / h
    elif seasonality_type == "sawtooth":
        seasonality = amplitude * 2 * (x / period - np.floor(x / period + 0.5))
    elif seasonality_type == "square":
        seasonality = amplitude * np.sign(np.sin(2 * np.pi * x / period + phase))
    else:
        raise ValueError(f"Unknown seasonality_type: {seasonality_type}")
    
    return seasonality


def generate_volatility(
    length: int,
    base_volatility: float = 1.0,
    volatility_type: str = "constant",
    volatility_period: Optional[int] = None,
    min_volatility: float = 0.1,
    max_volatility: float = 2.0
) -> np.ndarray:
    if volatility_type == "constant":
        volatility = np.full(length, base_volatility)
    elif volatility_type == "sine":
        if volatility_period is None:
            volatility_period = length // 4
        x = np.arange(length, dtype=float)
        volatility = min_volatility + (max_volatility - min_volatility) * \
                    (0.5 + 0.5 * np.sin(2 * np.pi * x / volatility_period))
    elif volatility_type == "linear":
        x = np.arange(length, dtype=float)
        volatility = min_volatility + (max_volatility - min_volatility) * x / (length - 1)
    elif volatility_type == "random_walk":
        volatility = np.zeros(length)
        volatility[0] = base_volatility
        for i in range(1, length):
            change = np.random.normal(0, 0.1)
            volatility[i] = np.clip(volatility[i-1] + change, min_volatility, max_volatility)
    else:
        raise ValueError(f"Unknown volatility_type: {volatility_type}")
    
    return volatility


def generate_noise(
    length: int,
    noise_type: str = "gaussian",
    mean: float = 0.0,
    std: float = 1.0
) -> np.ndarray:
    if noise_type == "gaussian":
        noise = np.random.normal(mean, std, length)
    elif noise_type == "uniform":
        noise = np.random.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), length)
    elif noise_type == "laplace":
        noise = np.random.laplace(mean, std / np.sqrt(2), length)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return noise


def synthesize_time_series(
    length: int,
    trend_params: Optional[dict] = None,
    seasonality_params: Optional[dict] = None,
    volatility_params: Optional[dict] = None,
    noise_params: Optional[dict] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    if seed is not None:
        np.random.seed(seed)
    
    components = {}
    ts = np.zeros(length)
    
    if trend_params is not None:
        trend = generate_trend(length, **trend_params)
        components['trend'] = trend
        ts += trend
    
    if seasonality_params is not None:
        seasonality = generate_seasonality(length, **seasonality_params)
        components['seasonality'] = seasonality
        ts += seasonality
    
    if volatility_params is not None:
        volatility = generate_volatility(length, **volatility_params)
        components['volatility'] = volatility
    
    if noise_params is not None:
        noise = generate_noise(length, **noise_params)
        if 'volatility' in components:
            noise = noise * components['volatility']
        components['noise'] = noise
        ts += noise
    
    return ts, components


def create_test_dataset(
    n_samples: int = 10,
    length: int = 100,
    patterns: List[str] = None,
    seed: Optional[int] = None
) -> List[dict]:
    if patterns is None:
        patterns = [
            "trend_up",
            "trend_down",
            "trend_up_seasonal",
            "trend_down_seasonal",
            "constant_seasonal",
            "trend_up_high_volatility",
            "trend_down_low_volatility",
            "trend_up_seasonal_high_volatility",
            "trend_down_seasonal_low_volatility",
            "constant_seasonal_varying_volatility"
        ]
    
    dataset = []
    
    for i, pattern in enumerate(patterns):
        if seed is not None:
            pattern_seed = seed + i
        else:
            pattern_seed = None
        
        if pattern == "trend_up":
            trend_params = {"slope": 0.5, "intercept": 10, "trend_type": "linear"}
            seasonality_params = None
            volatility_params = None
            noise_params = {"noise_type": "gaussian", "std": 0.5}
        
        elif pattern == "trend_down":
            trend_params = {"slope": -0.5, "intercept": 60, "trend_type": "linear"}
            seasonality_params = None
            volatility_params = None
            noise_params = {"noise_type": "gaussian", "std": 0.5}
        
        elif pattern == "trend_up_seasonal":
            trend_params = {"slope": 0.3, "intercept": 20, "trend_type": "linear"}
            seasonality_params = {"period": 12, "amplitude": 5, "seasonality_type": "sine"}
            volatility_params = None
            noise_params = {"noise_type": "gaussian", "std": 0.3}
        
        elif pattern == "trend_down_seasonal":
            trend_params = {"slope": -0.3, "intercept": 50, "trend_type": "linear"}
            seasonality_params = {"period": 12, "amplitude": 5, "seasonality_type": "sine"}
            volatility_params = None
            noise_params = {"noise_type": "gaussian", "std": 0.3}
        
        elif pattern == "constant_seasonal":
            trend_params = None
            seasonality_params = {"period": 12, "amplitude": 10, "seasonality_type": "sine"}
            volatility_params = None
            noise_params = {"noise_type": "gaussian", "std": 0.5}
        
        elif pattern == "trend_up_high_volatility":
            trend_params = {"slope": 0.5, "intercept": 10, "trend_type": "linear"}
            seasonality_params = None
            volatility_params = {"base_volatility": 2.0, "volatility_type": "constant"}
            noise_params = {"noise_type": "gaussian", "std": 1.0}
        
        elif pattern == "trend_down_low_volatility":
            trend_params = {"slope": -0.5, "intercept": 60, "trend_type": "linear"}
            seasonality_params = None
            volatility_params = {"base_volatility": 0.2, "volatility_type": "constant"}
            noise_params = {"noise_type": "gaussian", "std": 0.1}
        
        elif pattern == "trend_up_seasonal_high_volatility":
            trend_params = {"slope": 0.3, "intercept": 20, "trend_type": "linear"}
            seasonality_params = {"period": 12, "amplitude": 5, "seasonality_type": "sine"}
            volatility_params = {"base_volatility": 1.5, "volatility_type": "constant"}
            noise_params = {"noise_type": "gaussian", "std": 0.8}
        
        elif pattern == "trend_down_seasonal_low_volatility":
            trend_params = {"slope": -0.3, "intercept": 50, "trend_type": "linear"}
            seasonality_params = {"period": 12, "amplitude": 5, "seasonality_type": "sine"}
            volatility_params = {"base_volatility": 0.3, "volatility_type": "constant"}
            noise_params = {"noise_type": "gaussian", "std": 0.2}
        
        elif pattern == "constant_seasonal_varying_volatility":
            trend_params = None
            seasonality_params = {"period": 12, "amplitude": 10, "seasonality_type": "sine"}
            volatility_params = {"min_volatility": 0.2, "max_volatility": 2.0,
                               "volatility_type": "sine", "volatility_period": 30}
            noise_params = {"noise_type": "gaussian", "std": 1.0}
        
        else:
            continue
        
        ts, components = synthesize_time_series(
            length=length,
            trend_params=trend_params,
            seasonality_params=seasonality_params,
            volatility_params=volatility_params,
            noise_params=noise_params,
            seed=pattern_seed
        )
        
        dataset.append({
            "pattern": pattern,
            "time_series": ts,
            "components": components,
            "params": {
                "trend": trend_params,
                "seasonality": seasonality_params,
                "volatility": volatility_params,
                "noise": noise_params
            }
        })
    
    return dataset


def plot_synthetic_series(
    ts: np.ndarray,
    components: Optional[dict] = None,
    title: str = "Synthetic Time Series",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    if components is None or len(components) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(ts, label="Time Series", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        n_components = len(components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize)
        
        axes[0].plot(ts, label="Time Series", linewidth=1.5, color='black')
        axes[0].set_title(title)
        axes[0].set_ylabel("Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, (name, comp) in enumerate(components.items()):
            axes[i+1].plot(comp, label=name, linewidth=1.5, color=colors[i % len(colors)])
            axes[i+1].set_ylabel(name.capitalize())
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Time")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_metrics(
    original: np.ndarray,
    edited: np.ndarray,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> dict:
    if end_idx is None:
        end_idx = len(original)
    
    original_region = original[start_idx:end_idx]
    edited_region = edited[start_idx:end_idx]
    
    metrics = {
        "original_mean": np.mean(original_region),
        "edited_mean": np.mean(edited_region),
        "original_std": np.std(original_region),
        "edited_std": np.std(edited_region),
        "original_range": np.max(original_region) - np.min(original_region),
        "edited_range": np.max(edited_region) - np.min(edited_region),
        "mean_change": np.mean(edited_region) - np.mean(original_region),
        "std_change": np.std(edited_region) - np.std(original_region),
        "mse": np.mean((edited_region - original_region) ** 2),
        "mae": np.mean(np.abs(edited_region - original_region))
    }
    
    x = np.arange(len(original_region), dtype=float)
    original_trend = np.polyfit(x, original_region, 1)[0]
    edited_trend = np.polyfit(x, edited_region, 1)[0]
    
    metrics["original_trend"] = original_trend
    metrics["edited_trend"] = edited_trend
    metrics["trend_change"] = edited_trend - original_trend
    
    return metrics
