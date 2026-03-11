"""TEdit model wrapper for time series editing.

This module provides a unified interface to integrate TEdit (NeurIPS 2024)
diffusion-based time series editing model into the BetterTSE workflow.

Key Innovation: Soft-Boundary Temporal Injection
- Replaces hard array splicing with latent space blending
- Eliminates "cliff effect" at region boundaries
- Training-free attention region injection inspired by RePlan
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d


class TEditWrapper:
    """Wrapper class for TEdit diffusion model.

    This class encapsulates the TEdit model and provides a simple interface
    for editing time series based on attribute conditions.

    Attributes:
        model: The loaded TEdit ConditionalGenerator model
        device: Device where the model is loaded
        config: Model configuration dictionary
        is_loaded: Whether the model has been successfully loaded
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        tedit_root: Optional[str] = None,
    ):
        """Initialize TEdit wrapper.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            config_path: Path to the model configuration file (.yaml)
            device: Device to load the model on (default: "cuda:0")
            tedit_root: Root directory of TEdit project (if None, assumes TEdit-main/ in parent dir)
        """
        self.device = device
        self.model = None
        self.config = {}
        self.is_loaded = False

        if tedit_root is None:
            current_dir = Path(__file__).resolve().parent
            tedit_root = current_dir.parent / "TEdit-main"

        self.tedit_root = Path(tedit_root)

        if model_path and config_path:
            self.load_model(model_path, config_path)

    def load_model(
        self,
        model_path: str,
        config_path: str,
    ) -> None:
        """Load TEdit model from checkpoint and config.

        Args:
            model_path: Path to the model checkpoint file (.pth)
            config_path: Path to the model configuration file (.yaml)

        Raises:
            FileNotFoundError: If model or config file not found
            RuntimeError: If model loading fails
        """
        import yaml
        
        # Add TEdit-main to Python path
        if str(self.tedit_root) not in sys.path:
            sys.path.insert(0, str(self.tedit_root))
        
        try:
            from models.conditional_generator import ConditionalGenerator
        except ImportError as e:
            raise ImportError(
                f"Failed to import TEdit modules from {self.tedit_root}. Error: {e}"
            )

        model_path = Path(model_path)
        config_path = Path(config_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Recursively update all device settings
        def update_device(config, device):
            if isinstance(config, dict):
                for key in config:
                    if key == "device":
                        config[key] = device
                    elif isinstance(config[key], (dict, list)):
                        update_device(config[key], device)
            elif isinstance(config, list):
                for item in config:
                    if isinstance(item, (dict, list)):
                        update_device(item, device)
        
        update_device(self.config, self.device)

        try:
            self.model = ConditionalGenerator(self.config)
            
            # Load checkpoint with weights_only=False for compatibility
            # Note: Only use weights_only=False if you trust the source of the model file
            checkpoint = torch.load(
                model_path, 
                map_location=self.device,
                weights_only=False  # Required for PyTorch 2.6+ to load older model formats
            )
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load TEdit model: {e}")

    def edit_time_series(
        self,
        ts: np.ndarray,
        src_attrs: np.ndarray,
        tgt_attrs: np.ndarray,
        n_samples: int = 1,
        sampler: str = "ddim",
        edit_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Edit time series using TEdit model.

        Args:
            ts: Input time series to edit (shape: [L] or [1, L])
            src_attrs: Source attributes (shape: [n_attrs])
            tgt_attrs: Target attributes (shape: [n_attrs])
            n_samples: Number of samples to generate (default: 1)
            sampler: Sampler type ("ddim" or "ddpm", default: "ddim")
            edit_steps: Number of edit steps (default: from config)

        Returns:
            Edited time series (shape: [n_samples, L])

        Raises:
            RuntimeError: If model is not loaded or editing fails
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("TEdit model is not loaded. Call load_model() first.")

        ts_array = np.asarray(ts, dtype=np.float32)
        if ts_array.ndim == 1:
            ts_array = ts_array.reshape(1, -1)

        src_attrs_array = np.asarray(src_attrs, dtype=np.int64)
        tgt_attrs_array = np.asarray(tgt_attrs, dtype=np.int64)

        B, L = ts_array.shape
        n_attrs = src_attrs_array.shape[0]

        if src_attrs_array.shape != tgt_attrs_array.shape:
            raise ValueError(
                f"Source and target attributes must have same shape. "
                f"Got {src_attrs_array.shape} and {tgt_attrs_array.shape}"
            )

        with torch.no_grad():
            x = torch.from_numpy(ts_array).unsqueeze(1).to(self.device)
            src_attrs_tensor = (
                torch.from_numpy(src_attrs_array)
                .unsqueeze(0)
                .repeat(B, 1)
                .to(self.device)
            )
            tgt_attrs_tensor = (
                torch.from_numpy(tgt_attrs_array)
                .unsqueeze(0)
                .repeat(B, 1)
                .to(self.device)
            )

            tp = torch.zeros(B, L, device=self.device)

            batch = {
                "src_x": x.permute(0, 2, 1),
                "src_attrs": src_attrs_tensor,
                "tgt_attrs": tgt_attrs_tensor,
                "tgt_x": x.permute(0, 2, 1),
                "tp": tp,
            }

            if edit_steps is not None:
                self.model.edit_steps = edit_steps

            samples = self.model.generate(batch, n_samples=n_samples, mode="edit", sampler=sampler)

        edited_ts = samples.cpu().numpy().squeeze(1)

        return edited_ts

    def edit_region(
        self,
        ts: np.ndarray,
        start_idx: int,
        end_idx: int,
        src_attrs: np.ndarray,
        tgt_attrs: np.ndarray,
        n_samples: int = 1,
        sampler: str = "ddim",
    ) -> np.ndarray:
        """Edit a specific region of time series.

        Args:
            ts: Input time series (shape: [L])
            start_idx: Start index of region to edit (inclusive)
            end_idx: End index of region to edit (exclusive)
            src_attrs: Source attributes for the region
            tgt_attrs: Target attributes for the region
            n_samples: Number of samples to generate
            sampler: Sampler type

        Returns:
            Edited time series with region modified (shape: [L])
        """
        ts_array = np.asarray(ts, dtype=np.float32)

        if start_idx < 0 or end_idx > len(ts_array) or start_idx >= end_idx:
            raise ValueError(f"Invalid region indices: [{start_idx}, {end_idx})")

        region = ts_array[start_idx:end_idx].copy()

        edited_region = self.edit_time_series(
            region, src_attrs, tgt_attrs, n_samples, sampler
        )

        result = ts_array.copy()
        result[start_idx:end_idx] = edited_region[0]

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "config": self.config if self.is_loaded else {},
        }

        if self.is_loaded and self.model is not None:
            info.update({
                "num_steps": self.model.num_steps,
                "edit_steps": self.model.edit_steps,
                "bootstrap_ratio": self.model.bootstrap_ratio,
            })

        return info

    def set_edit_steps(self, steps: int) -> None:
        """Set the number of edit steps.

        Args:
            steps: Number of edit steps to use
        """
        if self.model is not None:
            self.model.edit_steps = steps

    def _generate_soft_mask(
        self,
        length: int,
        start_idx: int,
        end_idx: int,
        smooth_radius: float = 5.0,
    ) -> np.ndarray:
        """Generate a soft boundary mask using Gaussian smoothing.

        This creates a smooth transition from 0 to 1 at the region boundaries,
        eliminating the "cliff effect" that occurs with hard masks.

        Args:
            length: Total length of the time series
            start_idx: Start index of the edit region
            end_idx: End index of the edit region
            smooth_radius: Standard deviation for Gaussian kernel (default: 5.0)

        Returns:
            Soft mask array with values in [0, 1], shape: [length]
        """
        hard_mask = np.zeros(length, dtype=np.float32)
        hard_mask[start_idx:end_idx] = 1.0
        
        soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)
        
        return soft_mask

    def edit_region_soft(
        self,
        ts: np.ndarray,
        start_idx: int,
        end_idx: int,
        src_attrs: np.ndarray,
        tgt_attrs: np.ndarray,
        n_samples: int = 1,
        sampler: str = "ddim",
        smooth_radius: float = 3.0,
    ) -> np.ndarray:
        """Edit a specific region using Latent Blending (State-Space Mixing).

        Implements Latent Blending for Ground-Truth Preservation:
        z_{t-1} = M ⊙ z_{t-1}^{pred} + (1-M) ⊙ z_{t-1}^{GT}

        Key Mechanism:
        - Foreground (z^{pred}): Denoised from TEdit conditioned on edit prompt.
        - Background (z^{GT}): Forward-diffused directly from original time series.
        - Blending: Soft-mask fusion ensures seamless transition without 'cliff effect'.

        Why Latent Blending (NOT Noise Blending):
        - Noise Blending: ε_blend = M ⊙ ε_tgt + (1-M) ⊙ ε_src
          → Background is "predicted", causes reconstruction error
        - Latent Blending: z_blend = M ⊙ z^{pred} + (1-M) ⊙ z^{GT}
          → Background is "physical truth" (forward-diffused from original data)
          → Ensures 100% background fidelity (zero reconstruction error)

        Args:
            ts: Input time series (shape: [L] or [B, L])
            start_idx: Start index of region to edit (inclusive)
            end_idx: End index of region to edit (exclusive)
            src_attrs: Source attributes for background (Ground Truth condition)
            tgt_attrs: Target attributes for foreground (Edit condition)
            n_samples: Number of samples to generate
            sampler: Sampler type ("ddim" or "ddpm")
            smooth_radius: Radius for soft boundary smoothing (default: 3.0)

        Returns:
            Edited time series with smooth boundaries and 100% background fidelity
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("TEdit model is not loaded. Call load_model() first.")

        ts_array = np.asarray(ts, dtype=np.float32)
        if ts_array.ndim == 1:
            ts_array = ts_array.reshape(1, -1)

        B, L = ts_array.shape

        start_idx = max(0, start_idx)
        end_idx = min(L, end_idx)
        if start_idx >= end_idx:
            raise ValueError(f"Invalid region indices: [{start_idx}, {end_idx})")

        hard_mask = np.zeros(L, dtype=np.float32)
        hard_mask[start_idx:end_idx] = 1.0
        soft_mask = gaussian_filter1d(hard_mask, sigma=smooth_radius)

        src_attrs_array = np.asarray(src_attrs, dtype=np.int64)
        tgt_attrs_array = np.asarray(tgt_attrs, dtype=np.int64)

        with torch.no_grad():
            x = torch.from_numpy(ts_array).unsqueeze(1).to(self.device)
            
            src_attrs_tensor = torch.from_numpy(src_attrs_array).unsqueeze(0).repeat(B, 1).to(self.device)
            tgt_attrs_tensor = torch.from_numpy(tgt_attrs_array).unsqueeze(0).repeat(B, 1).to(self.device)
            tp = torch.zeros(B, L, device=self.device)

            batch = {
                "src_x": x.permute(0, 2, 1),
                "src_attrs": src_attrs_tensor,
                "tgt_attrs": tgt_attrs_tensor,
                "tgt_x": x.permute(0, 2, 1),
                "tp": tp,
            }

            samples = self.model.edit_soft(
                batch,
                n_samples=n_samples,
                sampler=sampler,
                soft_mask=soft_mask
            )

        edited_ts = samples.cpu().numpy().squeeze(1)
        
        if np.asarray(ts).ndim == 1:
            return edited_ts[0]
        return edited_ts


_tedit_instance: Optional[TEditWrapper] = None


def get_tedit_instance(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: str = "cuda:0",
    force_reload: bool = False,
) -> TEditWrapper:
    """Get or create a singleton TEdit instance.

    Args:
        model_path: Path to model checkpoint (required for first load)
        config_path: Path to config file (required for first load)
        device: Device to load model on
        force_reload: Force reload the model even if already loaded

    Returns:
        TEditWrapper instance
    """
    global _tedit_instance

    if _tedit_instance is None or force_reload:
        _tedit_instance = TEditWrapper(device=device)

        if model_path and config_path:
            _tedit_instance.load_model(model_path, config_path)

    return _tedit_instance
