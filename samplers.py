"""Diffusion Samplers for TEdit.

This module provides DDPM and DDIM samplers for diffusion-based time series editing.
"""

import torch
import torch.nn as nn
import numpy as np


class DDPMSampler:
    """DDPM (Denoising Diffusion Probabilistic Models) Sampler.
    
    Implements the forward and reverse diffusion process for DDPM.
    """
    
    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.device = device
        
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        elif schedule == "quad":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, device=device) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def forward(self, x, t, noise=None):
        """Forward diffusion process: add noise to x at timestep t.
        
        Args:
            x: Clean data tensor (B, K, L)
            t: Timestep tensor (B,)
            noise: Optional noise tensor, will be sampled if None
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
    
    def reverse(self, xt, pred_noise, t, noise=None):
        """Reverse diffusion process: denoise xt by one step.
        
        Args:
            xt: Noisy data at timestep t
            pred_noise: Predicted noise
            t: Timestep tensor
            noise: Optional noise for stochastic sampling
            
        Returns:
            Denoised data at timestep t-1
        """
        if noise is None:
            noise = torch.randn_like(xt)
        
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        mean = sqrt_recip_alpha_t * (xt - beta_t * pred_noise / sqrt_one_minus_alpha_t)
        
        if t[0] > 0:
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1)
            return mean + torch.sqrt(posterior_var_t) * noise
        else:
            return mean


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) Sampler.
    
    Implements the deterministic DDIM sampling process.
    """
    
    def __init__(
        self,
        num_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.device = device
        
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        elif schedule == "quad":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, device=device) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def forward(self, x, pred_noise, t):
        """DDIM forward/inversion process.
        
        Args:
            x: Data tensor
            pred_noise: Predicted noise
            t: Timestep tensor
            
        Returns:
            Noisy data at next timestep
        """
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        if isinstance(pred_noise, int) and pred_noise == 0:
            return sqrt_alpha_t * x
        else:
            return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * pred_noise
    
    def reverse(self, xt, pred_noise, t, noise=None, is_determin=True, eta=0.0):
        """DDIM reverse process: denoise xt by one step.
        
        Args:
            xt: Noisy data at timestep t
            pred_noise: Predicted noise
            t: Timestep tensor
            noise: Optional noise for stochastic DDIM
            is_determin: Whether to use deterministic sampling
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            
        Returns:
            Denoised data at timestep t-1
        """
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x0_pred = (xt - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
        
        if t[0] > 0:
            sqrt_alpha_t_prev = self.sqrt_alphas_cumprod[t - 1].view(-1, 1, 1)
            sqrt_one_minus_alpha_t_prev = self.sqrt_one_minus_alphas_cumprod[t - 1].view(-1, 1, 1)
            
            if is_determin or noise is None:
                return sqrt_alpha_t_prev * x0_pred
            else:
                sigma = eta * torch.sqrt(
                    (1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t]) * 
                    (1 - self.alphas_cumprod[t] / self.alphas_cumprod[t - 1])
                ).view(-1, 1, 1)
                return sqrt_alpha_t_prev * x0_pred + sqrt_one_minus_alpha_t_prev * pred_noise + sigma * noise
        else:
            return x0_pred
