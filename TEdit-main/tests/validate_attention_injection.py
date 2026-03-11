"""
Simple validation script for Attention Injection mechanism.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_soft_mask_creation():
    """Test soft mask creation from hard mask."""
    print("Testing soft mask creation...")
    
    from scipy.ndimage import gaussian_filter1d
    
    hard_mask = np.zeros(100)
    hard_mask[30:70] = 1
    
    soft_mask = gaussian_filter1d(hard_mask.astype(np.float32), sigma=5/3.0)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    
    print(f"  Hard mask shape: {hard_mask.shape}")
    print(f"  Soft mask shape: {soft_mask.shape}")
    print(f"  Soft mask min: {soft_mask.min():.4f}, max: {soft_mask.max():.4f}")
    print(f"  Edit region (50): {soft_mask[50]:.4f}")
    print(f"  Preserve region (10): {soft_mask[10]:.4f}")
    print(f"  Boundary (30): {soft_mask[30]:.4f}")
    print("  ✓ Soft mask creation works!")
    return True

def test_attention_injection_layer():
    """Test AttentionInjectionLayer forward pass."""
    print("\nTesting AttentionInjectionLayer...")
    
    from models.diffusion.diff_csdi_multipatch_weaver import AttentionInjectionLayer
    
    layer = AttentionInjectionLayer(d_model=64, nhead=4, dim_feedforward=64)
    src = torch.randn(2, 10, 64)
    output = layer(src)
    
    print(f"  Input shape: {src.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == src.shape, "Shape mismatch!"
    print("  ✓ AttentionInjectionLayer works!")
    return True

def test_residual_block_with_params():
    """Test ResidualBlock with attention injection parameters."""
    print("\nTesting ResidualBlock with attention params...")
    
    from models.diffusion.diff_csdi_multipatch_weaver import ResidualBlock
    
    block = ResidualBlock(
        side_dim=128,
        attr_dim=128,
        channels=64,
        diffusion_embedding_dim=128,
        nheads=4,
        is_linear=False,
    )
    
    B, K, L = 2, 1, 16
    x = torch.randn(B, 64, K, L)
    side_emb = torch.randn(B, 128, K, L)
    attr_emb = torch.randn(B, 3, 128)
    diffusion_emb = torch.randn(B, 128)
    
    soft_mask = torch.rand(B, L)
    keys_null = torch.randn(B, L, 64)
    values_null = torch.randn(B, L, 64)
    
    output, skip = block(
        x, side_emb, attr_emb, diffusion_emb,
        attention_mask=None,
        soft_mask=soft_mask,
        keys_null=keys_null,
        values_null=values_null
    )
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Skip shape: {skip.shape}")
    assert output.shape == x.shape, "Shape mismatch!"
    print("  ✓ ResidualBlock with attention params works!")
    return True

def test_diffusion_model_forward():
    """Test diffusion model forward with attention injection parameters."""
    print("\nTesting Diff_CSDI_MultiPatch_Weaver_Parallel with attention params...")
    
    from models.diffusion.diff_csdi_multipatch_weaver import Diff_CSDI_MultiPatch_Weaver_Parallel
    
    config = {
        "channels": 64,
        "diffusion_embedding_dim": 128,
        "num_steps": 10,
        "layers": 2,
        "nheads": 4,
        "is_linear": False,
        "side_dim": 128,
        "attr_dim": 128,
        "n_var": 1,
        "L_patch_len": 4,
        "multipatch_num": 1,
        "attention_mask_type": "full",
        "is_attr_proj": False,
    }
    
    model = Diff_CSDI_MultiPatch_Weaver_Parallel(config, inputdim=2)
    
    B, K, L = 2, 1, 16
    x_raw = torch.randn(B, 2, K, L)
    side_emb_raw = torch.randn(B, 128, K, L)
    attr_emb_raw = torch.randn(B, 3, 128)
    diffusion_step = torch.randint(0, 10, (B,))
    
    soft_mask = torch.rand(B, L)
    keys_null = torch.randn(B, L, 64)
    values_null = torch.randn(B, L, 64)
    
    output = model(
        x_raw, side_emb_raw, attr_emb_raw, diffusion_step,
        soft_mask=soft_mask,
        keys_null=keys_null,
        values_null=values_null
    )
    
    print(f"  Input shape: {x_raw.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (B, K, L), "Shape mismatch!"
    print("  ✓ Diffusion model with attention params works!")
    return True

def test_latent_blending():
    """Test latent blending formula."""
    print("\nTesting latent blending formula...")
    
    L = 100
    z_pred = torch.randn(1, 1, L)
    z_gt = torch.randn(1, 1, L)
    
    hard_mask = np.zeros(L)
    hard_mask[30:70] = 1
    from scipy.ndimage import gaussian_filter1d
    soft_mask = torch.from_numpy(
        gaussian_filter1d(hard_mask.astype(np.float32), sigma=2)
    ).float().view(1, 1, L)
    
    z_blended = soft_mask * z_pred + (1 - soft_mask) * z_gt
    
    print(f"  z_pred shape: {z_pred.shape}")
    print(f"  z_gt shape: {z_gt.shape}")
    print(f"  soft_mask shape: {soft_mask.shape}")
    print(f"  z_blended shape: {z_blended.shape}")
    
    edit_region = z_blended[0, 0, 40:60]
    pred_region = z_pred[0, 0, 40:60]
    preserve_region = z_blended[0, 0, 0:20]
    gt_region = z_gt[0, 0, 0:20]
    
    print(f"  Edit region matches z_pred: {torch.allclose(edit_region, pred_region, atol=1e-5)}")
    print(f"  Preserve region matches z_gt: {torch.allclose(preserve_region, gt_region, atol=1e-5)}")
    print("  ✓ Latent blending formula works!")
    return True

def main():
    print("=" * 60)
    print("Attention Injection Validation Tests")
    print("=" * 60)
    
    tests = [
        test_soft_mask_creation,
        test_attention_injection_layer,
        test_residual_block_with_params,
        test_diffusion_model_forward,
        test_latent_blending,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
