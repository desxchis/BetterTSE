from __future__ import annotations

import copy
import hashlib
import math
import re
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _format_scalar_key(value: float) -> str:
    return f"{float(value):.4f}"


def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2 or x.numel() != y.numel():
        return None
    x = x.detach().float().view(-1)
    y = y.detach().float().view(-1)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(torch.clamp(x_centered.square().sum() * y_centered.square().sum(), min=1.0e-12))
    if float(denom.item()) <= 1.0e-12:
        return None
    return float((x_centered * y_centered).sum().item() / denom.item())


def _safe_spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() < 2 or x.numel() != y.numel():
        return None
    x = x.detach().float().view(-1)
    y = y.detach().float().view(-1)
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    return _safe_corrcoef(x_rank, y_rank)


def pool_text_context(text_context: torch.Tensor) -> torch.Tensor:
    if text_context.dim() == 3:
        pooled = text_context.mean(dim=1)
    elif text_context.dim() == 2:
        pooled = text_context
    else:
        raise ValueError(f"Unexpected text_context shape: {tuple(text_context.shape)}")
    return pooled


def _stable_hash_token(token: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % max(1, int(num_buckets))


class InstructionContextEncoder(nn.Module):
    """Lightweight text encoder for pooled instruction context.

    This is intentionally small: token embedding + mean pooling. It provides the
    Kontext-style pooled text context without introducing a large standalone text
    encoder dependency.
    """

    def __init__(self, num_buckets: int, text_dim: int) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.text_dim = int(text_dim)
        self.token_emb = nn.Embedding(self.num_buckets, self.text_dim)
        self.null_context = nn.Parameter(torch.zeros(self.text_dim))

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(str(text).lower())

    def forward(self, text_list: Sequence[str], device: torch.device) -> torch.Tensor:
        pooled = []
        for text in text_list:
            tokens = self._tokenize(text)
            if not tokens:
                pooled.append(self.null_context)
                continue
            token_ids = torch.tensor(
                [_stable_hash_token(token, self.num_buckets) for token in tokens],
                dtype=torch.long,
                device=device,
            )
            pooled.append(self.token_emb(token_ids).mean(dim=0))
        return torch.stack(pooled, dim=0)


class StrengthProjector(nn.Module):
    """Project strength controls into a shared conditioning vector."""

    _diag_enabled = False
    _diag_records = []
    _diag_max_records = 32

    def __init__(
        self,
        num_strength_bins: int = 3,
        num_tasks: int = 8,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 64,
        use_text_context: bool = True,
        text_dim: int = 64,
        dropout: float = 0.0,
        use_task_id: bool = False,
        text_num_buckets: int = 4096,
        include_strength_scalar: bool = True,
    ) -> None:
        super().__init__()
        self.use_text_context = bool(use_text_context)
        self.use_task_id = bool(use_task_id)
        self.include_strength_scalar = bool(include_strength_scalar)

        self.strength_emb = nn.Embedding(num_strength_bins, emb_dim)
        self.strength_scalar_proj = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.task_emb = nn.Embedding(num_tasks, emb_dim) if self.use_task_id else None
        self.text_encoder = (
            InstructionContextEncoder(num_buckets=text_num_buckets, text_dim=text_dim)
            if self.use_text_context
            else None
        )

        in_dim = emb_dim
        if self.include_strength_scalar:
            in_dim += emb_dim
        if self.use_task_id:
            in_dim += emb_dim
        if self.use_text_context:
            in_dim += int(text_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.mlp[-1].bias, mean=0.0, std=1.0e-3)

    @classmethod
    def enable_diagnostics(cls, enabled: bool = True, max_records: int = 32) -> None:
        cls._diag_enabled = bool(enabled)
        cls._diag_max_records = max(1, int(max_records))
        cls._diag_records = []

    @classmethod
    def disable_diagnostics(cls) -> None:
        cls._diag_enabled = False

    @classmethod
    def consume_diagnostics(cls) -> list[dict[str, Any]]:
        records = copy.deepcopy(cls._diag_records)
        cls._diag_records = []
        return records

    def _record_diagnostics(
        self,
        strength_label: Optional[torch.Tensor],
        strength_scalar: Optional[torch.Tensor],
        projector_out: torch.Tensor,
        task_id: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor | Sequence[str] | str],
    ) -> None:
        if not self.__class__._diag_enabled:
            return
        if len(self.__class__._diag_records) >= self.__class__._diag_max_records:
            return

        with torch.no_grad():
            out_cpu = projector_out.detach().cpu().float()
            labels_cpu = None if strength_label is None else strength_label.detach().cpu().long().view(-1)
            scalar_cpu = None if strength_scalar is None else strength_scalar.detach().cpu().float().view(-1)
            pairwise = {}
            means = {}
            scalar_pairwise = {}
            scalar_means = {}
            full_norm_mean = float(out_cpu.norm(dim=-1).mean().item()) if out_cpu.numel() > 0 else 0.0
            full_norm_std = float(out_cpu.norm(dim=-1).std(unbiased=False).item()) if out_cpu.shape[0] > 1 else 0.0
            centered = out_cpu - out_cpu.mean(dim=0, keepdim=True)
            full_std_mean = float(centered.std(dim=0, unbiased=False).mean().item()) if out_cpu.shape[0] > 1 else 0.0
            if labels_cpu is not None:
                for label in sorted(set(labels_cpu.tolist())):
                    mask = labels_cpu == label
                    if int(mask.sum().item()) == 0:
                        continue
                    label_out = out_cpu[mask]
                    mean_vec = label_out.mean(dim=0)
                    means[str(label)] = {
                        "count": int(mask.sum().item()),
                        "norm": float(mean_vec.norm().item()),
                        "mean_abs": float(label_out.abs().mean().item()),
                        "std_mean": float(label_out.std(dim=0, unbiased=False).mean().item()) if label_out.shape[0] > 1 else 0.0,
                    }
                    pairwise[f"self_{label}"] = 0.0
                unique_labels = sorted(set(labels_cpu.tolist()))
                for idx, left in enumerate(unique_labels):
                    for right in unique_labels[idx + 1:]:
                        left_mask = labels_cpu == left
                        right_mask = labels_cpu == right
                        if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                            continue
                        left_mean = out_cpu[left_mask].mean(dim=0)
                        right_mean = out_cpu[right_mask].mean(dim=0)
                        diff = left_mean - right_mean
                        pairwise[f"{left}_{right}"] = float(diff.norm().item())
                        pairwise[f"{left}_{right}_mean_abs"] = float(diff.abs().mean().item())
            if scalar_cpu is not None:
                unique_scalars = sorted({float(value) for value in scalar_cpu.tolist()})
                for scalar_value in unique_scalars:
                    scalar_mask = torch.isclose(
                        scalar_cpu,
                        scalar_cpu.new_tensor(float(scalar_value)),
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                    if int(scalar_mask.sum().item()) == 0:
                        continue
                    scalar_out = out_cpu[scalar_mask]
                    mean_vec = scalar_out.mean(dim=0)
                    scalar_key = _format_scalar_key(scalar_value)
                    scalar_means[scalar_key] = {
                        "count": int(scalar_mask.sum().item()),
                        "norm": float(mean_vec.norm().item()),
                        "mean_abs": float(scalar_out.abs().mean().item()),
                        "std_mean": float(scalar_out.std(dim=0, unbiased=False).mean().item()) if scalar_out.shape[0] > 1 else 0.0,
                    }
                    scalar_pairwise[f"self_{scalar_key}"] = 0.0
                for idx, left in enumerate(unique_scalars):
                    for right in unique_scalars[idx + 1:]:
                        left_mask = torch.isclose(scalar_cpu, scalar_cpu.new_tensor(float(left)), atol=1.0e-6, rtol=0.0)
                        right_mask = torch.isclose(scalar_cpu, scalar_cpu.new_tensor(float(right)), atol=1.0e-6, rtol=0.0)
                        if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                            continue
                        left_mean = out_cpu[left_mask].mean(dim=0)
                        right_mean = out_cpu[right_mask].mean(dim=0)
                        diff = left_mean - right_mean
                        key = f"{_format_scalar_key(left)}_{_format_scalar_key(right)}"
                        scalar_pairwise[key] = float(diff.norm().item())
                        scalar_pairwise[f"{key}_mean_abs"] = float(diff.abs().mean().item())

            text_mode = "none"
            if isinstance(text_context, torch.Tensor):
                text_mode = "tensor"
            elif isinstance(text_context, str):
                text_mode = "string"
            elif isinstance(text_context, Sequence):
                text_mode = "sequence"

            record = {
                "labels": None if labels_cpu is None else labels_cpu.tolist(),
                "strength_scalar": None if scalar_cpu is None else scalar_cpu.tolist(),
                "task_ids": None if task_id is None else task_id.detach().cpu().long().view(-1).tolist(),
                "text_mode": text_mode,
                "projector_output_norm_mean": full_norm_mean,
                "projector_output_norm_std": full_norm_std,
                "projector_output_feature_std_mean": full_std_mean,
                "projector_output_mean_norm_by_strength": means,
                "projector_output_pairwise_l2": pairwise,
                "projector_output_mean_norm_by_scalar": scalar_means,
                "projector_output_pairwise_l2_by_scalar": scalar_pairwise,
            }
            if scalar_cpu is not None and scalar_cpu.numel() > 1:
                scalar_centered = scalar_cpu - scalar_cpu.mean()
                output_norms = out_cpu.norm(dim=-1)
                norm_centered = output_norms - output_norms.mean()
                denom = torch.sqrt(torch.clamp((scalar_centered.square().sum() * norm_centered.square().sum()), min=1.0e-12))
                record["projector_output_norm_scalar_corr"] = float((scalar_centered * norm_centered).sum().item() / denom.item())
            self.__class__._diag_records.append(record)

    def _encode_text_context(
        self,
        text_context: Optional[torch.Tensor | Sequence[str] | str],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_text_context:
            raise ValueError("text context requested but use_text_context is disabled")

        if text_context is None:
            return self.text_encoder.null_context.unsqueeze(0).expand(batch_size, -1)

        if isinstance(text_context, torch.Tensor):
            pooled = pool_text_context(text_context)
        elif isinstance(text_context, str):
            pooled = self.text_encoder([text_context] * batch_size, device=device)
        elif isinstance(text_context, Sequence):
            text_list = [str(item) for item in text_context]
            if len(text_list) != batch_size:
                raise ValueError(
                    f"text_context batch size mismatch: expected {batch_size}, got {len(text_list)}"
                )
            pooled = self.text_encoder(text_list, device=device)
        else:
            raise ValueError(f"Unsupported text_context type: {type(text_context)}")

        assert pooled.dim() == 2
        assert pooled.shape[0] == batch_size
        return pooled

    def forward(
        self,
        strength_label: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor | Sequence[str] | str] = None,
        strength_scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = None
        device = None

        if strength_label is not None:
            strength_label = strength_label.long().view(-1)
            batch_size = strength_label.shape[0]
            device = strength_label.device
            strength_vec = self.strength_emb(strength_label)
        else:
            if strength_scalar is not None:
                strength_scalar = strength_scalar.float().view(-1, 1)
                batch_size = strength_scalar.shape[0]
                device = strength_scalar.device
            elif task_id is not None:
                task_id = task_id.long().view(-1)
                batch_size = task_id.shape[0]
                device = task_id.device
            elif isinstance(text_context, torch.Tensor):
                pooled = pool_text_context(text_context)
                batch_size = pooled.shape[0]
                device = pooled.device
            elif isinstance(text_context, str):
                raise ValueError("String text_context requires strength_label or task_id for batch size")
            elif isinstance(text_context, Sequence):
                batch_size = len(text_context)
                if batch_size <= 0:
                    raise ValueError("text_context sequence must be non-empty")
                if self.text_encoder is not None:
                    device = self.text_encoder.null_context.device
                elif self.use_task_id and self.task_emb is not None:
                    device = self.task_emb.weight.device
                else:
                    raise ValueError("Cannot infer device for text-only strength conditioning")
            else:
                raise ValueError("StrengthProjector requires semantic conditioning input")
            if batch_size is None or device is None:
                raise ValueError("Could not infer batch/device for semantic strength conditioning")
            strength_vec = torch.zeros(batch_size, self.strength_emb.embedding_dim, device=device, dtype=self.strength_emb.weight.dtype)

        if strength_scalar is None:
            strength_scalar_vec = torch.zeros(batch_size, self.strength_emb.embedding_dim, device=device, dtype=self.strength_emb.weight.dtype)
        else:
            strength_scalar = strength_scalar.float().view(-1, 1).to(device)
            if strength_scalar.shape[0] != batch_size:
                raise ValueError(
                    f"strength_scalar batch size mismatch: expected {batch_size}, got {strength_scalar.shape[0]}"
                )
            strength_scalar_vec = self.strength_scalar_proj(strength_scalar)

        feats = [strength_vec]
        if self.include_strength_scalar:
            feats.append(strength_scalar_vec)

        if self.use_task_id:
            if task_id is None:
                task_id = torch.zeros(batch_size, device=device, dtype=torch.long)
            else:
                task_id = task_id.long().view(-1)
                if task_id.shape[0] != batch_size:
                    raise ValueError(
                        f"task_id batch size mismatch: expected {batch_size}, got {task_id.shape[0]}"
                    )
            feats.append(self.task_emb(task_id))

        if self.use_text_context:
            feats.append(
                self._encode_text_context(
                    text_context=text_context,
                    batch_size=batch_size,
                    device=device,
                )
            )

        x = torch.cat(feats, dim=-1)
        projector_out = self.mlp(x)
        self._record_diagnostics(
            strength_label=strength_label,
            strength_scalar=strength_scalar,
            projector_out=projector_out,
            task_id=task_id,
            text_context=text_context,
        )
        return projector_out
