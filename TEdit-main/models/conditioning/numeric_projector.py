from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


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
    ) -> None:
        super().__init__()
        self.use_text_context = bool(use_text_context)
        self.use_task_id = bool(use_task_id)

        self.strength_emb = nn.Embedding(num_strength_bins, emb_dim)
        self.task_emb = nn.Embedding(num_tasks, emb_dim) if self.use_task_id else None
        self.text_encoder = (
            InstructionContextEncoder(num_buckets=text_num_buckets, text_dim=text_dim)
            if self.use_text_context
            else None
        )

        in_dim = emb_dim
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

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

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
        strength_label: torch.Tensor,
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
            labels_cpu = strength_label.detach().cpu().long().view(-1)
            pairwise = {}
            means = {}
            full_norm_mean = float(out_cpu.norm(dim=-1).mean().item()) if out_cpu.numel() > 0 else 0.0
            full_norm_std = float(out_cpu.norm(dim=-1).std(unbiased=False).item()) if out_cpu.shape[0] > 1 else 0.0
            centered = out_cpu - out_cpu.mean(dim=0, keepdim=True)
            full_std_mean = float(centered.std(dim=0, unbiased=False).mean().item()) if out_cpu.shape[0] > 1 else 0.0
            for label in [0, 1, 2]:
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
            for left in [0, 1, 2]:
                for right in [left + 1, 2]:
                    if right <= left:
                        continue
                    left_mask = labels_cpu == left
                    right_mask = labels_cpu == right
                    if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
                        continue
                    left_mean = out_cpu[left_mask].mean(dim=0)
                    right_mean = out_cpu[right_mask].mean(dim=0)
                    diff = left_mean - right_mean
                    pairwise[f"{left}_{right}"] = float(diff.norm().item())
                    pairwise[f"{left}_{right}_mean_abs"] = float(diff.abs().mean().item())

            text_mode = "none"
            if isinstance(text_context, torch.Tensor):
                text_mode = "tensor"
            elif isinstance(text_context, str):
                text_mode = "string"
            elif isinstance(text_context, Sequence):
                text_mode = "sequence"

            self.__class__._diag_records.append(
                {
                    "labels": labels_cpu.tolist(),
                    "task_ids": None if task_id is None else task_id.detach().cpu().long().view(-1).tolist(),
                    "text_mode": text_mode,
                    "projector_output_norm_mean": full_norm_mean,
                    "projector_output_norm_std": full_norm_std,
                    "projector_output_feature_std_mean": full_std_mean,
                    "projector_output_mean_norm_by_strength": means,
                    "projector_output_pairwise_l2": pairwise,
                }
            )

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
        strength_label: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor | Sequence[str] | str] = None,
    ) -> torch.Tensor:
        strength_label = strength_label.long()
        strength_vec = self.strength_emb(strength_label)
        feats = [strength_vec]

        if self.use_task_id:
            if task_id is None:
                task_id = torch.zeros_like(strength_label)
            feats.append(self.task_emb(task_id.long()))

        if self.use_text_context:
            feats.append(
                self._encode_text_context(
                    text_context=text_context,
                    batch_size=strength_label.shape[0],
                    device=strength_label.device,
                )
            )

        x = torch.cat(feats, dim=-1)
        projector_out = self.mlp(x)
        self._record_diagnostics(
            strength_label=strength_label,
            projector_out=projector_out,
            task_id=task_id,
            text_context=text_context,
        )
        return projector_out
