from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

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
        return self.mlp(x)
