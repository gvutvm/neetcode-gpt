import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = KVCache()

        # how many tokens were already cached before this call
        prev_len = 0 if kv_cache.cache_k is None else kv_cache.cache_k.shape[1]
        new_len = x.shape[1]

        q = self.q_proj(x)          # (batch, new_len, d)
        k = self.k_proj(x)          # (batch, new_len, d)
        v = self.v_proj(x)          # (batch, new_len, d)

        full_k, full_v = kv_cache.update(k, v)   # (batch, prev_len+new_len, d)
        total_len = full_k.shape[1]

        # Scaled dot-product attention
        scores = (q @ full_k.transpose(-2, -1)) * (full_k.shape[-1] ** -0.5)
        # (batch, new_len, total_len)

        # Causal mask: query i (absolute position prev_len + i) may attend to
        # keys 0 .. prev_len + i (inclusive), not to anything after that.
        q_positions = torch.arange(prev_len, prev_len + new_len).unsqueeze(1)  # (new_len, 1)
        k_positions = torch.arange(total_len).unsqueeze(0)                     # (1, total_len)
        causal_mask = k_positions > q_positions                                # True = mask out

        scores = scores.masked_fill(causal_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ full_v

        return torch.round(output, decimals=4), kv_cache