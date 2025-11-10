import math
from functools import partial
from itertools import repeat
from collections.abc import Callable
from typing import Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flash_attention_2_jvp import flash_attn_func as fa2_func
# from flash_attention_3_jvp import flash_attn_func as fa3_func


"""MLP layer"""
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, act_layer: Callable[..., nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
"""Modulation operation for adaptive layer norm zero (adaLN-Zero)"""
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


"""RMS Layer Normalization"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use in-place safe variant for less memory overhead
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True, dtype=torch.float32) + self.eps)
        return x * norm_x.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)


"""attention operator"""
def attn_op(q: Tensor, k: Tensor, v: Tensor, op: str = "base") -> Tensor:
    """
    op: ["default", "fa2", "fa3", "torch_sdpa"]
        - default: base attention implementation
        - fa2: flash attention v2 with jvp support
        - fa3: flash attention v3 with jvp support
        - torch_sdpa: PyTorch built-in scaled dot-product attention (no jvp support)
    input: q, k, v (B, L, H, D)
    output: (B, L, H, D)
    """
    if op == "fa2":
        x = fa2_func(q, k, v)
    elif op == "fa3":
        # x = fa3_func(q, k, v)
        raise NotImplementedError("Flash Attention v3 is not implemented in this snippet.")
    else:
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # change to (B, H, L, D)
        if op == "torch_sdpa":
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,  # No dropout for simplicity
                is_causal=False  # Not causal for this example
            )
        elif op == "base":
            scale = q.shape[-1] ** -0.5
            q = q * scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)  #- torch.max(attn, dim=-1, keepdim=True).values.detach()
            x = attn @ v
        x = x.transpose(1, 2)
    return x


"""Custom Attention Layer"""
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_func: str = "torch_sdpa",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn = partial(attn_op, op=attn_func)

    # @torch.compile
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(
            B, L, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 1, 3, 4)
        # shape (B, H, L, D)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        x = self.attn(q, k, v)
        x = x.reshape(B, L, C)
        x = self.proj(x)
        return x
    

"""SiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
class SiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs.get("qk_norm", False),
            attn_func=block_kwargs.get("attn_func", "base"),
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    # @torch.compile
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
