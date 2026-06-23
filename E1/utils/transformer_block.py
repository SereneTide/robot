#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :transformer_block
# @Author    :SereneTide

r"""
  .--,       .--,
 ( (  \.---./  ) )
  '.__/o   o\__.'
     {=  ^  =}
      >  -  <
     /       \
    //       \\
   //|   .   |\\
   "'\       /'"_.-~^`'-.
      \  _  /--'         `
    ___)( )(___
   (((__) (__)))    高山仰止,景行行止.虽不能至,心向往之。

"""

import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .ffn import FeedForward
from .moe import LightweightMoE


class TransformerBlock(nn.Module):
    """Attention → MoE → FFN，每个子模块带残差 + LayerNorm。"""
    def __init__(self, d_model: int, n_heads: int, n_experts: int, top_k: int,
                 moe_hidden: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout,
        )
        self.moe = LightweightMoE(
            d_model=d_model, n_experts=n_experts,
            top_k=top_k, hidden_dim=moe_hidden,
            dropout=dropout,
        )
        self.ffn = FeedForward(
            d_model=d_model, hidden_dim=ffn_hidden,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.moe(x)
        x = self.ffn(x)
        return x
