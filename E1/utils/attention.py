#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @IDEName   :PyCharm
# @FileName  :attention
# @Time      :2026/6/13
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scale = self.d_k ** 0.5
        attn = (Q @ K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        return self.norm(residual + out)
