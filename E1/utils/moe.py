#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @IDEName   :PyCharm
# @FileName  :moe
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


class LightweightMoE(nn.Module):
    def __init__(self, d_model=256, n_experts=4, top_k=2, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, n_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.aux_loss = None          # 当前 batch 的负载均衡损失

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        router_logits = self.router(x)                                     # (B, L, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # ── 负载均衡辅助损失 (Switch Transformer 公式) ──
        #   loss = N * Σ(f_i * P_i)
        #   f_i: expert i 接收到的 token 比例
        #   P_i: expert i 的平均路由概率
        density = router_probs.mean(dim=(0, 1))                            # (n_experts,)  ← P_i
        dispatch_mask = F.one_hot(
            router_probs.argmax(dim=-1), num_classes=self.n_experts
        ).float()                                                           # (B, L, n_experts)
        frac = dispatch_mask.mean(dim=(0, 1))                               # (n_experts,)  ← f_i
        self.aux_loss = self.n_experts * (frac * density).sum()

        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        flat_x = x.view(-1, C)
        flat_w = topk_weights.view(-1, self.top_k)
        flat_idx = topk_indices.view(-1, self.top_k)

        output = torch.zeros_like(flat_x)

        for i in range(self.n_experts):
            mask = (flat_idx == i)
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[i](flat_x[token_idx])
            weights = flat_w[token_idx, k_idx].unsqueeze(-1)
            output.scatter_add_(0, token_idx.unsqueeze(-1).expand(-1, C), expert_out * weights)

        output = output.view(B, L, C)
        return self.norm(residual + output)
