#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model
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

  架构:
    Conv5 Backbone → 8×[Attention(12头) → MoE(8专家) → FFN] → Pool → Head

"""

from dataclasses import dataclass

import torch.nn as nn

from utils import TransformerBlock


# ── 模型配置 ──

@dataclass
class Config:
    # Conv backbone
    C1: int = 64; C2: int = 128; C3: int = 256; C4: int = 512; C5: int = 768
    kernels: tuple = (5, 5, 3, 3, 3)
    strides: tuple = (2, 2, 2, 2, 1)
    # Attention
    n_heads: int = 12
    d_model: int = 768               # = C5, 必须被 n_heads 整除
    # MoE
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_hidden: int = 768
    # FFN
    ffn_hidden: int = 768
    # Stacking
    num_blocks: int = 4               # 堆叠层数
    # Head
    head_hidden: int = 384
    # Regularization
    dropout: float = 0.1
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    grad_clip: float = 1.0


# ── 完整姿态估计网络 ──

class PoseEstimator(nn.Module):
    """基于 CSI 的人体姿态估计网络。

    Architecture:
        Conv5 Backbone → 8×TransformerBlock(Attention→MoE→FFN) → GlobalAvgPool → MLP Head
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        C = [config.C1, config.C2, config.C3, config.C4, config.C5]
        K = config.kernels
        S = config.strides

        # ── Conv Backbone (5层) ──
        in_chs = [20] + C[:4]
        conv_layers = []
        for i in range(5):
            conv_layers.append(nn.Conv1d(in_chs[i], C[i], kernel_size=K[i], stride=S[i]))
            conv_layers.append(nn.BatchNorm1d(C[i]))
            conv_layers.append(nn.ReLU())
        self.conv_backbone = nn.Sequential(*conv_layers)

        # ── 8× TransformerBlock ──
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_experts=config.moe_n_experts,
                top_k=config.moe_top_k,
                moe_hidden=config.moe_hidden,
                ffn_hidden=config.ffn_hidden,
                dropout=config.dropout,
            )
            for _ in range(config.num_blocks)
        ])

        # ── Regression Head ──
        self.head = nn.Sequential(
            nn.Linear(config.C5, config.head_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 30),
            nn.Sigmoid(),                            # 约束输出到 (0,1)，与归一化目标范围一致
        )

    def forward(self, x):
        # x: (B, 540, 20)
        x = x.permute(0, 2, 1)                        # (B, 20, 540)
        x = self.conv_backbone(x)                      # (B, C5, L5)
        x = x.permute(0, 2, 1)                         # (B, L5, C5)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)                              # (B, C5)
        x = self.head(x)                               # (B, 30)
        return x.view(-1, 15, 2)

    def collect_aux_losses(self) -> float:
        """收集所有 MoE block 的负载均衡辅助损失之和。"""
        total = 0.0
        for block in self.blocks:
            aux = getattr(block.moe, 'aux_loss', None)
            if aux is not None:
                total = total + aux
        return total
