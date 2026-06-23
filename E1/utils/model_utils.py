#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model_utils
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

import math

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数量。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def _conv1d_flops(c_in, c_out, k, l_out):
    """单层 Conv1d 的 MACs (Multiply-Accumulate)."""
    return c_in * c_out * k * l_out


def compute_macs(config, batch_size: int = 1) -> dict:
    """计算模型理论 MACs（以 batch_size=1 为基准）。

    返回各模块 MACs 及总计（单位：M）。"""
    C = [config.C1, config.C2, config.C3, config.C4, config.C5]
    K = list(config.kernels)
    S = list(config.strides)
    N = config.num_blocks
    d = config.d_model

    # ── 计算各卷积层输出长度 ──
    L = [540]
    for k, s in zip(K, S):
        L.append(math.floor((L[-1] - k) / s + 1))
    L_conv_out = L[1:]           # [L1, L2, L3, L4, L5]

    # ── Conv backbone MACs ──
    in_chs = [20] + C[:4]
    conv_macs = 0
    for i in range(5):
        conv_macs += _conv1d_flops(in_chs[i], C[i], K[i], L_conv_out[i])

    L_seq = L_conv_out[-1]        # L5

    # ── Self-Attention MACs (per block, xN) ──
    attn_proj_macs = 4 * L_seq * d * d
    attn_core_macs = 2 * L_seq * L_seq * d
    attn_macs = N * (attn_proj_macs + attn_core_macs)

    # ── MoE MACs (per block, xN) ──
    router_macs = L_seq * d * config.moe_n_experts
    expert_macs = config.moe_top_k * L_seq * (d * config.moe_hidden + config.moe_hidden * d)
    moe_macs = N * (router_macs + expert_macs)

    # ── FFN MACs (per block, xN) ──
    ffn_macs = N * L_seq * (d * config.ffn_hidden + config.ffn_hidden * d)

    # ── Regression Head MACs ──
    head_macs = (d * config.head_hidden
                 + config.head_hidden * 30)

    total_macs = conv_macs + attn_macs + moe_macs + ffn_macs + head_macs
    total_macs_batch = total_macs * batch_size

    return {
        'conv_macs': conv_macs / 1e6,
        'attn_macs': attn_macs / 1e6,
        'moe_macs': moe_macs / 1e6,
        'ffn_macs': ffn_macs / 1e6,
        'head_macs': head_macs / 1e6,
        'total_macs': total_macs / 1e6,
        'total_macs_batch': total_macs_batch / 1e6,
        'L_conv_out': L_conv_out,
    }


def print_model_info(model: nn.Module, config, device: str = 'cpu'):
    """打印模型参数与 FLOPs 统计。"""
    params = count_parameters(model)
    macs = compute_macs(config)
    flops = macs['total_macs'] * 2

    print(f"  Params: {params['total'] / 1e6:.2f}M  |  FLOPs: {flops:.2f}M  |  MACs: {macs['total_macs']:.2f}M")
