#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @IDEName   :PyCharm
# @FileName  :__init__.py
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

from .attention import MultiHeadSelfAttention
from .ffn import FeedForward
from .model_utils import compute_macs, count_parameters, print_model_info
from .moe import LightweightMoE
from .transformer_block import TransformerBlock
