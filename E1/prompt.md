## conda虚拟环境
work E:/anaconda3/envs/work/python.exe

## 实验项目结构
实验1均存放在E1目录下
文件：
- dataset.py 自定义数据集类
- dataset_real.pkl 数据文件（训练用数据）
- trainer.py 训练类（依赖注入式架构）
- run.py 主运行程序
自定义包：
- utils 存放神经网络模块，如注意力头、moe等
模型权重目录：
- model/ 存放训练好的模型权重

## 神经网络设计

### 整体架构
采用 **Conv1D + Multi-Head Self-Attention + Lightweight MoE** 混合架构。
设计原则：**可配置驱动**，通过修改 Config 类参数即可控制模型容量，便于 CPU 测试与 GPU 扩展切换。

### 数据流
```
Input: (B, 540, 20)
  │  540 = 18(天线链路) × 30(OFDM子载波)  ← 空间/特征维度
  │  20  = 对齐30FPS的时序帧内高频CSI包数  ← 时间维度
  │
  ├─ Step 1: Permute → (B, 20, 540)
  │   (将时间步作为通道维，空间特征作为序列长度，适配 Conv1d)
  │
  ├─ Step 2: Asymmetric Conv Backbone（空间轴下采样，时间特征保持）
  │   Conv1D(20 → C1, k, s) → BN → ReLU     : (B, C1, L1)
  │   Conv1D(C1 → C2, k, s) → BN → ReLU      : (B, C2, L2)
  │   Conv1D(C2 → C3, k, s) → BN → ReLU      : (B, C3, L3)
  │   Conv1D(C3 → C4, k, s) → BN → ReLU      : (B, C4, L4)
  │
  ├─ Step 3: Permute → (B, L4, C4)
  │   Multi-Head Self-Attention（空间位置间交互建模）
  │   → LayerNorm → (B, L4, C4)
  │
  ├─ Step 4: Lightweight MoE（增加模型容量）
  │   Token-Wise: top-2 路由，N 个专家 FFN
  │   → LayerNorm → (B, L4, C4)
  │
  ├─ Step 5: Global Average Pooling（空间维聚合）
  │   → (B, C4)
  │
  └─ Step 6: Regression Head
       Linear(C4 → H) → ReLU → Dropout → Linear(H → 30)
       → reshape → (B, 15, 2)    ← 15个关键点 × (x, y)
```

### 默认配置（CPU 测试用）
| 参数 | 值 | 说明 |
|------|-----|------|
| C1, C2, C3, C4 | 32, 64, 128, 256 | 各层通道数 |
| kernel / stride | [5,5,3,3] / [2,2,2,2] | 空间下采样步长 |
| n_heads | 4 | 注意力头数 |
| d_model | 256 | 注意力隐层维度（=C4） |
| moe_n_experts | 4 | MoE 专家数 |
| moe_top_k | 2 | 每 token 激活专家数 |
| moe_hidden | 512 | 专家 FFN 隐层维度 |
| head_hidden | 128 | 回归头隐层 |
| dropout | 0.1 | Dropout 率 |
| **参数量** | **~2.5M** | CPU 可训练 |

### 扩展方向（手动切换到 GPU 时使用）
只需修改 Config 中的数值即可扩容，无需改动代码结构：

| 扩展手段 | 修改对象 | 效果 |
|----------|---------|------|
| 加宽通道 | C1~C4 ×2（如 64→128→256→512） | 参数量约×4 |
| 加深卷积 | 增加 Conv Block 数 | 感受野扩大 |
| 加多头数 | n_heads: 4→8 | 注意力粒度更细 |
| 堆叠注意力层 | 重复 Step3-4 N 次 | 深层语义交互 |
| 增加专家 | moe_n_experts: 4→8 | 模型容量↑ |
| 增大专家容量 | moe_hidden: 512→1024 | 专家能力↑ |

### Loss 与优化器
- **损失函数**：SmoothL1Loss (Huber, beta=1.0) — 对离群点鲁棒
- **优化器**：AdamW (lr=1e-3, weight_decay=1e-4)
- **学习率调度**：CosineAnnealingLR → 0
- **Batch Size**：CPU 默认 8，GPU 可上调至 32~64

### 评估指标
- **PCK@0.2**：预测关键点与真值距离 < 0.2 × 躯干直径
- **PCK@0.5**：距离 < 0.5 × 躯干直径
- **MPJPE**：所有关键点平均欧氏距离（mm）

### 文件组织与输出
- `model/best_setting1.pth` / `model/best_setting2.pth` — 训练权重
- `test_predictions.csv` — 真实/预测坐标对比（含 sample_id）
- `keypoint_error_stats.csv` — 各关键点误差统计
- `comparison_poses.mp4` — 姿态对比视频（仅测试集首样本）

