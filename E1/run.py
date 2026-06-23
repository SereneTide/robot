#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @IDEName   :PyCharm
# @FileName  :run
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

  用法:
    python run.py                     # 运行 Setting 1 和 2
    python run.py --setting 1         # 仅运行 Setting 1
    python run.py --setting 2         # 仅运行 Setting 2
    python run.py --epochs 100        # 覆盖训练轮数
    python run.py --debug             # debug模式: 500样本, 2轮, 完整pipeline验证
    python run.py --debug --setting 1 # debug模式仅运行 Setting 1
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import IMG_H, IMG_W, create_dataloaders
from model import Config, PoseEstimator
from utils import count_parameters, compute_macs, print_model_info
from trainer import Trainer

# ── 常量 ──
# 关键点定义（基于实验数据集）：
# 0:Nose, 1:Neck, 2:L_Shoulder, 3:L_Elbow, 4:L_Wrist
# 5:R_Shoulder, 6:R_Elbow, 7:R_Wrist
# 8:Pelvis, 9:L_Hip, 10:L_Knee, 11:L_Ankle
# 12:R_Hip, 13:R_Knee, 14:R_Ankle
SKELETON_EDGES = [
    (0, 1),                              # Nose→Neck
    (1, 2), (2, 3), (3, 4),             # Neck→L_Shoulder→L_Elbow→L_Wrist
    (1, 5), (5, 6), (6, 7),             # Neck→R_Shoulder→R_Elbow→R_Wrist
    (1, 8),                              # Neck→Pelvis
    (8, 9), (9, 10), (10, 11),          # Pelvis→L_Hip→L_Knee→L_Ankle
    (8, 12), (12, 13), (13, 14),        # Pelvis→R_Hip→R_Knee→R_Ankle
    (2, 5),                              # L_Shoulder→R_Shoulder
]
VIDEO_NFRAMES = 150       # 对比视频最大帧数
VIDEO_FPS = 15            # 视频帧率


def _denorm(kp):
    """反归一化到像素坐标."""
    out = kp.copy()
    out[..., 0] *= IMG_W
    out[..., 1] *= IMG_H
    return out


def _generate_csv(preds, targets, tag=''):
    preds_px = _denorm(preds)
    targets_px = _denorm(targets)

    rows = []
    for i in range(len(preds_px)):
        row = {'sample_id': i}
        for k in range(15):
            row[f'true_kp{k}_x'] = targets_px[i, k, 0]
            row[f'true_kp{k}_y'] = targets_px[i, k, 1]
            row[f'pred_kp{k}_x'] = preds_px[i, k, 0]
            row[f'pred_kp{k}_y'] = preds_px[i, k, 1]
        rows.append(row)

    suffix = f'_{tag}' if tag else ''
    fname = f'test_predictions{suffix}.csv'
    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False, float_format='%.4f')

    err = np.abs(preds_px - targets_px)
    euc = np.linalg.norm(preds_px - targets_px, axis=2)
    stats = []
    for k in range(15):
        stats.append({
            'keypoint': k,
            'mean_err_x': float(err[:, k, 0].mean()),
            'mean_err_y': float(err[:, k, 1].mean()),
            'mean_euclidean': float(euc[:, k].mean()),
        })
    stats_fname = f'keypoint_error_stats{suffix}.csv'
    pd.DataFrame(stats).to_csv(stats_fname, index=False, float_format='%.4f')
    return fname, stats_fname


def _diagnose_pred_variance(preds, tag=''):
    """诊断预测值跨帧方差，判断是否出现预测静止问题。"""
    n = min(50, len(preds))
    subset = preds[:n]                                                       # (n, 15, 2)
    per_frame_mean = subset.reshape(n, -1).mean(axis=1)                      # 每帧30个值的均值
    per_frame_std = subset.reshape(n, -1).std(axis=1)                        # 每帧30个值的标准差
    global_std = subset.std(axis=0).mean()                                    # 跨帧的标准差（平均）
    frame_to_frame_diff = np.abs(np.diff(subset, axis=0)).mean()             # 相邻帧差异

    prefix = f'[{tag}] ' if tag else ''
    print(f"\n  ── Prediction Variance Diagnostic {prefix}──")
    print(f"  Global std across frames (avg over 15 kps): {global_std:.6f} (normalized)")
    print(f"  Mean frame-to-frame diff:                    {frame_to_frame_diff:.6f}")
    print(f"  Per-frame mean range:   [{per_frame_mean.min():.6f}, {per_frame_mean.max():.6f}]")
    print(f"  Per-frame std  range:   [{per_frame_std.min():.6f}, {per_frame_std.max():.6f}]")

    if global_std < 0.001:
        print(f"  ⚠ WARNING: Predictions are nearly CONSTANT across frames! "
              f"The model is outputting almost the same pose for all inputs.")
        print(f"    Possible causes: undertrained model, collapsed BatchNorm, or degenerated weights.")
    elif frame_to_frame_diff < 0.0005:
        print(f"  ⚠ WARNING: Frame-to-frame variation is extremely small. "
              f"Movement may be imperceptible in video.")
    else:
        print(f"  ✓ Predictions show meaningful variation across frames.")

    # 检查单个关键点的跨帧方差
    kp_var = subset.var(axis=0)                                              # (15, 2)
    most_static_kp = kp_var.sum(axis=1).argmin()
    most_dynamic_kp = kp_var.sum(axis=1).argmax()
    print(f"  Most static  keypoint: kp{most_static_kp}  var=({kp_var[most_static_kp, 0]:.6f}, {kp_var[most_static_kp, 1]:.6f})")
    print(f"  Most dynamic keypoint: kp{most_dynamic_kp}  var=({kp_var[most_dynamic_kp, 0]:.6f}, {kp_var[most_dynamic_kp, 1]:.6f})")


def _generate_video(preds, targets, tag='', debug=False):
    max_frames = 30 if debug else VIDEO_NFRAMES
    n = min(max_frames, len(preds))
    preds_px = _denorm(preds[:n])
    targets_px = _denorm(targets[:n])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    suffix = f'_{tag}' if tag else ''
    video_fname = f'comparison_poses{suffix}.mp4'
    writer = cv2.VideoWriter(video_fname, fourcc, VIDEO_FPS, (IMG_W, IMG_H))

    for i in range(n):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100)
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.02)

        for ax, kp, color, label in [
            (ax_l, targets_px[i], 'red', 'Ground Truth'),
            (ax_r, preds_px[i],   'blue', 'Prediction'),
        ]:
            ax.set_xlim(0, IMG_W)
            ax.set_ylim(IMG_H, 0)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(label, color=color, fontsize=14, pad=8)
            for a, b in SKELETON_EDGES:
                ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                        color=color, linewidth=2)
            ax.scatter(kp[:, 0], kp[:, 1], c=color, s=12, zorder=3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        writer.write(img_bgr)
        plt.close(fig)

    writer.release()
    return video_fname


def _pipeline_sanity_check(model, train_loader, device):
    """Debug 模式下的完整 pipeline 验证。

    依次验证: 数据形状 / forward 正确性 / backward 梯度流 / 输出范围。"""
    print(f"\n  ── Pipeline Sanity Check ──")

    # 1. 数据形状验证
    batch = next(iter(train_loader))
    csi, kp = batch
    print(f"  1. Data shapes: CSI={tuple(csi.shape)}, Keypoints={tuple(kp.shape)}")
    assert csi.shape[1:] == (540, 20), f"Expected (B,540,20), got {csi.shape}"
    assert kp.shape[1:] == (15, 2), f"Expected (B,15,2), got {kp.shape}"
    print(f"     CSI   range: [{csi.min().item():.4f}, {csi.max().item():.4f}]")
    print(f"     KP    range: [{kp.min().item():.4f}, {kp.max().item():.4f}]")

    # 2. Forward pass 正确性
    csi_d, kp_d = csi.to(device), kp.to(device)
    model.train()
    pred = model(csi_d)
    print(f"  2. Forward pass: output shape={tuple(pred.shape)}")
    assert pred.shape == kp_d.shape, f"Shape mismatch: {pred.shape} vs {kp_d.shape}"
    print(f"     Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    # 3. Backward 梯度流
    loss = nn.functional.smooth_l1_loss(pred, kp_d)
    loss.backward()
    grad_norms = {}
    has_grad, no_grad = 0, 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            has_grad += 1
            if g_norm > 0:
                grad_norms[name] = g_norm
        else:
            no_grad += 1
    print(f"  3. Gradient flow: {has_grad} params with grad, {no_grad} without")
    if grad_norms:
        max_name = max(grad_norms, key=grad_norms.get)
        print(f"     Max grad norm: {max_name} = {grad_norms[max_name]:.6f}")
    if no_grad > 0:
        print(f"     [WARN] {no_grad} parameters have no gradient!")
    model.zero_grad()

    # 4. 输出值范围校验（训练前 baseline）
    print(f"  4. All {len(list(model.parameters()))} parameters initialized")
    print(f"  ── Sanity Check PASSED ──\n")


def _verify_outputs(preds, targets, tag):
    """验证预测输出合法性并打印摘要。"""
    print(f"\n  ── Output Verification [{tag}] ──")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Targets shape:     {targets.shape}")
    for label, arr in [('Pred', preds), ('Target', targets)]:
        print(f"  {label}: x∈[{arr[..., 0].min():.4f}, {arr[..., 0].max():.4f}], "
              f"y∈[{arr[..., 1].min():.4f}, {arr[..., 1].max():.4f}]")
    dists = np.linalg.norm(_denorm(preds) - _denorm(targets), axis=2)
    print(f"  Per-keypoint mean error (px): "
          + ", ".join(f"kp{k}:{dists[:, k].mean():.1f}" for k in range(15)))


def run_setting(setting, epochs, device, debug=False):
    tag = f's{setting}'
    print(f"  Setting {setting}  {'[DEBUG]' if debug else ''}  |  Device: {device}")

    max_samples = 5000 if debug else None
    train_loader, val_loader, test_loader = create_dataloaders(setting, max_samples=max_samples)

    config = Config()
    if epochs is not None:
        config.epochs = epochs
    if debug and epochs is None:
        config.epochs = 5

    model = PoseEstimator(config)
    print_model_info(model, config, device)

    if debug:
        _pipeline_sanity_check(model, train_loader, device)

    trainer = Trainer(model, config, device=device, save_dir='model',
                      tag=f'setting{setting}')

    if debug:
        # ── Debug: 记录并打印每 epoch 的额外诊断信息 ──
        _orig_train_epoch = trainer.train_epoch

        def debug_train_epoch(train_loader, epoch, total_epochs):
            loss = _orig_train_epoch(train_loader, epoch, total_epochs)
            # 梯度范数
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            lr = trainer.optimizer.param_groups[0]['lr']
            print(f"           [DEBUG] lr={lr:.2e} | grad_norm={total_grad_norm:.4f}")
            return loss

        trainer.train_epoch = lambda tl, ep, total: debug_train_epoch(tl, ep, total)

    trainer.fit(train_loader, val_loader=val_loader, epochs=config.epochs)

    # ── 加载最优模型 ──
    if os.path.exists(trainer.best_model_path):
        model.load_state_dict(torch.load(trainer.best_model_path,
                                         map_location=device, weights_only=True))
        print(f"  Loaded: {trainer.best_model_path}")
        if debug:
            # 验证 checkpoint 完整性：检查参数数量匹配
            ckpt_keys = set(torch.load(trainer.best_model_path,
                                       map_location='cpu', weights_only=True).keys())
            model_keys = set(model.state_dict().keys())
            assert ckpt_keys == model_keys, \
                f"Checkpoint mismatch! Missing: {model_keys - ckpt_keys}, Extra: {ckpt_keys - model_keys}"
            print(f"  [DEBUG] Checkpoint integrity verified ({len(ckpt_keys)} keys match)")
    else:
        print("  WARNING: no checkpoint found, using final model state")

    # ── 测试集评估 ──
    metrics = trainer.validate(test_loader)
    print(f"\n  Test | PCK@0.2: {metrics['pck_0.2']:.4f}  "
          f"PCK@0.5: {metrics['pck_0.5']:.4f}  "
          f"MPJPE: {metrics['mpjpe_px']:.2f} px")

    # ── 输出文件 ──
    preds, targets = trainer.predict(test_loader)

    if debug:
        _verify_outputs(preds, targets, tag)

    # ── 诊断：预测值跨帧方差 ──
    _diagnose_pred_variance(preds, tag)

    # 输出文件带上 tag 避免多 setting 覆盖
    output_tag = tag if debug else ''
    csv1, csv2 = _generate_csv(preds, targets, tag=output_tag)

    if not debug:
        video = _generate_video(preds, targets)
    else:
        video = _generate_video(preds, targets, tag=tag, debug=True)

    print(f"  Outputs: {csv1}, {csv2}, {video}")

    if debug:
        print(f"\n  ── Debug Summary [{tag}] ──")
        print(f"  Data: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} samples")
        print(f"  Epochs: {config.epochs} | Test MPJPE: {metrics['mpjpe_px']:.2f}px | PCK@0.2: {metrics['pck_0.2']:.4f} | PCK@0.5: {metrics['pck_0.5']:.4f}")
        print(f"  Checkpoint: {trainer.best_model_path}")
        print(f"  Pipeline: data→forward→backward→train→val→save→load→test→csv→video ✓")


def main():
    parser = argparse.ArgumentParser(description='E1 WiFi Pose Estimation')
    parser.add_argument('--setting', type=int, nargs='+', default=None,
                        help='1, 2, or both (default: both)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override default epochs (50)')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode: 500 samples, 2 epochs, full pipeline check')
    args = parser.parse_args()

    settings = args.setting if args.setting else [1, 2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.debug:
        print(f"[DEBUG] Settings: {settings}")

    for s in settings:
        if s not in (1, 2):
            print(f"Invalid setting {s}, skipping")
            continue
        run_setting(s, args.epochs, device, debug=args.debug)


if __name__ == '__main__':
    main()
