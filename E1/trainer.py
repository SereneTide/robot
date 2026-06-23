#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :trainer
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

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import Config

# ── 常量 ──
IMG_W, IMG_H = 1280, 720
MOE_AUX_COEF = 0.01          # MoE 负载均衡损失系数


class Trainer:
    def __init__(self, model, config: Config, device='cpu', save_dir='model', tag=''):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 文件路径：tag 用于区分不同 setting
        suffix = f'_{tag}' if tag else ''
        self.best_model_path = os.path.join(save_dir, f'best{suffix}.pth')
        self.ckpt_path = os.path.join(save_dir, f'checkpoint{suffix}.pth')

        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    def train_epoch(self, train_loader, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}',
                    leave=False, ncols=100, unit='b')
        for csi, kp in pbar:
            csi, kp = csi.to(self.device), kp.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(csi)
            loss = self.criterion(pred, kp)
            # MoE 负载均衡辅助损失（跨所有 block 求和）
            moe_aux = self.model.collect_aux_losses()
            if moe_aux:
                loss = loss + MOE_AUX_COEF * moe_aux
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        self.scheduler.step()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for csi, kp in loader:
            csi, kp = csi.to(self.device), kp.to(self.device)
            pred = self.model(csi)
            loss = self.criterion(pred, kp)
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(kp.cpu())

        preds = torch.cat(all_preds, dim=0)            # (N, 15, 2)
        targets = torch.cat(all_targets, dim=0)        # (N, 15, 2)

        # Torso diameter: midpoint(shoulders) → midpoint(hips)
        shoulder_mid = (targets[:, 2] + targets[:, 5]) / 2   # 2=L_Shoulder, 5=R_Shoulder
        hip_mid = (targets[:, 9] + targets[:, 12]) / 2       # 9=L_Hip, 12=R_Hip
        torso_dia = torch.norm(shoulder_mid - hip_mid, dim=1)  # (N,)

        dists = torch.norm(preds - targets, dim=2)             # (N, 15)

        pck_0_2 = (dists < 0.2 * torso_dia.unsqueeze(1)).float().mean().item()
        pck_0_5 = (dists < 0.5 * torso_dia.unsqueeze(1)).float().mean().item()

        # MPJPE: denormalize to pixel coords
        preds_px = preds.clone()
        targets_px = targets.clone()
        preds_px[:, :, 0] *= IMG_W
        preds_px[:, :, 1] *= IMG_H
        targets_px[:, :, 0] *= IMG_W
        targets_px[:, :, 1] *= IMG_H
        mpjpe = torch.norm(preds_px - targets_px, dim=2).mean().item()

        return {
            'loss': total_loss / len(loader),
            'pck_0.2': pck_0_2,
            'pck_0.5': pck_0_5,
            'mpjpe_px': mpjpe,                     # unit: pixel
        }

    def fit(self, train_loader, val_loader=None, epochs=None):
        epochs = epochs or self.config.epochs
        start_epoch = 0
        best_mpjpe = float('inf')

        # ── 尝试恢复训练状态 ──
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mpjpe = ckpt['best_mpjpe']
            print(f"  Resumed from epoch {start_epoch}, best_mpjpe={best_mpjpe:.2f}")

        # (重新)创建 scheduler 以适配可能的 epochs 覆盖
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        if start_epoch > 0:
            # 将 scheduler 推进到正确位置
            for _ in range(start_epoch):
                self.scheduler.step()

        for epoch in range(start_epoch, epochs):
            train_loss = self.train_epoch(train_loader, epoch, epochs)

            # ── 每轮摘要行 ──
            parts = [f'train_loss={train_loss:.4f}']

            if val_loader is not None:
                m = self.validate(val_loader)
                parts += [
                    f"v_loss={m['loss']:.4f}",
                    f"pck.2={m['pck_0.2']:.3f}",
                    f"pck.5={m['pck_0.5']:.3f}",
                    f"mpjpe={m['mpjpe_px']:.1f}",
                ]
                if m['mpjpe_px'] < best_mpjpe:
                    best_mpjpe = m['mpjpe_px']
                    torch.save(self.model.state_dict(), self.best_model_path)
                    parts.append(f'*best={best_mpjpe:.1f}')
            else:
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, f'model_epoch_{epoch + 1:03d}.pth'))

            print(f'  {" | ".join(parts)}')

            # 保存训练状态（每轮保存，支持中断恢复）
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_mpjpe': best_mpjpe,
            }, self.ckpt_path)

    @torch.no_grad()
    def predict(self, loader) -> tuple:
        self.model.eval()
        all_preds, all_targets = [], []
        for csi, kp in loader:
            csi = csi.to(self.device)
            pred = self.model(csi)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(kp.numpy())
        return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)
