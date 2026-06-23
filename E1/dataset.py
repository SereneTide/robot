#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @IDEName   :PyCharm
# @FileName  :dataset
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

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── 常量 ──
IMG_W, IMG_H = 1280, 720
BATCH_SIZE = 8
RANDOM_SEED = 42


class E1PoseDataset(Dataset):
    def __init__(self, window_indices, csi_mmap, keypoints):
        self.window_indices = np.asarray(window_indices, dtype=np.int32)
        self.csi_mmap = csi_mmap
        self.keypoints = keypoints

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        win_idx = self.window_indices[idx]
        csi = self.csi_mmap[win_idx].copy()
        np.nan_to_num(csi, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(csi, 0.0, 1.0, out=csi)                                  # CSI已归一化，截断到[0,1]
        csi_tensor = torch.from_numpy(csi.astype(np.float32)).float()     # (540, 20)
        kp = self.keypoints[win_idx].astype(np.float32).copy()            # (15, 2) 已归一化到[0,1]
        kp_tensor = torch.from_numpy(kp).float()
        return csi_tensor, kp_tensor


def _verify_subject3_range(file_info, window_ranges):
    """验证 Subject 3 对应的样本范围是否正确（1-indexed Subject 3 = 样本 200-299）。

    优先使用 file_info / file_mappings 中的受试者标签验证；
    若无可用标签，则假定样本按受试者顺序排列（每受试者 100 样本）并打印提示。"""
    expected_start, expected_end = 200, 299

    # 尝试从 file_info 中查找受试者信息
    for key in ('subject_ids', 'subjects', 'subject_id', 'subject'):
        if key in file_info:
            subjects = np.asarray(file_info[key])
            if subjects.ndim == 1 and len(subjects) == 500:
                test_subjects = subjects[expected_start:expected_end + 1]
                if np.all(test_subjects == 3):
                    print(f"  [OK] Subject 3 范围验证通过 (样本 {expected_start}-{expected_end})")
                else:
                    print(f"  [WARN] Subject 3 范围不匹配! 预期样本 {expected_start}-{expected_end}"
                          f" 为 Subject 3, 实际标签: {np.unique(test_subjects)}")
                return

    # 尝试从 file_mappings.pkl 查找受试者信息
    mapping_path = os.path.join(os.path.dirname(__file__), "data", "file_mappings.pkl")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "rb") as f:
                mappings = pickle.load(f)
            if hasattr(mappings, 'get') and 'subject' in mappings:
                subjects = np.asarray(mappings['subject'])
                if len(subjects) == 500:
                    test_subjects = subjects[expected_start:expected_end + 1]
                    if np.all(test_subjects == 3):
                        print(f"  [OK] Subject 3 范围验证通过 (样本 {expected_start}-{expected_end})")
                    else:
                        print(f"  [WARN] Subject 3 范围不匹配!")
                    return
            elif isinstance(mappings, np.ndarray) and mappings.shape[0] == 500:
                print(f"  [INFO] file_mappings.pkl 已加载但结构未识别，使用默认范围")
                return
        except Exception:
            pass

    print(f"  [INFO] 未找到受试者标签元数据，"
          f"假定样本按顺序排列: Subject 3 = 样本 {expected_start}-{expected_end}")


def create_dataloaders(setting, max_samples=None):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csi_mmap = np.load(os.path.join(data_dir, "csi_windows.npy"), mmap_mode="r")
    keypoints = np.load(os.path.join(data_dir, "all_keypoints.npy"))           # (360000, 15, 2)
    file_info = dict(np.load(os.path.join(data_dir, "file_info.npz"),
                             allow_pickle=True))
    window_ranges = file_info["window_ranges"]                                  # (500, 2)

    all_indices = np.arange(len(keypoints))
    if setting == 1:
        # 随机划分: 72% 训练 / 8% 验证 / 20% 测试
        rng = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(all_indices)
        n_test = int(0.2 * len(all_indices))
        n_val = int(0.08 * len(all_indices))
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]
        print(f"  Setting 1 | train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    elif setting == 2:
        # LOSO: Subject 3（1-indexed）完全隔离为测试集
        _verify_subject3_range(file_info, window_ranges)
        test_start = window_ranges[200, 0]
        test_end = window_ranges[299, 1]
        test_idx = np.arange(test_start, test_end, dtype=np.int32)
        rest = np.setdiff1d(all_indices, test_idx)
        rng = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(rest)
        split = int(0.9 * len(rest))
        train_idx = perm[:split]
        val_idx = perm[split:]
        print(f"  Setting 2 | train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        raise ValueError(f"Unknown setting: {setting}")

    # ── 子集截取（debug 模式） ──
    if max_samples is not None and max_samples > 0:
        rng_sub = np.random.default_rng(RANDOM_SEED)
        train_idx = train_idx[rng_sub.choice(len(train_idx), min(max_samples, len(train_idx)), replace=False)]
        val_idx = val_idx[rng_sub.choice(len(val_idx), min(max_samples // 5, len(val_idx)), replace=False)]
        test_idx = test_idx[rng_sub.choice(len(test_idx), min(max_samples // 3, len(test_idx)), replace=False)]
        print(f"  [DEBUG] Subsampled → train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    common_kwargs = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    train_ds = E1PoseDataset(train_idx, csi_mmap, keypoints)
    train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)

    val_ds = E1PoseDataset(val_idx, csi_mmap, keypoints)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)

    test_ds = E1PoseDataset(test_idx, csi_mmap, keypoints)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
