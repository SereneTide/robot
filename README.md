# 智能决策实验 — 机械臂路径规划与避障

基于 **PyBullet** 的 KUKA IIWA 机械臂仿真，实现三维空间中的路径规划与碰撞避障。

## 实验内容

| 目录 | 说明 |
|------|------|
| `work1` | 基础仿真环境搭建 |
| `work2` | 二维路径规划（A* + RRT） |
| `work3` | 三维障碍物场景下的 RRT 路径规划与碰撞恢复 |

## 使用

```bash
python main.py
```

依赖：`pybullet`, `numpy`, `matplotlib`
