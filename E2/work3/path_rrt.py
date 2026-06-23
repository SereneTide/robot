"""
RRT快速扩展随机树避障路径规划算法。
"""
import numpy as np
import time
import random


def rrt_plan(grid_map, start_xy: tuple, goal_xy: tuple,
             step_size: float, max_iter: int,
             goal_threshold: float, goal_bias: float) -> tuple:
    """RRT路径规划。返回 (world_path, elapsed_time)。"""
    t0 = time.perf_counter()
    sx, sy = start_xy
    gx, gy = goal_xy

    su, sv = grid_map.world_to_grid(sx, sy)
    gu, gv = grid_map.world_to_grid(gx, gy)
    if grid_map.is_obstacle(su, sv):
        print("[RRT] 起点 (% .3f,% .3f) 位于障碍物上" % (sx, sy))
        return None, time.perf_counter() - t0
    if grid_map.is_obstacle(gu, gv):
        print("[RRT] 终点 (% .3f,% .3f) 位于障碍物上" % (gx, gy))
        return None, time.perf_counter() - t0

    x_range = (grid_map.x_min, grid_map.x_max)
    y_range = (grid_map.y_min, grid_map.y_max)

    nodes = [[sx, sy, -1]]

    for i in range(max_iter):
        if random.random() < goal_bias:
            rx, ry = gx, gy
        else:
            rx = random.uniform(*x_range)
            ry = random.uniform(*y_range)

        nearest_idx = _nearest(nodes, rx, ry)
        nx, ny = nodes[nearest_idx][0], nodes[nearest_idx][1]

        dx = rx - nx
        dy = ry - ny
        dist = np.hypot(dx, dy)
        if dist < 1e-9:
            continue
        ex = nx + (dx / dist) * min(step_size, dist)
        ey = ny + (dy / dist) * min(step_size, dist)

        if grid_map.is_line_collision(nx, ny, ex, ey):
            continue

        new_idx = len(nodes)
        nodes.append([ex, ey, nearest_idx])

        if np.hypot(ex - gx, ey - gy) <= goal_threshold:
            goal_idx = len(nodes)
            nodes.append([gx, gy, new_idx])
            path = _backtrack(nodes, goal_idx)
            elapsed = time.perf_counter() - t0
            print("[RRT] 规划成功, 路径长度=%d 节点, 迭代=%d, 耗时=%.4fs" %
                  (len(path), i + 1, elapsed))
            return path, elapsed

    elapsed = time.perf_counter() - t0
    print("[RRT] 规划失败, 达到最大迭代次数%d, 耗时=%.4fs" % (max_iter, elapsed))
    return None, elapsed


def _nearest(nodes: list, x: float, y: float) -> int:
    """寻找距离 (x,y) 最近的树节点索引。"""
    best_idx = 0
    best_dist = float('inf')
    for idx, (nx, ny, _) in enumerate(nodes):
        d = (nx - x) ** 2 + (ny - y) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _backtrack(nodes: list, goal_idx: int) -> list:
    """从目标节点回溯到起点，生成路径点列表 [(x,y), ...]。"""
    path = []
    cur = goal_idx
    while cur != -1:
        path.append((nodes[cur][0], nodes[cur][1]))
        cur = nodes[cur][2]
    path.reverse()
    return path
