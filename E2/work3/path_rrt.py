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


# ==================== 3D RRT ====================

def _aabb_line_collision(ax: float, ay: float, az: float,
                         bx: float, by: float, bz: float,
                         cube_centers: list,
                         half_extents: tuple,
                         clearance: float = 0.0) -> bool:
    """检测线段 AB 是否与任意一个AABB立方体相交（Slab方法，含安全包络）。"""
    hx, hy, hz = half_extents
    margin = clearance
    for cx, cy, cz in cube_centers:
        cmin = (cx - hx - margin, cy - hy - margin, cz - hz - margin)
        cmax = (cx + hx + margin, cy + hy + margin, cz + hz + margin)

        dir_x = bx - ax
        dir_y = by - ay
        dir_z = bz - az

        tmin, tmax = 0.0, 1.0

        for d, o, mn, mx in [
            (dir_x, ax, cmin[0], cmax[0]),
            (dir_y, ay, cmin[1], cmax[1]),
            (dir_z, az, cmin[2], cmax[2]),
        ]:
            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    tmin = float('inf')
                    break
            else:
                t1 = (mn - o) / d
                t2 = (mx - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    break

        if tmin <= tmax:
            return True
    return False


def _point_in_cubes(px: float, py: float, pz: float,
                    cube_centers: list, half_extents: tuple,
                    clearance: float = 0.0) -> bool:
    """检测点是否在任意一个立方体内部（含安全包络）。"""
    hx, hy, hz = half_extents
    margin = clearance
    for cx, cy, cz in cube_centers:
        if (abs(px - cx) <= hx + margin and
                abs(py - cy) <= hy + margin and
                abs(pz - cz) <= hz + margin):
            return True
    return False


def rrt_plan_3d(cube_positions: list, half_extents: tuple,
                start_xyz: tuple, goal_xyz: tuple,
                bounds: tuple,
                step_size: float, max_iter: int,
                goal_threshold: float, goal_bias: float,
                clearance: float = 0.0) -> tuple:
    """三维RRT路径规划。

    参数:
        cube_positions:  障碍物立方体中心坐标列表 [(cx,cy,cz), ...]
        half_extents:    立方体半边长 (hx, hy, hz)
        start_xyz:       起点世界坐标 (sx, sy, sz)
        goal_xyz:        终点世界坐标 (gx, gy, gz)
        bounds:          采样范围 (x_min, x_max, y_min, y_max, z_min, z_max)
        step_size:       每一步最大扩展步长
        max_iter:        最大迭代次数
        goal_threshold:  判定到达目标的距离阈值
        goal_bias:       直接采样终点的概率
        clearance:       安全包络/障碍物膨胀距离

    返回: (world_path_3d, elapsed_time) 或 (None, elapsed_time)
    """
    t0 = time.perf_counter()
    sx, sy, sz = start_xyz
    gx, gy, gz = goal_xyz
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    if _point_in_cubes(sx, sy, sz, cube_positions, half_extents, clearance):
        print("[RRT3D] 起点位于障碍物内部，规划失败")
        return None, time.perf_counter() - t0
    if _point_in_cubes(gx, gy, gz, cube_positions, half_extents, clearance):
        print("[RRT3D] 终点位于障碍物内部，规划失败")
        return None, time.perf_counter() - t0

    # 节点格式: [x, y, z, parent_idx]
    nodes = [[sx, sy, sz, -1]]

    for i in range(max_iter):
        if random.random() < goal_bias:
            rx, ry, rz = gx, gy, gz
        else:
            rx = random.uniform(x_min, x_max)
            ry = random.uniform(y_min, y_max)
            rz = random.uniform(z_min, z_max)

        nearest_idx = _nearest_3d(nodes, rx, ry, rz)
        nx, ny, nz = nodes[nearest_idx][0], nodes[nearest_idx][1], nodes[nearest_idx][2]

        dx = rx - nx
        dy = ry - ny
        dz = rz - nz
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-9:
            continue

        ex = nx + (dx / dist) * min(step_size, dist)
        ey = ny + (dy / dist) * min(step_size, dist)
        ez = nz + (dz / dist) * min(step_size, dist)

        if _aabb_line_collision(nx, ny, nz, ex, ey, ez,
                                cube_positions, half_extents, clearance):
            continue

        new_idx = len(nodes)
        nodes.append([ex, ey, ez, nearest_idx])

        if np.sqrt((ex - gx)**2 + (ey - gy)**2 + (ez - gz)**2) <= goal_threshold:
            goal_idx = len(nodes)
            nodes.append([gx, gy, gz, new_idx])
            path = _backtrack_3d(nodes, goal_idx)
            path_len = sum(
                np.sqrt((path[i][0]-path[i-1][0])**2 +
                       (path[i][1]-path[i-1][1])**2 +
                       (path[i][2]-path[i-1][2])**2)
                for i in range(1, len(path))
            )
            elapsed = time.perf_counter() - t0
            print("[RRT3D] 规划成功, 路径点数=%d, 总长=%.4f m, 迭代=%d, 耗时=%.4fs" %
                  (len(path), path_len, i + 1, elapsed))
            return path, elapsed

    elapsed = time.perf_counter() - t0
    print("[RRT3D] 规划失败, 达到最大迭代次数%d, 耗时=%.4fs" % (max_iter, elapsed))
    return None, elapsed


def _nearest_3d(nodes: list, x: float, y: float, z: float) -> int:
    """寻找距离 (x,y,z) 最近的树节点索引（三维）。"""
    best_idx = 0
    best_dist = float('inf')
    for idx, (nx, ny, nz, _) in enumerate(nodes):
        d = (nx - x)**2 + (ny - y)**2 + (nz - z)**2
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return best_idx


def _backtrack_3d(nodes: list, goal_idx: int) -> list:
    """从目标节点回溯到起点，生成3D路径点列表 [(x,y,z), ...]。"""
    path = []
    cur = goal_idx
    while cur != -1:
        path.append((nodes[cur][0], nodes[cur][1], nodes[cur][2]))
        cur = nodes[cur][3]
    path.reverse()
    return path
