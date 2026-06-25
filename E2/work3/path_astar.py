"""
A*避障路径规划算法。
"""
import heapq
import numpy as np
import time


def astar_plan(grid_map, start_uv: tuple, goal_uv: tuple,
               heuristic: str = 'euclidean', eight_dir: bool = True) -> tuple:
    """A*网格路径规划。返回 (grid_path, elapsed_time)。

    参数:
        eight_dir:  True=8方向（含对角线，斜向步长√2），False=4方向（曼哈顿邻域）
    """
    t0 = time.perf_counter()
    su, sv = start_uv
    gu, gv = goal_uv

    if not grid_map.is_free(su, sv):
        print("[A*] 起点 (%d,%d) 位于障碍物上，规划失败" % (su, sv))
        return None, time.perf_counter() - t0
    if not grid_map.is_free(gu, gv):
        print("[A*] 终点 (%d,%d) 位于障碍物上，规划失败" % (gu, gv))
        return None, time.perf_counter() - t0

    def _h(u, v):
        if heuristic == 'manhattan':
            return abs(u - gu) + abs(v - gv)
        else:
            return np.hypot(u - gu, v - gv)

    # 邻域定义: (du, dv, cost)
    if eight_dir:
        SQRT2 = np.sqrt(2)
        neighbors = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
        ]
        print("[A*] 使用8方向邻域（含对角线√2权重）")
    else:
        neighbors = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        ]

    open_heap = []
    g_cost = {start_uv: 0.0}
    parent = {start_uv: None}
    closed = set()

    heapq.heappush(open_heap, (_h(su, sv), 0.0, su, sv))

    while open_heap:
        f, g, cu, cv = heapq.heappop(open_heap)

        if (cu, cv) in closed:
            continue
        if g_cost.get((cu, cv), float('inf')) < g:
            continue

        if (cu, cv) == goal_uv:
            elapsed = time.perf_counter() - t0
            path = _reconstruct_path(parent, goal_uv)
            path_dist = sum(
                np.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                for i in range(1, len(path))
            )
            print("[A*] 规划成功, 路径长度=%d 节点, 网格距离=%.1f, 耗时=%.4fs" %
                  (len(path), path_dist, elapsed))
            return path, elapsed

        closed.add((cu, cv))

        for du, dv, step_cost in neighbors:
            nu, nv = cu + du, cv + dv
            if not grid_map.is_free(nu, nv) or (nu, nv) in closed:
                continue
            ng = g + step_cost
            if ng < g_cost.get((nu, nv), float('inf')):
                g_cost[(nu, nv)] = ng
                parent[(nu, nv)] = (cu, cv)
                nf = ng + _h(nu, nv)
                heapq.heappush(open_heap, (nf, ng, nu, nv))

    elapsed = time.perf_counter() - t0
    print("[A*] 规划失败, OpenSet已空, 耗时=%.4fs" % elapsed)
    return None, elapsed


def _reconstruct_path(parent: dict, goal_uv: tuple) -> list:
    """回溯父节点生成从起点到终点的完整网格路径。"""
    path = []
    cur = goal_uv
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path
