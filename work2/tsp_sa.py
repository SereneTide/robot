"""
模拟退火算法TSP求解 —— 随机交换路径节点，以概率接受差解。
"""
import numpy as np
from tsp_base import TSP

class TSPSA(TSP):
    """模拟退火：初始随机路径→交换扰动→Metropolis准则→降温迭代。"""

    def solve(self, start_idx=0, T_start=1000.0, T_end=0.01,
              alpha=0.98, max_iter=5000):
        """
        start_idx: 固定为路径首节点的索引
        返回: (route, total_distance)
        """
        others = [i for i in range(self.n) if i != start_idx]
        np.random.shuffle(others)
        curr_route = [start_idx] + others
        curr_dist = self.router_distance(curr_route)

        best_route, best_dist = curr_route[:], curr_dist
        T = T_start

        for _ in range(max_iter):
            i, j = np.random.randint(1, self.n, size=2)
            if i == j:
                continue
            new_route = curr_route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            new_dist = self.router_distance(new_route)

            delta = new_dist - curr_dist
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                curr_route, curr_dist = new_route, new_dist
                if curr_dist < best_dist:
                    best_route, best_dist = curr_route[:], curr_dist

            T *= alpha
            if T < T_end:
                break

        return best_route, best_dist
