"""
贪心算法TSP求解 —— 每步选择最近未访问节点。
"""
from tsp_base import TSP

class TSPGreedy(TSP):
    """贪心策略：以指定节点为起点，每步取距离当前点最近的未访问节点。"""

    def solve(self, start_idx=0):
        """
        start_idx: 起始节点索引
        返回: (route, total_distance)
        """
        n = self.n
        visited = [False] * n
        route = [start_idx]
        visited[start_idx] = True
        current = start_idx

        for _ in range(n - 1):
            nearest, min_d = None, float('inf')
            for j in range(n):
                if not visited[j] and self.adj_matrix[current][j] < min_d:
                    min_d = self.adj_matrix[current][j]
                    nearest = j
            route.append(nearest)
            visited[nearest] = True
            current = nearest

        return route, self.router_distance(route)
