"""
遗传算法TSP求解 —— 种群初始化、锦标赛选择、有序交叉、交换变异。
"""
import numpy as np
from tsp_base import TSP

class TSPGA(TSP):
    """遗传算法：精英保留 + 锦标赛选择 + OX交叉 + 交换变异。"""

    def solve(self, pop_size=100, generations=500,
              mut_rate=0.15, elite_rate=0.1):
        """
        返回: (route, total_distance)
        """
        pop = self._init_pop(pop_size)

        for _ in range(generations):
            fitness = np.array([1.0 / self.router_distance(ind) for ind in pop])
            n_elite = max(1, int(pop_size * elite_rate))
            elite_idx = np.argsort(fitness)[-n_elite:]
            new_pop = [pop[i] for i in elite_idx]

            while len(new_pop) < pop_size:
                p1 = self._tournament(pop, fitness)
                p2 = self._tournament(pop, fitness)
                child = self._ox_crossover(p1, p2)
                if np.random.random() < mut_rate:
                    self._swap_mutate(child)
                new_pop.append(child)
            pop = new_pop

        best = min(pop, key=lambda r: self.router_distance(r))
        return best, self.router_distance(best)

    # ---- 内部算子 ----

    def _init_pop(self, size):
        """随机生成初始种群，个体以节点0为固定起点。"""
        pop = []
        others = list(range(1, self.n))
        for _ in range(size):
            np.random.shuffle(others)
            pop.append([0] + others[:])
        return pop

    def _tournament(self, pop, fitness, k=3):
        """锦标赛选择：随机k个个体，取适应度最高者。"""
        idx = np.random.choice(len(pop), k)
        return pop[idx[np.argmax(fitness[idx])]]

    def _ox_crossover(self, p1, p2):
        """
        有序交叉(OX)：从p1截取子序列，其余按p2顺序填充，保持起始点不变。
        """
        n = len(p1)
        a, b = sorted(np.random.choice(range(1, n), 2, replace=False))
        child = [-1] * n
        child[a:b] = p1[a:b]
        used = set(child[a:b]) | {0}
        fill = [x for x in p2 if x not in used]
        idx = 0
        for i in range(1, n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        child[0] = 0
        return child

    def _swap_mutate(self, route):
        """交换变异：随机交换路径中两个非起始节点。"""
        i, j = np.random.randint(1, self.n, size=2)
        route[i], route[j] = route[j], route[i]
