"""
TSP基础模块 —— 邻接矩阵构建、路径距离计算、图可视化。
"""
import numpy as np
import matplotlib.pyplot as plt

class TSP:
    """TSP问题父类：提供邻接矩阵、距离计算、图可视化等公共接口。"""

    def __init__(self, points):
        self.points = list(points)
        self.n = len(self.points)
        self.adj_matrix = self._build_adjacency()

    def get_all_points(self):
        return self.points

    @staticmethod
    def _euclidean_dist(p1, p2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _build_adjacency(self):
        """构建n×n邻接矩阵。"""
        mat = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    mat[i][j] = self._euclidean_dist(self.points[i], self.points[j])
        return mat

    def router_distance(self, route):
        """计算路径序列总长度。"""
        total = 0.0
        for k in range(len(route) - 1):
            total += self._euclidean_dist(
                self.points[route[k]], self.points[route[k + 1]])
        return total

    def draw_graph(self, title="TSP Complete Graph"):
        """matplotlib可视化：节点位置、完全图边、边权标注。"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]

        fig, ax = plt.subplots(figsize=(7, 6))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        'gray', linewidth=0.8, alpha=0.6)
                mx, my = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
                ax.annotate("%.3f" % self.adj_matrix[i][j], (mx, my),
                            fontsize=7, color='blue', alpha=0.8,
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.1',
                                     facecolor='white', alpha=0.7))

        ax.scatter(xs, ys, s=300, c='red', zorder=3, edgecolors='black')
        for i in range(self.n):
            ax.annotate("n%d" % i, (xs[i], ys[i]),
                        fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white')

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.show(block=False)
