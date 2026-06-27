import numpy as np


class MapFromPoints:
    """
    二维网格地图：输入障碍物点集，生成二值网格 (0=通路, 1=障碍物)。
    提供世界坐标与网格UV坐标的双向转换、网格文本可视化。
    """

    def __init__(self, obstacle_points: list, resolution: float, margin: float,
                 half_extents: tuple, clearance: float = 0.0):
        """根据障碍物点集创建二维二值网格地图。

        参数:
            clearance: 障碍物膨胀距离（安全包络），用于把机械臂本体体积
                       近似考虑到网格中。
        """
        self.resolution = resolution
        self.half_extents = half_extents[:2]  # (hx, hy)
        self.clearance = clearance

        obs_xy = np.array([(p[0], p[1]) for p in obstacle_points])

        self.x_min = obs_xy[:, 0].min() - margin
        self.x_max = obs_xy[:, 0].max() + margin
        self.y_min = obs_xy[:, 1].min() - margin
        self.y_max = obs_xy[:, 1].max() + margin

        self.cols = int(np.ceil((self.x_max - self.x_min) / resolution)) + 1
        self.rows = int(np.ceil((self.y_max - self.y_min) / resolution)) + 1

        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self._mark_obstacles(obs_xy)

        print("[网格地图] 创建完成: %dx%d 网格, 分辨率=%.3fm, "
              "范围x[%.3f,%.3f] y[%.3f,%.3f]" %
              (self.rows, self.cols, resolution,
               self.x_min, self.x_max, self.y_min, self.y_max))

    def _mark_obstacles(self, obs_xy: np.ndarray):
        """将障碍物占据的网格单元标记为1（含安全包络膨胀）。"""
        hx, hy = self.half_extents
        margin = self.clearance
        for ox, oy in obs_xy:
            u_min = max(0, self.world_to_grid_u(ox - hx - margin))
            u_max = min(self.cols - 1, self.world_to_grid_u(ox + hx + margin))
            v_min = max(0, self.world_to_grid_v(oy - hy - margin))
            v_max = min(self.rows - 1, self.world_to_grid_v(oy + hy + margin))
            for v in range(v_min, v_max + 1):
                for u in range(u_min, u_max + 1):
                    self.grid[v, u] = 1
        obstacle_count = np.sum(self.grid)
        print("[网格地图] 障碍物网格数: %d / %d (%.1f%%)" %
              (obstacle_count, self.grid.size,
               100.0 * obstacle_count / self.grid.size))

    # 坐标转换

    def world_to_grid(self, x: float, y: float) -> tuple:
        """世界坐标 (x, y) → 网格UV坐标 (u, v)。"""
        return (self.world_to_grid_u(x), self.world_to_grid_v(y))

    def world_to_grid_u(self, x: float) -> int:
        return int(round((x - self.x_min) / self.resolution))

    def world_to_grid_v(self, y: float) -> int:
        return int(round((y - self.y_min) / self.resolution))

    def grid_to_world(self, u: int, v: int) -> tuple:
        """网格UV坐标 → 世界坐标 (x, y)。"""
        x = self.x_min + u * self.resolution
        y = self.y_min + v * self.resolution
        return (x, y)

    def grid_to_world_path(self, grid_path: list) -> list:
        """网格UV路径 → 世界坐标路径点列表 [(x, y), ...]。"""
        return [self.grid_to_world(u, v) for (u, v) in grid_path]

    # 网格查询

    def in_bounds(self, u: int, v: int) -> bool:
        """判断网格UV坐标是否在地图范围内。"""
        return 0 <= u < self.cols and 0 <= v < self.rows

    def is_obstacle(self, u: int, v: int) -> bool:
        """判断指定网格单元是否为障碍物。"""
        if not self.in_bounds(u, v):
            return True
        return self.grid[v, u] == 1

    def is_free(self, u: int, v: int) -> bool:
        """判断指定网格单元是否可通行。"""
        return self.in_bounds(u, v) and self.grid[v, u] == 0

    def is_line_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """检测世界坐标线段 (x1,y1)→(x2,y2) 是否穿过障碍物网格。"""
        dist = np.hypot(x2 - x1, y2 - y1)
        if dist < 1e-9:
            return False
        n_samples = max(2, int(dist / (self.resolution * 0.5)) + 1)
        for i in range(n_samples):
            t = i / (n_samples - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            u, v = self.world_to_grid(x, y)
            if self.is_obstacle(u, v):
                return True
        return False

    # 可视化

    def print_grid(self):
        """打印二维网格矩阵文本可视化 (0=. 1=#)。"""
        print("\n[网格地图] 文本可视化 ('.'=通路 '#'=障碍物):")
        print("  网格尺寸: %d行 x %d列, 世界范围x[%.3f,%.3f] y[%.3f,%.3f]" %
              (self.rows, self.cols, self.x_min, self.x_max, self.y_min, self.y_max))
        header_step = max(1, self.cols // 20)
        col_header = "    " + "".join(str(c % 10) if c % header_step == 0 else " "
                                      for c in range(self.cols))
        print(col_header)
        for v in range(self.rows):
            row_str = "".join("#" if self.grid[v, u] else "." for u in range(self.cols))
            print(" %3d|%s" % (v, row_str))
        print()

    def _draw_base_grid(self, ax):
        """绘制网格背景与网格线，返回legend基础图例句柄列表。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        cmap = plt.cm.colors.ListedColormap(['white', '#444444'])
        ax.imshow(self.grid, cmap=cmap, origin='lower',
                  extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                  aspect='equal', interpolation='nearest')

        x_ticks = np.arange(self.x_min, self.x_max + self.resolution, self.resolution)
        y_ticks = np.arange(self.y_min, self.y_max + self.resolution, self.resolution)
        ax.set_xticks(x_ticks, minor=True)
        ax.set_yticks(y_ticks, minor=True)
        ax.grid(True, which='minor', color='#cccccc', linewidth=0.3, alpha=0.5)

        return [
            mpatches.Patch(color='#444444', label='Obstacle'),
            mpatches.Patch(color='white', label='Free'),
        ]

    def _draw_start_goal(self, ax, start_xy, goal_xy):
        """在图上标记起点和终点，返回legend句柄列表。"""
        import matplotlib.patches as mpatches
        handles = []
        if start_xy is not None:
            ax.scatter(*start_xy, c='limegreen', s=120, marker='o',
                       edgecolors='black', linewidths=1.0, zorder=5)
            handles.append(mpatches.Patch(color='limegreen', label='Start'))
        if goal_xy is not None:
            ax.scatter(*goal_xy, c='red', s=120, marker='X',
                       edgecolors='black', linewidths=1.0, zorder=5)
            handles.append(mpatches.Patch(color='red', label='Goal'))
        return handles

    def draw_path_astar(self, astar_grid_path: list,
                        start_xy: tuple = None, goal_xy: tuple = None):
        """单独绘制A*路径网格图。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 7))
        legend_handles = self._draw_base_grid(ax)

        if astar_grid_path is not None and len(astar_grid_path) > 0:
            awp = self.grid_to_world_path(astar_grid_path)
            ax.plot([p[0] for p in awp], [p[1] for p in awp],
                    color='steelblue', linewidth=2.5, marker='s',
                    markersize=5, zorder=4)
            legend_handles.append(mpatches.Patch(color='steelblue', label='A* Path'))

        legend_handles += self._draw_start_goal(ax, start_xy, goal_xy)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.9)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('A* Path Planning (Grid Map)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)

    def draw_path_rrt(self, rrt_world_path: list,
                      start_xy: tuple = None, goal_xy: tuple = None):
        """单独绘制RRT路径网格图。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 7))
        legend_handles = self._draw_base_grid(ax)

        if rrt_world_path is not None and len(rrt_world_path) > 0:
            ax.plot([p[0] for p in rrt_world_path],
                    [p[1] for p in rrt_world_path],
                    color='darkorange', linewidth=2.0, marker='.',
                    markersize=4, zorder=4)
            legend_handles.append(mpatches.Patch(color='darkorange', label='RRT Path'))

        legend_handles += self._draw_start_goal(ax, start_xy, goal_xy)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.9)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('RRT Path Planning (Grid Map)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)

    def draw_grid(self, start_xy: tuple = None, goal_xy: tuple = None,
                  title: str = "Grid Map"):
        """绘制网格地图的Matplotlib可视化（不含路径，仅网格背景+可选起终点）。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 7))
        legend_handles = self._draw_base_grid(ax)
        legend_handles += self._draw_start_goal(ax, start_xy, goal_xy)

        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.9)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        print("[网格地图] 网格可视化窗口已弹出")

    def draw_paths(self, astar_grid_path: list = None,
                   rrt_world_path: list = None,
                   start_xy: tuple = None, goal_xy: tuple = None):
        """分别弹出A*和RRT两张独立网格路径图。"""
        if astar_grid_path is not None and len(astar_grid_path) > 0:
            self.draw_path_astar(astar_grid_path, start_xy, goal_xy)
            print("[网格地图] A* 可视化窗口已弹出")
        if rrt_world_path is not None and len(rrt_world_path) > 0:
            self.draw_path_rrt(rrt_world_path, start_xy, goal_xy)
            print("[网格地图] RRT 可视化窗口已弹出")

    def draw_paths_combined(self, astar_grid_path: list = None,
                            rrt_world_path: list = None,
                            start_xy: tuple = None, goal_xy: tuple = None,
                            title: str = "A* vs RRT Path Comparison"):
        """在同一张网格地图上叠加绘制A*和RRT两条路径，用于直观对比。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 7))
        legend_handles = self._draw_base_grid(ax)

        if astar_grid_path is not None and len(astar_grid_path) > 0:
            awp = self.grid_to_world_path(astar_grid_path)
            ax.plot([p[0] for p in awp], [p[1] for p in awp],
                    color='steelblue', linewidth=2.5, marker='s',
                    markersize=4, zorder=4, label='_A*')
            legend_handles.append(mpatches.Patch(color='steelblue', label='A* Path'))

        if rrt_world_path is not None and len(rrt_world_path) > 0:
            ax.plot([p[0] for p in rrt_world_path],
                    [p[1] for p in rrt_world_path],
                    color='darkorange', linewidth=2.0, marker='.',
                    markersize=3, zorder=3, label='_RRT')
            legend_handles.append(mpatches.Patch(color='darkorange', label='RRT Path'))

        legend_handles += self._draw_start_goal(ax, start_xy, goal_xy)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.9)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        print("[网格地图] A*/RRT 叠加对比图窗口已弹出")


def draw_rrt_3d(cube_positions: list, half_extents: tuple,
                rrt3d_world_path: list = None,
                start_xyz: tuple = None, goal_xyz: tuple = None,
                title: str = "RRT 3D Path Planning"):
    """绘制三维RRT路径可视化：物块(半透明)+路径线+起终点。

    参数:
        cube_positions:  障碍物立方体中心坐标 [(cx,cy,cz), ...]
        half_extents:    立方体半边长 (hx, hy, hz)
        rrt3d_world_path: RRT 3D路径点 [(x,y,z), ...]
        start_xyz:       起点 (sx, sy, sz)
        goal_xyz:        终点 (gx, gy, gz)
        title:           图表标题
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    hx, hy, hz = half_extents

    # 绘制所有障碍物立方体（半透明灰色）
    for cx, cy, cz in cube_positions:
        xc = [cx - hx, cx + hx]
        yc = [cy - hy, cy + hy]
        zc = [cz - hz, cz + hz]

        verts = []
        # 6个面
        for xs, xe in [(xc[0], xc[1])]:
            for ys, ye in [(yc[0], yc[1])]:
                verts.append([
                    (xs, ys, zc[0]), (xe, ys, zc[0]),
                    (xe, ye, zc[0]), (xs, ye, zc[0])
                ])
                verts.append([
                    (xs, ys, zc[1]), (xe, ys, zc[1]),
                    (xe, ye, zc[1]), (xs, ye, zc[1])
                ])
            for zs, ze in [(zc[0], zc[1])]:
                verts.append([
                    (xs, yc[0], zs), (xe, yc[0], zs),
                    (xe, yc[0], ze), (xs, yc[0], ze)
                ])
                verts.append([
                    (xs, yc[1], zs), (xe, yc[1], zs),
                    (xe, yc[1], ze), (xs, yc[1], ze)
                ])

        # 区分底层(红色系)和上层(绿色系)障碍物颜色
        if cz < 0.06:
            face_color = (0.8, 0.3, 0.3, 0.35)
            edge_color = (0.6, 0.2, 0.2, 0.6)
        else:
            face_color = (0.3, 0.7, 0.3, 0.35)
            edge_color = (0.2, 0.5, 0.2, 0.6)

        poly = Poly3DCollection(verts, alpha=0.35, facecolors=face_color,
                                edgecolors=edge_color, linewidths=0.5)
        ax.add_collection3d(poly)

    # 绘制RRT 3D路径
    if rrt3d_world_path is not None and len(rrt3d_world_path) > 0:
        xs = [p[0] for p in rrt3d_world_path]
        ys = [p[1] for p in rrt3d_world_path]
        zs = [p[2] for p in rrt3d_world_path]
        ax.plot(xs, ys, zs, color='darkorange', linewidth=2.5,
                marker='.', markersize=2, zorder=10, label='RRT 3D Path')

    # 起点/终点
    if start_xyz is not None:
        ax.scatter(*start_xyz, c='limegreen', s=150, marker='o',
                   edgecolors='black', linewidths=1.2, zorder=20,
                   label='Start')
    if goal_xyz is not None:
        ax.scatter(*goal_xyz, c='red', s=150, marker='X',
                   edgecolors='black', linewidths=1.2, zorder=20,
                   label='Goal')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.show(block=False)
    print("[3D可视化] RRT 3D轨迹窗口已弹出")
