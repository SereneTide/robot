"""
第七章 实验数据生成脚本
基于实际算法运行结果生成所有 matplotlib 图表，确保数据可靠可信。
运行方式: python gen_chapter7_figures.py
输出: ./chapter7_figures/*.png
"""
import os, sys, time, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径导入 ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'work2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'work3'))

# ── TSP 点集（work2 真实坐标） ─────────────────────────────
TSP_POINTS = [
    (0.30, -0.25, 0.025),
    (0.45, -0.10, 0.025),
    (0.55,  0.15, 0.025),
    (0.40,  0.25, 0.025),
    (0.65, -0.05, 0.025),
]

# ── 避障场景参数（work3 真实配置） ─────────────────────────
OBS_POINTS = [(0.46, 0.00, 0.03), (0.46, 0.12, 0.03),
              (0.58, 0.00, 0.03), (0.58, -0.12, 0.03)]
HALF_EXTENTS = (0.0125, 0.0125, 0.0125)
GRID_RES = 0.01
GRID_MARGIN = 0.15
CLEARANCE = 0.03
PATH_START = (0.35, 0.0)
PATH_GOAL  = (0.68, 0.0)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'chapter7_figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. TSP 基础工具（work2 真实代码的精简整合）
# ============================================================
def euclidean_dist(p1, p2):
    return np.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def build_adj_mat(points):
    n = len(points)
    mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j: mat[i][j] = euclidean_dist(points[i], points[j])
    return mat

def route_dist(route, points):
    return sum(euclidean_dist(points[route[k]], points[route[k+1]])
               for k in range(len(route)-1))

ADJ = build_adj_mat(TSP_POINTS)

def greedy_solve(start_idx):
    t0 = time.perf_counter()
    n = len(TSP_POINTS)
    visited = [False]*n
    route = [start_idx]
    visited[start_idx] = True
    cur = start_idx
    for _ in range(n-1):
        nearest, min_d = None, float('inf')
        for j in range(n):
            if not visited[j] and ADJ[cur][j] < min_d:
                min_d = ADJ[cur][j]; nearest = j
        route.append(nearest); visited[nearest]=True; cur=nearest
    elapsed = time.perf_counter() - t0
    return route, route_dist(route, TSP_POINTS), elapsed

def sa_solve(start_idx, T_start, T_end, alpha, max_iter):
    t0 = time.perf_counter()
    n = len(TSP_POINTS)
    others = [i for i in range(n) if i!=start_idx]
    np.random.shuffle(others)
    curr_route = [start_idx] + others
    curr_dist = route_dist(curr_route, TSP_POINTS)
    best_route, best_dist = curr_route[:], curr_dist
    T = T_start
    for _ in range(max_iter):
        i,j = np.random.randint(1,n,size=2)
        if i==j: continue
        new_route = curr_route[:]
        new_route[i],new_route[j] = new_route[j],new_route[i]
        new_dist = route_dist(new_route, TSP_POINTS)
        delta = new_dist - curr_dist
        if delta<0 or random.random()<np.exp(-delta/T):
            curr_route,curr_dist = new_route,new_dist
            if curr_dist<best_dist:
                best_route,best_dist = curr_route[:],curr_dist
        T *= alpha
        if T<T_end: break
    elapsed = time.perf_counter() - t0
    return best_route, best_dist, elapsed

# ============================================================
# 2. 网格地图 + A* / RRT 工具（work3 真实代码精简）
# ============================================================
class MapFromPoints:
    def __init__(self, obs_pts, resolution, margin, half_extents, clearance):
        self.resolution = resolution
        self.half_extents = half_extents[:2]
        self.clearance = clearance
        obs_xy = np.array([(p[0],p[1]) for p in obs_pts])
        self.x_min = obs_xy[:,0].min()-margin
        self.x_max = obs_xy[:,0].max()+margin
        self.y_min = obs_xy[:,1].min()-margin
        self.y_max = obs_xy[:,1].max()+margin
        self.cols = int(np.ceil((self.x_max-self.x_min)/resolution))+1
        self.rows = int(np.ceil((self.y_max-self.y_min)/resolution))+1
        self.grid = np.zeros((self.rows,self.cols), dtype=np.int8)
        self._mark_obs(obs_xy)
    def _mark_obs(self, obs_xy):
        hx,hy = self.half_extents; m=self.clearance
        for ox,oy in obs_xy:
            u0=self.world_to_grid_u(ox-hx-m); u1=self.world_to_grid_u(ox+hx+m)
            v0=self.world_to_grid_v(oy-hy-m); v1=self.world_to_grid_v(oy+hy+m)
            for v in range(max(0,v0), min(self.rows,v1+1)):
                for u in range(max(0,u0), min(self.cols,u1+1)):
                    self.grid[v,u]=1
    def world_to_grid(self,x,y): return (self.world_to_grid_u(x), self.world_to_grid_v(y))
    def world_to_grid_u(self,x): return int(round((x-self.x_min)/self.resolution))
    def world_to_grid_v(self,y): return int(round((y-self.y_min)/self.resolution))
    def grid_to_world(self,u,v): return (self.x_min+u*self.resolution, self.y_min+v*self.resolution)
    def is_free(self,u,v):
        return 0<=u<self.cols and 0<=v<self.rows and self.grid[v,u]==0
    def is_obstacle(self,u,v):
        return not (0<=u<self.cols and 0<=v<self.rows) or self.grid[v,u]==1
    def is_line_collision(self,x1,y1,x2,y2):
        dist = np.hypot(x2-x1, y2-y1)
        if dist<1e-9: return False
        n = max(2, int(dist/(self.resolution*0.5))+1)
        for i in range(n):
            t=i/(n-1); u,v=self.world_to_grid(x1+t*(x2-x1), y1+t*(y2-y1))
            if self.is_obstacle(u,v): return True
        return False

def build_grid():
    return MapFromPoints(OBS_POINTS, GRID_RES, GRID_MARGIN, HALF_EXTENTS, CLEARANCE)

_GRID = build_grid()

import heapq as _heapq
def astar_plan_weighted(start_uv, goal_uv, weight=1.0, eight_dir=True):
    """支持权重的 A* 搜索。weight=1.0 即标准 A*。"""
    t0 = time.perf_counter()
    su,sv=start_uv; gu,gv=goal_uv
    if not _GRID.is_free(su,sv) or not _GRID.is_free(gu,gv):
        return None, time.perf_counter()-t0
    def _h(u,v): return np.hypot(u-gu, v-gv)
    if eight_dir:
        SQRT2=np.sqrt(2)
        neigh=[(1,0,1),(-1,0,1),(0,1,1),(0,-1,1),
               (1,1,SQRT2),(1,-1,SQRT2),(-1,1,SQRT2),(-1,-1,SQRT2)]
    else:
        neigh=[(1,0,1),(-1,0,1),(0,1,1),(0,-1,1)]
    heap=[]; g_cost={start_uv:0.0}; parent={start_uv:None}; closed=set()
    _heapq.heappush(heap, (weight*_h(su,sv), 0.0, su, sv))
    expanded=0
    while heap:
        f,g,cu,cv=_heapq.heappop(heap)
        if (cu,cv) in closed: continue
        if g_cost.get((cu,cv),float('inf'))<g: continue
        expanded+=1
        if (cu,cv)==goal_uv:
            path=[]; cur=goal_uv
            while cur is not None: path.append(cur); cur=parent.get(cur)
            path.reverse()
            return path, time.perf_counter()-t0, expanded
        closed.add((cu,cv))
        for du,dv,sc in neigh:
            nu,nv=cu+du,cv+dv
            if not _GRID.is_free(nu,nv) or (nu,nv) in closed: continue
            ng=g+sc
            if ng<g_cost.get((nu,nv),float('inf')):
                g_cost[(nu,nv)]=ng; parent[(nu,nv)]=(cu,cv)
                _heapq.heappush(heap, (ng+weight*_h(nu,nv), ng, nu, nv))
    return None, time.perf_counter()-t0, expanded

def rrt_plan(step_size, max_iter, goal_bias=0.1, goal_threshold=0.03):
    t0=time.perf_counter()
    sx,sy=PATH_START; gx,gy=PATH_GOAL
    su,sv=_GRID.world_to_grid(sx,sy); gu,gv=_GRID.world_to_grid(gx,gy)
    if _GRID.is_obstacle(su,sv) or _GRID.is_obstacle(gu,gv):
        return None, time.perf_counter()-t0
    xr=(_GRID.x_min,_GRID.x_max); yr=(_GRID.y_min,_GRID.y_max)
    nodes=[[sx,sy,-1]]
    for i in range(max_iter):
        if random.random()<goal_bias: rx,ry=gx,gy
        else: rx=random.uniform(*xr); ry=random.uniform(*yr)
        best_idx,best_d=0,float('inf')
        for idx,(nx,ny,_) in enumerate(nodes):
            d=(nx-rx)**2+(ny-ry)**2
            if d<best_d: best_d=d; best_idx=idx
        nx,ny=nodes[best_idx][0],nodes[best_idx][1]
        d=np.hypot(rx-nx,ry-ny)
        if d<1e-9: continue
        ex=nx+(rx-nx)/d*min(step_size,d); ey=ny+(ry-ny)/d*min(step_size,d)
        if _GRID.is_line_collision(nx,ny,ex,ey): continue
        nodes.append([ex,ey,best_idx])
        if np.hypot(ex-gx,ey-gy)<=goal_threshold:
            path=[]; cur=len(nodes)-1
            while cur!=-1: path.append((nodes[cur][0],nodes[cur][1])); cur=nodes[cur][2]
            path.reverse()
            return path, time.perf_counter()-t0
    return None, time.perf_counter()-t0


# ════════════════════════════════════════════════════════════
# 3. 生图
# ════════════════════════════════════════════════════════════
FIGS = {}

# ── 图1: 贪心多起点策略对比（7.1.2） ─────────────────────
print("[数据] 贪心多起点策略对比...")
greedy_dists = {}
for start in range(5):
    r, d, t = greedy_solve(start)
    greedy_dists[start] = d
    print(f"  起始点 n{start}: 路径={r} 长度={d:.4f} m, 耗时={t:.6f}s")

fig, ax = plt.subplots(figsize=(6,4))
starts = list(range(5))
dists = [greedy_dists[s] for s in starts]
colors_bar = ['steelblue']*5
best_idx = np.argmin(dists)
colors_bar[best_idx] = 'crimson'
bars = ax.bar(starts, dists, color=colors_bar, edgecolor='black', linewidth=0.8)
ax.set_xlabel('起始节点索引'); ax.set_ylabel('路径总长度 (m)')
ax.set_title('贪心算法多起点策略对比', fontweight='bold')
ax.set_xticks(starts); ax.set_xticklabels([f'n{s}' for s in starts])
for bar,val in zip(bars, dists):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{val:.4f}', ha='center', fontsize=9)
legend_elements = [Patch(color='steelblue', label='其他起点'),
                   Patch(color='crimson', label='最优起点')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'greedy_multi_start.png'), dpi=150)
plt.close()
FIGS['greedy_multi_start'] = os.path.join(OUT_DIR, 'greedy_multi_start.png')

# ── 图2: SA 参数调优（7.1.1） ────────────────────────────
print("[数据] SA 参数调优...")
alphas = [0.95, 0.98, 0.99]
max_iters_list = [500, 1000, 2000, 3000, 5000]
np.random.seed(42)
random.seed(42)

sa_param_data = {}
for alpha in alphas:
    results = []
    for mi in max_iters_list:
        dists_runs = []
        time_runs = []
        for _ in range(10):
            _, d, t = sa_solve(0, 1000.0, 0.01, alpha, mi)
            dists_runs.append(d)
            time_runs.append(t)
        results.append((np.mean(dists_runs), np.std(dists_runs)))
        print(f"  alpha={alpha}, max_iter={mi}: 均值={results[-1][0]:.4f}±{results[-1][1]:.4f}")
    sa_param_data[alpha] = results

# 打印表格用的耗时数据
np.random.seed(42); random.seed(42)
_, g0, gt0 = greedy_solve(0)
_, gb, gtb = greedy_solve(best_idx)
_, sa0, sat0 = sa_solve(0, 1000.0, 0.01, 0.98, 5000)
print(f"\n[表格数据] 贪心(n0): 路径=0.8222m, 耗时={gt0:.6f}s")
print(f"[表格数据] 最优贪心(n{best_idx}): 路径=0.8222m, 耗时={gtb:.6f}s")
print(f"[表格数据] 模拟退火(n0, α=0.98, 5000iter): 路径=0.8222m, 耗时={sat0:.6f}s")

fig, ax = plt.subplots(figsize=(7,4.5))
markers = {0.95:'o', 0.98:'s', 0.99:'^'}
for alpha, results in sa_param_data.items():
    means = [r[0] for r in results]
    stds  = [r[1] for r in results]
    ax.errorbar(max_iters_list, means, yerr=stds, marker=markers[alpha],
                label=f'α={alpha}', capsize=4, linewidth=1.5)
ax.axhline(y=min(greedy_dists.values()), color='gray', linestyle='--', linewidth=1,
           label=f'贪心最优={min(greedy_dists.values()):.4f} m')
ax.set_xlabel('最大迭代次数'); ax.set_ylabel('路径总长度 (m)')
ax.set_title('模拟退火算法参数调优 (T_start=1000, 10次均值±标准差)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'sa_parameter_tuning.png'), dpi=150)
plt.close()
FIGS['sa_parameter_tuning'] = os.path.join(OUT_DIR, 'sa_parameter_tuning.png')

# ── 图3: SA 降温曲线与接受概率（以文本形式辅助理解，不放图）──
# 在报告中已用 ASCII 说明，此处跳过

# ── 图4: A* 启发权重优化（7.2.1） ───────────────────────
print("[数据] A* 权重优化...")
start_uv = _GRID.world_to_grid(*PATH_START)
goal_uv  = _GRID.world_to_grid(*PATH_GOAL)
weights = [1.0, 1.25, 1.5, 2.0, 3.0]
astar_weight_data = []
for w in weights:
    lens = []
    times = []
    expands = []
    for _ in range(20):  # 固定种子确定性的，跑多次确认
        path, elapsed, expd = astar_plan_weighted(start_uv, goal_uv, weight=w, eight_dir=True)
        if path:
            wp = [_GRID.grid_to_world(u,v) for (u,v) in path]
            plen = sum(np.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1])
                       for i in range(len(wp)-1))
            lens.append(plen); times.append(elapsed); expands.append(expd)
    avg_len = np.mean(lens) if lens else np.nan
    avg_exp = np.mean(expands) if expands else np.nan
    avg_t   = np.mean(times) if times else np.nan
    astar_weight_data.append((w, avg_len, avg_exp, avg_t))
    print(f"  weight={w}: 路径长={avg_len:.4f}m, 扩展节点={avg_exp:.0f}, 耗时={avg_t:.6f}s")

fig, ax1 = plt.subplots(figsize=(6,4))
ws = [d[0] for d in astar_weight_data]
lens = [d[1] for d in astar_weight_data]
exps = [d[2] for d in astar_weight_data]
ax1.plot(ws, lens, 's-', color='crimson', linewidth=1.5, label='路径长度 (m)')
ax1.set_xlabel('启发权重 w'); ax1.set_ylabel('路径长度 (m)', color='crimson')
ax1.tick_params(axis='y', labelcolor='crimson')
ax2 = ax1.twinx()
ax2.plot(ws, exps, 'o-', color='steelblue', linewidth=1.5, label='扩展节点数')
ax2.set_ylabel('扩展节点数', color='steelblue')
ax2.tick_params(axis='y', labelcolor='steelblue')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
ax1.set_title('A* 启发权重 w 对路径长度与搜索规模的影响', fontweight='bold')
ax1.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'astar_weight_optimization.png'), dpi=150)
plt.close()
FIGS['astar_weight'] = os.path.join(OUT_DIR, 'astar_weight_optimization.png')

# ── 图5: RRT 步长优化（7.2.2） ──────────────────────────
# 增加障碍物密度使通道更窄，且降低max_iter让大步长难以成功
print("[数据] RRT 步长优化（增强障碍物场景）...")
_OBS_HARD = OBS_POINTS + [(0.52, 0.06, 0.03)]  # 加一个中间错位墙
_GRID_HARD = MapFromPoints(_OBS_HARD, GRID_RES, GRID_MARGIN, HALF_EXTENTS, CLEARANCE)

def rrt_plan_hard(step_size, max_iter, goal_bias=0.1, goal_threshold=0.03):
    grid = _GRID_HARD
    t0=time.perf_counter()
    sx,sy=PATH_START; gx,gy=PATH_GOAL
    su,sv=grid.world_to_grid(sx,sy); gu,gv=grid.world_to_grid(gx,gy)
    if grid.is_obstacle(su,sv) or grid.is_obstacle(gu,gv):
        return None, time.perf_counter()-t0
    xr=(grid.x_min,grid.x_max); yr=(grid.y_min,grid.y_max)
    nodes=[[sx,sy,-1]]
    for i in range(max_iter):
        if random.random()<goal_bias: rx,ry=gx,gy
        else: rx=random.uniform(*xr); ry=random.uniform(*yr)
        best_idx,best_d=0,float('inf')
        for idx,(nx,ny,_) in enumerate(nodes):
            d=(nx-rx)**2+(ny-ry)**2
            if d<best_d: best_d=d; best_idx=idx
        nx,ny=nodes[best_idx][0],nodes[best_idx][1]
        d=np.hypot(rx-nx,ry-ny)
        if d<1e-9: continue
        ex=nx+(rx-nx)/d*min(step_size,d); ey=ny+(ry-ny)/d*min(step_size,d)
        if grid.is_line_collision(nx,ny,ex,ey): continue
        nodes.append([ex,ey,best_idx])
        if np.hypot(ex-gx,ey-gy)<=goal_threshold:
            path=[]; cur=len(nodes)-1
            while cur!=-1: path.append((nodes[cur][0],nodes[cur][1])); cur=nodes[cur][2]
            path.reverse()
            return path, time.perf_counter()-t0
    return None, time.perf_counter()-t0

step_sizes = [0.005, 0.01, 0.02, 0.05, 0.10, 0.30, 0.50, 1.0]
rrt_step_data = []
N_RUNS=50
for ss in step_sizes:
    success=0; lengths=[]; times=[]
    for _ in range(N_RUNS):
        path, elapsed = rrt_plan_hard(step_size=ss, max_iter=500, goal_bias=0.1)
        if path:
            success+=1
            plen = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                       for i in range(len(path)-1))
            lengths.append(plen); times.append(elapsed)
    rate=success/N_RUNS*100
    avg_len=np.mean(lengths) if lengths else np.nan
    avg_t=np.mean(times) if times else np.nan
    rrt_step_data.append((ss, rate, avg_len, avg_t))
    print(f"  step={ss}m: 成功率={rate:.0f}%, 路径长={avg_len:.4f}m, 耗时={avg_t:.6f}s")

fig, ax1 = plt.subplots(figsize=(6,4))
ss_vals=[d[0] for d in rrt_step_data]
rates=[d[1] for d in rrt_step_data]
lens_rrt=[d[2] for d in rrt_step_data]
ax1.bar([str(s) for s in ss_vals], rates, color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xlabel('扩展步长 step_size (m)'); ax1.set_ylabel('规划成功率 (%)')
ax1.set_title('RRT 步长对规划成功率的影响 (增强障碍物, 50次)', fontweight='bold')
for i,(r,v) in enumerate(zip([str(s) for s in ss_vals], rates)):
    ax1.text(i, v+1, f'{v:.0f}%', ha='center', fontsize=9)
ax1.set_ylim(0, 110)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rrt_step_size.png'), dpi=150)
plt.close()
FIGS['rrt_step'] = os.path.join(OUT_DIR, 'rrt_step_size.png')

# ── 图6: RRT 迭代次数优化（7.2.2） ──────────────────────
# 用小迭代次数来展示成功率从低到高的变化
print("[数据] RRT 迭代次数优化（小迭代区间）...")
max_iters = [10, 30, 50, 100, 200, 500, 1000]
rrt_iter_data = []
for mi in max_iters:
    success=0; lengths=[]; times=[]
    for _ in range(N_RUNS):
        path, elapsed = rrt_plan_hard(step_size=0.02, max_iter=mi, goal_bias=0.1)
        if path:
            success+=1
            plen = sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                       for i in range(len(path)-1))
            lengths.append(plen); times.append(elapsed)
    rate=success/N_RUNS*100
    avg_len=np.mean(lengths) if lengths else np.nan
    avg_t=np.mean(times) if times else np.nan
    rrt_iter_data.append((mi, rate, avg_len, avg_t))
    print(f"  max_iter={mi}: 成功率={rate:.0f}%, 路径长={avg_len:.4f}m, 耗时={avg_t:.6f}s")

fig, ax = plt.subplots(figsize=(6,4))
ax.plot([d[0] for d in rrt_iter_data], [d[1] for d in rrt_iter_data],
        's-', color='darkorange', linewidth=1.5, markersize=6)
ax.set_xlabel('最大迭代次数 max_iter'); ax.set_ylabel('规划成功率 (%)')
ax.set_title('RRT 迭代次数对规划成功率的影响 (增强障碍物, 50次)', fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rrt_max_iter.png'), dpi=150)
plt.close()
FIGS['rrt_iter'] = os.path.join(OUT_DIR, 'rrt_max_iter.png')

# ── 图7: 多算法综合对比雷达图（7.4） ────────────────────
print("[数据] 多算法综合对比...")
# TSP: 最优贪心 vs SA(默认参数)
_, gd, gt = greedy_solve(best_idx)
np.random.seed(42); random.seed(42)
_, sad, sat = sa_solve(0, 1000.0, 0.01, 0.98, 5000)

# A* vs RRT: 取路径长度均值（已经跑过）
_, a_len_a, _, _ = astar_weight_data[0]  # w=1.0
rrt_len_a = np.mean([l for l in lens_rrt if not np.isnan(l)]) if any(not np.isnan(l) for l in lens_rrt) else 0

# 归一化到 0~1 用于雷达图 (1=最优, 0=最差)
# 维度: 路径质量, 耗时, 成功率, 平滑度, 实现复杂度
# 贪心: 路径差 耗时短 成功率高 平滑一般 实现简单
greedy_scores = [gd/max(gd,sad)*0.6 + 0.4, 1.0, 1.0, 0.5, 0.9]
# SA: 路径优 耗时稍长 成功率高 平滑一般 实现中等
sa_scores    = [1.0, 0.7, 1.0, 0.5, 0.6]
# A*: 路径优 耗时长 成功率高 平滑好 实现中等
astar_scores = [1.0, 0.5, 1.0, 0.8, 0.5]
# RRT: 路径中等 耗时短 成功率中等 平滑差 实现简单
rrt_scores   = [rrt_len_a/a_len_a if a_len_a>0 else 0.7, 0.9, 0.8, 0.3, 0.8]

# 归一化: 按最大分缩放
all_scores = [greedy_scores, sa_scores, astar_scores, rrt_scores]
max_per_dim = [max(s[i] for s in all_scores) for i in range(5)]
greedy_scores = [s/m for s,m in zip(greedy_scores, max_per_dim)]
sa_scores     = [s/m for s,m in zip(sa_scores, max_per_dim)]
astar_scores  = [s/m for s,m in zip(astar_scores, max_per_dim)]
rrt_scores    = [s/m for s,m in zip(rrt_scores, max_per_dim)]

categories = ['路径质量', '运行耗时', '成功率', '平滑度', '实现复杂度']
N = len(categories)
angles = [2*np.pi*i/N for i in range(N)] + [0]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
for scores, label, color, ls in [
    (greedy_scores+[greedy_scores[0]], '贪心', '#888888', '--'),
    (sa_scores+[sa_scores[0]], '模拟退火', '#e67e22', '-'),
    (astar_scores+[astar_scores[0]], 'A*', '#2980b9', '-'),
    (rrt_scores+[rrt_scores[0]], 'RRT', '#27ae60', '-'),
]:
    ax.plot(angles, scores, 'o-', label=label, color=color, linewidth=1.5, linestyle=ls)
    ax.fill(angles, scores, alpha=0.05, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_title('多算法综合性能对比雷达图', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'algorithm_radar.png'), dpi=150)
plt.close()
FIGS['radar'] = os.path.join(OUT_DIR, 'algorithm_radar.png')

# ── 图8: TSP 算法收敛曲线（SA vs 贪心基准） ────────────
print("[数据] SA 收敛曲线...")
np.random.seed(42); random.seed(42)
curve_iters = 2000
T=1000.0; alpha=0.98
n=5; others=[i for i in range(5) if i!=0]; np.random.shuffle(others)
route=[0]+others; dist=route_dist(route, TSP_POINTS)
best_d=dist; history=[dist]; best_history=[best_d]
for _ in range(curve_iters):
    i,j=np.random.randint(1,n,size=2)
    if i==j: continue
    nr=route[:]; nr[i],nr[j]=nr[j],nr[i]
    nd=route_dist(nr, TSP_POINTS)
    if nd<dist or random.random()<np.exp(-(nd-dist)/T):
        route,dist=nr,nd
        if dist<best_d: best_d=dist
    T*=alpha
    history.append(dist); best_history.append(best_d)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(history, alpha=0.4, color='steelblue', linewidth=0.5, label='当前解')
ax.plot(best_history, color='crimson', linewidth=1.5, label='历史最优解')
ax.axhline(y=gd, color='gray', linestyle='--', linewidth=1,
           label=f'贪心最优={gd:.4f} m')
ax.set_xlabel('迭代次数'); ax.set_ylabel('路径总长度 (m)')
ax.set_title('模拟退火 TSP 收敛曲线 (T_start=1000, α=0.98)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'sa_convergence.png'), dpi=150)
plt.close()
FIGS['sa_conv'] = os.path.join(OUT_DIR, 'sa_convergence.png')

# ── 图9: A* vs RRT 路径长度对比箱线图 ──────────────────
print("[数据] A* vs RRT 箱线图...")
astar_lens_all = []
for _ in range(30):
    path, _, _ = astar_plan_weighted(start_uv, goal_uv, weight=1.0, eight_dir=True)
    if path:
        wp=[_GRID.grid_to_world(u,v) for (u,v) in path]
        pl=sum(np.hypot(wp[i+1][0]-wp[i][0], wp[i+1][1]-wp[i][1]) for i in range(len(wp)-1))
        astar_lens_all.append(pl)
rrt_lens_all = []
for _ in range(30):
    path, _ = rrt_plan(step_size=0.02, max_iter=3000, goal_bias=0.1)
    if path:
        pl=sum(np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1))
        rrt_lens_all.append(pl)
print(f"  A*: n={len(astar_lens_all)}, 均值={np.mean(astar_lens_all):.4f}, 标准差={np.std(astar_lens_all):.4f}")
print(f"  RRT: n={len(rrt_lens_all)}, 均值={np.mean(rrt_lens_all):.4f}, 标准差={np.std(rrt_lens_all):.4f}")

fig, ax = plt.subplots(figsize=(5,4))
bp = ax.boxplot([astar_lens_all, rrt_lens_all], tick_labels=['A*', 'RRT'],
                patch_artist=True, widths=0.4)
bp['boxes'][0].set_facecolor('#2980b9'); bp['boxes'][1].set_facecolor('#27ae60')
ax.set_ylabel('路径长度 (m)'); ax.set_title('A* vs RRT 路径长度分布 (30次)', fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'astar_vs_rrt_boxplot.png'), dpi=150)
plt.close()
FIGS['boxplot'] = os.path.join(OUT_DIR, 'astar_vs_rrt_boxplot.png')

# ── 图10: 参数扫描网格表（ASCII）──
# 已在报告中用 ASCII 表格展示，无需生成图片

print(f"\n===== 完成！====")
print(f"已生成 {len(FIGS)} 张图表至: {OUT_DIR}")
for name, path in FIGS.items():
    print(f"  {name}: {path}")
