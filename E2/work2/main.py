"""
主程序 —— TSP分拣路径优化与机械臂联动。
三种算法独立求解 → 对比输出 → Z键触发机械臂按最优路径抓取。
"""
import pybullet as p
import time
from config import (
    PYBULLET_GUI, SAFE_Z, GRASP_OFFSET_Z, CUBE_NUM,
    SA_T_START, SA_T_END, SA_ALPHA, SA_MAX_ITER,
    GA_POP_SIZE, GA_GENERATIONS, GA_MUT_RATE, GA_ELITE_RATE,
)
from scene import Scene
from sim_robot import SimRobot
from tsp_base import TSP
from tsp_greedy import TSPGreedy
from tsp_sa import TSPSA
from tsp_ga import TSPGA


def run_tsp_comparison(points):
    """
    统一输入相同物料节点，分别运行三种TSP算法。
    返回: (best_route, results_dict)
    """
    tsp = TSP(points)
    tsp.draw_graph("TSP Complete Graph | n=%d" % tsp.n)
    print("\n[图结构] matplotlib窗口已弹出，节点数=%d 边数=%d" %
          (tsp.n, tsp.n * (tsp.n - 1) // 2))

    results = {}

    # 贪心算法
    t0 = time.perf_counter()
    g = TSPGreedy(points)
    route_g, dist_g = g.solve(start_idx=0)
    t_g = time.perf_counter() - t0
    results['贪心算法'] = (route_g, dist_g, t_g)

    # 模拟退火
    t0 = time.perf_counter()
    sa = TSPSA(points)
    route_sa, dist_sa = sa.solve(
        start_idx=0, T_start=SA_T_START, T_end=SA_T_END,
        alpha=SA_ALPHA, max_iter=SA_MAX_ITER)
    t_sa = time.perf_counter() - t0
    results['模拟退火'] = (route_sa, dist_sa, t_sa)

    # 遗传算法
    t0 = time.perf_counter()
    ga = TSPGA(points)
    route_ga, dist_ga = ga.solve(
        pop_size=GA_POP_SIZE, generations=GA_GENERATIONS,
        mut_rate=GA_MUT_RATE, elite_rate=GA_ELITE_RATE)
    t_ga = time.perf_counter() - t0
    results['遗传算法'] = (route_ga, dist_ga, t_ga)

    # 打印对比表格
    print("\n算法        | 路径长度(m)  | 耗时(s)       | 路径序列")
    for name, (route, dist, t_elapsed) in results.items():
        route_str = " -> ".join(str(i) for i in route)
        print("  %-10s | %12.4f | %12.8f | %s" %
              (name, dist, t_elapsed, route_str))

    # 确定最优
    best_name = min(results, key=lambda k: results[k][1])
    best_route, best_dist, _ = results[best_name]
    print("\n[最优] %s, 路径长度=%.4f m, 序列=%s" %
          (best_name, best_dist, str(best_route)))
    return best_route, results


def execute_tsp_path(robot, points, route):
    """按TSP最优路径控制机械臂逐个抓取物料。"""
    print("\n[TSP抓取] 序列: %s, 总长: %.4f m" %
          (str(route), TSP(points).router_distance(route)))

    robot.go_to_rest(wait=True)

    for seq, idx in enumerate(route):
        cx, cy, cz = points[idx]
        print("\n[TSP抓取 %d/%d] 物块#%d (%.3f, %.3f, %.3f)" %
              (seq + 1, len(route), idx, cx, cy, cz))

        # 水平移动至物块正上方
        robot.move_one_point(cx, cy, SAFE_Z)
        # 垂直下落
        robot.move_one_point(cx, cy, cz + GRASP_OFFSET_Z)
        # 抬升回安全高度
        robot.move_one_point(cx, cy, SAFE_Z)

        ee = robot.get_end_effector_pos()
        print("[TSP抓取] 物块#%d 完成, 末端=(%.3f, %.3f, %.3f)" %
              (idx, ee[0], ee[1], ee[2]))

    robot.go_to_rest(wait=True)
    print("\n[TSP抓取] 全部%d个物块抓取完成" % len(route))


def main():
    print("[主程序] TSP分拣路径优化, 模式=%s | 物块数=%d" %
          ("GUI" if PYBULLET_GUI else "DIRECT", CUBE_NUM))

    scene = Scene()
    scene.draw_workspace()
    robot = SimRobot(scene_client_id=scene.client_id)

    # 物理稳定
    print("\n[主程序] 等待物理稳定...")
    for _ in range(200):
        p.stepSimulation(physicsClientId=scene.client_id)
    robot.go_to_rest(wait=True)

    # 获取物料坐标
    points = scene.get_cube_positions()
    print("\n[物料节点坐标] (n=%d)" % len(points))
    for i, pt in enumerate(points):
        print("  n%d: (% .3f, % .3f, % .3f)" % (i, pt[0], pt[1], pt[2]))

    # TSP求解与对比
    t0_total = time.time()
    best_route, results = run_tsp_comparison(points)
    t_total = time.time() - t0_total
    print("\n[TSP对比] 三种算法总耗时: %.4f s" % t_total)

    # 操作提示
    print("\n  操作提示: [Z]键抓取  [ESC]键退出\n")

    tsp_executed = False

    try:
        while True:
            scene.step(1)

            keys = p.getKeyboardEvents(physicsClientId=scene.client_id)

            # Z键触发TSP抓取
            if ord('z') in keys:
                key_state = keys[ord('z')]
                if isinstance(key_state, int):
                    triggered = bool(key_state & p.KEY_WAS_TRIGGERED)
                elif isinstance(key_state, tuple):
                    triggered = (key_state[0] & p.KEY_WAS_TRIGGERED
                                if key_state else False)
                else:
                    triggered = False

                if triggered and not tsp_executed:
                    print("\n[主程序] [TRIGGER] Z键按下，启动TSP最优路径抓取\n")
                    execute_tsp_path(robot, points, best_route)
                    tsp_executed = True
                    print("\n[主程序] TSP抓取完毕。按ESC退出。")
                elif triggered and tsp_executed:
                    print("[主程序] 已执行过TSP抓取，按ESC退出。")

            # ESC键退出
            if 27 in keys:
                key_state = keys[27]
                if isinstance(key_state, int):
                    triggered = bool(key_state & p.KEY_WAS_TRIGGERED)
                elif isinstance(key_state, tuple):
                    triggered = (key_state[0] & p.KEY_WAS_TRIGGERED
                                if key_state else False)
                else:
                    triggered = False
                if triggered:
                    print("[主程序] 退出仿真。")
                    break

    except KeyboardInterrupt:
        print("\n[主程序] KeyboardInterrupt, 退出。")
    finally:
        scene.close()


if __name__ == '__main__':
    main()
