import pybullet as p
import time
from config import (
    PYBULLET_GUI, CUBE_NUM, CUBE_HALF_EXTENTS,
    GRID_RESOLUTION, GRID_MARGIN,
    ASTAR_HEURISTIC,
    RRT_STEP_SIZE, RRT_MAX_ITER, RRT_GOAL_THRESHOLD, RRT_GOAL_BIAS,
    PATH_START, PATH_GOAL, PATH_Z, SAFE_Z,
)
from scene import Scene
from sim_robot import SimRobot
from map_tools import MapFromPoints
from path_astar import astar_plan
from path_rrt import rrt_plan


def _check_collision_with_cubes(grid_map, wx: float, wy: float,
                                 cube_positions: list,
                                 half_extents: tuple) -> bool:
    """检测末端 (wx,wy,PATH_Z) 是否与任何物块碰撞（水平投影重合即判碰撞）。"""
    hx, hy = half_extents[:2]
    for cx, cy, _ in cube_positions:
        if abs(wx - cx) <= hx and abs(wy - cy) <= hy:
            return True
    return False


def _draw_path_marker(pos: tuple, color: tuple, size: float = 0.02,
                      client_id: int = 0):
    """在场景中放置一个彩色小球作为路径标记。"""
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=size,
                                rgbaColor=color,
                                physicsClientId=client_id)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                              basePosition=pos, physicsClientId=client_id)
    return body


def execute_path(robot, grid_map, world_path: list, label: str,
                 cube_positions: list, half_extents: tuple,
                 client_id: int):
    """
    驱动机械臂末端沿路径点列表移动，含完整起降流程与碰撞检测。
    流程: 安全位 → 起点正上方(SAFE_Z) → 下降至(PATH_Z) → 沿路径运动 → 终点抬升(SAFE_Z)。
    """
    print("\n[路径执行] %s: 点数=%d, 高度=%.3f m" %
          (label, len(world_path), PATH_Z))

    sx, sy = world_path[0]
    gx, gy = world_path[-1]
    collision_count = 0
    unreachable_count = 0
    # 阶段1: 从待机位移至起点正上方(SAFE_Z)
    print("[%s] 阶段1: 移至起点正上方 (%.3f, %.3f, %.3f)" % (label, sx, sy, SAFE_Z))
    robot.move_one_point(sx, sy, SAFE_Z, log=True)
    # 阶段2: 垂直下降至执行高度(PATH_Z)
    print("[%s] 阶段2: 下降至执行高度 (%.3f, %.3f, %.3f)" % (label, sx, sy, PATH_Z))
    robot.move_one_point(sx, sy, PATH_Z, log=True)
    # 阶段3: 沿路径逐点运动
    print("[%s] 阶段3: 沿路径运动 (%d个点)" % (label, len(world_path)))
    for i, (wx, wy) in enumerate(world_path):
        if i % max(1, len(world_path) // 10) == 0:
            print("[%s] 第%d/%d点: (%.3f, %.3f)" %
                  (label, i + 1, len(world_path), wx, wy))
        if _check_collision_with_cubes(grid_map, wx, wy, cube_positions, half_extents):
            collision_count += 1
            print("[碰撞日志] %s 第%d点 (%.3f,%.3f) 与物块碰撞!" %
                  (label, i + 1, wx, wy))
        success, _ = robot.move_one_point(wx, wy, PATH_Z, log=False)
        if not success:
            unreachable_count += 1
            print("[碰撞日志] %s 第%d点 末端未达目标, 偏差超容差" % (label, i + 1))
    # 阶段4: 终点抬升至安全高度
    print("[%s] 阶段4: 终点抬升至安全高度 (%.3f, %.3f, %.3f)" % (label, gx, gy, SAFE_Z))
    robot.move_one_point(gx, gy, SAFE_Z, log=True)
    # 汇总
    pass_ok = (collision_count == 0 and unreachable_count == 0)
    print("[%s] 路径点数=%d, 碰撞=%d, 不可达=%d, 避障=%s" %
          (label, len(world_path), collision_count, unreachable_count,
           "PASS" if pass_ok else "FAIL"))
    return pass_ok

def main():
    # 初始化场景与机械臂
    scene = Scene()
    scene.draw_workspace()
    robot = SimRobot(scene_client_id=scene.client_id)
    print("\n[主程序] 等待物理稳定...")
    for _ in range(200):
        p.stepSimulation(physicsClientId=scene.client_id)
    robot.go_to_rest(wait=True)
    # 绘制起点/终点标记
    _draw_path_marker((PATH_START[0], PATH_START[1], 0.01),
                      (0.0, 1.0, 0.0, 1.0), 0.025, scene.client_id)
    _draw_path_marker((PATH_GOAL[0], PATH_GOAL[1], 0.01),
                      (1.0, 0.0, 0.0, 1.0), 0.025, scene.client_id)
    print("[主程序] 起点(绿球) 终点(红球) 标记已绘制")
    # 获取物料坐标，构建网格地图
    cube_positions = scene.get_cube_positions()
    print("\n[物料节点] (n=%d)" % len(cube_positions))
    for i, pt in enumerate(cube_positions):
        print("  n%d: (% .3f, % .3f, % .3f)" % (i, pt[0], pt[1], pt[2]))
    grid_map = MapFromPoints(
        cube_positions, GRID_RESOLUTION, GRID_MARGIN, CUBE_HALF_EXTENTS)
    grid_map.print_grid()
    grid_map.draw_grid(start_xy=PATH_START, goal_xy=PATH_GOAL,
                       title="Grid Map | resolution=%.2fcm" %
                             (GRID_RESOLUTION * 100))
    # 路径规划参数
    start_uv = grid_map.world_to_grid(*PATH_START)
    goal_uv = grid_map.world_to_grid(*PATH_GOAL)
    print("[路径规划] 起点世界(% .3f,% .3f) -> UV(%d,%d)" %
          (PATH_START[0], PATH_START[1], start_uv[0], start_uv[1]))
    print("[路径规划] 终点世界(% .3f,% .3f) -> UV(%d,%d)" %
          (PATH_GOAL[0], PATH_GOAL[1], goal_uv[0], goal_uv[1]))
    # A* 规划
    print("\n[A*] 启发=%s" % ASTAR_HEURISTIC)
    astar_grid_path, astar_time = astar_plan(
        grid_map, start_uv, goal_uv, ASTAR_HEURISTIC)
    astar_world_path = None
    if astar_grid_path:
        astar_world_path = grid_map.grid_to_world_path(astar_grid_path)
        astar_length = sum(
            ((astar_world_path[i + 1][0] - astar_world_path[i][0]) ** 2 +
             (astar_world_path[i + 1][1] - astar_world_path[i][1]) ** 2) ** 0.5
            for i in range(len(astar_world_path) - 1)
        )
        a_collisions = 0
        for wx, wy in astar_world_path:
            if _check_collision_with_cubes(grid_map, wx, wy, cube_positions, CUBE_HALF_EXTENTS):
                a_collisions += 1
        print("[A*] 世界路径点数=%d, 路径总长=%.4f m, 碰撞点=%d" %
              (len(astar_world_path), astar_length, a_collisions))
    # RRT 规划
    print("\n[RRT]")
    t0 = time.perf_counter()
    rrt_world_path, rrt_time = rrt_plan(
        grid_map, PATH_START, PATH_GOAL,
        RRT_STEP_SIZE, RRT_MAX_ITER, RRT_GOAL_THRESHOLD, RRT_GOAL_BIAS)
    rrt_length = None
    if rrt_world_path:
        rrt_length = sum(
            ((rrt_world_path[i + 1][0] - rrt_world_path[i][0]) ** 2 +
             (rrt_world_path[i + 1][1] - rrt_world_path[i][1]) ** 2) ** 0.5
            for i in range(len(rrt_world_path) - 1)
        )
        r_collisions = 0
        for wx, wy in rrt_world_path:
            if _check_collision_with_cubes(grid_map, wx, wy, cube_positions, CUBE_HALF_EXTENTS):
                r_collisions += 1
        print("[RRT] 世界路径点数=%d, 路径总长=%.4f m, 碰撞点=%d" %
              (len(rrt_world_path), rrt_length, r_collisions))
    # 对比表格
    print("\n路径规划对比:")
    print("算法     | 路径总长(m) | 耗时(s)    | 节点数   | 碰撞验证")
    a_len = sum(
        ((astar_world_path[i + 1][0] - astar_world_path[i][0]) ** 2 +
         (astar_world_path[i + 1][1] - astar_world_path[i][1]) ** 2) ** 0.5
        for i in range(len(astar_world_path) - 1)
    ) if astar_world_path else float('inf')
    print("  %-7s | %11.4f | %10.6f | %8d | %s" %
          ("A*", a_len, astar_time,
           len(astar_world_path) if astar_world_path else 0,
           "PASS" if a_collisions == 0 else "FAIL(%d)" % a_collisions))
    print("  %-7s | %11.4f | %10.6f | %8d | %s" %
          ("RRT", rrt_length if rrt_world_path else -1, rrt_time,
           len(rrt_world_path) if rrt_world_path else 0,
           "PASS" if r_collisions == 0 else "FAIL(%d)" % r_collisions))
    # 二维可视化路径图
    grid_map.draw_paths(
        astar_grid_path=astar_grid_path,
        rrt_world_path=rrt_world_path,
        start_xy=PATH_START,
        goal_xy=PATH_GOAL,
    )
    grid_map.draw_paths_combined(
        astar_grid_path=astar_grid_path,
        rrt_world_path=rrt_world_path,
        start_xy=PATH_START,
        goal_xy=PATH_GOAL,
    )
    # 操作提示
    print("\n  操作提示: [A]键A*路径  [R]键RRT路径  [ESC]键退出\n")
    astar_executed = False
    rrt_executed = False
    def _key_triggered(ks) -> bool:
        if isinstance(ks, int):
            return bool(ks & p.KEY_WAS_TRIGGERED)
        if isinstance(ks, tuple) and ks:
            return bool(ks[0] & p.KEY_WAS_TRIGGERED)
        return False

    try:
        while True:
            scene.step(1)
            keys = p.getKeyboardEvents(physicsClientId=scene.client_id)

            # A键触发A*路径执行
            if ord('a') in keys and _key_triggered(keys[ord('a')]):
                if not astar_executed and astar_world_path:
                    print("\n[主程序] [TRIGGER] A键按下，沿A*路径运动\n")
                    execute_path(robot, grid_map, astar_world_path, "A*",
                                 cube_positions, CUBE_HALF_EXTENTS,
                                 scene.client_id)
                    astar_executed = True
                elif astar_executed:
                    print("[主程序] A*路径已执行过")

            # R键触发RRT路径执行
            if ord('r') in keys and _key_triggered(keys[ord('r')]):
                if not rrt_executed and rrt_world_path:
                    print("\n[主程序] [TRIGGER] R键按下，沿RRT路径运动\n")
                    execute_path(robot, grid_map, rrt_world_path, "RRT",
                                 cube_positions, CUBE_HALF_EXTENTS,
                                 scene.client_id)
                    rrt_executed = True
                elif rrt_executed:
                    print("[主程序] RRT路径已执行过")

            # ESC退出
            if 27 in keys and _key_triggered(keys[27]):
                print("[主程序] ESC按下，退出仿真。")
                break

    except KeyboardInterrupt:
        print("\n[主程序] KeyboardInterrupt, 退出。")
    finally:
        scene.close()


if __name__ == '__main__':
    main()
