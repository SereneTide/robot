import pybullet as p
import time
import numpy as np
from config import (
    PYBULLET_GUI, CUBE_NUM, CUBE_HALF_EXTENTS,
    GRID_RESOLUTION, GRID_MARGIN,
    ASTAR_HEURISTIC, ASTAR_8DIR,
    RRT_STEP_SIZE, RRT_MAX_ITER, RRT_GOAL_THRESHOLD, RRT_GOAL_BIAS,
    RRT3D_STEP_SIZE, RRT3D_MAX_ITER, RRT3D_GOAL_THRESHOLD, RRT3D_GOAL_BIAS,
    RRT3D_Z_MIN, RRT3D_Z_MAX,
    PATH_START, PATH_GOAL, PATH_Z, SAFE_Z,
    PATH3D_START, PATH3D_GOAL,
    ROBOT_CLEARANCE,
)
from scene import Scene
from sim_robot import SimRobot
from map_tools import MapFromPoints, draw_rrt_3d
from path_astar import astar_plan
from path_rrt import rrt_plan, rrt_plan_3d, _point_in_cubes


def _check_collision_with_cubes_2d(grid_map, wx: float, wy: float,
                                    cube_positions: list,
                                    half_extents: tuple,
                                    clearance: float = 0.0) -> bool:
    """检测末端 (wx,wy,PATH_Z) 是否与任何物块碰撞（含安全包络）。"""
    hx, hy = half_extents[:2]
    margin = clearance
    for cx, cy, _ in cube_positions:
        if abs(wx - cx) <= hx + margin and abs(wy - cy) <= hy + margin:
            return True
    return False


def _recover_to_safe(robot, cube_positions: list, half_extents: tuple,
                     clearance: float, is_3d: bool = False):
    """碰撞恢复：抬升至SAFE_Z清除物理卡死状态。"""
    curr = robot.get_end_effector_pos()
    print("[恢复] 抬升至安全高度 (%.3f, %.3f, %.3f) → z=%.3f" %
          (curr[0], curr[1], curr[2], SAFE_Z))
    robot.move_one_point(curr[0], curr[1], SAFE_Z, log=False)


def _find_next_safe_waypoint_2d(world_path: list, start_idx: int,
                                  cube_positions: list, half_extents: tuple,
                                  clearance: float) -> int:
    """从start_idx开始扫描，返回第一个无碰撞的2D路径点索引。找不到返回-1。"""
    for j in range(start_idx, len(world_path)):
        wx, wy = world_path[j]
        if not _check_collision_with_cubes_2d(None, wx, wy, cube_positions,
                                               half_extents, clearance):
            return j
    return -1


def _find_next_safe_waypoint_3d(world_path: list, start_idx: int,
                                  cube_positions: list, half_extents: tuple,
                                  clearance: float) -> int:
    """从start_idx开始扫描，返回第一个无碰撞的3D路径点索引。找不到返回-1。"""
    for j in range(start_idx, len(world_path)):
        wx, wy, wz = world_path[j]
        if not _point_in_cubes(wx, wy, wz, cube_positions, half_extents, clearance):
            return j
    return -1


def _draw_path_marker(pos: tuple, color: tuple, size: float = 0.02,
                      client_id: int = 0):
    """在场景中放置一个彩色小球作为路径标记。"""
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=size,
                                rgbaColor=color,
                                physicsClientId=client_id)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                              basePosition=pos, physicsClientId=client_id)
    return body


def execute_path_3d(robot, world_path_3d: list, label: str,
                    cube_positions: list, half_extents: tuple,
                    client_id: int):
    """
    驱动机械臂末端沿3D路径点列表移动（含碰撞预检+恢复）。
    流程: 安全位 → 起点SAFE_Z → 下降至起点Z → 沿3D路径运动 → 终点抬升SAFE_Z。
    路径中遇到碰撞点时自动抬升SAFE_Z恢复并跳过。
    """
    print("\n[路径执行3D] %s: 点数=%d" % (label, len(world_path_3d)))

    sx, sy, sz = world_path_3d[0]
    gx, gy, gz = world_path_3d[-1]
    collision_count = 0
    unreachable_count = 0

    # 阶段1: 移至起点正上方(SAFE_Z)
    print("[%s-3D] 阶段1: 移至起点正上方 (%.3f, %.3f, %.3f)" % (label, sx, sy, SAFE_Z))
    robot.move_one_point(sx, sy, SAFE_Z, log=True)

    # 阶段2: 垂直下降至起点Z
    print("[%s-3D] 阶段2: 下降至起点高度 (%.3f, %.3f, %.3f)" % (label, sx, sy, sz))
    robot.move_one_point(sx, sy, sz, log=True)

    # 阶段3: 沿3D路径逐点运动（含碰撞预检与恢复）
    print("[%s-3D] 阶段3: 沿3D路径运动 (%d个点)" % (label, len(world_path_3d)))
    i = 0
    while i < len(world_path_3d):
        wx, wy, wz = world_path_3d[i]
        if i % max(1, len(world_path_3d) // 10) == 0:
            print("[%s-3D] 第%d/%d点: (%.3f, %.3f, %.3f)" %
                  (label, i + 1, len(world_path_3d), wx, wy, wz))

        # 碰撞预检
        if _point_in_cubes(wx, wy, wz, cube_positions, half_extents, ROBOT_CLEARANCE):
            collision_count += 1
            print("[碰撞预警3D] %s 第%d点 (%.3f,%.3f,%.3f) 与物块碰撞!" %
                  (label, i + 1, wx, wy, wz))
            _recover_to_safe(robot, cube_positions, half_extents, ROBOT_CLEARANCE)
            # 跳过后续连续碰撞点
            i = _find_next_safe_waypoint_3d(world_path_3d, i + 1,
                                             cube_positions, half_extents, ROBOT_CLEARANCE)
            if i < 0:
                print("[%s-3D] 剩余路径点全部碰撞，终止执行" % label)
                break
            wjx, wjy, wjz = world_path_3d[i]
            print("[%s-3D] 恢复: 跳过碰撞段，从第%d点 (%.3f,%.3f,%.3f) 继续" %
                  (label, i + 1, wjx, wjy, wjz))
            robot.move_one_point(wjx, wjy, SAFE_Z, log=False)
            robot.move_one_point(wjx, wjy, wjz, log=False)
            i += 1
            continue

        success, _ = robot.move_one_point(wx, wy, wz, log=False)
        if not success:
            unreachable_count += 1
            print("[不可达3D] %s 第%d点 末端未达目标, 偏差超容差" % (label, i + 1))
            _recover_to_safe(robot, cube_positions, half_extents, ROBOT_CLEARANCE)
            i = _find_next_safe_waypoint_3d(world_path_3d, i + 1,
                                             cube_positions, half_extents, ROBOT_CLEARANCE)
            if i < 0:
                print("[%s-3D] 剩余路径点全部不可达，终止执行" % label)
                break
            wjx, wjy, wjz = world_path_3d[i]
            print("[%s-3D] 恢复: 跳过不可达段，从第%d点继续" % (label, i + 1))
            robot.move_one_point(wjx, wjy, SAFE_Z, log=False)
            robot.move_one_point(wjx, wjy, wjz, log=False)
            i += 1
            continue

        i += 1

    # 阶段4: 终点抬升至安全高度
    print("[%s-3D] 阶段4: 终点抬升至安全高度 (%.3f, %.3f, %.3f)" % (label, gx, gy, SAFE_Z))
    robot.move_one_point(gx, gy, SAFE_Z, log=True)

    pass_ok = (collision_count == 0 and unreachable_count == 0)
    print("[%s-3D] 3D路径执行完成, 碰撞预警=%d, 不可达=%d, 避障=%s" %
          (label, collision_count, unreachable_count,
           "PASS" if pass_ok else "RECOVERED(%d/%d)" % (collision_count, unreachable_count)))
    return pass_ok


def execute_path(robot, grid_map, world_path: list, label: str,
                 cube_positions: list, half_extents: tuple,
                 client_id: int):
    """
    驱动机械臂末端沿路径点列表移动，含碰撞预检与自动恢复。
    流程: 安全位 → 起点正上方(SAFE_Z) → 下降至(PATH_Z) → 沿路径运动 → 终点抬升(SAFE_Z)。
    碰撞/不可达时自动抬升SAFE_Z恢复并跳过问题路径段。
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
    # 阶段3: 沿路径逐点运动（含碰撞预检与恢复）
    print("[%s] 阶段3: 沿路径运动 (%d个点)" % (label, len(world_path)))
    i = 0
    while i < len(world_path):
        wx, wy = world_path[i]
        if i % max(1, len(world_path) // 10) == 0:
            print("[%s] 第%d/%d点: (%.3f, %.3f)" %
                  (label, i + 1, len(world_path), wx, wy))

        # 碰撞预检
        if _check_collision_with_cubes_2d(grid_map, wx, wy, cube_positions,
                                           half_extents, ROBOT_CLEARANCE):
            collision_count += 1
            print("[碰撞预警] %s 第%d点 (%.3f,%.3f) 与物块碰撞!" %
                  (label, i + 1, wx, wy))
            _recover_to_safe(robot, cube_positions, half_extents, ROBOT_CLEARANCE)
            # 跳过后续连续碰撞点
            i = _find_next_safe_waypoint_2d(world_path, i + 1,
                                             cube_positions, half_extents, ROBOT_CLEARANCE)
            if i < 0:
                print("[%s] 剩余路径点全部碰撞，终止执行" % label)
                break
            wjx, wjy = world_path[i]
            print("[%s] 恢复: 跳过碰撞段，从第%d点 (%.3f,%.3f) 继续" %
                  (label, i + 1, wjx, wjy))
            robot.move_one_point(wjx, wjy, SAFE_Z, log=False)
            robot.move_one_point(wjx, wjy, PATH_Z, log=False)
            i += 1
            continue

        success, _ = robot.move_one_point(wx, wy, PATH_Z, log=False)
        if not success:
            unreachable_count += 1
            print("[不可达] %s 第%d点 末端未达目标, 偏差超容差" % (label, i + 1))
            _recover_to_safe(robot, cube_positions, half_extents, ROBOT_CLEARANCE)
            i = _find_next_safe_waypoint_2d(world_path, i + 1,
                                             cube_positions, half_extents, ROBOT_CLEARANCE)
            if i < 0:
                print("[%s] 剩余路径点全部不可达，终止执行" % label)
                break
            wjx, wjy = world_path[i]
            print("[%s] 恢复: 跳过不可达段，从第%d点继续" % (label, i + 1))
            robot.move_one_point(wjx, wjy, SAFE_Z, log=False)
            robot.move_one_point(wjx, wjy, PATH_Z, log=False)
            i += 1
            continue

        i += 1

    # 阶段4: 终点抬升至安全高度
    print("[%s] 阶段4: 终点抬升至安全高度 (%.3f, %.3f, %.3f)" % (label, gx, gy, SAFE_Z))
    robot.move_one_point(gx, gy, SAFE_Z, log=True)
    # 汇总
    pass_ok = (collision_count == 0 and unreachable_count == 0)
    print("[%s] 路径点数=%d, 碰撞预警=%d, 不可达=%d, 避障=%s" %
          (label, len(world_path), collision_count, unreachable_count,
           "PASS" if pass_ok else "RECOVERED(%d/%d)" % (collision_count, unreachable_count)))
    return pass_ok

def main():
    print("[主程序] ====== 启动 ======", flush=True)
    # 初始化场景与机械臂
    print("[主程序] 初始化场景...", flush=True)
    scene = Scene()
    scene.draw_workspace()
    print("[主程序] 加载机械臂...", flush=True)
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
        cube_positions, GRID_RESOLUTION, GRID_MARGIN,
        CUBE_HALF_EXTENTS, ROBOT_CLEARANCE)
    # 用于报告可视化的地图：只显示真实 cube 大小，不带膨胀
    grid_map_viz = MapFromPoints(
        cube_positions, GRID_RESOLUTION, GRID_MARGIN,
        CUBE_HALF_EXTENTS, clearance=0.0)
    grid_map.print_grid()
    # 路径规划参数
    start_uv = grid_map.world_to_grid(*PATH_START)
    goal_uv = grid_map.world_to_grid(*PATH_GOAL)
    print("[路径规划] 起点世界(% .3f,% .3f) -> UV(%d,%d)" %
          (PATH_START[0], PATH_START[1], start_uv[0], start_uv[1]))
    print("[路径规划] 终点世界(% .3f,% .3f) -> UV(%d,%d)" %
          (PATH_GOAL[0], PATH_GOAL[1], goal_uv[0], goal_uv[1]))
    # A* 规划
    print("\n[A*] 启发=%s 8方向=%s" % (ASTAR_HEURISTIC, ASTAR_8DIR))
    a_len = float('inf')
    a_collisions = -1
    astar_grid_path, astar_time = astar_plan(
        grid_map, start_uv, goal_uv, ASTAR_HEURISTIC, eight_dir=ASTAR_8DIR)
    astar_world_path = None
    if astar_grid_path:
        astar_world_path = grid_map.grid_to_world_path(astar_grid_path)
        a_len = sum(
            ((astar_world_path[i + 1][0] - astar_world_path[i][0]) ** 2 +
             (astar_world_path[i + 1][1] - astar_world_path[i][1]) ** 2) ** 0.5
            for i in range(len(astar_world_path) - 1)
        )
        a_collisions = 0
        for wx, wy in astar_world_path:
            if _check_collision_with_cubes_2d(grid_map, wx, wy, cube_positions,
                                           CUBE_HALF_EXTENTS, ROBOT_CLEARANCE):
                a_collisions += 1
        print("[A*] 世界路径点数=%d, 路径总长=%.4f m, 碰撞点=%d" %
              (len(astar_world_path), a_len, a_collisions))
    # RRT 规划
    print("\n[RRT]")
    t0 = time.perf_counter()
    rrt_world_path, rrt_time = rrt_plan(
        grid_map, PATH_START, PATH_GOAL,
        RRT_STEP_SIZE, RRT_MAX_ITER, RRT_GOAL_THRESHOLD, RRT_GOAL_BIAS)
    rrt_length = None
    r_collisions = -1
    if rrt_world_path:
        rrt_length = sum(
            ((rrt_world_path[i + 1][0] - rrt_world_path[i][0]) ** 2 +
             (rrt_world_path[i + 1][1] - rrt_world_path[i][1]) ** 2) ** 0.5
            for i in range(len(rrt_world_path) - 1)
        )
        r_collisions = 0
        for wx, wy in rrt_world_path:
            if _check_collision_with_cubes_2d(grid_map, wx, wy, cube_positions,
                                           CUBE_HALF_EXTENTS, ROBOT_CLEARANCE):
                r_collisions += 1
        print("[RRT] 世界路径点数=%d, 路径总长=%.4f m, 碰撞点=%d" %
              (len(rrt_world_path), rrt_length, r_collisions))
    # RRT 3D 规划
    print("\n[RRT3D] 三维空间规划 (z∈[%.3f,%.3f])" %
          (RRT3D_Z_MIN, RRT3D_Z_MAX), flush=True)
    bounds_3d = (grid_map.x_min, grid_map.x_max,
                 grid_map.y_min, grid_map.y_max,
                 RRT3D_Z_MIN, RRT3D_Z_MAX)
    print("[RRT3D] 开始搜索 max_iter=%d ..." % RRT3D_MAX_ITER, flush=True)
    rrt3d_world_path, rrt3d_time = rrt_plan_3d(
        cube_positions, CUBE_HALF_EXTENTS,
        PATH3D_START, PATH3D_GOAL, bounds_3d,
        RRT3D_STEP_SIZE, RRT3D_MAX_ITER,
        RRT3D_GOAL_THRESHOLD, RRT3D_GOAL_BIAS,
        clearance=ROBOT_CLEARANCE)
    print("[RRT3D] 搜索完成, 耗时=%.4fs" % rrt3d_time, flush=True)
    rrt3d_length = None
    rrt3d_collisions = 0
    if rrt3d_world_path:
        rrt3d_length = sum(
            np.sqrt((rrt3d_world_path[i+1][0]-rrt3d_world_path[i][0])**2 +
                    (rrt3d_world_path[i+1][1]-rrt3d_world_path[i][1])**2 +
                    (rrt3d_world_path[i+1][2]-rrt3d_world_path[i][2])**2)
            for i in range(len(rrt3d_world_path)-1)
        )
        # AABB碰撞验证
        for wx, wy, wz in rrt3d_world_path:
            if _point_in_cubes(wx, wy, wz, cube_positions,
                               CUBE_HALF_EXTENTS, ROBOT_CLEARANCE):
                rrt3d_collisions += 1
        print("[RRT3D] 路径点数=%d, 总长=%.4f m, 碰撞点=%d" %
              (len(rrt3d_world_path), rrt3d_length, rrt3d_collisions))
    # 对比表格
    print("\n路径规划对比:")
    print("算法     | 路径总长(m) | 耗时(s)    | 节点数   | 碰撞验证")
    print("  %-7s | %11.4f | %10.6f | %8d | %s" %
          ("A*", a_len, astar_time,
           len(astar_world_path) if astar_world_path else 0,
           "PASS" if a_collisions == 0 else "FAIL(%d)" % a_collisions))
    print("  %-7s | %11.4f | %10.6f | %8d | %s" %
          ("RRT", rrt_length if rrt_world_path else -1, rrt_time,
           len(rrt_world_path) if rrt_world_path else 0,
           "PASS" if r_collisions == 0 else "FAIL(%d)" % r_collisions))
    print("  %-7s | %11.4f | %10.6f | %8d | %s" %
          ("RRT3D", rrt3d_length if rrt3d_world_path else -1, rrt3d_time,
           len(rrt3d_world_path) if rrt3d_world_path else 0,
           "PASS" if rrt3d_collisions == 0 else "FAIL(%d)" % rrt3d_collisions))
    print("\n[主程序] 规划完成，打开可视化窗口...\n", flush=True)
    # 二维/三维可视化（规划完成后统一弹出，使用零膨胀的显示地图）
    grid_map_viz.draw_grid(start_xy=PATH_START, goal_xy=PATH_GOAL,
                           title="Grid Map | resolution=%.2fcm" %
                                 (GRID_RESOLUTION * 100))
    if rrt3d_world_path:
        draw_rrt_3d(cube_positions, CUBE_HALF_EXTENTS,
                    rrt3d_world_path=rrt3d_world_path,
                    start_xyz=PATH3D_START, goal_xyz=PATH3D_GOAL,
                    title="RRT 3D Obstacle Avoidance | %d cubes" % CUBE_NUM)
    grid_map_viz.draw_paths(
        astar_grid_path=astar_grid_path,
        rrt_world_path=rrt_world_path,
        start_xy=PATH_START,
        goal_xy=PATH_GOAL,
    )
    grid_map_viz.draw_paths_combined(
        astar_grid_path=astar_grid_path,
        rrt_world_path=rrt_world_path,
        start_xy=PATH_START,
        goal_xy=PATH_GOAL,
    )
    # 操作提示
    print("\n  操作提示: [A]键A*路径  [R]键RRT路径  [T]键RRT3D路径  [ESC]键退出\n")
    astar_executed = False
    rrt_executed = False
    rrt3d_executed = False
    # 边缘检测：跟踪每帧按键状态，上升沿触发
    prev_keys_down = set()
    def _get_key_state(ks):
        """从getKeyboardEvents返回值中提取按键状态int。"""
        if isinstance(ks, int):
            return ks
        if isinstance(ks, tuple) and ks:
            return ks[0]
        return 0

    try:
        while True:
            scene.step(1)
            raw_keys = p.getKeyboardEvents(physicsClientId=scene.client_id)
            cur_keys_down = set()
            for kc, ks in raw_keys.items():
                if _get_key_state(ks) & p.KEY_IS_DOWN:
                    cur_keys_down.add(kc)
            # 上升沿：本帧按下但上帧未按下
            triggered = cur_keys_down - prev_keys_down

            # A键触发A*路径执行
            if ord('a') in triggered:
                if not astar_executed and astar_world_path:
                    print("\n[主程序] [TRIGGER] A键按下，沿A*路径运动\n")
                    execute_path(robot, grid_map, astar_world_path, "A*",
                                 cube_positions, CUBE_HALF_EXTENTS,
                                 scene.client_id)
                    astar_executed = True
                elif astar_executed:
                    print("[主程序] A*路径已执行过")
                else:
                    print("[主程序] A*路径不存在，无法执行")

            # R键触发RRT路径执行
            if ord('r') in triggered:
                if not rrt_executed and rrt_world_path:
                    print("\n[主程序] [TRIGGER] R键按下，沿RRT路径运动\n")
                    execute_path(robot, grid_map, rrt_world_path, "RRT",
                                 cube_positions, CUBE_HALF_EXTENTS,
                                 scene.client_id)
                    rrt_executed = True
                elif rrt_executed:
                    print("[主程序] RRT路径已执行过")
                else:
                    print("[主程序] RRT路径不存在，无法执行")

            # T键触发RRT3D路径执行
            if ord('t') in triggered:
                if not rrt3d_executed and rrt3d_world_path:
                    print("\n[主程序] [TRIGGER] T键按下，沿RRT3D路径运动\n")
                    execute_path_3d(robot, rrt3d_world_path, "RRT3D",
                                    cube_positions, CUBE_HALF_EXTENTS,
                                    scene.client_id)
                    rrt3d_executed = True
                elif rrt3d_executed:
                    print("[主程序] RRT3D路径已执行过")
                else:
                    print("[主程序] RRT3D路径不存在，无法执行")

            # ESC退出
            if 27 in triggered:
                print("[主程序] ESC按下，退出仿真。")
                break

            prev_keys_down = cur_keys_down

    except KeyboardInterrupt:
        print("\n[主程序] KeyboardInterrupt, 退出。")
    finally:
        scene.close()


if __name__ == '__main__':
    main()
