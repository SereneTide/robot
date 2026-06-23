import pybullet as p
import time
import json
import os

from config import (
    SAFE_Z, GRASP_OFFSET_Z,
    CUBE_POS_LIST, CUBE_NUM,
)
from scene import Scene
from sim_robot import SimRobot

class TrajectoryLogger:
    """轨迹数据记录器，记录每个物块各阶段运动轨迹。"""

    def __init__(self):
        """初始化空记录列表。"""
        self.records = []
        self._start_time_global = time.time()

    def _now(self) -> float:
        """获取自仿真开始以来的相对时间（秒）。"""
        return time.time() - self._start_time_global

    def log_phase(self, block_id: int, phase: str,
                  start_pos: tuple, end_pos: tuple,
                  start_ee: tuple = None, end_ee: tuple = None):
        """
        记录一个运动阶段的轨迹数据。

        参数:
            block_id: 物块编号 (0~4)
            phase: 阶段名称
            start_pos: 阶段起始目标坐标
            end_pos: 阶段结束目标坐标
            start_ee: 阶段起始末端实际坐标
            end_ee: 阶段结束末端实际坐标
        """
        record = {
            'block_id': block_id,
            'phase': phase,
            'start_time': self._now(),
            'end_time': None,
            'start_target_pos': tuple(round(v, 4) for v in start_pos),
            'end_target_pos': tuple(round(v, 4) for v in end_pos),
            'start_ee_pos': tuple(round(v, 4) for v in start_ee) if start_ee else None,
            'end_ee_pos': None,
        }
        self.records.append(record)

    def complete_last(self, end_ee: tuple):
        """完成最后一条记录：填入结束时间和末端实际位置。"""
        if self.records:
            self.records[-1]['end_time'] = self._now()
            self.records[-1]['end_ee_pos'] = tuple(round(v, 4) for v in end_ee)

    def print_summary(self):
        """打印轨迹数据汇总表格。"""
        print("\n【轨迹数据汇总】")
        print("物块 | 阶段                 | 起始目标坐标               | 结束目标坐标               | 耗时(s)")
        for r in self.records:
            duration = r['end_time'] - r['start_time'] if r['end_time'] else 0
            print("  %d   | %-22s | (% 5.2f, % 5.2f, % 5.2f) | "
                  "(% 5.2f, % 5.2f, % 5.2f) | %.3f" %
                  (r['block_id'], r['phase'],
                   r['start_target_pos'][0], r['start_target_pos'][1], r['start_target_pos'][2],
                   r['end_target_pos'][0], r['end_target_pos'][1], r['end_target_pos'][2],
                   duration))
        total_time = sum(r['end_time'] - r['start_time']
                         for r in self.records if r['end_time'])
        print("总耗时: %.3f 秒 | 记录数: %d\n" % (total_time, len(self.records)))

    def save_to_file(self, filepath: str = "trajectory_data.json"):
        """将轨迹数据保存为JSON文件。"""
        save_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(save_dir, filepath)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        print("[轨迹] 数据已保存至: %s" % full_path)


class PickAndPlacePipeline:
    """物块遍历抓取流水线，按指定路径模式依次遍历所有物块，记录全程轨迹。"""

    def __init__(self, robot: SimRobot, logger: TrajectoryLogger):
        self.robot = robot
        self.logger = logger
        self.cube_positions = CUBE_POS_LIST

    def run(self):
        """
        执行完整抓取流水线：
        对每个物块依次执行水平接近 -> 垂直下落 -> 垂直上升，物块间水平平移过渡。
        """
        print("\n[流水线] 启动: 物块数=%d, 安全高度=%.3f m, 抓取偏移=%.3f m\n" %
              (CUBE_NUM, SAFE_Z, GRASP_OFFSET_Z))

        print("[流水线] 机械臂回归待机位...")
        self.robot.go_to_rest(wait=True)

        for i, cube_pos in enumerate(self.cube_positions):
            cx, cy, cz = cube_pos
            safe_pos = (cx, cy, SAFE_Z)
            grasp_pos = (cx, cy, cz + GRASP_OFFSET_Z)

            print("\n[物块 %d/%d] 目标坐标: (%.3f, %.3f, %.3f)" %
                  (i + 1, CUBE_NUM, cx, cy, cz))
            if i == 0:
                phase_name = 'horizontal_approach'
            else:
                phase_name = 'horizontal_transfer'

            start_ee = self.robot.get_end_effector_pos()
            print("[流水线] 阶段%s -> 移动到物块#%d正上方 (%.3f, %.3f, %.3f)" %
                  (phase_name, i + 1, cx, cy, SAFE_Z))
            self.logger.log_phase(
                block_id=i, phase=phase_name,
                start_pos=start_ee, end_pos=safe_pos,
                start_ee=start_ee,
            )

            self.robot.move_one_point(cx, cy, SAFE_Z, log=False)
            end_ee = self.robot.get_end_effector_pos()
            self.logger.complete_last(end_ee)
            start_ee = self.robot.get_end_effector_pos()
            print("[流水线] 阶段vertical_descend -> 下落至物块#%d上方 "
                  "(%.3f, %.3f, %.3f)" % (i + 1, cx, cy, cz + GRASP_OFFSET_Z))
            self.logger.log_phase(
                block_id=i, phase='vertical_descend',
                start_pos=start_ee, end_pos=grasp_pos,
                start_ee=start_ee,
            )
            self.robot.move_one_point(cx, cy, cz + GRASP_OFFSET_Z, log=False)
            end_ee = self.robot.get_end_effector_pos()
            self.logger.complete_last(end_ee)
            start_ee = self.robot.get_end_effector_pos()
            print("[流水线] 阶段vertical_ascend -> 抬升至安全高度 "
                  "(%.3f, %.3f, %.3f)" % (cx, cy, SAFE_Z))
            self.logger.log_phase(
                block_id=i, phase='vertical_ascend',
                start_pos=start_ee, end_pos=safe_pos,
                start_ee=start_ee,
            )
            self.robot.move_one_point(cx, cy, SAFE_Z, log=False)
            end_ee = self.robot.get_end_effector_pos()
            self.logger.complete_last(end_ee)
            current_ee = self.robot.get_end_effector_pos()
            print("[流水线] 物块#%d 完成, 当前末端坐标: (%.3f, %.3f, %.3f)" %
                  (i + 1, current_ee[0], current_ee[1], current_ee[2]))

        print("\n[流水线] 全部%d个物块遍历完成，回归待机位..." % CUBE_NUM)
        self.robot.go_to_rest(wait=True)
        self.logger.print_summary()

def main():
    """主入口：初始化场景 -> 加载机械臂 -> 启动仿真循环 -> 监听键盘Z键触发流水线。"""
    print("[主程序] 机械臂视觉控制仿真, 物块数=5")

    scene = Scene()
    scene.draw_workspace()
    robot = SimRobot(scene_client_id=scene.client_id)
    traj_logger = TrajectoryLogger()
    pipeline = PickAndPlacePipeline(robot, traj_logger)

    print("\n[主程序] 仿真初始化完成，等待物理稳定...")
    for _ in range(200):
        p.stepSimulation(physicsClientId=scene.client_id)
    print("[主程序] 物理稳定完成。")

    robot.go_to_rest(wait=True)
    print("  操作提示:")
    print("    按下 [Z] 键 -> 启动完整抓取流水线")
    print("    按下 [ESC] 键 -> 退出仿真")

    pipeline_executed = False

    try:
        while True:
            scene.step(1)

            keys = p.getKeyboardEvents(physicsClientId=scene.client_id)
            z_key = ord('z')
            if z_key in keys:
                key_state = keys[z_key]
                if isinstance(key_state, int):
                    triggered = bool(key_state & p.KEY_WAS_TRIGGERED)
                elif isinstance(key_state, tuple):
                    triggered = key_state[0] & p.KEY_WAS_TRIGGERED if key_state else False
                else:
                    triggered = False
                if triggered:
                    if not pipeline_executed:
                        print("\n[主程序] Z键按下，启动抓取流水线\n")
                        pipeline.run()
                        pipeline_executed = True
                        print("\n[主程序] 流水线执行完毕。按ESC退出，或继续观察仿真。")
                    else:
                        print("[主程序] 流水线已执行过，按ESC退出。")
            esc_key = 27
            if esc_key in keys:
                key_state = keys[esc_key]
                if isinstance(key_state, int):
                    triggered = bool(key_state & p.KEY_WAS_TRIGGERED)
                elif isinstance(key_state, tuple):
                    triggered = key_state[0] & p.KEY_WAS_TRIGGERED if key_state else False
                else:
                    triggered = False
                if triggered:
                    print("\n[主程序] ESC键按下，退出仿真。")
                    break
    except KeyboardInterrupt:
        print("\n[主程序] KeyboardInterrupt, 退出仿真。")
    finally:
        if pipeline_executed:
            traj_logger.save_to_file()
        scene.close()

if __name__ == '__main__':
    main()
