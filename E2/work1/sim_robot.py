"""
机械臂控制模块 (sim_robot.py)
SimRobot类 —— 封装KUKA IIWA机械臂的初始化、运动控制、逆运动学、末端位置获取。
"""

import pybullet as p
import numpy as np
import time

from config import (
    ROBOT_URDF_PATH, ROBOT_BASE_POS, ROBOT_BASE_ORN,
    JOINT_NUM, JOINT_INDICES,
    REST_JOINTS, PHOTO_JOINTS,
    JOINT_TOLERANCE, POSITION_TOLERANCE,
    MAX_FORCE, MAX_VELOCITY,
    MOVE_STEPS, SIM_STEPS_PER_CMD,
    SAFE_Z, GRASP_OFFSET_Z,
)


class SimRobot:
    """KUKA IIWA 机械臂控制类，封装机械臂加载、姿态控制、末端移动、抓取动作等接口。"""

    def __init__(self, scene_client_id: int):
        """
        初始化机械臂对象。

        参数:
            scene_client_id: PyBullet物理客户端ID
        """
        self.client_id = scene_client_id

        self.robot_id = p.loadURDF(
            ROBOT_URDF_PATH,
            basePosition=ROBOT_BASE_POS,
            baseOrientation=ROBOT_BASE_ORN,
            useFixedBase=True,       # 固定基座，防止机械臂倾倒
            physicsClientId=self.client_id,
        )

        self.ee_link_index = 6

        self._disable_default_damping()

        print("[机械臂] KUKA IIWA加载完成, 机器人ID=%d, 末端Link索引=%d" %
              (self.robot_id, self.ee_link_index))
        print("[机械臂] 关节数=%d, 基座位置=%s" % (JOINT_NUM, str(ROBOT_BASE_POS)))

    def _disable_default_damping(self):
        """禁用PyBullet默认关节阻尼，预设位置控制模式，使关节完全由位置控制驱动。"""
        for j in JOINT_INDICES:
            p.changeDynamics(
                self.robot_id, j,
                jointDamping=0.0,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self.client_id,
            )
            current_pos = p.getJointState(self.robot_id, j,
                                          physicsClientId=self.client_id)[0]
            p.setJointMotorControl2(
                self.robot_id, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=current_pos,
                force=MAX_FORCE,
                maxVelocity=MAX_VELOCITY,
                physicsClientId=self.client_id,
            )

    def _set_joint_positions(self, joint_angles: list):
        """设置所有关节的目标角度（位置控制模式）。"""
        for j in JOINT_INDICES:
            p.setJointMotorControl2(
                self.robot_id, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[j],
                force=MAX_FORCE,
                maxVelocity=MAX_VELOCITY,
                physicsClientId=self.client_id,
            )

    def _get_joint_positions(self) -> list:
        """获取当前所有关节的实际角度。"""
        joint_states = p.getJointStates(
            self.robot_id,
            JOINT_INDICES,
            physicsClientId=self.client_id,
        )
        return [js[0] for js in joint_states]

    def _joints_reached(self, target_joints: list) -> bool:
        """判断所有关节是否在容差范围内到达目标角度。"""
        current = self._get_joint_positions()
        for cur, tgt in zip(current, target_joints):
            if abs(cur - tgt) > JOINT_TOLERANCE:
                return False
        return True

    def go_to_rest(self, wait: bool = True):
        """机械臂回到待机安全位，使用宽松容差快速归位。"""
        print("[机械臂] 执行 go_to_rest() -> 待机安全位")
        self._set_joint_positions(REST_JOINTS)

        if wait:
            for _ in range(800):
                p.stepSimulation(physicsClientId=self.client_id)
            print("[机械臂] 待机位归位完成（宽松容差）")

    def go_to_photo(self, wait: bool = True):
        """机械臂移动至拍照位。"""
        print("[机械臂] 执行 go_to_photo() -> 拍照位")
        self._set_joint_positions(PHOTO_JOINTS)

        if wait:
            self._wait_until_joints_reached(PHOTO_JOINTS, label="拍照位")

    def goto_joint_pose(self, joint_angles: list, tolerance: float = None,
                        trail_color: tuple = None) -> bool:
        """
            输入关节弧度，控制各关节运动至目标角度。

            参数:
                joint_angles: 目标关节角度列表 (弧度, 长度=7)
                tolerance: 到达容差，默认使用全局JOINT_TOLERANCE
                trail_color: 若提供(R,G,B)元组，绘制末端轨迹线条。
            返回: True成功到达 / False超时未到达
        """
        if tolerance is None:
            tolerance = JOINT_TOLERANCE

        print("[机械臂] 执行 goto_joint_pose() -> 目标关节=%s" %
              str([round(a, 3) for a in joint_angles]))
        self._set_joint_positions(joint_angles)

        return self._wait_until_joints_reached(joint_angles, label="关节目标",
                                               trail_color=trail_color)

    def go_to_point(self, x: float, y: float, z: float,
                    orn: tuple = None) -> tuple:
        """
        输入三维笛卡尔坐标，通过逆运动学求解并运动至目标末端点位。

        参数:
            x, y, z: 目标末端世界坐标
            orn: 目标末端姿态四元数 (默认None = 保持朝下)
        返回: (ik_success, collision_free)
        """
        if orn is None:
            orn = p.getQuaternionFromEuler([0, np.pi, 0])

        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            targetPosition=[x, y, z],
            targetOrientation=orn,
            physicsClientId=self.client_id,
        )

        ik_success = len(joint_angles) == JOINT_NUM

        if ik_success:
            print("[机械臂] 逆解成功 -> 目标点(% .3f, % .3f, % .3f)" % (x, y, z))
            print("[机械臂] 逆解关节角=%s" % str([round(a, 3) for a in joint_angles]))
            reached = self.goto_joint_pose(list(joint_angles))
            if not reached:
                print("[机械臂] [WARNING] 目标点(% .3f, % .3f, % .3f) 未完全到达" % (x, y, z))
        else:
            print("[机械臂] [FAIL] 逆解失败 -> 目标点(% .3f, % .3f, % .3f)" % (x, y, z))

        collision_free = True
        return (ik_success, collision_free)

    def move_one_point(self, x: float, y: float, z: float,
                       steps: int = None, orn: tuple = None,
                       log: bool = True,
                       collision_callback: callable = None,
                       trail_color: tuple = None) -> tuple:
        """
        机械臂末端平滑移动至单个空间坐标，采用线性插值->IK求解->逐点位置控制。

        参数:
            x, y, z: 目标末端世界坐标
            steps: 插值步数 (默认MOVE_STEPS)
            orn: 目标末端姿态四元数
            log: 是否打印日志
            collision_callback: 每插值步回调，签名为 callback(interp_pos: tuple) -> bool，
                                返回True表示可提前终止运动。
            trail_color: 若提供(R,G,B)元组，则在末端移动路径上绘制可视化轨迹线条。
        返回: (success, trajectory)
        """
        if steps is None:
            steps = MOVE_STEPS

        if orn is None:
            orn = p.getQuaternionFromEuler([0, np.pi, 0])

        start_pos = self.get_end_effector_pos()
        start = np.array(start_pos)
        target = np.array([x, y, z])

        if log:
            print("[机械臂] move_one_point: (% .3f, % .3f, % .3f) -> (% .3f, % .3f, % .3f), 步数=%d" %
                  (start[0], start[1], start[2], target[0], target[1], target[2], steps))

        trajectory = []
        early_stop = False
        prev_trail_pos = start_pos
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            interp = start + alpha * (target - start)
            trajectory.append(tuple(interp))

            joint_angles = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                targetPosition=interp.tolist(),
                targetOrientation=orn,
                physicsClientId=self.client_id,
            )

            self._set_joint_positions(list(joint_angles))
            for _ in range(SIM_STEPS_PER_CMD):
                p.stepSimulation(physicsClientId=self.client_id)
                if trail_color is not None:
                    cur_pos = self.get_end_effector_pos()
                    p.addUserDebugLine(
                        prev_trail_pos, cur_pos,
                        lineColorRGB=trail_color,
                        lineWidth=2,
                        lifeTime=0,
                        physicsClientId=self.client_id,
                    )
                    prev_trail_pos = cur_pos
                if collision_callback is not None:
                    actual_pos = self.get_end_effector_pos()
                    if collision_callback(actual_pos):
                        early_stop = True
                        break
            if early_stop:
                break

        final_pos = self.get_end_effector_pos()
        error = np.linalg.norm(np.array(final_pos) - target)
        success = (error < POSITION_TOLERANCE) or early_stop

        if log:
            tag = "[EARLY_STOP]" if early_stop else ("[OK]" if error < POSITION_TOLERANCE else "[WARN]偏差过大")
            print("[机械臂] %s -> 最终位置(% .3f, % .3f, % .3f), 误差=% .4f m" %
                  (tag, final_pos[0], final_pos[1], final_pos[2], error))

        return (success, trajectory)

    def pick_one_object(self, obj_pos: tuple) -> bool:
        """
        执行完整抓取动作流程：移动到物块正上方 -> 下落 -> 抬升回安全高度。

        参数:
            obj_pos: 物块世界坐标 (x, y, z)
        返回: True 动作完成
        """
        x, y, z = obj_pos
        safe_pos = (x, y, SAFE_Z)
        grasp_pos = (x, y, z + GRASP_OFFSET_Z)

        print("\n[机械臂] ==== 开始抓取流程, 目标物块位置(% .3f, % .3f, % .3f) ====" % (x, y, z))

        print("[机械臂] 移动到物块正上方 SAFE_Z=%.3f" % SAFE_Z)
        self.move_one_point(*safe_pos)

        print("[机械臂] 下落至物块上方")
        self.move_one_point(*grasp_pos)

        print("[机械臂] 抬升至安全高度")
        self.move_one_point(*safe_pos)

        print("[机械臂] ==== 抓取流程完成 ====\n")
        return True

    def get_end_effector_pos(self) -> tuple:
        """获取机械臂末端执行器当前世界坐标。"""
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return tuple(link_state[4])

    def _wait_until_joints_reached(self, target_joints: list,
                                    label: str = "",
                                    max_iter: int = 5000,
                                    trail_color: tuple = None) -> bool:
        """
        阻塞等待直到所有关节到达目标角度（或超时）。

        参数:
            target_joints: 目标关节角度
            label: 日志标签
            max_iter: 最大等待迭代次数
            trail_color: 若提供(R,G,B)元组，绘制末端轨迹线条。
        返回: True到达 / False超时
        """
        steps_per_check = 5
        prev_pos = self.get_end_effector_pos()
        for check_count in range(max_iter // steps_per_check):
            for _ in range(steps_per_check):
                p.stepSimulation(physicsClientId=self.client_id)
                if trail_color is not None:
                    cur_pos = self.get_end_effector_pos()
                    p.addUserDebugLine(
                        prev_pos, cur_pos,
                        lineColorRGB=trail_color,
                        lineWidth=2,
                        lifeTime=0,
                        physicsClientId=self.client_id,
                    )
                    prev_pos = cur_pos
            if self._joints_reached(target_joints):
                total_steps = (check_count + 1) * steps_per_check
                if label:
                    print("[机械臂] %s 到位, 耗时步数=%d" % (label, total_steps))
                return True
        if label:
            print("[机械臂] [TIMEOUT] %s 超时未到达 (max_iter=%d)" % (label, max_iter))
        return False


def trans_point_from_camera_to_base(point_cam: tuple,
                                     cam_world_pos: tuple,
                                     cam_world_orn: tuple) -> tuple:
    """
    相机坐标系三维点 -> 机械臂基座坐标系世界坐标。

    参数:
        point_cam: 相机坐标系下的点 (x_c, y_c, z_c)
        cam_world_pos: 相机在世界坐标系下的位置
        cam_world_orn: 相机在世界坐标系下的姿态四元数
    返回: (x_w, y_w, z_w) 世界坐标
    """
    cam_rot_matrix = np.array(
        p.getMatrixFromQuaternion(cam_world_orn)).reshape(3, 3)
    point_cam_arr = np.array(point_cam)
    point_world = cam_rot_matrix.dot(point_cam_arr) + np.array(cam_world_pos)
    return tuple(point_world)
