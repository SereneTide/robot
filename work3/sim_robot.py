"""
机械臂控制模块 —— KUKA IIWA的初始化、运动控制、逆运动学、末端位置获取。
"""
import pybullet as p
import numpy as np
from config import (
    ROBOT_URDF_PATH, ROBOT_BASE_POS, ROBOT_BASE_ORN,
    JOINT_NUM, JOINT_INDICES, REST_JOINTS,
    JOINT_TOLERANCE, POSITION_TOLERANCE,
    MAX_FORCE, MAX_VELOCITY,
    MOVE_STEPS, SIM_STEPS_PER_CMD,
    EE_TARGET_ORIENTATION,
)

class SimRobot:
    """KUKA IIWA机械臂控制：加载、姿态控制、末端移动。"""

    def __init__(self, scene_client_id: int):
        self.client_id = scene_client_id
        self.robot_id = p.loadURDF(
            ROBOT_URDF_PATH, basePosition=ROBOT_BASE_POS,
            baseOrientation=ROBOT_BASE_ORN, useFixedBase=True,
            physicsClientId=self.client_id)
        self.ee_link_index = 6
        self._disable_default_damping()
        print("[机械臂] KUKA IIWA加载完成, ID=%d, 末端Link=%d" %
              (self.robot_id, self.ee_link_index))

    def _disable_default_damping(self):
        for j in JOINT_INDICES:
            p.changeDynamics(self.robot_id, j, jointDamping=0.0,
                             linearDamping=0.0, angularDamping=0.0,
                             physicsClientId=self.client_id)
            current_pos = p.getJointState(
                self.robot_id, j, physicsClientId=self.client_id)[0]
            p.setJointMotorControl2(
                self.robot_id, j, controlMode=p.POSITION_CONTROL,
                targetPosition=current_pos, force=MAX_FORCE,
                maxVelocity=MAX_VELOCITY, physicsClientId=self.client_id)

    def _set_joint_positions(self, joint_angles: list):
        for j in JOINT_INDICES:
            p.setJointMotorControl2(
                self.robot_id, j, controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angles[j], force=MAX_FORCE,
                maxVelocity=MAX_VELOCITY, physicsClientId=self.client_id)

    def go_to_rest(self, wait: bool = True):
        """机械臂回到待机安全位。"""
        print("[机械臂] go_to_rest()")
        self._set_joint_positions(REST_JOINTS)
        if wait:
            for _ in range(800):
                p.stepSimulation(physicsClientId=self.client_id)
            print("[机械臂] 待机位归位完成")

    def move_one_point(self, x: float, y: float, z: float,
                       orientation: tuple = None,
                       steps: int = None, log: bool = True) -> tuple:
        """末端平滑移动至目标空间坐标（Cartesian线性插值→IK→逐点控制）。返回 (success, trajectory)。"""
        if orientation is None:
            orientation = EE_TARGET_ORIENTATION
        if steps is None:
            steps = MOVE_STEPS
        start = np.array(self.get_end_effector_pos())
        target = np.array([x, y, z])
        if log:
            print("[机械臂] move: (% .3f,% .3f,% .3f)->(% .3f,% .3f,% .3f)" %
                  (start[0], start[1], start[2], target[0], target[1], target[2]))
        trajectory = []
        for i in range(1, steps + 1):
            alpha = i / float(steps)
            interp = start + alpha * (target - start)
            trajectory.append(tuple(interp))
            j_angles = p.calculateInverseKinematics(
                self.robot_id, self.ee_link_index,
                targetPosition=interp.tolist(),
                targetOrientation=orientation,
                physicsClientId=self.client_id)
            self._set_joint_positions(list(j_angles))
            for _ in range(SIM_STEPS_PER_CMD):
                p.stepSimulation(physicsClientId=self.client_id)
        final = self.get_end_effector_pos()
        error = np.linalg.norm(np.array(final) - target)
        success = error < POSITION_TOLERANCE
        if log:
            print("[机械臂] %s 最终=(% .3f,% .3f,% .3f) 误差=%.4f" %
                  ("[OK]" if success else "[WARN]", final[0], final[1], final[2], error))
        return (success, trajectory)

    def get_end_effector_pos(self) -> tuple:
        """获取末端执行器当前世界坐标 (x, y, z)。"""
        state = p.getLinkState(self.robot_id, self.ee_link_index,
                                computeForwardKinematics=True,
                                physicsClientId=self.client_id)
        return tuple(state[4])
