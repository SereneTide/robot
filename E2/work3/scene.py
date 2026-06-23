"""
场景构建模块 —— 地面、货架、物块批量生成、相机视角设置。
"""
import pybullet as p
import pybullet_data
from config import (
    PYBULLET_GUI, GRAVITY, SIM_STEP_DT,
    PLANE_URDF, SHELF_URDF, SHELF_POS, SHELF_ORN,
    CUBE_NUM, CUBE_HALF_EXTENTS, CUBE_MASS, CUBE_POS_LIST, CUBE_COLORS,
    CAMERA_TARGET_POS,
)

class Scene:
    """仿真场景管理：物理客户端初始化、场景元素加载、可视化设置。"""

    def __init__(self):
        if PYBULLET_GUI:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*GRAVITY, physicsClientId=self.client_id)
        p.setTimeStep(SIM_STEP_DT, physicsClientId=self.client_id)
        self.plane_id = None
        self.shelf_id = None
        self.cube_ids = []
        print("[场景] 初始化完成, 模式=%s, 重力=%s" %
              ("GUI" if PYBULLET_GUI else "DIRECT", str(GRAVITY)))

    def draw_workspace(self):
        """绘制完整工作场景：地面 + 货架 + 物块 + 相机。"""
        self._load_ground()
        self._load_shelf()
        self._create_blocks()
        self._setup_camera()
        print("[场景] 场景绘制完成: 地面+货架+%d个物块" % CUBE_NUM)

    def _load_ground(self):
        self.plane_id = p.loadURDF(PLANE_URDF, basePosition=(0, 0, 0),
                                   physicsClientId=self.client_id)
        print("[场景] 地面加载完成, ID=%d" % self.plane_id)

    def _load_shelf(self):
        self.shelf_id = p.loadSDF(SHELF_URDF, physicsClientId=self.client_id)[0]
        p.resetBasePositionAndOrientation(
            self.shelf_id, SHELF_POS,
            p.getQuaternionFromEuler(SHELF_ORN[:3]),
            physicsClientId=self.client_id)
        print("[场景] 货架加载完成, 位置=%s" % str(SHELF_POS))

    def _create_blocks(self):
        self.cube_ids = []
        for i in range(CUBE_NUM):
            pos, color = CUBE_POS_LIST[i], CUBE_COLORS[i]
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=CUBE_HALF_EXTENTS,
                physicsClientId=self.client_id)
            vis_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=CUBE_HALF_EXTENTS, rgbaColor=color,
                physicsClientId=self.client_id)
            cube_id = p.createMultiBody(
                baseMass=CUBE_MASS, baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape, basePosition=pos,
                physicsClientId=self.client_id)
            self.cube_ids.append(cube_id)
            print("[场景] 物块#%d 创建, 位置=%s" % (i + 1, str(pos)))

    def _setup_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0, cameraYaw=45, cameraPitch=-35,
            cameraTargetPosition=CAMERA_TARGET_POS,
            physicsClientId=self.client_id)

    def get_cube_positions(self) -> list:
        """获取所有物块当前世界坐标，返回 [(x, y, z), ...]"""
        positions = []
        for cube_id in self.cube_ids:
            pos, _ = p.getBasePositionAndOrientation(
                cube_id, physicsClientId=self.client_id)
            positions.append(pos)
        return positions

    def step(self, steps: int = 1):
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def close(self):
        p.disconnect(physicsClientId=self.client_id)
        print("[场景] 仿真连接已关闭。")
