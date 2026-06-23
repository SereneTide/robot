import pybullet as p
import pybullet_data

from config import (
    PYBULLET_GUI, GRAVITY, SIM_STEP_DT,
    PLANE_URDF,
    SHELF_URDF, SHELF_POS, SHELF_ORN,
    CUBE_NUM, CUBE_HALF_EXTENTS, CUBE_MASS, CUBE_POS_LIST, CUBE_COLORS,
    CAMERA_TARGET_POS,
)


class Scene:
    """仿真场景管理类，封装物理客户端初始化、场景元素加载与相机设置。"""

    def __init__(self):
        """初始化PyBullet物理仿真客户端，设置重力与时间步长。"""
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

        print("[场景] PyBullet物理仿真客户端初始化完成，模式: %s" %
              ("GUI" if PYBULLET_GUI else "DIRECT"))
        print("[场景] 重力: %s, 时间步长: %.4f s" % (str(GRAVITY), SIM_STEP_DT))

    def draw_workspace(self):
        """加载地面、货架、批量生成物块，设置全局相机视角。"""
        self._load_ground()
        self._load_shelf()
        self._create_blocks()
        self._setup_camera()
        print("[场景] 工作场景绘制完成：地面 + 货架 + %d个物块（地面摆放）" % CUBE_NUM)

    def _load_ground(self):
        """加载地面平面。"""
        self.plane_id = p.loadURDF(
            PLANE_URDF,
            basePosition=(0, 0, 0),
            physicsClientId=self.client_id,
        )
        print("[场景] 地面加载完成, ID=%d" % self.plane_id)

    def _load_shelf(self):
        """加载货架作为工业场景元素。"""
        self.shelf_id = p.loadSDF(
            SHELF_URDF,
            physicsClientId=self.client_id,
        )[0]  # loadSDF返回列表，取第一个物体ID
        p.resetBasePositionAndOrientation(
            self.shelf_id,
            SHELF_POS,
            p.getQuaternionFromEuler(SHELF_ORN[:3]),
            physicsClientId=self.client_id,
        )
        print("[场景] 货架加载完成, 位置=%s, ID=%d" % (str(SHELF_POS), self.shelf_id))

    def _create_blocks(self):
        """批量生成分拣物料方块，使用PyBullet碰撞与视觉形状创建彩色立方体。"""
        self.cube_ids = []
        for i in range(CUBE_NUM):
            pos = CUBE_POS_LIST[i]
            color = CUBE_COLORS[i]

            col_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=CUBE_HALF_EXTENTS,
                physicsClientId=self.client_id,
            )

            vis_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=CUBE_HALF_EXTENTS,
                rgbaColor=color,
                physicsClientId=self.client_id,
            )

            cube_id = p.createMultiBody(
                baseMass=CUBE_MASS,
                baseCollisionShapeIndex=col_shape_id,
                baseVisualShapeIndex=vis_shape_id,
                basePosition=pos,
                physicsClientId=self.client_id,
            )

            self.cube_ids.append(cube_id)
            print("[场景] 物块#%d 创建, 位置=%s, 颜色=%s, ID=%d" %
                  (i + 1, str(pos), str(color), cube_id))

    def _setup_camera(self):
        """设置全局可视化相机视角。"""
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=CAMERA_TARGET_POS,
            physicsClientId=self.client_id,
        )
        print("[场景] 相机视角设置完成, 注视点=%s" % str(CAMERA_TARGET_POS))

    def get_cube_positions(self) -> list:
        """获取所有物块当前世界坐标。"""
        positions = []
        for cube_id in self.cube_ids:
            pos, _ = p.getBasePositionAndOrientation(
                cube_id, physicsClientId=self.client_id)
            positions.append(pos)
        return positions

    def step(self, steps: int = 1):
        """推进仿真指定步数。"""
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def close(self):
        """断开仿真连接。"""
        p.disconnect(physicsClientId=self.client_id)
        print("[场景] 仿真连接已关闭。")
