# PyBullet仿真配置
PYBULLET_GUI = True
GRAVITY = (0, 0, -9.81)
SIM_STEP_DT = 1. / 240.

# 机械臂配置
ROBOT_URDF_PATH = "kuka_iiwa/model.urdf"
ROBOT_BASE_POS = (0.0, 0.0, 0.0)
ROBOT_BASE_ORN = (0, 0, 0, 1)

JOINT_NUM = 7
JOINT_INDICES = list(range(JOINT_NUM))

# 待机位关节角度 —— 末端朝上、高抬的安全姿态
REST_JOINTS = [0.0, 0.6, 0.0, -1.2, 0.0, 1.0, 0.0]
# 拍照位关节角度 —— 末端朝向工作台
PHOTO_JOINTS = [0.0, 0.8, 0.0, -0.8, 0.0, 1.2, 0.0]

# 运动控制参数
JOINT_TOLERANCE = 0.01
POSITION_TOLERANCE = 0.015
MAX_FORCE = 300.0
MAX_VELOCITY = 2.0
MOVE_STEPS = 120
SIM_STEPS_PER_CMD = 12

# 工作场景配置
PLANE_URDF = "plane.urdf"

TABLE_URDF = "table_square/table_square.urdf"
TABLE_POS = (0.45, 0.0, 0.0)
TABLE_ORN = (0, 0, 0, 1)

# 物块配置
CUBE_NUM = 5
CUBE_HALF_EXTENTS = (0.025, 0.025, 0.025)
CUBE_MASS = 0.1

# 5个物块地面坐标，z为物块半高使底面贴地，坐标在KUKA IIWA可达空间内
CUBE_POS_LIST = [
    (0.30, -0.15, 0.025),
    (0.40, -0.05, 0.025),
    (0.50,  0.05, 0.025),
    (0.55,  0.12, 0.025),
    (0.60,  0.17, 0.025),
]

# 物块颜色 (RGBA)
CUBE_COLORS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.2, 1.0, 1.0),
    (1.0, 1.0, 0.2, 1.0),
    (1.0, 0.5, 0.0, 1.0),
]

# 抓取运动参数
SAFE_Z = 0.50  # 安全高度（米），需保证所有物块xy位置下均在机械臂可达空间内
GRASP_OFFSET_Z = 0.03
GRASP_THRESHOLD = 0.05  # 末端与物块碰撞检测的距离阈值（米）

# 相机配置 (视觉模块预留)
CAMERA_EYE_POS = (0.5, -0.8, 1.2)
CAMERA_TARGET_POS = (0.45, 0.0, 0.15)
