# PyBullet仿真
PYBULLET_GUI = True
GRAVITY = (0, 0, -9.81)
SIM_STEP_DT = 1. / 240.

# 机械臂
ROBOT_URDF_PATH = "kuka_iiwa/model.urdf"
ROBOT_BASE_POS = (0.0, 0.0, 0.0)
ROBOT_BASE_ORN = (0, 0, 0, 1)
JOINT_NUM = 7
JOINT_INDICES = list(range(JOINT_NUM))
REST_JOINTS = [0.0, 0.6, 0.0, -1.2, 0.0, 1.0, 0.0]
PHOTO_JOINTS = [0.0, 0.8, 0.0, -0.8, 0.0, 1.2, 0.0]
JOINT_TOLERANCE = 0.01
POSITION_TOLERANCE = 0.015
MAX_FORCE = 300.0
MAX_VELOCITY = 2.0
MOVE_STEPS = 120
SIM_STEPS_PER_CMD = 12

# 工作场景
PLANE_URDF = "plane.urdf"
SHELF_URDF = "kiva_shelf/model.sdf"
SHELF_POS = (0.45, 1.2, 0.0)
SHELF_ORN = (0, 0, 0, 1)
CUBE_NUM = 5
CUBE_HALF_EXTENTS = (0.025, 0.025, 0.025)
CUBE_MASS = 0.1

# 5个物块世界坐标 (x, y, z), z取半高使底面贴地
CUBE_POS_LIST = [
    (0.30, -0.25, 0.025),
    (0.45, -0.10, 0.025),
    (0.55,  0.15, 0.025),
    (0.40,  0.25, 0.025),
    (0.65, -0.05, 0.025),
]

CUBE_COLORS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 1.0, 0.2, 1.0),
    (0.2, 0.2, 1.0, 1.0),
    (1.0, 1.0, 0.2, 1.0),
    (1.0, 0.5, 0.0, 1.0),
]

# 抓取运动
SAFE_Z = 0.85
GRASP_OFFSET_Z = 0.03

# 相机
CAMERA_EYE_POS = (0.5, -0.8, 1.2)
CAMERA_TARGET_POS = (0.45, 0.0, 0.15)

# TSP算法超参数
SA_T_START = 1000.0
SA_T_END = 0.01
SA_ALPHA = 0.98
SA_MAX_ITER = 5000

GA_POP_SIZE = 100
GA_GENERATIONS = 500
GA_MUT_RATE = 0.15
GA_ELITE_RATE = 0.1
