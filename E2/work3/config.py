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
REST_JOINTS = [0.0, 0.6, 0.0, -1.2, 0.0, 1.0, 0.0]
JOINT_TOLERANCE = 0.01
POSITION_TOLERANCE = 0.015
MAX_FORCE = 300.0
MAX_VELOCITY = 2.0
MOVE_STEPS = 120
SIM_STEPS_PER_CMD = 12

# 工作场景配置
PLANE_URDF = "plane.urdf"
SHELF_URDF = "kiva_shelf/model.sdf"
SHELF_POS = (0.45, 1.2, 0.0)
SHELF_ORN = (0, 0, 0, 1)
CUBE_NUM = 5
CUBE_HALF_EXTENTS = (0.025, 0.025, 0.025)
CUBE_MASS = 0.1
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

# 抓取运动参数
SAFE_Z = 0.85
GRASP_OFFSET_Z = 0.03

# 相机配置
CAMERA_TARGET_POS = (0.45, 0.0, 0.15)

# 网格地图配置
GRID_RESOLUTION = 0.01
GRID_MARGIN = 0.15

# A*算法配置
ASTAR_HEURISTIC = 'euclidean'

# RRT算法配置
RRT_STEP_SIZE = 0.02
RRT_MAX_ITER = 5000
RRT_GOAL_THRESHOLD = 0.03
RRT_GOAL_BIAS = 0.1

# 路径规划起终点
PATH_START = (0.35, 0.0)
PATH_GOAL = (0.68, 0.0)
PATH_Z = 0.25
