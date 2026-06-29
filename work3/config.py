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
CUBE_NUM = 4
CUBE_HALF_EXTENTS = (0.0125, 0.0125, 0.0125)  # 2.5cm³ 物块（缩小50%，给绕行留空间）
CUBE_MASS = 0  # 质量=0，物块固定不动，不会被撞飞
CUBE_POS_LIST = [
    # ====== 底层 (z=0.03): 4个，两道交错墙迫使Z字绕行 ======
    # 物块直接放在 y=0 上，挡住起点→终点的直线
    # 两堵墙缺口交错：Wall1缺口在上方(y>0)，Wall2缺口在下方(y<0)
    # Wall1从x=0.42移到x=0.46，远离起点(0.35)，避免机械臂下降时撞击
    # 墙间距 0.12m，物块中心距 ≥ 0.12m
    # Wall 1 @ x=0.46 — 中心物块挡y=0，缺口在 y>0.0125
    (0.46,  0.00, 0.03), (0.46,  0.12, 0.03),
    # Wall 2 @ x=0.58 — 中心物块挡y=0，缺口在 y<-0.0125
    (0.58,  0.00, 0.03), (0.58, -0.12, 0.03),
]
CUBE_COLORS = (
    [(0.75, 0.25, 0.25, 1.0)] * 4   # 红系
)

# 抓取运动参数
SAFE_Z = 0.85
GRASP_OFFSET_Z = 0.03

# 末端执行器姿态：垂直指向地面（绕Y轴旋转180°）
# 该四元数使末端法兰的局部Z轴与世界-Z轴对齐
EE_TARGET_ORIENTATION = (0.0, 1.0, 0.0, 0.0)

# 相机配置
CAMERA_TARGET_POS = (0.45, 0.0, 0.15)

# 网格地图配置
GRID_RESOLUTION = 0.01
GRID_MARGIN = 0.15

# 机械臂安全包络/障碍物膨胀距离（米）
# 以末端为中心，向外预留的半径，用于粗略考虑连杆、关节、夹爪的体积
ROBOT_CLEARANCE = 0.03   # 3cm，保证路径从两堵墙的缝隙中绕行，避免大图绕行

# A*算法配置
ASTAR_HEURISTIC = 'euclidean'
ASTAR_8DIR = True

# RRT算法配置（2D）
RRT_STEP_SIZE = 0.02
RRT_MAX_ITER = 3000
RRT_GOAL_THRESHOLD = 0.03
RRT_GOAL_BIAS = 0.1

# RRT 3D算法配置
RRT3D_STEP_SIZE = 0.02
RRT3D_MAX_ITER = 2000
RRT3D_GOAL_THRESHOLD = 0.03
RRT3D_GOAL_BIAS = 0.15
RRT3D_Z_MIN = 0.04
RRT3D_Z_MAX = 0.15

# 路径规划起终点（2D）
PATH_START = (0.35, 0.0)
PATH_GOAL = (0.68, 0.0)
PATH_Z = 0.06  # 略高于物块顶层，避免末端贴地抖动

# 3D路径起终点（z 高于障碍物膨胀包络顶面 0.0725m，避免起点附近被困住）
PATH3D_START = (0.35, 0.0, 0.08)
PATH3D_GOAL = (0.68, 0.0, 0.08)
