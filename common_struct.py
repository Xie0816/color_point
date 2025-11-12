import numpy as np

# 激光雷达参数
HORIZONTAL_ANGULAR_RESOLUTION = 0.2 # 雷达水平角分辨率

# 传感器匹配
TSS_GAP = 50   #传感器时间间隔 ms

# 点云累积
TIME_LEN = 100    # 累积帧数
TIME_GAP = 1      # 累积间隔帧数
Z_max = 10        # 点云最大高度
Z_min = -10       # 点云最低高度

#BEV
MAP_SIZE = 500      # bev地图大小 pixel
RESOLUTION =0.2     # bev地图分辨率 m/pixel

# 图像
SCALE_FACTOR = 1     #图像缩放尺寸


# LM camera calib
LM_AR0231_Front = np.array(
    [[1004.55, -1944.58, 93.9691, 48470.2],
[571.708, -30.8496, -1926.77, 7056.09],
[0.998301, 0.0125654, 0.0568946, 74.0718]]
)
