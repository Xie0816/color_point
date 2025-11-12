"""
激光雷达数据处理与可视化模块
主要功能：激光雷达点云处理、运动补偿、图像投影、BEV地图生成
作者：xie
日期：2025-11-12
"""

import math
import os
import time
import numpy as np
import torch
from torch_scatter import scatter_min
import open3d as o3d
from common_struct import *
import cv2
import matplotlib.pyplot as plt 


class Lidar():
    """
    激光雷达数据处理类
    功能：点云加载、运动补偿、坐标变换、点云累积
    """
    
    def __init__(self):
        """初始化激光雷达处理器"""
        self.device = torch.device("cuda")
        self.horizontal_angular_resolution = HORIZONTAL_ANGULAR_RESOLUTION

    def load_lidarbin(self, lidar_path):
        """
        加载并预处理激光雷达二进制文件
        
        输入:
            lidar_path (str): 激光雷达二进制文件路径
            
        输出:
            torch.Tensor: 过滤后的点云数据 [N, 3]，包含有效的(x,y,z)坐标
        """
        # 1. 加载到CPU并转为Tensor
        points_cpu = torch.from_numpy(np.fromfile(lidar_path, dtype=np.int32)).view(-1, 4)

        # 2. 转换到GPU (单位转换: cm -> m)
        points_gpu = points_cpu.float().to(self.device) / 100.0

        # 3. 分离坐标
        x, y, z, _ = points_gpu.unbind(dim=1)

        # 4. 使用数学比较替代按位操作
        mask = (
                torch.isfinite(x).float() *
                torch.isfinite(y).float() *
                torch.isfinite(z).float() *
                (~((x == 0) & (y == 0) & (z == 0))).float() *
                (x.abs() < 1000).float() *
                (y.abs() < 1000).float() *
                (z < Z_max).float() *
                (z > Z_min).float() *
                ((x.pow(2) + y.pow(2)).sqrt() >= 3).float()
        ).bool()

        return points_gpu[mask, :3]

    def angle_index(self, points):
        """
        将点云的(x,y)坐标转换为角度编号（0-1800）
        
        输入:
            points (torch.Tensor): 点云数据 [N, 3] (x,y,z)
            
        输出:
            torch.Tensor: 带角度编号的点云数据 [N, 4] (x,y,z,角度编号)
        """
        x, y = points[:, 0], points[:, 1]
        # 计算水平角度（-90°起始，顺时针扫描，范围[0°, 360°]）
        angles_deg = torch.atan2(y, -x) * (180 / math.pi)  # [-180°, 180°]
        angles_deg = torch.where(angles_deg < -180, angles_deg + 360, angles_deg)  # 转换到[0°, 360°]

        # 映射到0-1800的整数编号（0.2°分辨率）
        angle_indices = (angles_deg / self.horizontal_angular_resolution).long().clamp(0, 1800)  # [N]
        return torch.cat([points, angle_indices.unsqueeze(-1)], dim=-1)  # [N, 4]

    def inter_pose(self, pre_pose, pre_quat, cur_pose, cur_quat, t):
        """
        线性插值位姿和四元数
        
        输入:
            pre_pose (torch.Tensor or np.ndarray): 前一帧位姿 [3]
            pre_quat (torch.Tensor or np.ndarray): 前一帧四元数 [4] (w,x,y,z)
            cur_pose (torch.Tensor or np.ndarray): 当前帧位姿 [3]
            cur_quat (torch.Tensor or np.ndarray): 当前帧四元数 [4] (w,x,y,z)
            t (torch.Tensor or np.ndarray): 时间比例 [N] 或标量，范围[0,1]
            
        输出:
            tuple: (interp_pose, interp_quat)
                interp_pose (torch.Tensor): 插值后的位移 [N,3]
                interp_quat (torch.Tensor): 插值后的四元数 [N,4]
        """

        # 位移线性插值 (LERP)
        inter_pose = pre_pose + t.unsqueeze(-1) * (cur_pose - pre_pose)  # [N,3]

        # 四元数球面线性插值 (SLERP) 的改进版本
        dot = (pre_quat * cur_quat).sum(dim=-1, keepdim=True)

        # 处理四元数方向不一致的情况（保证取最短路径）
        mask = dot < 0
        cur_quat = torch.where(mask, -cur_quat, cur_quat)
        dot = torch.where(mask, -dot, dot)

        # 当四元数非常接近时，退化为线性插值
        theta = torch.acos(torch.clamp(dot, -1 + 1e-6, 1 - 1e-6))  # 添加微小缓冲避免NaN

        # 判断是否需要退化为线性插值
        small_angle = theta < 1e-6
        interp_linear = (1 - t.unsqueeze(-1)) * pre_quat + t.unsqueeze(-1) * cur_quat

        # 正常SLERP计算
        sin_theta = torch.sin(theta)
        interp_slerp = (torch.sin((1 - t.unsqueeze(-1)) * theta) / sin_theta) * pre_quat + \
                       (torch.sin(t.unsqueeze(-1) * theta) / sin_theta) * cur_quat

        # 根据角度选择插值方式
        inter_quat = torch.where(small_angle, interp_linear, interp_slerp)

        # 最后归一化四元数（确保单位四元数）
        inter_quat = inter_quat / torch.norm(inter_quat, dim=-1, keepdim=True)

        return inter_pose, inter_quat

    def motion_compensation(self, points, pre_odom, cur_odom):
        """
        运动补偿主函数：将点云从扫描时刻校正到当前帧坐标系
        
        输入:
            points (torch.Tensor): 原始点云数据 [N,3] (x,y,z)
            pre_odom (torch.Tensor or np.ndarray): 前一帧位姿 (pose [3], quat [4])
            cur_odom (torch.Tensor or np.ndarray): 当前帧位姿 (pose [3], quat [4])
            
        输出:
            torch.Tensor: 校正后的点云数据 [N,3]
        """
        # Step 1: 分配角度编号 -> [N,4]
        points_with_indices = self.angle_index(points)  # [N,4]
        angle_indices = points_with_indices[:, -1].float()  # [N]

        # Step 2: 计算时间比例 (0=扫描开始时刻, 1=扫描结束时刻)
        t = angle_indices / 1800.0  # [N], 归一化到[0,1]

        # Step 3: 插值位姿
        pre_pose = pre_odom[:3] # [3]
        pre_quat = pre_odom[3:]  # [4]
        cur_pose = cur_odom[:3]  # [3]
        cur_quat = cur_odom[3:]  # [4]

        inter_pose, inter_quat = self.inter_pose(
            pre_pose, pre_quat, cur_pose, cur_quat, t
        )  # inter_pose [N,3], inter_quat [N,4]

        # Step 4: 投影到当前帧坐标系 (直接调用已有投影函数)
        compensated_points = self.projection(
            points_with_indices[:, :3],  # [N,3]
            inter_pose,  # [N,3]
            inter_quat,  # [N,4]
            cur_pose.unsqueeze(0),  # [1,3]
            cur_quat.unsqueeze(0)  # [1,4]
        )
        return compensated_points

    def quat_to_mat(self, quat):
        """
        批量四元数转旋转矩阵
        
        输入:
            quat (torch.Tensor): 四元数 [...,4]
            
        输出:
            torch.Tensor: 旋转矩阵 [...,3,3]
        """
        w, x, y, z = quat.unbind(-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_mat = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
        ], dim=-1).view(*quat.shape[:-1], 3, 3)
        return rot_mat

    def projection(self, points, ori_pose, ori_quat, target_pose, target_quat):
        """
        坐标投影函数：将点云从一个坐标系投影到另一个坐标系
        
        输入:
            points (torch.Tensor): 原始点云 [N,3]
            ori_pose (torch.Tensor or np.ndarray): 原始坐标系位姿 [N,3] 或 [3]
            ori_quat (torch.Tensor or np.ndarray): 原始坐标系四元数 [N,4] 或 [4]
            target_pose (torch.Tensor or np.ndarray): 目标坐标系位姿 [N,3] 或 [3]
            target_quat (torch.Tensor or np.ndarray): 目标坐标系四元数 [N,4] 或 [4]
            
        输出:
            torch.Tensor: 投影后的点云 [N,3]
        """

        # Compute rotation matrices
        ori_R = self.quat_to_mat(ori_quat)  # [N,3,3] or [3,3]
        target_R_inv = self.quat_to_mat(target_quat).transpose(-2, -1)  # [N,3,3] or [3,3]

        # Project to world coordinates
        world_points = torch.matmul(points.unsqueeze(-2), ori_R.transpose(-2, -1)).squeeze(-2) + ori_pose

        # Project to target coordinates
        target_points = torch.matmul((world_points - target_pose).unsqueeze(-2),
                                     target_R_inv.transpose(-2, -1)).squeeze(-2)
        return target_points

    def projection_accumulation(self, points_xyzrgb, odom_data, time_windows=TIME_LEN, accumulate_gap=TIME_GAP):
        """
        点云累积投影：将多帧点云投影到当前坐标系
        
        输入:
            points_xyzrgb (dict): 带颜色的点云数据字典 {时间戳: 点云}
            odom_data (dict): 位姿数据字典 {时间戳: 位姿}
            time_windows (int): 时间窗口大小
            accumulate_gap (int): 累积间隔
            
        输出:
            torch.Tensor: 累积后的点云数据
        """
        torch.cuda.synchronize()
        # 计算待投影的历史-未来时段
        time_stamps = sorted([key for key in points_xyzrgb.keys()], reverse=True)
        cur_stamp = time_stamps[0]

        # 当前位姿(转移到GPU)
        cur_odom = odom_data[cur_stamp]
        cur_pose = cur_odom[:3]
        cur_quat = cur_odom[3:]

        # 投影累积
        accumulated_points_list = []
        end = min(time_windows, len(time_stamps))
        for i in range(0, end, accumulate_gap):
            # 获取t_i处的点云
            stamp = time_stamps[i]
            points_i = points_xyzrgb[stamp][:, :3]
            colors_i = points_xyzrgb[stamp][:, 3:]

            # 获取t_i处的点云位姿信息
            odom_i = odom_data[stamp]
            pose_i = odom_i[:3]
            quat_i = odom_i[3:]

            # 计算投影矩阵
            projected_points = self.projection(points_i, pose_i, quat_i, cur_pose, cur_quat)

            projected_points = torch.cat((projected_points, colors_i), dim=-1)

            filtered_points = self.points_filter(projected_points)

            accumulated_points_list.append(filtered_points)

        # 在GPU上合并所有点云
        accumulated_points = torch.cat(accumulated_points_list, dim=0)
        torch.cuda.synchronize()
        return accumulated_points

    def points_filter(self, points):
        """
        GPU加速的点云边界滤波
        
        输入:
            points (torch.Tensor or np.ndarray): 输入点云，shape (N,3)
            
        输出:
            torch.Tensor: 过滤后的点云
        """
        # 转换为GPU Tensor
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float().to(self.device)

        # 计算边界距离(使用GPU并行计算)
        distance = MAP_SIZE * RESOLUTION
        mask = (points[:, 0] > -distance/2 + RESOLUTION) & (points[:, 0] < distance/2) & \
               (points[:, 1] > -distance/2 + RESOLUTION) & (points[:, 1] < distance/2)

        # 应用滤波
        filtered_points = points[mask]

        return filtered_points

    def accumulated_show(self, accumulated_points):
        """
        可视化累积点云
        
        输入:
            accumulated_points (torch.Tensor): 累积的点云数据
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(accumulated_points.cpu().numpy())
        o3d.visualization.draw_geometries([pcd])


class fusion():
    """
    多传感器融合类
    功能：激光雷达与相机数据融合、时间同步、点云着色、BEV地图生成
    """
    
    def __init__(self):
        self.device = torch.device("cuda")
        # 文件来源
        self.img_files = dict()
        self.lidar_files = dict()
        self.lidar_odom_files = dict()

        # 传感器时间帧
        self.img_stamps = list()
        self.lidar_stamps = list()
        self.lidar_odom_stamps = list()

        # 时序数据
        self.lidar_data = dict()
        self.odom_data = dict()

        # 着色点云
        self.points_xyzrgb = dict()
    
    def read_folder(self, folder_path):
        """
        读取文件夹中的文件并构建时间戳到文件路径的映射
        
        输入:
            folder_path (str): 文件夹路径
            
        输出:
            dict: 时间戳到文件路径的映射 {时间戳: 文件路径}
        """
        folder_dic = {}
        files_path = os.listdir(folder_path)
        for file_path in files_path:
            file_name = file_path.split(".")[0]
            folder_dic[int(file_name)] = os.path.join(folder_path, file_path)
        return folder_dic

    def read_file(self, file_path):
        """
        读取文本文件并构建时间戳到数据的映射
        
        输入:
            file_path (str): 文本文件路径
            
        输出:
            dict: 时间戳到数据的映射 {时间戳: 数据数组}
        """
        txt = np.loadtxt(file_path)
        file_dic = {}
        for i in range(txt.shape[0]):
            local_time = int(txt[i,0])
            file_dic[local_time] = torch.from_numpy(txt[i,1:]).float().to(self.device)
        return file_dic
    
    def data_loader(self, path):
        """
        通用数据加载器：根据路径类型自动选择文件或文件夹加载方式
        
        输入:
            path (str): 文件路径或文件夹路径
            
        输出:
            dict: 时间戳到数据/文件路径的映射
        """
        if os.path.isfile(path):
            folder_dic = self.read_file(path)
            return folder_dic
        elif os.path.isdir(path):
            file_dic = self.read_folder(path)
            return file_dic
        else:
            return {}
    
    def time_match(self, stamps, target_stamp, tss_gap = TSS_GAP):
        """
        时间戳匹配：在时间戳列表中查找与目标时间戳最接近的匹配项
        
        输入:
            stamps (list): 时间戳列表
            target_stamp (int): 目标时间戳
            tss_gap (int): 最大允许时间差阈值
            
        输出:
            tuple: (匹配索引, 匹配时间戳) 或 (None, None) 如果没有找到有效匹配
        """
        diff = [abs(stamp - target_stamp) for stamp in stamps]
        min_diff = min(diff)
        min_index = diff.index(min_diff)
        if min_diff <= tss_gap:
            return min_index, stamps[min_index]
        else:
            return None, None

    def data_init(self, root):
        """
        数据初始化：加载并组织所有传感器数据
        
        输入:
            root (str): 数据集根目录路径
            
            - self.img_files: 图像文件路径字典
            - self.lidar_files: 激光雷达文件路径字典  
            - self.lidar_odom_files: 激光雷达里程计数据字典
            - self.img_stamps: 排序后的图像时间戳列表
            - self.lidar_stamps: 排序后的激光雷达时间戳列表
            - self.lidar_odom_stamps: 排序后的里程计时间戳列表
        """
        images_path = os.path.join(root, "undistort_Image")
        lidar_path = os.path.join(root,  "LidarData_original_bin")
        lidarodometry_path = os.path.join(root, "LidarOdometry.txt")

        self.img_files = self.data_loader(images_path)
        self.lidar_files = self.data_loader(lidar_path)
        self.lidar_odom_files = self.data_loader(lidarodometry_path)

        self.img_stamps = list(self.img_files.keys())
        self.img_stamps.sort()

        self.lidar_stamps = list(self.lidar_files.keys())
        self.lidar_stamps.sort()

        self.lidar_odom_stamps = list(self.lidar_odom_files.keys())
        self.lidar_odom_stamps.sort()

    def lidarTocamera(self, img, points, img_points, proj_mat=LM_AR0231_Front, VISUALIZE=False, scale_factor=SCALE_FACTOR):
        """
        GPU加速的激光雷达到相机坐标变换
        将激光雷达点云投影到相机图像平面并提取颜色信息
        
        输入:
            img: 相机图像 (OpenCV BGR格式)
            points: 原始点云坐标 [N,3]
            img_points: 投影后的点云坐标 [N,3] 
            proj_mat: 投影矩阵 [3,4]
            VISUALIZE: 是否可视化投影结果
            scale_factor: 图像缩放因子
            
        输出:
            torch.Tensor: 带颜色的点云数据 [M,6] (x,y,z,r,g,b)
        """

        # 计时
        start = time.perf_counter()
        # 初始化
        torch.cuda.synchronize()

        img_resized = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)),
                                 interpolation=cv2.INTER_LINEAR)
        proj_mat_gpu = torch.from_numpy(proj_mat).float().to(self.device)


        points_gpu = points
        img_points_gpu = img_points

        # 坐标变换 (m->mm并添加齐次坐标)
        points_homo = torch.cat([
            img_points_gpu * 1000.0,
            torch.ones(img_points_gpu.shape[0], 1, device=self.device)
        ], dim=1)

        # 投影计算 (批量矩阵乘法)
        img_points = torch.matmul(proj_mat_gpu, points_homo.t()).t()

        # 归一化坐标
        imgx = (img_points[:, 0] / img_points[:, 2]* scale_factor).round().to(torch.int32)
        imgy = (img_points[:, 1] / img_points[:, 2]* scale_factor).round().to(torch.int32)

        # 边界检查
        height, width = img_resized.shape[:2]
        valid_mask = (
                (imgx >= 0) & (imgx < width) &
                (imgy >= 0) & (imgy < height) &
                (points_homo[:, 0] >= 0))

        valid_indices = torch.where(valid_mask)[0]                                
        if len(valid_indices) > 0:
            # 创建唯一像素键
            pixel_keys = imgx[valid_indices] * width + imgy[valid_indices]
            depths = img_points[:, 2].clone()[valid_indices]

            # 使用argmin找到每个像素的最浅深度点
            unique_keys, inverse_indices = torch.unique(pixel_keys, return_inverse=True)
            _, min_indices = scatter_min(depths, inverse_indices)

            # 提取最终点
            final_indices = valid_indices[min_indices]
            
            # 优化：批量颜色提取，避免GPU-CPU传输
            imgx_final = imgx[final_indices]  # 保持在GPU上
            imgy_final = imgy[final_indices]  # 保持在GPU上
            
            # 批量颜色提取 - 避免Python循环和GPU-CPU传输
            rgb_colors_tensor = self._batch_extract_colors(img_resized, imgx_final, imgy_final, width, height)

            # 创建包含RGB信息的img_points
            img_points = torch.stack([
                points_gpu[final_indices, 0],  # x
                points_gpu[final_indices, 1],  # y
                points_gpu[final_indices, 2],  # z
                rgb_colors_tensor[:, 0],  # r
                rgb_colors_tensor[:, 1],  # g
                rgb_colors_tensor[:, 2]   # b
            ], dim=1)
        else:
            img_points = torch.empty((0, 8), device=self.device)  # 现在有8列

        torch.cuda.synchronize()
        # 计时
        end = time.perf_counter()
        gap = end - start
        print("投影至图像计算耗时:%.6f s" % gap)

        # 可视化 (可选)
        if VISUALIZE:
            self._visualize_projection(img_resized, imgx, imgy, final_indices, img_points)
        
        return img_points.float()

    def _batch_extract_colors(self, img_resized, imgx, imgy, width, height):
        """
        批量颜色提取：从图像中批量提取像素颜色，避免GPU-CPU传输
        
        输入:
            img_resized: 调整大小后的图像
            imgx: 像素x坐标张量
            imgy: 像素y坐标张量  
            width: 图像宽度
            height: 图像高度
            
        输出:
            torch.Tensor: 提取的颜色张量 [N,3] (RGB格式)
        """
        # 创建有效的像素坐标掩码
        valid_mask = (imgx >= 0) & (imgx < width) & (imgy >= 0) & (imgy < height)
        valid_imgx = imgx[valid_mask]
        valid_imgy = imgy[valid_mask]
        
        if len(valid_imgx) == 0:
            return torch.zeros((len(imgx), 3), device=self.device)
        
        # 将图像转换为Tensor（一次性操作）
        img_tensor = torch.from_numpy(img_resized).to(self.device).float() / 255.0
        
        # 批量索引获取颜色 (BGR -> RGB)
        colors = img_tensor[valid_imgy, valid_imgx]  # [N, 3] in BGR
        rgb_colors = colors[:, [2, 1, 0]]  # 转换为RGB
        
        # 创建完整的结果张量
        result = torch.zeros((len(imgx), 3), device=self.device)
        result[valid_mask] = rgb_colors
        
        return result

    def _visualize_projection(self, img_resized, imgx, imgy, final_indices, img_points):
        """
        投影可视化：在图像上可视化点云投影结果
        
        输入:
            img_resized: 调整大小后的图像
            imgx: 像素x坐标张量
            imgy: 像素y坐标张量
            final_indices: 最终选择的点索引
            img_points: 带颜色的点云数据
            
        输出:
            无，但会显示可视化窗口
        """
        img_vis = img_resized.copy()
        
        # 批量传输数据到CPU（一次性操作）
        if len(final_indices) > 0:
            u = imgx[final_indices].cpu().numpy().astype(np.int32)
            v = imgy[final_indices].cpu().numpy().astype(np.int32)
            
            # 使用深度颜色而不是真实颜色
            depths = torch.sqrt(img_points[:, 0]**2 + img_points[:, 1] ** 2).cpu().numpy()   # z坐标作为深度
            
            # 创建深度颜色映射 (热图)
            min_depth = depths.min() if len(depths) > 0 else 0
            max_depth = depths.max() if len(depths) > 0 else 1
            if max_depth > min_depth:
                normalized_depths = (depths - min_depth) / (max_depth - min_depth)
            else:
                normalized_depths = np.zeros_like(depths)
            
            # 使用热图颜色映射 (蓝->绿->红)
            colors = np.zeros((len(normalized_depths), 3), dtype=np.uint8)
            for i, depth_ratio in enumerate(normalized_depths):
                if depth_ratio < 0.5:
                    # 蓝色到绿色
                    r = 0
                    g = int(255 * (depth_ratio * 2))
                    b = int(255 * (1 - depth_ratio * 2))
                else:
                    # 绿色到红色
                    r = int(255 * ((depth_ratio - 0.5) * 2))
                    g = int(255 * (1 - (depth_ratio - 0.5) * 2))
                    b = 0
                colors[i] = [b, g, r]  # OpenCV使用BGR格式
            
            for u_val, v_val, color in zip(u, v, colors):
                cv2.circle(img_vis, (u_val, v_val), 1, color.tolist(), -1)  # 使用深度颜色
        
        cv2.namedWindow("Projection", cv2.WINDOW_NORMAL)
        cv2.imshow("Projection", img_vis)
        cv2.waitKey(1)

    def raster(self, colored_points, map_size=MAP_SIZE, resolution=RESOLUTION):
        """
        GPU加速的BEV地图生成函数
        将累积的着色点云投影到地图上，生成着色的BEV地图
        按照图像坐标：纵轴为x，横轴为y
        选择z轴高度最大的点进行着色
        
        输入:
            colored_points: [N, 6] Tensor, 包含 [x, y, z, b, g, r] (BGR格式)
            map_size: 地图尺寸 (像素)
            resolution: 分辨率 (米/像素)
            
        输出:
            bev_image: [map_size, map_size, 3] 彩色BEV图像 (BGR格式)
        """

        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 确保输入是GPU Tensor
        if not isinstance(colored_points, torch.Tensor):
            colored_points = torch.from_numpy(colored_points).float().to(self.device)
        
        # 提取坐标和颜色信息
        points_xy = colored_points[:, :2]  # [N, 2] (x, y)
        points_z = colored_points[:, 2]    # [N] (z坐标，用于高度选择)
        colors = colored_points[:, 3:]    # [N, 3] (r, g, b)
        
        # 计算地图边界 (以原点为中心)
        map_half_size = (map_size * resolution) / 2
        
        # 过滤超出地图边界的点
        valid_mask = (
            (points_xy[:, 0] >= -map_half_size) & 
            (points_xy[:, 0] < map_half_size) &
            (points_xy[:, 1] >= -map_half_size) & 
            (points_xy[:, 1] < map_half_size)
        )
        
        if not torch.any(valid_mask):
            print("警告: 没有点在地图边界内")
            return np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        points_xy = points_xy[valid_mask]
        points_z = points_z[valid_mask]
        colors = colors[valid_mask]
        
        # 将世界坐标转换为像素坐标
        # 按照图像坐标：纵轴为x，横轴为y
        # 原点在图像中心，x向下，y向右
        pixel_x = ((map_half_size - points_xy[:, 0]) / resolution).long()  # [N] (x轴翻转)
        pixel_y = ((points_xy[:, 1] + map_half_size) / resolution).long()  # [N]
        
        # 确保像素坐标在有效范围内
        valid_pixel_mask = (
            (pixel_x >= 0) & (pixel_x < map_size) &
            (pixel_y >= 0) & (pixel_y < map_size)
        )
        
        pixel_x = pixel_x[valid_pixel_mask]
        pixel_y = pixel_y[valid_pixel_mask]
        points_z = points_z[valid_pixel_mask]
        colors = colors[valid_pixel_mask]
        
        if len(pixel_x) == 0:
            print("警告: 没有有效的像素坐标")
            return np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # 创建像素键用于唯一性检查
        pixel_keys = pixel_x * map_size + pixel_y
        
        # 使用scatter_max找到每个像素的z轴最大点 (选择高度最大的点)
        unique_keys, inverse_indices = torch.unique(pixel_keys, return_inverse=True)
        _, max_indices = scatter_min(-points_z, inverse_indices)  # 使用负号实现scatter_max效果
        
        # 提取最终像素和颜色
        final_pixel_x = pixel_x[max_indices]
        final_pixel_y = pixel_y[max_indices]
        final_colors = colors[max_indices]
        
        # 优化：使用向量化操作创建BEV图像
        # 将颜色转换为uint8格式并保持在GPU上
        final_colors_uint8 = (final_colors * 255).byte()  # [M, 3] uint8 on GPU (RGB格式)
        
        # 创建BEV图像张量 (保持在GPU上)
        bev_image_tensor = torch.zeros((map_size, map_size, 3), dtype=torch.uint8, device=self.device)
        
        # 使用向量化索引赋值 (一次性操作)
        bev_image_tensor[final_pixel_x, final_pixel_y] = final_colors_uint8
        
        # 将结果传输到CPU (一次性操作)
        bev_image = bev_image_tensor.cpu().numpy()
        
        # 确保颜色格式正确：RGB -> BGR (OpenCV使用BGR格式)
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"优化版BEV地图生成耗时: {end - start:.6f} s")
        print(f"BEV地图有效像素数: {len(final_pixel_x)}")
        
        return bev_image
    
    def run(self, root, VIS = False, SAVE = False):
        """
        融合处理主函数：执行完整的激光雷达-相机融合流程
        
        输入:
            root (str): 数据集根目录路径
            VIS (bool): 是否启用可视化
            SAVE (bool): 是否保存BEV地图
        """
        lidar_sample = Lidar()
        self.data_init(root)
        
        for img_stamp in self.img_stamps:
            # 时间匹配
            lidar_idx, lidar_stamp = self.time_match(self.lidar_stamps, img_stamp)
            odom_idx, odom_stamp = self.time_match(self.lidar_odom_stamps, img_stamp)
            if not lidar_idx or not lidar_stamp or not odom_idx or not odom_stamp:
                continue

            # 数据读取
            cur_img = cv2.imread(self.img_files[img_stamp])
            cur_points = lidar_sample.load_lidarbin(self.lidar_files[lidar_stamp])
            cur_odom = self.lidar_odom_files[odom_stamp]

            # 单帧点云数据帧内补偿,投影到末帧
            lidar_pre_stamp = self.lidar_stamps[lidar_idx - 1]
            odom_pre_stamp = self.lidar_odom_stamps[odom_idx - 1]
            pre_odom = self.lidar_odom_files[odom_pre_stamp]
            compensated_points = lidar_sample.motion_compensation(cur_points, pre_odom, cur_odom)

            self.lidar_data[lidar_stamp] = compensated_points
            self.odom_data[lidar_stamp] = cur_odom

            # 点云与图像配准，将点云投影到图像时间戳
            # 插值 img 位姿
            t = torch.from_numpy(np.array((img_stamp - lidar_stamp) / (lidar_stamp - lidar_pre_stamp))).to(self.device)
            img_pose, img_quat = lidar_sample.inter_pose(pre_odom[:3], pre_odom[3:], cur_odom[:3], cur_odom[3:], t)
            img_points = lidar_sample.projection(compensated_points, cur_odom[:3], cur_odom[3:], img_pose, img_quat)

            # 单张图片投影着色（获取RGB颜色）
            img_points = self.lidarTocamera(cur_img, img_points, compensated_points, proj_mat=LM_AR0231_Front, VISUALIZE=VIS, scale_factor=SCALE_FACTOR)
            self.points_xyzrgb[lidar_stamp] = img_points


            # 如果成功着色，将所有着色后的点云合并
            if self.points_xyzrgb is not None and len(self.points_xyzrgb) > 0:
                accumulated_colored_points = lidar_sample.projection_accumulation(self.points_xyzrgb, self.odom_data)
                print(f"累积着色点云数量: {len(accumulated_colored_points)}")
                
                # 生成BEV地图
                bev_image = self.raster(accumulated_colored_points)
                
                # 保存BEV地图
                if SAVE:
                    os.makedirs('./bev_map', exist_ok= True)
                    bev_path = f"./bev_map/{img_stamp}.png"
                    cv2.imwrite(bev_path, bev_image)
                    print(f"BEV地图已保存: {bev_path}")
                
                # 显示BEV地图
                if VIS:
                    cv2.namedWindow("BEV Map", cv2.WINDOW_NORMAL)
                    cv2.imshow("BEV Map", bev_image)
                    cv2.waitKey(1)  # 短暂显示，不阻塞
                
                    # 可视化累积结果
                    if len(accumulated_colored_points) > 0:
                        # 提取坐标和颜色信息
                        points_3d = accumulated_colored_points[:, :3].cpu().numpy()
                        colors = (accumulated_colored_points[:, 3:].cpu().numpy() * 255).astype(np.uint8)
                        
                        # 创建Open3D点云
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points_3d)
                        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
                        
                        # 可视化
                        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    root = "./Dataset/2025-02-25-14-22-23"

    fusion_sample = fusion()
    fusion_sample.run(root, VIS=False, SAVE=True)
