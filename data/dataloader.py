"""This file contains basic dataset class, used in the AutoLens project."""

import glob
import os
import zipfile

import cv2 as cv
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ======================================
# Basic dataset class
# ======================================
class ImageDataset(Dataset):
    def __init__(self, img_dir, img_res=None):
        super(ImageDataset, self).__init__()
        # 查找所有子文件夹
        self.group_folders = sorted(glob.glob(f"{img_dir}/group*"))
        print(f"Found {len(self.group_folders)} data groups in {img_dir}")

        # 验证每个子文件夹是否包含31张图像
        self.valid_groups = []
        for group in self.group_folders:
            # 生成预期的文件名列表
            expected_files = [f"{group}/{i}.png" for i in range(400, 710, 10)]
            # 检查所有文件是否存在
            if all(os.path.exists(f) for f in expected_files):
                self.valid_groups.append(expected_files)
        print(f"Valid data groups: {len(self.valid_groups)}")

        if isinstance(img_res, int):
            img_res = [img_res, img_res]

        # 定义预处理流程（移除AutoAugment，因为它不支持多通道）
        self.transform = transforms.Compose([
            transforms.Resize(img_res),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
        ])

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        # 获取当前组的31张图像路径
        img_paths = self.valid_groups[idx]

        # 读取并堆叠31张图像为一个张量
        img_stack = []
        for path in img_paths:
            img = Image.open(path).convert('L')  # 转为单通道灰度图
            img = self.transform(img)  # 应用变换
            img_stack.append(img)

        # 堆叠为 (31, H, W) 的张量
        img = torch.cat(img_stack, dim=0)  # 沿通道维度拼接

        return img



# import os
# import torch
# import numpy as np
# import cv2 as cv
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from typing import List, Tuple, Optional
#
#
# class StackedImageDataset(Dataset):
#     """
#     加载并处理堆叠图像数据的自定义数据集类
#     格式为: 31张大小为(5120, 5120, 1)的png图像组成一个数据样本
#     """
#
#     def __init__(self, data_dir: str, sensor_res: Tuple[int, int],
#                  normalize: bool = True, transform: Optional[transforms.Compose] = None):
#         """
#         初始化数据集
#
#         参数:
#             data _dir: 数据目录路径 (应指向包含group1, group2等文件夹的目录)
#             sensor_res: 传感器分辨率，用于调整图像大小
#             normalize: 是否对图像进行归一化
#             transform: 可选的图像转换操作
#         """
#         self.data_dir = data_dir
#         self.sensor_res = sensor_res
#         self.normalize = normalize
#         self.transform = transform
#         self.samples = self._find_samples()
#
#     def _find_samples(self) -> List[List[str]]:
#         """查找并组织数据样本"""
#         samples = []
#
#         # 查找所有groupXX文件夹
#         group_folders = sorted(
#             [f for f in os.listdir(self.data_dir)
#              if os.path.isdir(os.path.join(self.data_dir, f)) and f.startswith('group')]
#         )
#
#         for group_folder in group_folders:
#             folder_path = os.path.join(self.data_dir, group_folder)
#             # 获取文件夹内所有png图像
#             images = sorted(
#                 [os.path.join(folder_path, f)
#                  for f in os.listdir(folder_path)
#                  if f.lower().endswith('.png')]
#             )
#
#             # 确保每个样本包含31张图像
#             if len(images) == 31:
#                 samples.append(images)
#
#         return samples
#
#     def __len__(self) -> int:
#         """返回数据集大小"""
#         return len(self.samples)
#
#     def __getitem__(self, idx: int) -> torch.Tensor:
#         """获取单个数据样本"""
#         img_paths = self.samples[idx]
#         # 创建一个空的张量来存储堆叠的图像
#         img_stack = torch.zeros(31, self.sensor_res[0], self.sensor_res[1])
#
#         # 读取并处理每张图像
#         for i, img_path in enumerate(img_paths):
#             # 读取图像 (默认为灰度图)
#             img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
#
#             # 调整图像大小
#             img = cv.resize(img, self.sensor_res)
#
#             # 转换为张量
#             img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
#
#             # 归一化
#             if self.normalize:
#                 img_tensor /= 255.0
#
#             img_stack[i] = img_tensor[0]
#
#         # 应用额外的转换
#         if self.transform:
#             img_stack = self.transform(img_stack)
#
#         return img_stack
#
#
# def create_data_loader(data_dir: str, sensor_res: Tuple[int, int],
#                        batch_size: int = 4, num_workers: int = 4,
#                        shuffle: bool = True) -> DataLoader:
#     """
#     创建数据加载器
#
#     参数:
#         data_dir: 数据目录路径 (应指向包含group1, group2等文件夹的目录)
#         sensor_res: 传感器分辨率
#         batch_size: 批次大小
#         num_workers: 工作线程数
#         shuffle: 是否打乱数据
#
#     返回:
#         DataLoader: 数据加载器实例
#     """
#     # 定义数据转换
#     transform = transforms.Compose([
#         # 可以添加更多的数据增强操作
#     ])
#
#     # 创建数据集
#     dataset = StackedImageDataset(
#         data_dir=data_dir,
#         sensor_res=sensor_res,
#         normalize=True,
#         transform=transform
#     )
#
#     # 创建数据加载器
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     return data_loader