# 作者: qzf
import os
import json
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Callable
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class STUDataset(Dataset):
    """
    STUDataset数据集加载器，支持无人机姿态信息的加载
    """
    def __init__(self, root_dir: str, split: str = "train", transform: Optional[Callable] = None):
        """
        初始化数据集
        Args:
            root_dir: 数据集根目录
            split: 数据集分割（train/val/test）
            transform: 数据转换函数
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 数据集路径设置
        self.image_dir = os.path.join(root_dir, "images", split)
        self.detection_dir = os.path.join(root_dir, "annotations", "detection", split)
        self.segmentation_dir = os.path.join(root_dir, "annotations", "segmentation", split)
        self.ego_info_path = os.path.join(root_dir, "annotations", "ego_information.json")
        
        # 加载图像ID列表
        self.image_ids = self._load_image_ids()
        
        # 加载无人机姿态信息
        self.ego_information = self._load_ego_information()
        
        # 类别定义
        self.detection_classes = ['pedestrian', 'car', 'excavator', 'bulldozer', 'truck', 'bus', 'bicycle', 'motorcycle', 'other']
        self.segmentation_classes = ['background', 'road', 'building', 'green_space', 'bare_land', 'developed_area']
        
        # 检查数据集完整性
        self._validate_dataset()
    
    def _load_image_ids(self) -> List[str]:
        """
        加载图像ID列表
        Returns:
            图像ID列表
        """
        # 尝试从splits目录加载
        splits_dir = os.path.join(self.root_dir, "splits")
        split_file = os.path.join(splits_dir, f"{self.split}.txt")
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            # 如果没有split文件，直接从图像目录获取
            image_extensions = ['.jpg', '.jpeg', '.png']
            image_files = [f for f in os.listdir(self.image_dir) 
                          if any(f.lower().endswith(ext) for ext in image_extensions)]
            return [os.path.splitext(f)[0] for f in image_files]
    
    def _load_ego_information(self) -> Dict[str, Dict[str, float]]:
        """
        加载无人机姿态信息
        Returns:
            姿态信息字典
        """
        if os.path.exists(self.ego_info_path):
            with open(self.ego_info_path, 'r') as f:
                return json.load(f)
        else:
            # 如果没有姿态信息文件，返回空字典
            print(f"Warning: Ego information file not found at {self.ego_info_path}")
            return {}
    
    def _validate_dataset(self) -> None:
        """
        验证数据集完整性
        """
        for image_id in self.image_ids:
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str]]:
        """
        获取数据项
        Args:
            idx: 数据索引
        Returns:
            数据字典
        """
        image_id = self.image_ids[idx]
        
        # 加载图像
        image = self._load_image(image_id)
        
        # 加载标注
        detection_boxes, detection_labels = self._load_detection_annotations(image_id)
        segmentation_mask = self._load_segmentation_mask(image_id)
        
        # 加载姿态信息
        ego_info = self.get_ego_information(idx)
        
        # 构建数据字典
        data = {
            'image': image,
            'image_id': image_id,
            'detection_boxes': detection_boxes,
            'detection_labels': detection_labels,
            'segmentation_mask': segmentation_mask,
            'ego_info': ego_info
        }
        
        # 应用转换
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _load_image(self, image_id: str) -> np.ndarray:
        """
        加载图像
        Args:
            image_id: 图像ID
        Returns:
            图像数组
        """
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            # 尝试其他扩展名
            for ext in ['.jpeg', '.png']:
                alt_path = os.path.join(self.image_dir, f"{image_id}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_detection_annotations(self, image_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载检测标注
        Args:
            image_id: 图像ID
        Returns:
            边界框和标签张量
        """
        # 支持YOLO格式
        yolo_path = os.path.join(self.detection_dir, f"{image_id}.txt")
        if os.path.exists(yolo_path):
            boxes = []
            labels = []
            with open(yolo_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO格式: [class, x_center, y_center, width, height] (归一化)
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为[xmin, ymin, xmax, ymax]格式
                        xmin = x_center - width / 2
                        ymin = y_center - height / 2
                        xmax = x_center + width / 2
                        ymax = y_center + height / 2
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
            
            if boxes:
                return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        
        # 如果没有标注，返回空张量
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)
    
    def _load_segmentation_mask(self, image_id: str) -> torch.Tensor:
        """
        加载分割掩码
        Args:
            image_id: 图像ID
        Returns:
            分割掩码张量
        """
        mask_path = os.path.join(self.segmentation_dir, f"{image_id}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return torch.tensor(mask, dtype=torch.long)
        else:
            # 如果没有掩码，返回空张量
            return torch.zeros((0, 0), dtype=torch.long)
    
    def get_ego_information(self, idx: int) -> Dict[str, float]:
        """
        获取无人机姿态信息
        Args:
            idx: 数据索引
        Returns:
            姿态信息字典（包含pitch, roll, yaw, altitude）
        """
        image_id = self.image_ids[idx]
        if image_id in self.ego_information:
            return self.ego_information[image_id]
        else:
            # 返回默认值
            return {
                'pitch': 0.0,
                'roll': 0.0,
                'yaw': 0.0,
                'altitude': 0.0,
                'timestamp': 0.0
            }