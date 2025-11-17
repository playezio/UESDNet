# 作者: qzf
import numpy as np
import torch
from typing import Dict, Any
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UAVTransforms:
    """
    无人机图像处理转换类
    """
    
    @staticmethod
    def train_transforms(size=(1080, 1920)):
        """
        训练集转换
        Args:
            size: 输出尺寸 (height, width)
        Returns:
            转换函数
        """
        return A.Compose([
            A.Resize(height=size[0], width=size[1], interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # 使用[xmin, ymin, xmax, ymax]格式
            label_fields=['detection_labels']
        ))
    
    @staticmethod
    def val_transforms(size=(1080, 1920)):
        """
        验证集转换
        Args:
            size: 输出尺寸 (height, width)
        Returns:
            转换函数
        """
        return A.Compose([
            A.Resize(height=size[0], width=size[1], interpolation=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['detection_labels']
        ))

class STUDatasetTransform:
    """
    STUDataset专用转换类，处理整个数据字典
    """
    def __init__(self, is_train=True, size=(1080, 1920)):
        """
        初始化转换类
        Args:
            is_train: 是否为训练模式
            size: 输出尺寸
        """
        if is_train:
            self.transform = UAVTransforms.train_transforms(size)
        else:
            self.transform = UAVTransforms.val_transforms(size)
        self.is_train = is_train
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用转换
        Args:
            data: 数据字典
        Returns:
            转换后的数据字典
        """
        image = data['image']
        boxes = data['detection_boxes']
        labels = data['detection_labels']
        mask = data['segmentation_mask']
        
        # 准备bbox格式转换（YOLO到Pascal VOC）
        if len(boxes) > 0 and (boxes.max() <= 1.0):
            # 如果是归一化的YOLO格式，转换为像素坐标
            h, w = image.shape[:2]
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        
        # 转换为Pascal VOC格式的列表
        bboxes = boxes.tolist() if len(boxes) > 0 else []
        class_labels = labels.tolist() if len(labels) > 0 else []
        
        # 应用albumentations转换
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            detection_labels=class_labels,
            mask=mask
        )
        
        # 更新数据字典
        transformed_data = {
            'image': transformed['image'],
            'image_id': data['image_id'],
            'detection_boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32) if transformed['bboxes'] else torch.zeros((0, 4), dtype=torch.float32),
            'detection_labels': torch.tensor(transformed['detection_labels'], dtype=torch.long) if transformed['detection_labels'] else torch.zeros(0, dtype=torch.long),
            'segmentation_mask': transformed['mask'],
            'ego_info': data['ego_info']
        }
        
        # 处理姿态信息
        transformed_data['uav_attitude'] = torch.tensor([
            transformed_data['ego_info'].get('pitch', 0.0),
            transformed_data['ego_info'].get('roll', 0.0),
            transformed_data['ego_info'].get('yaw', 0.0)
        ], dtype=torch.float32)
        
        return transformed_data

class NormalizeAttitude:
    """
    姿态信息归一化转换
    """
    def __init__(self, pitch_range=(-30.0, 30.0), roll_range=(-30.0, 30.0), yaw_range=(-180.0, 180.0)):
        """
        初始化归一化参数
        Args:
            pitch_range: 俯仰角范围
            roll_range: 横滚角范围
            yaw_range: 偏航角范围
        """
        self.pitch_min, self.pitch_max = pitch_range
        self.roll_min, self.roll_max = roll_range
        self.yaw_min, self.yaw_max = yaw_range
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用归一化
        Args:
            data: 数据字典
        Returns:
            归一化后的数据字典
        """
        attitude = data['uav_attitude']
        
        # 归一化到[-1, 1]范围
        normalized_pitch = 2 * (attitude[0] - self.pitch_min) / (self.pitch_max - self.pitch_min) - 1
        normalized_roll = 2 * (attitude[1] - self.roll_min) / (self.roll_max - self.roll_min) - 1
        normalized_yaw = 2 * (attitude[2] - self.yaw_min) / (self.yaw_max - self.yaw_min) - 1
        
        # 裁剪到[-1, 1]范围
        normalized_pitch = torch.clamp(normalized_pitch, -1.0, 1.0)
        normalized_roll = torch.clamp(normalized_roll, -1.0, 1.0)
        normalized_yaw = torch.clamp(normalized_yaw, -1.0, 1.0)
        
        data['uav_attitude_normalized'] = torch.tensor([normalized_pitch, normalized_roll, normalized_yaw])
        
        return data