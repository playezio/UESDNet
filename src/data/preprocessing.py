# 作者: qzf
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

# 定义数据预处理函数
def studataset_preprocessing(image: np.ndarray, 
                            detection_boxes: Optional[torch.Tensor], 
                            segmentation_mask: Optional[torch.Tensor], 
                            ego_info: Dict[str, float], 
                            training: bool = True) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    STUDataset数据预处理流程
    Args:
        image: 原始图像
        detection_boxes: 检测边界框
        segmentation_mask: 分割掩码
        ego_info: 无人机姿态信息
        training: 是否为训练模式
    Returns:
        预处理后的数据
    """
    # 步骤1: 调整大小到1080p
    image = resize_to_1080p(image)
    
    # 步骤2: 归一化姿态信息
    ego_info = normalize_attitude_angles(ego_info)
    
    # 步骤3: 转换标注为张量
    if detection_boxes is None:
        detection_boxes = torch.zeros((0, 4), dtype=torch.float32)
    
    if segmentation_mask is None:
        segmentation_mask = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.long)
    else:
        # 调整掩码大小以匹配图像
        segmentation_mask = resize_mask(segmentation_mask, image.shape[:2])
    
    # 步骤4: 应用数据增强
    if training:
        image, detection_boxes, segmentation_mask = apply_augmentations(
            image, detection_boxes, segmentation_mask
        )
    
    return image, detection_boxes, segmentation_mask, ego_info

def resize_to_1080p(image: np.ndarray, target_height: int = 1080, target_width: int = 1920) -> np.ndarray:
    """
    调整图像大小到1080p，保持宽高比
    Args:
        image: 输入图像
        target_height: 目标高度
        target_width: 目标宽度
    Returns:
        调整大小后的图像
    """
    # 计算调整后的尺寸，保持宽高比
    h, w = image.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio > target_width / target_height:
        # 宽度是限制因素
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        # 高度是限制因素
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    
    # 调整大小
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 如果需要，添加填充以达到目标尺寸
    if new_width != target_width or new_height != target_height:
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        start_x = (target_width - new_width) // 2
        start_y = (target_height - new_height) // 2
        padded[start_y:start_y+new_height, start_x:start_x+new_width] = resized
        return padded
    
    return resized

def resize_mask(mask: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    调整分割掩码大小
    Args:
        mask: 输入掩码
        target_shape: 目标形状 (height, width)
    Returns:
        调整大小后的掩码
    """
    # 转换为numpy进行调整
    mask_np = mask.numpy()
    # 使用最近邻插值保持类别标签
    resized = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return torch.tensor(resized, dtype=torch.long)

def normalize_attitude_angles(ego_info: Dict[str, float]) -> Dict[str, float]:
    """
    归一化无人机姿态角度
    Args:
        ego_info: 包含姿态信息的字典
    Returns:
        归一化后的姿态信息
    """
    # 定义姿态角范围
    pitch_min, pitch_max = -30.0, 30.0
    roll_min, roll_max = -30.0, 30.0
    yaw_min, yaw_max = -180.0, 180.0
    
    # 归一化到[-1, 1]范围
    if 'pitch' in ego_info:
        ego_info['pitch_normalized'] = 2 * (ego_info['pitch'] - pitch_min) / (pitch_max - pitch_min) - 1
    if 'roll' in ego_info:
        ego_info['roll_normalized'] = 2 * (ego_info['roll'] - roll_min) / (roll_max - roll_min) - 1
    if 'yaw' in ego_info:
        ego_info['yaw_normalized'] = 2 * (ego_info['yaw'] - yaw_min) / (yaw_max - yaw_min) - 1
    
    # 裁剪到有效范围
    for key in ['pitch_normalized', 'roll_normalized', 'yaw_normalized']:
        if key in ego_info:
            ego_info[key] = max(-1.0, min(1.0, ego_info[key]))
    
    return ego_info

def apply_augmentations(image: np.ndarray, 
                        detection_boxes: torch.Tensor, 
                        segmentation_mask: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    应用数据增强
    Args:
        image: 输入图像
        detection_boxes: 检测边界框
        segmentation_mask: 分割掩码
    Returns:
        增强后的数据
    """
    # 这里实现简单的数据增强，实际使用时可以与transforms.py中的albumentations结合
    h, w = image.shape[:2]
    
    # 随机水平翻转 (50%概率)
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        segmentation_mask = torch.flip(segmentation_mask, dims=[1])
        
        # 调整边界框
        if len(detection_boxes) > 0:
            detection_boxes[:, [0, 2]] = w - detection_boxes[:, [2, 0]]
    
    # 随机亮度和对比度调整
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.8, 1.2)  # 对比度因子
        beta = np.random.uniform(-10, 10)    # 亮度偏移
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image, detection_boxes, segmentation_mask

def boxes_to_tensor(detection_boxes: List[List[float]]) -> torch.Tensor:
    """
    将边界框列表转换为张量
    Args:
        detection_boxes: 边界框列表
    Returns:
        边界框张量
    """
    if not detection_boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(detection_boxes, dtype=torch.float32)

def mask_to_tensor(segmentation_mask: np.ndarray) -> torch.Tensor:
    """
    将分割掩码转换为张量
    Args:
        segmentation_mask: 分割掩码
    Returns:
        分割掩码张量
    """
    return torch.tensor(segmentation_mask, dtype=torch.long)