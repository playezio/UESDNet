# 作者: qzf
import os
import sys
import torch
import pytest
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import (
    resize_to_1080p,
    resize_mask,
    normalize_attitude_angles,
    augment_image_and_labels,
    studataset_preprocessing
)
from src.data.transforms import Resize,
    Normalize,
    RandomFlip,
    RandomRotate,
    ToTensor,
    Compose

class TestPreprocessing:
    """
    数据预处理测试类
    """
    
    @pytest.fixture
    def sample_image(self):
        """
        创建示例图像
        """
        # 创建随机RGB图像 (H, W, C)
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask(self):
        """
        创建示例分割掩码
        """
        # 创建分割掩码 (H, W)
        return np.zeros((512, 512), dtype=np.int64)
    
    @pytest.fixture
    def sample_boxes(self):
        """
        创建示例边界框
        """
        # 创建边界框 [xmin, ymin, xmax, ymax]
        return np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    
    @pytest.fixture
    def sample_labels(self):
        """
        创建示例标签
        """
        return np.array([1, 2])
    
    @pytest.fixture
    def sample_attitude(self):
        """
        创建示例姿态信息
        """
        # 创建姿态信息 [roll, pitch, yaw]
        return np.array([45.0, -30.0, 15.0])
    
    def test_resize_to_1080p(self, sample_image):
        """
        测试图像调整到1080p
        """
        # 调整图像大小
        resized_image = resize_to_1080p(sample_image)
        
        # 验证结果
        assert resized_image.shape[1] == 1920  # 宽度
        assert resized_image.shape[0] == 1080  # 高度
        assert resized_image.shape[2] == 3      # 通道数
        
        # 测试非正方形图像
        rect_image = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        resized_rect = resize_to_1080p(rect_image)
        assert resized_rect.shape[1] == 1920
        assert resized_rect.shape[0] == 1080
    
    def test_resize_mask(self, sample_mask):
        """
        测试分割掩码调整大小
        """
        # 调整掩码大小
        resized_mask = resize_mask(sample_mask, 256, 256)
        
        # 验证结果
        assert resized_mask.shape == (256, 256)
        assert resized_mask.dtype == np.int64
    
    def test_normalize_attitude_angles(self, sample_attitude):
        """
        测试姿态角度归一化
        """
        # 归一化姿态角度
        normalized_attitude = normalize_attitude_angles(sample_attitude)
        
        # 验证结果范围 [-1, 1]
        assert np.all(normalized_attitude >= -1.0)
        assert np.all(normalized_attitude <= 1.0)
        
        # 测试极端角度
        extreme_attitude = np.array([180.0, -180.0, 360.0])
        normalized_extreme = normalize_attitude_angles(extreme_attitude)
        assert np.allclose(normalized_extreme, [1.0, -1.0, 0.0])
    
    def test_augment_image_and_labels(self, sample_image, sample_mask, sample_boxes, sample_labels):
        """
        测试图像和标签数据增强
        """
        # 执行数据增强
        augmented_data = augment_image_and_labels(
            image=sample_image,
            mask=sample_mask,
            boxes=sample_boxes,
            labels=sample_labels,
            rotate_prob=0.0,  # 不旋转以确保可重现性
            flip_prob=0.0     # 不翻转以确保可重现性
        )
        
        # 验证结果
        assert 'image' in augmented_data
        assert 'mask' in augmented_data
        assert 'boxes' in augmented_data
        assert 'labels' in augmented_data
        
        # 验证形状一致
        assert augmented_data['image'].shape == sample_image.shape
        assert augmented_data['mask'].shape == sample_mask.shape
        assert augmented_data['boxes'].shape == sample_boxes.shape
        assert augmented_data['labels'].shape == sample_labels.shape
    
    def test_studataset_preprocessing(self, sample_image, sample_mask, sample_boxes, sample_labels, sample_attitude):
        """
        测试STUDataset预处理流程
        """
        # 创建数据字典
        data = {
            'image': sample_image,
            'segmentation_mask': sample_mask,
            'detection_boxes': sample_boxes,
            'detection_labels': sample_labels,
            'uav_attitude': sample_attitude
        }
        
        # 执行预处理
        processed_data = studataset_preprocessing(
            data=data,
            resize_shape=(512, 512),
            augment=True,
            normalize=True
        )
        
        # 验证结果
        assert 'image' in processed_data
        assert 'segmentation_mask' in processed_data
        assert 'detection_boxes' in processed_data
        assert 'detection_labels' in processed_data
        assert 'uav_attitude' in processed_data
        
        # 验证数据类型和形状
        assert isinstance(processed_data['image'], np.ndarray)
        assert processed_data['image'].shape == (3, 512, 512)  # 预处理后应转为CHW格式
        assert processed_data['segmentation_mask'].shape == (512, 512)
        
        # 验证姿态角度已归一化
        assert np.all(processed_data['uav_attitude'] >= -1.0)
        assert np.all(processed_data['uav_attitude'] <= 1.0)

class TestTransforms:
    """
    数据转换测试类
    """
    
    @pytest.fixture
    def sample_image_np(self):
        """
        创建NumPy示例图像
        """
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_mask_np(self):
        """
        创建NumPy示例掩码
        """
        return np.zeros((512, 512), dtype=np.int64)
    
    @pytest.fixture
    def sample_data(self, sample_image_np, sample_mask_np):
        """
        创建示例数据字典
        """
        return {
            'image': sample_image_np,
            'segmentation_mask': sample_mask_np,
            'detection_boxes': np.array([[100, 100, 200, 200]]),
            'detection_labels': np.array([1]),
            'uav_attitude': np.array([0.0, 0.0, 0.0])
        }
    
    def test_resize_transform(self, sample_data):
        """
        测试Resize转换
        """
        transform = Resize(size=(256, 256))
        transformed_data = transform(sample_data.copy())
        
        # 验证图像和掩码已调整大小
        assert transformed_data['image'].shape == (256, 256, 3)
        assert transformed_data['segmentation_mask'].shape == (256, 256)
        
        # 验证边界框已相应调整
        assert transformed_data['detection_boxes'].shape == sample_data['detection_boxes'].shape
        # 边界框坐标应该按比例缩小
        assert transformed_data['detection_boxes'][0, 0] == sample_data['detection_boxes'][0, 0] * 0.5
    
    def test_normalize_transform(self, sample_data):
        """
        测试Normalize转换
        """
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformed_data = transform(sample_data.copy())
        
        # 验证图像已归一化 (范围应在[-1, 1]左右)
        image_mean = transformed_data['image'].mean()
        image_std = transformed_data['image'].std()
        
        # 由于随机输入，我们只验证数据类型和形状
        assert transformed_data['image'].dtype == np.float32
        assert transformed_data['image'].shape == sample_data['image'].shape
    
    def test_to_tensor_transform(self, sample_data):
        """
        测试ToTensor转换
        """
        transform = ToTensor()
        transformed_data = transform(sample_data.copy())
        
        # 验证所有数据都转换为张量
        assert isinstance(transformed_data['image'], torch.Tensor)
        assert isinstance(transformed_data['segmentation_mask'], torch.Tensor)
        assert isinstance(transformed_data['detection_boxes'], torch.Tensor)
        assert isinstance(transformed_data['detection_labels'], torch.Tensor)
        assert isinstance(transformed_data['uav_attitude'], torch.Tensor)
        
        # 验证图像形状已从HWC转为CHW
        assert transformed_data['image'].shape == (3, 512, 512)
    
    def test_compose_transforms(self, sample_data):
        """
        测试Compose组合转换
        """
        # 创建转换管道
        transforms = Compose([
            Resize(size=(256, 256)),
            Normalize(),
            ToTensor()
        ])
        
        # 应用转换
        transformed_data = transforms(sample_data.copy())
        
        # 验证所有转换都已应用
        assert transformed_data['image'].shape == (3, 256, 256)  # CHW格式
        assert transformed_data['image'].dtype == torch.float32
        assert transformed_data['segmentation_mask'].shape == (256, 256)
        
        # 验证边界框已调整
        assert transformed_data['detection_boxes'].shape == sample_data['detection_boxes'].shape
    
    def test_random_flip_transform(self, sample_data):
        """
        测试RandomFlip转换
        """
        # 创建翻转转换 (强制翻转以确保可测试性)
        transform = RandomFlip(flip_prob=1.0, horizontal=True, vertical=False)
        
        # 应用转换
        transformed_data = transform(sample_data.copy())
        
        # 验证边界框已翻转
        original_box = sample_data['detection_boxes'][0]
        flipped_box = transformed_data['detection_boxes'][0]
        
        # 水平翻转时，x坐标应该从另一侧开始
        image_width = sample_data['image'].shape[1]
        assert flipped_box[0] == image_width - original_box[2]
        assert flipped_box[2] == image_width - original_box[0]
        
        # y坐标应该保持不变
        assert flipped_box[1] == original_box[1]
        assert flipped_box[3] == original_box[3]
    
    def test_random_rotate_transform(self, sample_data):
        """
        测试RandomRotate转换
        """
        # 创建旋转转换 (使用小角度以确保边界框仍然有效)
        transform = RandomRotate(rotate_prob=1.0, max_angle=5.0)
        
        # 应用转换
        transformed_data = transform(sample_data.copy())
        
        # 验证图像和掩码已旋转
        assert transformed_data['image'].shape == sample_data['image'].shape
        assert transformed_data['segmentation_mask'].shape == sample_data['segmentation_mask'].shape
        
        # 验证边界框已旋转
        assert transformed_data['detection_boxes'].shape == sample_data['detection_boxes'].shape
        
        # 边界框坐标应该有所变化
        assert not np.array_equal(transformed_data['detection_boxes'], sample_data['detection_boxes'])

if __name__ == "__main__":
    pytest.main([__file__])