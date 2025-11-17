# 作者: qzf
import os
import sys
import pytest
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入UAV自定位模块
from uav_self import UAVSelfLocalizer, get_uav_self_localizer, process_uav_image
from src.config import UESDNetConfig, UAVSelfConfig


class TestUAVSelfLocalizer:
    """
    测试UAV自定位模块
    """
    
    @pytest.fixture
    def sample_image(self):
        """
        创建一个示例图像
        """
        # 创建一个随机的3通道图像 (H, W, C) - BGR格式
        return np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_config(self):
        """
        创建一个示例配置
        """
        return {
            'enabled': True,
            'use_extension': False,  # 测试时不使用C扩展
            'april_tag_size': 0.135
        }
    
    def test_localizer_initialization(self, sample_config):
        """
        测试自定位器初始化
        """
        # 测试默认初始化
        localizer1 = UAVSelfLocalizer()
        assert localizer1 is not None
        assert isinstance(localizer1, UAVSelfLocalizer)
        
        # 测试带配置初始化
        localizer2 = UAVSelfLocalizer(sample_config)
        assert localizer2 is not None
        assert isinstance(localizer2, UAVSelfLocalizer)
        
        # 测试工厂函数
        localizer3 = get_uav_self_localizer(sample_config)
        assert localizer3 is not None
        assert isinstance(localizer3, UAVSelfLocalizer)
    
    def test_process_image(self, sample_image, sample_config):
        """
        测试图像处理功能
        """
        localizer = UAVSelfLocalizer(sample_config)
        
        # 测试处理图像
        result = localizer.process_image(sample_image)
        
        # 验证结果格式
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert 'z' in result
        assert 'yaw' in result
        
        # 验证结果类型
        assert isinstance(result['x'], float)
        assert isinstance(result['y'], float)
        assert isinstance(result['z'], float)
        assert isinstance(result['yaw'], float)
    
    def test_get_ego_information(self, sample_config):
        """
        测试获取无人机自信息
        """
        localizer = UAVSelfLocalizer(sample_config)
        
        # 测试获取自信息
        ego_info = localizer.get_ego_information()
        
        # 验证结果格式
        assert isinstance(ego_info, dict)
        assert 'position' in ego_info
        assert 'attitude' in ego_info
        assert 'velocity' in ego_info
        
        # 验证位置信息
        position = ego_info['position']
        assert isinstance(position, dict)
        assert 'x' in position
        assert 'y' in position
        assert 'z' in position
        
        # 验证姿态信息
        attitude = ego_info['attitude']
        assert isinstance(attitude, dict)
        assert 'pitch' in attitude
        assert 'roll' in attitude
        assert 'yaw' in attitude
    
    def test_convenience_function(self, sample_image, sample_config):
        """
        测试便捷函数
        """
        # 测试便捷函数处理图像
        result = process_uav_image(sample_image, sample_config)
        
        # 验证结果格式
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert 'z' in result
        assert 'yaw' in result
    
    def test_integration_with_config(self):
        """
        测试与项目配置的集成
        """
        # 创建项目配置对象
        config = UESDNetConfig()
        
        # 验证UAV自定位配置存在
        assert hasattr(config, 'uav_self')
        assert isinstance(config.uav_self, UAVSelfConfig)
        
        # 测试配置参数
        assert config.uav_self.enabled is True
        assert config.uav_self.use_c_extension is False
        assert config.uav_self.april_tag_size > 0
        assert len(config.uav_self.camera_matrix) == 3
    
    def test_error_handling(self, sample_image):
        """
        测试错误处理
        """
        # 测试无效配置
        invalid_config = {'use_extension': True}  # 强制使用C扩展，但可能不可用
        localizer = UAVSelfLocalizer(invalid_config)
        
        # 处理图像应该不会崩溃，会回退到Python实现
        try:
            result = localizer.process_image(sample_image)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"处理图像时发生异常: {e}")


# 如果直接运行测试
if __name__ == "__main__":
    pytest.main([__file__])