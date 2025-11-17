# 作者: qzf
import os
import sys
import torch
import pytest
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.uesdnet import UESDNet
from src.models.expertsync import apply_expertsync
from src.data.studataset import STUDataset
from src.config import load_config, Config

class TestModel:
    """
    模型测试类
    """
    
    @pytest.fixture
    def sample_config(self):
        """
        创建一个示例配置
        """
        config = {
            "model": {
                "num_classes": 8,
                "pretrained": False,
                "freeze_backbone": False,
                "use_expertsync": False,
                "backbone": {
                    "use_attitude_attention": True
                }
            }
        }
        return Config(**config)
    
    @pytest.fixture
    def sample_image(self):
        """
        创建一个示例图像
        """
        # 创建一个随机的3通道图像 (B, C, H, W)
        return torch.randn(2, 3, 512, 512)
    
    @pytest.fixture
    def sample_attitude(self):
        """
        创建示例无人机姿态信息
        """
        # 创建随机的姿态信息 (B, 3) - [roll, pitch, yaw]
        return torch.randn(2, 3)
    
    def test_model_initialization(self, sample_config):
        """
        测试模型初始化
        """
        # 初始化模型
        model = UESDNet(
            num_classes=sample_config.model.num_classes,
            pretrained=sample_config.model.pretrained,
            freeze_backbone=sample_config.model.freeze_backbone,
            use_expertsync=sample_config.model.use_expertsync
        )
        
        # 验证模型是否正确初始化
        assert model is not None
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'detection_head')
        assert hasattr(model, 'segmentation_head')
    
    def test_model_forward(self, sample_config, sample_image):
        """
        测试模型前向传播
        """
        # 初始化模型
        model = UESDNet(
            num_classes=sample_config.model.num_classes,
            pretrained=False,  # 测试时不使用预训练权重
            freeze_backbone=False,
            use_expertsync=False
        )
        
        # 设置模型为评估模式
        model.eval()
        
        # 前向传播 (无姿态信息)
        with torch.no_grad():
            outputs = model(sample_image)
        
        # 验证输出
        assert 'detection' in outputs
        assert 'segmentation' in outputs
        assert 'cls_logits' in outputs['detection']
        assert 'bbox_regression' in outputs['detection']
        assert 'logits' in outputs['segmentation']
    
    def test_model_with_attitude(self, sample_config, sample_image, sample_attitude):
        """
        测试带姿态信息的模型前向传播
        """
        # 初始化支持姿态信息的模型
        model = UESDNet(
            num_classes=sample_config.model.num_classes,
            pretrained=False,
            freeze_backbone=False,
            use_expertsync=False,
            use_attitude_attention=True
        )
        
        # 设置模型为评估模式
        model.eval()
        
        # 前向传播 (带姿态信息)
        with torch.no_grad():
            outputs = model(sample_image, sample_attitude)
        
        # 验证输出
        assert 'detection' in outputs
        assert 'segmentation' in outputs
    
    def test_expertsync_integration(self, sample_config, sample_image):
        """
        测试ExpertSync模块集成
        """
        # 初始化模型
        model = UESDNet(
            num_classes=sample_config.model.num_classes,
            pretrained=False,
            freeze_backbone=False,
            use_expertsync=True
        )
        
        # 应用ExpertSync
        model = apply_expertsync(model)
        
        # 设置模型为评估模式
        model.eval()
        
        # 前向传播
        with torch.no_grad():
            outputs = model(sample_image)
        
        # 验证输出包含ExpertSync结果
        assert 'expert_sync' in outputs
    
    def test_model_freeze(self, sample_config):
        """
        测试模型冻结功能
        """
        # 初始化模型
        model = UESDNet(
            num_classes=sample_config.model.num_classes,
            pretrained=False,
            freeze_backbone=True,
            use_expertsync=False
        )
        
        # 冻结backbone
        model.freeze_features(freeze_backbone=True)
        
        # 检查参数是否冻结
        for param in model.backbone.parameters():
            assert not param.requires_grad
        
        # 解冻backbone
        model.freeze_features(freeze_backbone=False)
        
        # 检查参数是否解冻
        for param in model.backbone.parameters():
            assert param.requires_grad

class TestDataLoader:
    """
    数据加载器测试类
    """
    
    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """
        创建一个模拟的数据集目录结构
        """
        # 创建图像目录
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        
        # 创建标注目录
        anno_dir = tmp_path / "annotations"
        anno_dir.mkdir()
        
        # 创建姿态信息目录
        attitude_dir = tmp_path / "attitude"
        attitude_dir.mkdir()
        
        # 创建一个示例图像
        sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        np.save(str(image_dir / "sample_001.npy"), sample_image)
        
        # 创建示例姿态信息
        sample_attitude = np.array([10.0, -5.0, 30.0])
        np.save(str(attitude_dir / "sample_001.npy"), sample_attitude)
        
        # 创建示例分割标注
        sample_mask = np.zeros((512, 512), dtype=np.int64)
        np.save(str(anno_dir / "sample_001_mask.npy"), sample_mask)
        
        # 创建示例检测标注
        sample_boxes = np.array([[100, 100, 200, 200]])
        sample_labels = np.array([1])
        np.save(str(anno_dir / "sample_001_boxes.npy"), sample_boxes)
        np.save(str(anno_dir / "sample_001_labels.npy"), sample_labels)
        
        # 创建图像ID列表
        with open(tmp_path / "train.txt", "w") as f:
            f.write("sample_001\n")
        
        return tmp_path
    
    def test_dataset_initialization(self, mock_dataset_dir):
        """
        测试数据集初始化
        """
        # 初始化数据集
        dataset = STUDataset(
            root_dir=str(mock_dataset_dir),
            split="train",
            transform=None,
            use_attitude=True
        )
        
        # 验证数据集长度
        assert len(dataset) == 1
    
    def test_dataset_getitem(self, mock_dataset_dir):
        """
        测试数据集getitem方法
        """
        # 初始化数据集
        dataset = STUDataset(
            root_dir=str(mock_dataset_dir),
            split="train",
            transform=None,
            use_attitude=True
        )
        
        # 获取第一个数据项
        data = dataset[0]
        
        # 验证数据项内容
        assert 'image' in data
        assert 'detection_boxes' in data
        assert 'detection_labels' in data
        assert 'segmentation_mask' in data
        assert 'uav_attitude' in data
        
        # 验证数据形状
        assert data['image'].shape == (3, 512, 512)
        assert data['uav_attitude'].shape == (3,)

class TestConfig:
    """
    配置系统测试类
    """
    
    @pytest.fixture
    def config_file(self, tmp_path):
        """
        创建一个临时配置文件
        """
        config_content = """
model:
  num_classes: 8
  pretrained: true
  freeze_backbone: false
  use_expertsync: true
  
data:
  dataset:
    root_dir: "./data"
    train_split: "train.txt"
  
  loader:
    batch_size: 4
    """
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            f.write(config_content)
        
        return config_path
    
    def test_config_loading(self, config_file):
        """
        测试配置加载
        """
        # 加载配置
        config = load_config(config_file)
        
        # 验证配置内容
        assert config.model.num_classes == 8
        assert config.model.pretrained is True
        assert config.model.use_expertsync is True
        assert config.data.dataset.root_dir == "./data"
        assert config.data.loader.batch_size == 4
    
    def test_config_validation(self, sample_config):
        """
        测试配置验证
        """
        # 验证配置是否有效
        assert hasattr(sample_config, 'model')
        assert hasattr(sample_config.model, 'num_classes')
        assert sample_config.model.num_classes > 0

if __name__ == "__main__":
    pytest.main([__file__])