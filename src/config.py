# 作者: qzf
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """模型配置类"""
    backbone_type: str = "C2F2"
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_fa: bool = True

@dataclass
class NeckConfig:
    """颈部配置类"""
    detection: str = "AFF"
    segmentation: str = "AAFF"

@dataclass
class ExpertSyncConfig:
    """ExpertSync模块配置类"""
    enabled: bool = True
    det_threshold: float = 0.5
    attitude_gating: bool = True

@dataclass
class TrainingConfig:
    """训练配置类"""
    warmup_epochs: int = 30
    total_epochs: int = 200
    optimizer_type: str = "Adam"
    lr: float = 1e-4
    batch_size: int = 8
    num_workers: int = 4

@dataclass
class LossConfig:
    """损失函数配置类"""
    det: float = 1.0
    seg: float = 1.0
    consistency: float = 0.5
    det_in_seg: float = 1.0
    seg_in_det: float = 1.0

@dataclass
class UAVSelfConfig:
    """UAV自定位配置类"""
    enabled: bool = True
    use_c_extension: bool = False
    april_tag_size: float = 0.135  # AprilTag的实际大小（米）
    camera_matrix: List[List[float]] = field(default_factory=lambda: [[1399.008608, 0.0, 912.8436694988], 
                                                                     [0.0, 1398.06775013, 560.97156462088], 
                                                                     [0.0, 0.0, 1.0]])

@dataclass
class DataConfig:
    """数据配置类"""
    dataset_path: str = "/path/to/STUDataset"
    input_size: List[int] = field(default_factory=lambda: [1920, 1080])

@dataclass
class UESDNetConfig:
    """UESDNet整体配置类"""
    model: ModelConfig = field(default_factory=ModelConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    expertsync: ExpertSyncConfig = field(default_factory=ExpertSyncConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    uav_self: UAVSelfConfig = field(default_factory=UAVSelfConfig)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")

def create_config_from_dict(config_dict: Dict[str, Any]) -> UESDNetConfig:
    """
    从字典创建配置对象
    Args:
        config_dict: 配置字典
    Returns:
        UESDNetConfig对象
    """
    # 处理嵌套配置结构
    model_dict = config_dict.get('model', {})
    neck_dict = config_dict.get('neck', {})
    expertsync_dict = config_dict.get('expertsync', {})
    training_dict = config_dict.get('training', {})
    loss_dict = config_dict.get('loss', {})
    data_dict = config_dict.get('data', {})
    uav_self_dict = config_dict.get('uav_self', {})
    
    return UESDNetConfig(
        model=ModelConfig(**model_dict),
        neck=NeckConfig(**neck_dict),
        expertsync=ExpertSyncConfig(**expertsync_dict),
        training=TrainingConfig(**training_dict),
        loss=LossConfig(**loss_dict),
        data=DataConfig(**data_dict),
        uav_self=UAVSelfConfig(**uav_self_dict)
    )

def validate_config(config: UESDNetConfig) -> None:
    """
    验证配置的有效性
    Args:
        config: 配置对象
    """
    # 验证模型配置
    assert config.model.backbone_type in ["C2F2"], "Unsupported backbone type"
    assert len(config.model.channels) == 4, "Channels must have 4 elements"
    
    # 验证颈部配置
    assert config.neck.detection in ["AFF"], "Unsupported detection neck type"
    assert config.neck.segmentation in ["AAFF"], "Unsupported segmentation neck type"
    
    # 验证训练配置
    assert config.training.warmup_epochs > 0, "Warmup epochs must be positive"
    assert config.training.total_epochs > config.training.warmup_epochs, "Total epochs must be larger than warmup epochs"
    assert config.training.lr > 0, "Learning rate must be positive"
    assert config.training.batch_size > 0, "Batch size must be positive"
    
    # 验证损失配置
    assert all(w >= 0 for w in [config.loss.det, config.loss.seg, config.loss.consistency]), "Loss weights must be non-negative"
    
    # 验证UAV自定位配置
    assert config.uav_self.april_tag_size > 0, "AprilTag size must be positive"
    assert len(config.uav_self.camera_matrix) == 3, "Camera matrix must be 3x3"