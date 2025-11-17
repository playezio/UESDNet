# 作者: qzf
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

# 导入模型组件
from .backbone import UAVResNet50, FeaturePyramid
from .neck import FeatureFusionNeck
from .heads import MultiTaskHead

class UESDNet(nn.Module):
    """
    UESDNet模型 - 无人机环境感知与同步检测网络
    结合了特征提取、融合和多任务学习功能
    """
    
    def __init__(self, 
                 num_classes: int = 1,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 use_expertsync: bool = True):
        super().__init__()
        
        # 创建网络组件
        self.backbone = UAVResNet50(pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.feature_pyramid = FeaturePyramid()  # 特征金字塔
        self.neck = FeatureFusionNeck()  # 特征融合层
        self.head = MultiTaskHead(in_channels=256, num_classes=num_classes)
        
        # 配置参数
        self.use_expertsync = use_expertsync
        self.num_classes = num_classes
    
    def forward(self, images, uav_attitude=None, return_features=False):
        # 标准前向传播流程
        # 1. 提取基础特征
        backbone_features = self.backbone(images, uav_attitude)
        
        # 2. 构建特征金字塔
        pyramid_features = self.feature_pyramid(backbone_features)
        
        # 3. 特征融合增强
        neck_features = self.neck(pyramid_features)
        
        # 4. 生成检测和分割结果
        outputs = self.head(neck_features)
        
        # 调试或分析时可以返回中间特征
        if return_features:
            outputs.update({
                'backbone_features': backbone_features,
                'pyramid_features': pyramid_features,
                'neck_features': neck_features
            })
        
        return outputs
    
    def freeze_features(self, freeze_backbone=True, freeze_neck=False):
        # 冻结指定组件的参数，用于分阶段训练
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if freeze_neck:
            for param in self.neck.parameters():
                param.requires_grad = False
    
    def unfreeze_features(self):
        # 解冻所有参数，用于微调阶段
        for param in self.parameters():
            param.requires_grad = True
    
    def get_model_parameters(self):
        # 计算模型参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @torch.no_grad()
    def inference(self, images, uav_attitude=None, threshold=0.5):
        # 推理阶段使用的函数
        self.eval()
        
        outputs = self.forward(images, uav_attitude)
        
        # 简单后处理 - 应用阈值过滤低置信度预测
        if 'cls_p3' in outputs:
            # 应用sigmoid激活
            outputs['cls_p3'] = torch.sigmoid(outputs['cls_p3'])
            outputs['cls_p4'] = torch.sigmoid(outputs['cls_p4'])
            
            # 阈值过滤
            outputs['cls_p3'][outputs['cls_p3'] < threshold] = 0
            outputs['cls_p4'][outputs['cls_p4'] < threshold] = 0
        
        # 分割结果处理
        if 'segmentation' in outputs:
            outputs['segmentation'] = torch.softmax(outputs['segmentation'], dim=1)
        
        return outputs

def get_uesdnet_model(config=None):
    # 辅助函数：根据配置创建模型实例
    if config is None:
        config = {
            'num_classes': 1,
            'pretrained': True,
            'freeze_backbone': False,
            'use_expertsync': True
        }
    
    # 创建模型
    model = UESDNet(
        num_classes=config.get('num_classes', 1),
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        use_expertsync=config.get('use_expertsync', True)
    )
    
    return model

def count_parameters(model):
    # 计算模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def print_model_summary(model, input_shape=(1, 3, 1080, 1920)):
    # 打印模型结构和参数统计
    try:
        from torchsummary import summary
        
        device = next(model.parameters()).device
        summary(model, input_shape, device=str(device))
        
        # 参数统计
        params = count_parameters(model)
        print(f"\n模型参数量:")
        print(f"总参数量: {params['total']:,}")
        print(f"可训练: {params['trainable']:,}")
        print(f"冻结: {params['frozen']:,}")
    except ImportError:
        print("需要安装torchsummary库来查看模型摘要")
        # 仅打印参数统计
        params = count_parameters(model)
        print(f"\n模型参数量:")
        print(f"总参数量: {params['total']:,}")