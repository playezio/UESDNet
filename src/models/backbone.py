# 作者: qzf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List

# 导入预训练模型
from torchvision.models import resnet50, ResNet50_Weights

class UAVResNet50(nn.Module):
    """
    无人机图像特征提取器 - 基于ResNet50改进
    适配无人机图像的特征提取网络，添加姿态感知能力
    """
    
    def __init__(self, pretrained=True, freeze_backbone=False, out_indices=[2, 3, 4]):
        super().__init__()
        
        # 加载预训练的ResNet50或随机初始化
        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = resnet50()
        
        # 移除分类头，只保留特征提取部分
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        
        # 配置输出层索引
        self.out_indices = out_indices
        
        # 根据需求冻结骨干网络参数
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 添加姿态感知模块，适应无人机拍摄角度变化
        self.pose_attention = PoseAwareAttention()
    
    def forward(self, x, uav_attitude=None):
        # 提取特征并应用姿态感知注意力
        features = {}
        
        # 基础ResNet50前处理
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # 逐层提取特征
        x = self.resnet.layer1(x)
        features['C1'] = x
        
        x = self.resnet.layer2(x)
        features['C2'] = x
        
        # 当提供姿态信息时，应用姿态感知注意力
        if uav_attitude is not None:
            x = self.pose_attention(x, uav_attitude)
        
        x = self.resnet.layer3(x)
        features['C3'] = x
        
        x = self.resnet.layer4(x)
        features['C4'] = x
        
        # 构建输出特征集
        output_features = {}
        for i in self.out_indices:
            key = f'C{i}'
            if key in features:
                output_features[key] = features[key]
        
        return output_features

class PoseAwareAttention(nn.Module):
    """
    姿态感知注意力模块
    根据无人机姿态动态调整特征提取策略
    """
    
    def __init__(self, in_channels=512, reduction_ratio=4):
        super().__init__()
        
        # 将姿态信息(俯仰/横滚/偏航)映射到特征维度
        self.pose_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_channels)
        )
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, uav_attitude):
        # 根据无人机姿态调整特征
        batch_size = x.size(0)
        
        # 处理姿态信息并生成特征级权重
        pose_features = self.pose_proj(uav_attitude)
        pose_features = pose_features.view(batch_size, -1, 1, 1)
        
        # 计算通道注意力
        channel_attention = self.channel_attention(x)
        
        # 融合姿态信息和通道注意力
        combined_attention = channel_attention * pose_features.sigmoid()
        
        # 应用注意力到特征图
        x = x * combined_attention
        
        return x

class FeaturePyramid(nn.Module):
    """
    特征金字塔网络
    构建多尺度特征表示
    """
    
    def __init__(self):
        super().__init__()
        
        # 通道调整卷积层
        self.conv_c2 = nn.Conv2d(512, 256, kernel_size=1)  # 通道降维
        self.conv_c3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv_c4 = nn.Conv2d(2048, 256, kernel_size=1)
        
        # 平滑卷积层
        self.conv_p3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 3x3卷积平滑
        self.conv_p4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_p5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    
    def forward(self, features):
        # 构建特征金字塔
        # 获取骨干网络各层特征
        c2 = features.get('C2')
        c3 = features.get('C3')
        c4 = features.get('C4')
        
        # 检查必需的特征
        if c2 is None or c3 is None or c4 is None:
            raise ValueError('骨干网络特征不完整')
        
        # 调整各层通道数
        p4 = self.conv_c4(c4)
        p3 = self.conv_c3(c3)
        p2 = self.conv_c2(c2)
        
        # 自上而下融合多尺度特征
        # 先将上层特征上采样与下层特征相加
        upsampled_p4 = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = p3 + upsampled_p4
        
        upsampled_p3 = F.interpolate(p3, scale_factor=2, mode='nearest')
        p2 = p2 + upsampled_p3
        
        # 应用卷积平滑融合特征
        p3 = self.conv_p3(p3)
        p4 = self.conv_p4(p4)
        p5 = self.conv_p5(p4)  # P5通过对P4再次下采样得到
        
        # 组织输出
        pyramid_outputs = {
            'P2': p2,  # 高分辨率特征
            'P3': p3,
            'P4': p4,
            'P5': p5   # 低分辨率高语义特征
        }
        
        return pyramid_outputs