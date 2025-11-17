# 作者: qzf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class DetectionHead(nn.Module):
    """
    检测头模块
    生成目标检测所需的边界框和类别预测
    """
    def __init__(self, in_channels=256, num_classes=1, num_anchors=9):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 边界框回归网络
        self.bbox_regressor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)  # 每个锚框4个坐标
        )
        
        # 分类预测网络
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _init_weights(self):
        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 前向传播处理检测任务
        # 提取不同尺度特征
        neck_p3 = x['neck_p3']
        neck_p4 = x['neck_p4']
        
        # 计算边界框回归预测
        bbox_reg_p3 = self.bbox_regressor(neck_p3)
        bbox_reg_p4 = self.bbox_regressor(neck_p4)
        
        # 计算类别预测
        cls_p3 = self.classifier(neck_p3)
        cls_p4 = self.classifier(neck_p4)
        
        # 整合结果
        detection_outputs = {
            'bbox_reg_p3': bbox_reg_p3,
            'bbox_reg_p4': bbox_reg_p4,
            'cls_p3': cls_p3,
            'cls_p4': cls_p4
        }
        
        return detection_outputs

class SegmentationHead(nn.Module):
    """
    分割头模块
    生成像素级分类结果
    """
    def __init__(self, in_channels=256, num_classes=1, feature_levels=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_levels = feature_levels if feature_levels else ['neck_p1', 'neck_p2', 'neck_p3']
        
        # 特征融合层
        self.feature_convs = nn.ModuleDict()
        for level in self.feature_levels:
            self.feature_convs[level] = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        # 上采样层
        self.upsample_p2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_p3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        # 融合后的处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 * len(self.feature_levels), in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 分割输出层
        self.segmentation_out = nn.Conv2d(in_channels, num_classes + 1, kernel_size=1)  # +1 for background
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 前向传播处理分割任务
        processed_features = []
        
        # 处理不同层级特征并上采样到相同尺寸
        for level in self.feature_levels:
            feat = self.feature_convs[level](x[level])
            
            # 根据特征层级进行适当的上采样
            if level == 'neck_p2':
                feat = self.upsample_p2(feat)
            elif level == 'neck_p3':
                feat = self.upsample_p3(feat)
            
            processed_features.append(feat)
        
        # 融合多尺度特征
        fused = torch.cat(processed_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # 生成分割结果
        segmentation = self.segmentation_out(fused)
        
        return segmentation

class UnifiedHead(nn.Module):
    """
    统一的检测和分割头模块
    整合两个任务的输出
    """
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        
        # 初始化检测头和分割头
        self.detection_head = DetectionHead(in_channels, num_classes)
        self.segmentation_head = SegmentationHead(in_channels, num_classes)
    
    def forward(self, x):
        # 前向传播处理统一输出
        # 获取检测结果
        detection_results = self.detection_head(x)
        
        # 获取分割结果
        segmentation = self.segmentation_head(x)
        
        # 整合输出结果
        results = {
            **detection_results,
            'segmentation': segmentation
        }
        
        return results

class MultiTaskHead(nn.Module):
    """
    多任务头模块
    支持检测和分割任务的交互学习
    """
    def __init__(self, in_channels=256, num_classes=1, use_shared_features=True):
        super().__init__()
        
        self.use_shared_features = use_shared_features
        
        # 共享特征层
        if use_shared_features:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # 任务特定层
        self.detection_specific = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.segmentation_specific = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # 初始化检测头和分割头
        self.detection_head = DetectionHead(in_channels, num_classes)
        self.segmentation_head = SegmentationHead(in_channels, num_classes)
        
        # 任务交互门控机制
        self.task_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 前向传播处理多任务学习
        processed_features = {}
        
        # 处理每个层级的特征
        for key, feat in x.items():
            # 共享特征提取
            if self.use_shared_features:
                shared_feat = self.shared_conv(feat)
            else:
                shared_feat = feat
            
            # 生成任务特定特征
            detection_feat = self.detection_specific(shared_feat)
            segmentation_feat = self.segmentation_specific(shared_feat)
            
            # 在高分辨率特征上应用任务交互
            if key in ['neck_p1', 'neck_p2']:
                gate_input = torch.cat([detection_feat, segmentation_feat], dim=1)
                gate = self.task_gate(gate_input)
                detection_feat = detection_feat * (1 + gate)
                segmentation_feat = segmentation_feat * (1 + gate)
            
            # 保存处理后的特征
            processed_features[f'{key}_detection'] = detection_feat
            processed_features[f'{key}_segmentation'] = segmentation_feat
        
        # 准备任务特定输入
        detection_input = {
            'neck_p3': processed_features['neck_p3_detection'],
            'neck_p4': processed_features['neck_p4_detection']
        }
        
        segmentation_input = {
            'neck_p1': processed_features['neck_p1_segmentation'],
            'neck_p2': processed_features['neck_p2_segmentation'],
            'neck_p3': processed_features['neck_p3_segmentation']
        }
        
        # 分别计算各任务结果
        detection_results = self.detection_head(detection_input)
        segmentation = self.segmentation_head(segmentation_input)
        
        # 整合输出
        results = {
            **detection_results,
            'segmentation': segmentation
        }
        
        return results