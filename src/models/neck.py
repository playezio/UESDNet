# 作者: qzf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class FeatureFusionNeck(nn.Module):
    """
    特征融合模块
    融合不同尺度的特征并增强其表达能力
    """
    def __init__(self, in_channels=None, out_channels=256, use_attention=True):
        super().__init__()
        
        # 默认输入通道数
        if in_channels is None:
            in_channels = {
                'P1': 256,
                'P2': 256,
                'P3': 256,
                'P4': 256
            }
        
        self.out_channels = out_channels
        self.use_attention = use_attention
        
        # 1×1卷积用于统一通道数
        self.conv_p1 = nn.Conv2d(in_channels['P1'], out_channels, kernel_size=1)
        self.conv_p2 = nn.Conv2d(in_channels['P2'], out_channels, kernel_size=1)
        self.conv_p3 = nn.Conv2d(in_channels['P3'], out_channels, kernel_size=1)
        self.conv_p4 = nn.Conv2d(in_channels['P4'], out_channels, kernel_size=1)
        
        # 上采样层
        self.up_p4_to_p3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_p3_to_p2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_p2_to_p1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 特征融合卷积层
        self.fusion_p3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.fusion_p2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.fusion_p1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        
        # 通道注意力模块
        if use_attention:
            self.attention_p1 = ChannelAttention(out_channels)
            self.attention_p2 = ChannelAttention(out_channels)
            self.attention_p3 = ChannelAttention(out_channels)
            self.attention_p4 = ChannelAttention(out_channels)
        
        # 最终特征处理
        self.final_conv_p1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.final_conv_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.final_conv_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.final_conv_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 批归一化和激活函数
        self.bn_p1 = nn.BatchNorm2d(out_channels)
        self.bn_p2 = nn.BatchNorm2d(out_channels)
        self.bn_p3 = nn.BatchNorm2d(out_channels)
        self.bn_p4 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features):
        # 特征融合流程
        P1, P2, P3, P4 = features['P1'], features['P2'], features['P3'], features['P4']
        
        # 统一通道数
        P1 = self.conv_p1(P1)
        P2 = self.conv_p2(P2)
        P3 = self.conv_p3(P3)
        P4 = self.conv_p4(P4)
        
        # 应用注意力机制
        if self.use_attention:
            P1 = self.attention_p1(P1) * P1
            P2 = self.attention_p2(P2) * P2
            P3 = self.attention_p3(P3) * P3
            P4 = self.attention_p4(P4) * P4
        
        # 自顶向下的特征融合
        up_p4 = self.up_p4_to_p3(P4)
        fuse_p3 = torch.cat([P3, up_p4], dim=1)
        fuse_p3 = self.fusion_p3(fuse_p3)
        fuse_p3 = self.bn_p3(fuse_p3)
        fuse_p3 = self.relu(fuse_p3)
        
        up_p3 = self.up_p3_to_p2(fuse_p3)
        fuse_p2 = torch.cat([P2, up_p3], dim=1)
        fuse_p2 = self.fusion_p2(fuse_p2)
        fuse_p2 = self.bn_p2(fuse_p2)
        fuse_p2 = self.relu(fuse_p2)
        
        up_p2 = self.up_p2_to_p1(fuse_p2)
        fuse_p1 = torch.cat([P1, up_p2], dim=1)
        fuse_p1 = self.fusion_p1(fuse_p1)
        fuse_p1 = self.bn_p1(fuse_p1)
        fuse_p1 = self.relu(fuse_p1)
        
        # 最终特征处理
        final_p1 = self.final_conv_p1(fuse_p1)
        final_p2 = self.final_conv_p2(fuse_p2)
        final_p3 = self.final_conv_p3(fuse_p3)
        final_p4 = self.final_conv_p4(P4)
        
        # 返回融合后的特征
        neck_features = {
            'neck_p1': final_p1,
            'neck_p2': final_p2,
            'neck_p3': final_p3,
            'neck_p4': final_p4
        }
        
        return neck_features

class ChannelAttention(nn.Module):
    """
    通道注意力模块
    关注特征中的重要通道信息
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        # 全局池化 - 捕获全局上下文
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的通道压缩-扩张结构
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # 生成注意力权重
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道注意力计算
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    空间注意力模块
    识别和增强特征中的重要空间区域
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        # 卷积层用于生成空间注意力图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 空间注意力计算
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块
    建立不同尺度特征之间的关联关系
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # 查询、键、值的投影
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, context):
        # 跨尺度特征关联
        batch_size, channels, height, width = x.size()
        
        # 投影
        q = self.query_proj(x).view(batch_size, channels, -1)
        k = self.key_proj(context).view(batch_size, channels, -1)
        v = self.value_proj(context).view(batch_size, channels, -1)
        
        # 计算注意力
        attention = torch.bmm(q.transpose(1, 2), k)
        attention = self.softmax(attention)
        
        # 应用注意力
        out = torch.bmm(v, attention.transpose(1, 2)).view(batch_size, channels, height, width)
        out = self.out_proj(out)
        
        # 残差连接
        out = out + x
        
        return out