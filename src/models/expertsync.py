# 作者: qzf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

class ExpertSync(nn.Module):
    """
    多任务专家同步学习模块
    协调不同专家网络之间的知识共享
    """
    
    def __init__(self, in_channels=256, num_experts=2, sync_threshold=0.5):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.sync_threshold = sync_threshold
        
        # 各专家网络的注意力模块
        self.expert_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for _ in range(num_experts)
        ])
        
        # 知识蒸馏模块
        self.knowledge_distiller = KnowledgeDistillation(in_channels)
        
        # 专家协同模块
        self.expert_coordination = ExpertCoordination(in_channels, num_experts)
        
        # 同步门控机制
        self.sync_gate = nn.Sequential(
            nn.Conv2d(in_channels * num_experts, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, expert_features, expert_outputs, global_feature=None):
        # 前向传播实现专家同步
        batch_size, _, height, width = expert_features[0].size()
        
        # 1. 应用专家特定注意力
        attended_features = []
        for i, feat in enumerate(expert_features):
            attention = self.expert_attention[i](feat)
            attended = feat * attention
            attended_features.append(attended)
        
        # 2. 计算同步门控
        combined_features = torch.cat(attended_features, dim=1)
        sync_gates = self.sync_gate(combined_features)
        
        # 3. 知识蒸馏和专家协同
        distilled_knowledge = self.knowledge_distiller(expert_features, expert_outputs)
        coordinated_features = self.expert_coordination(attended_features, global_feature)
        
        # 4. 应用同步机制增强特征
        enhanced_features = []
        for i in range(self.num_experts):
            # 获取当前专家的同步门控
            gate = sync_gates[:, i:i+1]  # [B, 1, H, W]
            
            # 收集其他专家的知识
            other_experts = [j for j in range(self.num_experts) if j != i]
            other_knowledge = sum(distilled_knowledge[j][i] for j in other_experts)
            
            # 门控融合知识
            enhanced = attended_features[i] + gate * other_knowledge
            enhanced_features.append(enhanced)
        
        # 5. 保存中间结果用于分析
        intermediate_results = {
            'sync_gates': sync_gates,
            'distilled_knowledge': distilled_knowledge,
            'coordinated_features': coordinated_features,
            'attended_features': attended_features
        }
        
        return enhanced_features, intermediate_results

class KnowledgeDistillation(nn.Module):
    """
    知识蒸馏模块
    在不同专家网络之间传递有用知识
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # 特征空间转换
        self.feature_transformer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 软目标温度参数
        self.temperature = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, expert_features, expert_outputs):
        # 实现专家间知识蒸馏
        num_experts = len(expert_features)
        
        # 初始化知识转移矩阵
        distilled_knowledge = [[None for _ in range(num_experts)] for _ in range(num_experts)]
        
        # 遍历所有专家对
        for source_idx in range(num_experts):
            for target_idx in range(num_experts):
                if source_idx != target_idx:
                    # 1. 特征空间转换
                    transformed_feature = self.feature_transformer(expert_features[source_idx])
                    
                    # 2. 特征尺寸对齐
                    if transformed_feature.shape != expert_features[target_idx].shape:
                        transformed_feature = F.interpolate(
                            transformed_feature,
                            size=expert_features[target_idx].shape[2:],
                            mode='bilinear',
                            align_corners=True
                        )
                    
                    # 3. 计算注意力权重
                    attention = self._compute_attention(
                        expert_features[target_idx], 
                        transformed_feature
                    )
                    
                    # 4. 应用注意力权重提取相关知识
                    distilled = transformed_feature * attention
                    distilled_knowledge[source_idx][target_idx] = distilled
        
        return distilled_knowledge
    
    def _compute_attention(self, target, source):
        # 计算注意力权重，找出最相关的知识
        # 归一化特征向量
        target_norm = F.normalize(target, dim=1)
        source_norm = F.normalize(source, dim=1)
        
        # 计算特征相似度
        similarity = torch.sum(target_norm * source_norm, dim=1, keepdim=True)
        
        # 应用温度参数控制分布平滑度
        attention = F.softmax(similarity / self.temperature, dim=2)
        
        return attention

class ExpertCoordination(nn.Module):
    """
    专家协同模块
    促进不同专家之间的协作学习
    """
    def __init__(self, in_channels, num_experts):
        super().__init__()
        
        # 多专家特征融合
        self.coordination_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_experts, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 专家权重生成
        self.expert_weights = nn.Conv2d(in_channels, num_experts, kernel_size=1)
    
    def forward(self, expert_features, global_feature=None):
        # 实现专家协同机制
        num_experts = len(expert_features)
        
        # 1. 合并所有专家特征
        combined = torch.cat(expert_features, dim=1)
        
        # 2. 计算协同特征
        coordination = self.coordination_conv(combined)
        
        # 3. 结合全局特征（如果提供）
        if global_feature is not None:
            if global_feature.shape != coordination.shape:
                global_feature = F.interpolate(
                    global_feature,
                    size=coordination.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            coordination = coordination + global_feature
        
        # 4. 动态计算专家权重
        weights = self.expert_weights(coordination)
        weights = F.softmax(weights, dim=1)
        
        # 5. 应用协同权重增强每个专家特征
        coordinated_features = []
        for i in range(num_experts):
            weight = weights[:, i:i+1]
            coordinated = expert_features[i] * (1 + weight)  # 加性增强
            coordinated_features.append(coordinated)
        
        return coordinated_features

class TaskSpecificExpert(nn.Module):
    """
    任务特定专家网络
    根据任务类型定制特征提取
    """
    def __init__(self, in_channels, out_channels, task_type='detection'):
        super().__init__()
        
        self.task_type = task_type
        
        # 构建任务特定特征提取器
        # 检测和分割任务结构相似但可根据需要调整
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 特征提取
        return self.feature_extractor(x)

def apply_expertsync(model, use_expertsync=True):
    """
    将ExpertSync模块应用到模型中
    增强多任务学习能力
    """
    if not use_expertsync:
        return model
    
    # 第一种情况：在neck部分应用ExpertSync
    if hasattr(model, 'backbone') and hasattr(model, 'neck'):
        original_neck = model.neck
        expertsync = ExpertSync(in_channels=256, num_experts=2, sync_threshold=0.5)
        
        class ExpertSyncWrapper(nn.Module):
            def __init__(self, neck, expertsync_module):
                super().__init__()
                self.neck = neck
                self.expertsync = expertsync_module
            
            def forward(self, x):
                # 先通过neck获取特征
                neck_out = self.neck(x)
                # 处理多专家特征情况
                if isinstance(neck_out, (list, tuple)) and len(neck_out) >= 2:
                    # 假设前两个特征分别用于检测和分割
                    det_feat, seg_feat = neck_out[0], neck_out[1]
                    expert_features = [det_feat, seg_feat]
                    
                    # 使用零占位输出进行同步（简化处理）
                    dummy_det_out = torch.zeros_like(det_feat)
                    dummy_seg_out = torch.zeros_like(seg_feat)
                    expert_outputs = [dummy_det_out, dummy_seg_out]
                    
                    # 应用专家同步
                    enhanced_features, _ = self.expertsync(expert_features, expert_outputs)
                    
                    # 返回增强后的特征，保留其他特征
                    return enhanced_features + list(neck_out[2:]) if len(neck_out) > 2 else enhanced_features
                else:
                    return neck_out
        
        model.neck = ExpertSyncWrapper(original_neck, expertsync)
    
    # 第二种情况：在检测头和分割头之间应用ExpertSync
    if hasattr(model, 'det_head') and hasattr(model, 'seg_head'):
        original_det_head = model.det_head
        original_seg_head = model.seg_head
        
        class ExpertSyncHead(nn.Module):
            def __init__(self, det_head, seg_head, expertsync_module):
                super().__init__()
                self.det_head = det_head
                self.seg_head = seg_head
                self.expertsync = expertsync_module
            
            def forward(self, x):
                # 处理多特征输入
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    det_feat, seg_feat = x[0], x[1]
                    expert_features = [det_feat, seg_feat]
                    
                    # 先获取专家输出
                    det_out = self.det_head(det_feat)
                    seg_out = self.seg_head(seg_feat)
                    expert_outputs = [det_out, seg_out]
                    
                    # 应用专家同步
                    enhanced_features, _ = self.expertsync(expert_features, expert_outputs)
                    
                    # 使用增强特征重新计算输出
                    new_det_out = self.det_head(enhanced_features[0])
                    new_seg_out = self.seg_head(enhanced_features[1])
                    return [new_det_out, new_seg_out]
                else:
                    # 单一特征输入的处理方式
                    det_out = self.det_head(x)
                    seg_out = self.seg_head(x)
                    return [det_out, seg_out]
        
        # 替换为新的融合头
        expertsync = ExpertSync(in_channels=256, num_experts=2, sync_threshold=0.5)
        model.heads = ExpertSyncHead(original_det_head, original_seg_head, expertsync)
        
        # 移除原始头属性避免冲突
        if hasattr(model, 'det_head'):
            delattr(model, 'det_head')
        if hasattr(model, 'seg_head'):
            delattr(model, 'seg_head')
    
    return model