# 作者: qzf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class DetectionLoss(nn.Module):
    """
    目标检测损失函数
    结合边界框回归损失和分类损失
    """
    def __init__(self,
                 cls_loss_weight: float = 1.0,
                 reg_loss_weight: float = 1.0,
                 reg_loss_type: str = 'l1'):
        """
        初始化检测损失
        Args:
            cls_loss_weight: 分类损失权重
            reg_loss_weight: 回归损失权重
            reg_loss_type: 回归损失类型，支持'l1'、'smooth_l1'、'giou'
        """
        super(DetectionLoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.reg_loss_type = reg_loss_type
        
        # 分类损失
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # 回归损失
        if reg_loss_type == 'l1':
            self.reg_loss_fn = nn.L1Loss(reduction='mean')
        elif reg_loss_type == 'smooth_l1':
            self.reg_loss_fn = nn.SmoothL1Loss(reduction='mean')
        elif reg_loss_type == 'giou':
            self.reg_loss_fn = self._giou_loss
        else:
            raise ValueError(f"Unsupported regression loss type: {reg_loss_type}")
    
    def _giou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算GIoU损失
        Args:
            pred_boxes: 预测边界框，形状 [B, N, 4]，格式 [x1, y1, x2, y2]
            target_boxes: 目标边界框，形状 [B, N, 4]，格式 [x1, y1, x2, y2]
        Returns:
            GIoU损失值
        """
        # 计算交集
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # 计算并集
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union = pred_area + target_area - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-8)
        
        # 计算最小包围矩形
        x1_c = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
        y1_c = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
        x2_c = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
        y2_c = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
        
        c_area = (x2_c - x1_c) * (y2_c - y1_c)
        
        # 计算GIoU
        giou = iou - (c_area - union) / (c_area + 1e-8)
        
        # 计算GIoU损失
        loss = 1 - giou
        
        return loss.mean()
    
    def forward(self, 
                pred_cls: torch.Tensor, 
                pred_boxes: torch.Tensor,
                target_cls: torch.Tensor,
                target_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播计算损失
        Args:
            pred_cls: 预测类别，形状 [B, N, C]
            pred_boxes: 预测边界框，形状 [B, N, 4]
            target_cls: 目标类别，形状 [B, N]
            target_boxes: 目标边界框，形状 [B, N, 4]
        Returns:
            损失字典
        """
        # 计算分类损失
        cls_loss = self.cls_loss_fn(pred_cls.permute(0, 2, 1), target_cls)
        
        # 计算回归损失（只考虑正样本）
        positive_mask = target_cls > 0  # 类别索引从1开始
        if positive_mask.any():
            # 过滤出正样本的边界框
            pred_boxes_pos = pred_boxes[positive_mask]
            target_boxes_pos = target_boxes[positive_mask]
            
            if self.reg_loss_type == 'giou':
                reg_loss = self.reg_loss_fn(pred_boxes_pos, target_boxes_pos)
            else:
                reg_loss = self.reg_loss_fn(pred_boxes_pos, target_boxes_pos)
        else:
            reg_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # 总损失
        total_loss = self.cls_loss_weight * cls_loss + self.reg_loss_weight * reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }

class SegmentationLoss(nn.Module):
    """
    语义分割损失函数
    支持交叉熵、Dice损失和Focal损失
    """
    def __init__(self,
                 loss_type: str = 'ce',
                 weight: Optional[torch.Tensor] = None,
                 dice_weight: float = 0.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        初始化分割损失
        Args:
            loss_type: 损失类型，支持'ce'、'dice'、'focal'
            weight: 类别权重
            dice_weight: Dice损失权重
            focal_alpha: Focal损失的alpha参数
            focal_gamma: Focal损失的gamma参数
        """
        super(SegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if loss_type == 'ce':
            self.main_loss_fn = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        elif loss_type == 'focal':
            self.main_loss_fn = self._focal_loss
        else:
            self.main_loss_fn = None
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Focal损失
        Args:
            pred: 预测输出，形状 [B, C, H, W]
            target: 目标标签，形状 [B, H, W]
        Returns:
            Focal损失值
        """
        ce_loss = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return loss.mean()
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Dice损失
        Args:
            pred: 预测输出（经过softmax后），形状 [B, C, H, W]
            target: 目标标签，形状 [B, H, W]
        Returns:
            Dice损失值
        """
        B, C, H, W = pred.shape
        
        # 将目标转换为one-hot编码
        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # 计算交集和并集
        intersection = torch.sum(pred * target_onehot, dim=(0, 2, 3))
        union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target_onehot, dim=(0, 2, 3))
        
        # 计算Dice系数
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        
        # 计算Dice损失
        loss = 1 - dice.mean()
        
        return loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播计算损失
        Args:
            pred: 预测输出，形状 [B, C, H, W]
            target: 目标标签，形状 [B, H, W]
        Returns:
            损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 计算主损失
        if self.loss_type == 'ce' or self.loss_type == 'focal':
            main_loss = self.main_loss_fn(pred, target)
            total_loss += main_loss
            loss_dict['main_loss'] = main_loss
        elif self.loss_type == 'dice':
            # Dice损失需要softmax输出
            pred_softmax = F.softmax(pred, dim=1)
            dice_loss = self._dice_loss(pred_softmax, target)
            total_loss += dice_loss
            loss_dict['dice_loss'] = dice_loss
        
        # 添加Dice损失（如果需要）
        if self.dice_weight > 0 and self.loss_type != 'dice':
            pred_softmax = F.softmax(pred, dim=1)
            dice_loss = self._dice_loss(pred_softmax, target)
            total_loss += self.dice_weight * dice_loss
            loss_dict['dice_loss'] = dice_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict

class ExpertSyncLoss(nn.Module):
    """
    ExpertSync模块的损失函数
    用于计算专家之间的知识蒸馏和协同损失
    """
    def __init__(self,
                 distillation_weight: float = 1.0,
                 coordination_weight: float = 1.0,
                 temperature: float = 4.0):
        """
        初始化ExpertSync损失
        Args:
            distillation_weight: 知识蒸馏损失权重
            coordination_weight: 协同损失权重
            temperature: 温度参数，用于知识蒸馏
        """
        super(ExpertSyncLoss, self).__init__()
        self.distillation_weight = distillation_weight
        self.coordination_weight = coordination_weight
        self.temperature = temperature
    
    def _knowledge_distillation_loss(self, teacher_logits: torch.Tensor, 
                                     student_logits: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失
        Args:
            teacher_logits: 教师模型的输出
            student_logits: 学生模型的输出
        Returns:
            蒸馏损失值
        """
        # 软标签
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        # 学生模型的log_softmax输出
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # 计算KL散度
        loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        return loss
    
    def _coordination_loss(self, expert_features: List[torch.Tensor]) -> torch.Tensor:
        """
        计算专家协同损失
        Args:
            expert_features: 多个专家的特征列表
        Returns:
            协同损失值
        """
        if len(expert_features) < 2:
            return torch.tensor(0.0, device=expert_features[0].device)
        
        loss = 0.0
        num_experts = len(expert_features)
        
        # 计算所有专家对之间的一致性损失
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算特征之间的MSE损失
                loss += F.mse_loss(expert_features[i], expert_features[j])
        
        # 平均所有专家对的损失
        loss /= (num_experts * (num_experts - 1) / 2)
        
        return loss
    
    def forward(self, expert_sync_outputs: Dict) -> Dict[str, torch.Tensor]:
        """
        前向传播计算损失
        Args:
            expert_sync_outputs: ExpertSync模块的输出字典
        Returns:
            损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 知识蒸馏损失
        if 'distillation_pairs' in expert_sync_outputs:
            distillation_loss = 0.0
            for teacher_logits, student_logits in expert_sync_outputs['distillation_pairs']:
                distillation_loss += self._knowledge_distillation_loss(teacher_logits, student_logits)
            
            if expert_sync_outputs['distillation_pairs']:
                distillation_loss /= len(expert_sync_outputs['distillation_pairs'])
                total_loss += self.distillation_weight * distillation_loss
                loss_dict['distillation_loss'] = distillation_loss
        
        # 协同损失
        if 'expert_features' in expert_sync_outputs:
            coordination_loss = self._coordination_loss(expert_sync_outputs['expert_features'])
            total_loss += self.coordination_weight * coordination_loss
            loss_dict['coordination_loss'] = coordination_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数，整合检测和分割损失
    """
    def __init__(self,
                 detection_weight: float = 1.0,
                 segmentation_weight: float = 1.0,
                 expert_sync_weight: float = 0.1):
        """
        初始化多任务损失
        Args:
            detection_weight: 检测损失权重
            segmentation_weight: 分割损失权重
            expert_sync_weight: ExpertSync损失权重
        """
        super(MultiTaskLoss, self).__init__()
        
        # 初始化各任务损失
        self.detection_loss_fn = DetectionLoss()
        self.segmentation_loss_fn = SegmentationLoss()
        self.expert_sync_loss_fn = ExpertSyncLoss()
        
        # 损失权重
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.expert_sync_weight = expert_sync_weight
    
    def update_weights(self,
                      detection: Optional[float] = None,
                      segmentation: Optional[float] = None,
                      expert_sync: Optional[float] = None):
        """
        更新损失权重
        Args:
            detection: 新的检测损失权重
            segmentation: 新的分割损失权重
            expert_sync: 新的ExpertSync损失权重
        """
        if detection is not None:
            self.detection_weight = detection
        if segmentation is not None:
            self.segmentation_weight = segmentation
        if expert_sync is not None:
            self.expert_sync_weight = expert_sync
    
    def forward(self,
                model_outputs: Dict,
                target_boxes: torch.Tensor,
                target_labels: torch.Tensor,
                target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播计算多任务损失
        Args:
            model_outputs: 模型输出字典
            target_boxes: 目标边界框，形状 [B, N, 4]
            target_labels: 目标标签，形状 [B, N]
            target_mask: 目标分割掩码，形状 [B, H, W]
        Returns:
            综合损失字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 计算检测损失
        if 'detection' in model_outputs:
            detection_outputs = model_outputs['detection']
            detection_loss = self.detection_loss_fn(
                pred_cls=detection_outputs['cls_logits'],
                pred_boxes=detection_outputs['bbox_regression'],
                target_cls=target_labels,
                target_boxes=target_boxes
            )
            
            total_loss += self.detection_weight * detection_loss['total_loss']
            loss_dict['detection_loss'] = detection_loss['total_loss']
            loss_dict.update({f'detection_{k}': v for k, v in detection_loss.items()})
        
        # 计算分割损失
        if 'segmentation' in model_outputs:
            segmentation_outputs = model_outputs['segmentation']
            segmentation_loss = self.segmentation_loss_fn(
                pred=segmentation_outputs['logits'],
                target=target_mask
            )
            
            total_loss += self.segmentation_weight * segmentation_loss['total_loss']
            loss_dict['segmentation_loss'] = segmentation_loss['total_loss']
            loss_dict.update({f'segmentation_{k}': v for k, v in segmentation_loss.items()})
        
        # 计算ExpertSync损失
        if 'expert_sync' in model_outputs:
            expert_sync_loss = self.expert_sync_loss_fn(model_outputs['expert_sync'])
            
            total_loss += self.expert_sync_weight * expert_sync_loss['total_loss']
            loss_dict['expert_sync_loss'] = expert_sync_loss['total_loss']
            loss_dict.update({f'expert_sync_{k}': v for k, v in expert_sync_loss.items()})
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict