# 作者: qzf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple
import os
import time
import logging
from tqdm import tqdm

from ..models.uesdnet import UESDNet
from ..models.expertsync import ExpertSync, apply_expertsync
from .losses import DetectionLoss, SegmentationLoss, MultiTaskLoss
from ..data.studataset import STUDataset
from ..config import Config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PhasedTrainer')

class PhasedTrainer:
    """
    分阶段训练器，实现UESDNet的多阶段训练策略
    """
    
    def __init__(self, config: Config):
        """
        初始化训练器
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化损失函数
        self.loss_fn = MultiTaskLoss(
            detection_weight=config.training.detection_weight,
            segmentation_weight=config.training.segmentation_weight,
            expert_sync_weight=config.training.expert_sync_weight
        )
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._init_scheduler()
        
        # 训练状态
        self.current_phase = 0
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # 创建输出目录
        self.output_dir = config.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _init_model(self) -> UESDNet:
        """
        初始化模型
        Returns:
            UESDNet模型实例
        """
        model = UESDNet(
            num_classes=self.config.model.num_classes,
            pretrained=self.config.model.pretrained,
            freeze_backbone=self.config.model.freeze_backbone,
            use_expertsync=self.config.model.use_expertsync
        )
        
        # 移动到设备
        model.to(self.device)
        
        # 如果需要启用ExpertSync
        if self.config.model.use_expertsync:
            model = apply_expertsync(model)
        
        return model
    
    def _init_optimizer(self) -> optim.Optimizer:
        """
        初始化优化器
        Returns:
            优化器实例
        """
        if self.config.training.optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer_type}")
        
        return optimizer
    
    def _init_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        初始化学习率调度器
        Returns:
            学习率调度器实例
        """
        if self.config.training.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.total_epochs,
                eta_min=self.config.training.min_learning_rate
            )
        elif self.config.training.scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.gamma,
                patience=self.config.training.patience
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _get_current_phase_config(self) -> Dict:
        """
        获取当前阶段的配置
        Returns:
            阶段配置字典
        """
        phases = self.config.training.phases
        if self.current_phase < len(phases):
            return phases[self.current_phase]
        return phases[-1]  # 返回最后一个阶段的配置
    
    def _update_phase(self) -> bool:
        """
        更新训练阶段
        Returns:
            是否成功更新阶段
        """
        phase_config = self._get_current_phase_config()
        if self.current_epoch >= phase_config['end_epoch']:
            if self.current_phase < len(self.config.training.phases) - 1:
                self.current_phase += 1
                logger.info(f"Transitioning to training phase {self.current_phase + 1}")
                
                # 应用新阶段的配置
                new_phase_config = self._get_current_phase_config()
                self._apply_phase_config(new_phase_config)
                return True
        return False
    
    def _apply_phase_config(self, phase_config: Dict):
        """
        应用阶段配置
        Args:
            phase_config: 阶段配置字典
        """
        # 调整模型冻结状态
        if 'freeze_backbone' in phase_config:
            self.model.freeze_features(
                freeze_backbone=phase_config['freeze_backbone'],
                freeze_neck=phase_config.get('freeze_neck', False)
            )
        
        # 调整损失权重
        if 'loss_weights' in phase_config:
            self.loss_fn.update_weights(
                detection=phase_config['loss_weights'].get('detection', 1.0),
                segmentation=phase_config['loss_weights'].get('segmentation', 1.0),
                expert_sync=phase_config['loss_weights'].get('expert_sync', 0.1)
            )
        
        # 调整学习率
        if 'learning_rate' in phase_config:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = phase_config['learning_rate']
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        Args:
            train_loader: 训练数据加载器
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in progress_bar:
            # 准备数据
            images = batch['image'].to(self.device)
            detection_boxes = batch['detection_boxes'].to(self.device)
            detection_labels = batch['detection_labels'].to(self.device)
            segmentation_mask = batch['segmentation_mask'].to(self.device)
            uav_attitude = batch.get('uav_attitude_normalized', None)
            
            if uav_attitude is not None:
                uav_attitude = uav_attitude.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, uav_attitude)
            
            # 计算损失
            loss_dict = self.loss_fn(
                outputs,
                detection_boxes,
                detection_labels,
                segmentation_mask
            )
            
            # 反向传播和优化
            total_batch_loss = loss_dict['total_loss']
            total_batch_loss.backward()
            
            # 梯度裁剪
            if self.config.training.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_norm)
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += total_batch_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_batch_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        Args:
            val_loader: 验证数据加载器
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # 准备数据
                images = batch['image'].to(self.device)
                detection_boxes = batch['detection_boxes'].to(self.device)
                detection_labels = batch['detection_labels'].to(self.device)
                segmentation_mask = batch['segmentation_mask'].to(self.device)
                uav_attitude = batch.get('uav_attitude_normalized', None)
                
                if uav_attitude is not None:
                    uav_attitude = uav_attitude.to(self.device)
                
                # 前向传播
                outputs = self.model(images, uav_attitude)
                
                # 计算损失
                loss_dict = self.loss_fn(
                    outputs,
                    detection_boxes,
                    detection_labels,
                    segmentation_mask
                )
                
                total_loss += loss_dict['total_loss'].item()
        
        # 返回验证结果
        metrics = {
            'val_loss': total_loss / len(val_loader)
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        开始训练过程
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        start_time = time.time()
        
        # 初始阶段配置
        self._apply_phase_config(self._get_current_phase_config())
        
        logger.info(f"Starting training for {self.config.training.total_epochs} epochs")
        
        while self.current_epoch < self.config.training.total_epochs:
            # 检查阶段更新
            self._update_phase()
            
            logger.info(f"Phase {self.current_phase + 1}, Epoch {self.current_epoch + 1}/{self.config.training.total_epochs}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 保存检查点
            self._save_checkpoint(train_loss, val_metrics)
            
            # 打印日志
            logger.info(
                f"Epoch {self.current_epoch + 1} completed - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            self.current_epoch += 1
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
    
    def _save_checkpoint(self, train_loss: float, val_metrics: Dict[str, float]):
        """
        保存检查点
        Args:
            train_loss: 训练损失
            val_metrics: 验证指标
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'best_metric': self.best_metric
        }
        
        # 保存最新检查点
        latest_checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_checkpoint_path)
        
        # 保存最佳检查点（基于验证损失）
        if val_metrics['val_loss'] < self.best_metric or self.current_epoch == 0:
            self.best_metric = val_metrics['val_loss']
            best_checkpoint_path = os.path.join(self.output_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"New best model saved with val loss: {self.best_metric:.4f}")
        
        # 保存定期检查点
        if (self.current_epoch + 1) % self.config.training.save_interval == 0:
            periodic_checkpoint_path = os.path.join(
                self.output_dir,
                f'checkpoint_epoch_{self.current_epoch + 1}.pth'
            )
            torch.save(checkpoint, periodic_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练状态
            self.current_epoch = checkpoint['epoch']
            self.current_phase = checkpoint['phase']
            self.best_metric = checkpoint['best_metric']
            
            logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch + 1}")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return False