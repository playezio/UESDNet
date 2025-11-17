# 作者: qzf
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

class DetectionMetrics:
    """
    目标检测评估指标计算
    支持mAP、AP50、AP75等指标
    """
    def __init__(self,
                 num_classes: int,
                 iou_thresholds: Optional[List[float]] = None,
                 class_names: Optional[List[str]] = None):
        """
        初始化检测指标计算器
        Args:
            num_classes: 类别数量
            iou_thresholds: IoU阈值列表，默认[0.5:0.05:0.95]
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        
        # 存储预测结果和真实标注
        self.reset()
    
    def reset(self):
        """
        重置指标状态
        """
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)
        self.gt_available = defaultdict(list)  # 标记是否有对应的真实标注
    
    def add_batch(self,
                  pred_boxes: torch.Tensor,
                  pred_scores: torch.Tensor,
                  pred_labels: torch.Tensor,
                  gt_boxes: torch.Tensor,
                  gt_labels: torch.Tensor):
        """
        添加一个批次的预测结果和真实标注
        Args:
            pred_boxes: 预测边界框，形状 [N, 4]，格式 [x1, y1, x2, y2]
            pred_scores: 预测分数，形状 [N]
            pred_labels: 预测类别，形状 [N]
            gt_boxes: 真实边界框，形状 [M, 4]
            gt_labels: 真实类别，形状 [M]
        """
        # 转换为numpy数组
        if torch.is_tensor(pred_boxes):
            pred_boxes = pred_boxes.cpu().numpy()
        if torch.is_tensor(pred_scores):
            pred_scores = pred_scores.cpu().numpy()
        if torch.is_tensor(pred_labels):
            pred_labels = pred_labels.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()
        if torch.is_tensor(gt_labels):
            gt_labels = gt_labels.cpu().numpy()
        
        # 按类别分组存储预测结果
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            self.predictions[int(label)].append((box, score))
        
        # 按类别分组存储真实标注
        for box, label in zip(gt_boxes, gt_labels):
            self.ground_truths[int(label)].append(box)
            self.gt_available[int(label)].append(False)  # 初始标记为未匹配
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个边界框的IoU
        Args:
            box1: 第一个边界框，格式 [x1, y1, x2, y2]
            box2: 第二个边界框，格式 [x1, y1, x2, y2]
        Returns:
            IoU值
        """
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def compute_ap(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        计算平均精度AP
        Args:
            precision: 精确率数组
            recall: 召回率数组
        Returns:
            AP值
        """
        # 按召回率排序
        sorted_indices = np.argsort(recall)
        precision = precision[sorted_indices]
        recall = recall[sorted_indices]
        
        # 计算插值精度（从右到左取最大值）
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # 计算AP
        ap = 0.0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i - 1]) * precision[i]
        
        return ap
    
    def compute_ap_per_class(self, class_id: int, iou_threshold: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        计算特定类别和IoU阈值下的AP
        Args:
            class_id: 类别ID
            iou_threshold: IoU阈值
        Returns:
            AP值、精确率数组、召回率数组
        """
        # 获取该类别的预测结果和真实标注
        class_preds = self.predictions[class_id]
        class_gts = self.ground_truths[class_id]
        num_gts = len(class_gts)
        
        if num_gts == 0:
            return 0.0, np.array([]), np.array([])
        
        if len(class_preds) == 0:
            return 0.0, np.array([0.0]), np.array([0.0])
        
        # 按分数降序排序预测结果
        class_preds.sort(key=lambda x: x[1], reverse=True)
        
        # 初始化TP和FP数组
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        # 标记已匹配的真实标注
        matched_gts = np.zeros(num_gts, dtype=bool)
        
        # 遍历所有预测结果
        for i, (pred_box, pred_score) in enumerate(class_preds):
            best_iou = 0.0
            best_gt_idx = -1
            
            # 寻找最佳匹配的真实标注
            for j, gt_box in enumerate(class_gts):
                if not matched_gts[j]:
                    iou = self.compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # 判断是否为TP
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_gts[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # 计算累积的TP和FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = cum_tp / (cum_tp + cum_fp + 1e-8)
        recall = cum_tp / num_gts
        
        # 计算AP
        ap = self.compute_ap(precision, recall)
        
        return ap, precision, recall
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标
        Returns:
            指标字典
        """
        metrics = {}
        aps = defaultdict(list)
        
        # 计算每个类别在每个IoU阈值下的AP
        for class_id in range(self.num_classes):
            for iou_threshold in self.iou_thresholds:
                ap, _, _ = self.compute_ap_per_class(class_id, iou_threshold)
                aps[class_id].append(ap)
        
        # 计算mAP
        mAP_all = []
        for class_id in range(self.num_classes):
            if len(aps[class_id]) > 0:
                class_mAP = np.mean(aps[class_id])
                mAP_all.append(class_mAP)
                metrics[f'AP_{self.class_names[class_id]}'] = class_mAP
        
        # 计算整体mAP
        if mAP_all:
            metrics['mAP'] = np.mean(mAP_all)
        else:
            metrics['mAP'] = 0.0
        
        # 计算特定IoU阈值的mAP
        iou_to_name = {
            0.5: 'AP50',
            0.75: 'AP75'
        }
        
        for iou_threshold, name in iou_to_name.items():
            if iou_threshold in self.iou_thresholds:
                ap_at_iou = []
                for class_id in range(self.num_classes):
                    idx = list(self.iou_thresholds).index(iou_threshold)
                    if idx < len(aps[class_id]):
                        ap_at_iou.append(aps[class_id][idx])
                
                if ap_at_iou:
                    metrics[name] = np.mean(ap_at_iou)
                else:
                    metrics[name] = 0.0
        
        # 计算小、中、大目标的mAP（可选）
        # 这里可以根据需要实现
        
        return metrics

class SegmentationMetrics:
    """
    语义分割评估指标计算
    支持mIoU、精确率、召回率等指标
    """
    def __init__(self,
                 num_classes: int,
                 class_names: Optional[List[str]] = None,
                 ignore_index: Optional[int] = None):
        """
        初始化分割指标计算器
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
            ignore_index: 忽略的类别索引
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # 混淆矩阵
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def reset(self):
        """
        重置混淆矩阵
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def add_batch(self, pred: torch.Tensor, target: torch.Tensor):
        """
        添加一个批次的预测结果和真实标注
        Args:
            pred: 预测标签，形状 [B, H, W]
            target: 真实标签，形状 [B, H, W]
        """
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # 展平为一维数组
        pred = pred.flatten()
        target = target.flatten()
        
        # 忽略指定索引
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        
        # 更新混淆矩阵
        for p, t in zip(pred, target):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute_iou(self, class_id: int) -> float:
        """
        计算特定类别的IoU
        Args:
            class_id: 类别ID
        Returns:
            IoU值
        """
        # 真正例
        tp = self.confusion_matrix[class_id, class_id]
        # 假正例
        fp = np.sum(self.confusion_matrix[:, class_id]) - tp
        # 假负例
        fn = np.sum(self.confusion_matrix[class_id, :]) - tp
        
        # 计算IoU
        iou = tp / (tp + fp + fn + 1e-8)
        
        return iou
    
    def compute_precision(self, class_id: int) -> float:
        """
        计算特定类别的精确率
        Args:
            class_id: 类别ID
        Returns:
            精确率值
        """
        # 真正例
        tp = self.confusion_matrix[class_id, class_id]
        # 假正例
        fp = np.sum(self.confusion_matrix[:, class_id]) - tp
        
        # 计算精确率
        precision = tp / (tp + fp + 1e-8)
        
        return precision
    
    def compute_recall(self, class_id: int) -> float:
        """
        计算特定类别的召回率
        Args:
            class_id: 类别ID
        Returns:
            召回率值
        """
        # 真正例
        tp = self.confusion_matrix[class_id, class_id]
        # 假负例
        fn = np.sum(self.confusion_matrix[class_id, :]) - tp
        
        # 计算召回率
        recall = tp / (tp + fn + 1e-8)
        
        return recall
    
    def compute_f1_score(self, class_id: int) -> float:
        """
        计算特定类别的F1分数
        Args:
            class_id: 类别ID
        Returns:
            F1分数
        """
        precision = self.compute_precision(class_id)
        recall = self.compute_recall(class_id)
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1
    
    def compute_accuracy(self) -> float:
        """
        计算总体准确率
        Returns:
            准确率值
        """
        # 正确预测的总数
        correct = np.trace(self.confusion_matrix)
        # 总预测数
        total = np.sum(self.confusion_matrix)
        
        # 计算准确率
        accuracy = correct / (total + 1e-8)
        
        return accuracy
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标
        Returns:
            指标字典
        """
        metrics = {}
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # 计算每个类别的指标
        for class_id in range(self.num_classes):
            if self.ignore_index is not None and class_id == self.ignore_index:
                continue
            
            iou = self.compute_iou(class_id)
            precision = self.compute_precision(class_id)
            recall = self.compute_recall(class_id)
            f1 = self.compute_f1_score(class_id)
            
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            # 记录每个类别的指标
            metrics[f'IoU_{self.class_names[class_id]}'] = iou
            metrics[f'precision_{self.class_names[class_id]}'] = precision
            metrics[f'recall_{self.class_names[class_id]}'] = recall
            metrics[f'f1_{self.class_names[class_id]}'] = f1
        
        # 计算平均指标
        metrics['mIoU'] = np.mean(ious) if ious else 0.0
        metrics['mPrecision'] = np.mean(precisions) if precisions else 0.0
        metrics['mRecall'] = np.mean(recalls) if recalls else 0.0
        metrics['mF1'] = np.mean(f1_scores) if f1_scores else 0.0
        
        # 计算总体准确率
        metrics['accuracy'] = self.compute_accuracy()
        
        return metrics

class MultiTaskEvaluator:
    """
    多任务评估器，整合检测和分割评估
    """
    def __init__(self,
                 num_classes: int,
                 class_names: Optional[List[str]] = None,
                 ignore_index: Optional[int] = None):
        """
        初始化多任务评估器
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
            ignore_index: 分割任务中忽略的类别索引
        """
        # 初始化检测和分割指标计算器
        self.detection_metrics = DetectionMetrics(
            num_classes=num_classes,
            class_names=class_names
        )
        
        self.segmentation_metrics = SegmentationMetrics(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index
        )
    
    def reset(self):
        """
        重置所有指标
        """
        self.detection_metrics.reset()
        self.segmentation_metrics.reset()
    
    def add_batch(self,
                  detection_preds: Dict,
                  segmentation_preds: torch.Tensor,
                  detection_targets: Dict,
                  segmentation_targets: torch.Tensor):
        """
        添加一个批次的预测结果和真实标注
        Args:
            detection_preds: 检测预测结果字典，包含'boxes'、'scores'、'labels'
            segmentation_preds: 分割预测标签，形状 [B, H, W]
            detection_targets: 检测真实标注字典，包含'boxes'、'labels'
            segmentation_targets: 分割真实标签，形状 [B, H, W]
        """
        # 添加检测结果
        self.detection_metrics.add_batch(
            pred_boxes=detection_preds['boxes'],
            pred_scores=detection_preds['scores'],
            pred_labels=detection_preds['labels'],
            gt_boxes=detection_targets['boxes'],
            gt_labels=detection_targets['labels']
        )
        
        # 添加分割结果
        self.segmentation_metrics.add_batch(
            pred=segmentation_preds,
            target=segmentation_targets
        )
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有任务的评估指标
        Returns:
            综合指标字典
        """
        # 计算检测指标
        detection_metrics = self.detection_metrics.compute_metrics()
        
        # 计算分割指标
        segmentation_metrics = self.segmentation_metrics.compute_metrics()
        
        # 合并指标
        all_metrics = {}
        all_metrics.update({f'detection_{k}': v for k, v in detection_metrics.items()})
        all_metrics.update({f'segmentation_{k}': v for k, v in segmentation_metrics.items()})
        
        # 计算整体性能分数：检测与分割加权综合评分
        w_det = 0.6  # 检测任务权重
        w_seg = 0.4  # 分割任务权重
        overall_score = (
            w_det * detection_metrics.get('mAP', 0.0) +
            w_seg * segmentation_metrics.get('mIoU', 0.0)
        )
        all_metrics['overall_score'] = overall_score
        
        return all_metrics
    
    def visualize_results(self,
                         images: torch.Tensor,
                         detection_preds: Dict,
                         segmentation_preds: torch.Tensor,
                         num_samples: int = 4):
        """
        可视化评估结果
        Args:
            images: 输入图像，形状 [B, C, H, W]
            detection_preds: 检测预测结果
            segmentation_preds: 分割预测结果
            num_samples: 可视化的样本数量
        """
        # 创建可视化图
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)

        # 转换图像到numpy并反归一化
        if torch.is_tensor(images):
            images = images.cpu().numpy()
        # 假设输入为CHW格式，转为HWC
        if images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        # 简单反归一化到0-1
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)

        for i in range(num_samples):
            ax_img = axes[0, i]
            ax_seg = axes[1, i]

            # 绘制原图+检测框
            ax_img.imshow(images[i])
            ax_img.set_title(f"Sample {i+1} - Detection")
            ax_img.axis('off')

            # 绘制检测框
            boxes = detection_preds['boxes'][i].cpu().numpy()
            labels = detection_preds['labels'][i].cpu().numpy()
            scores = detection_preds['scores'][i].cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                if score < 0.3:  # 只显示置信度>0.3的框
                    continue
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
                ax_img.add_patch(rect)
                ax_img.text(x1, y1 - 5, f"{self.detection_metrics.class_names[int(label)]}:{score:.2f}",
                            color='yellow', fontsize=8, weight='bold')

            # 绘制分割掩码
            seg_pred = segmentation_preds[i].cpu().numpy()
            ax_seg.imshow(images[i], alpha=0.7)
            ax_seg.imshow(seg_pred, cmap='tab20', alpha=0.3)
            ax_seg.set_title(f"Sample {i+1} - Segmentation")
            ax_seg.axis('off')

        plt.tight_layout()
        plt.show()
        # 引入可视化所需库
        import matplotlib.patches as patches
        import matplotlib.cm as cm
        pass

# 辅助函数
def evaluate_model(model,
                  dataloader,
                  device,
                  num_classes,
                  class_names=None,
                  ignore_index=None):
    """
    评估模型性能的便捷函数
    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 设备
        num_classes: 类别数量
        class_names: 类别名称列表
        ignore_index: 忽略的类别索引
    Returns:
        评估指标字典
    """
    # 初始化评估器
    evaluator = MultiTaskEvaluator(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index
    )
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # 准备数据
            images = batch['image'].to(device)
            uav_attitude = batch.get('uav_attitude_normalized', None)
            
            if uav_attitude is not None:
                uav_attitude = uav_attitude.to(device)
            
            # 前向传播
            outputs = model(images, uav_attitude)
            
            # 处理检测结果
            detection_preds = {
                'boxes': outputs['detection']['bboxes'],
                'scores': outputs['detection']['scores'],
                'labels': outputs['detection']['labels']
            }
            
            # 处理分割结果
            segmentation_preds = outputs['segmentation']['labels']
            
            # 准备真实标注
            detection_targets = {
                'boxes': batch['detection_boxes'],
                'labels': batch['detection_labels']
            }
            
            segmentation_targets = batch['segmentation_mask']
            
            # 添加到评估器
            evaluator.add_batch(
                detection_preds=detection_preds,
                segmentation_preds=segmentation_preds,
                detection_targets=detection_targets,
                segmentation_targets=segmentation_targets
            )
    
    # 计算指标
    metrics = evaluator.compute_metrics()
    
    return metrics