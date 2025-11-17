# UESDNet: 无人机专家引导先验与分割检测网络

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **注**：[English version](README_EN.md) 也可用

## 项目概述

这里是论文《UESDNet: UAV Expert-Guided Prior and Segmentation-Detection Network》的官方代码实现。我们提出了一个多任务感知框架，特别针对无人机视觉系统设计，能同时处理目标检测和语义分割。

主要创新点包括：
- 融合无人机姿态信息（俯仰、横滚、偏航角）来提高感知效果
- 设计了CTFMB跨任务特征互促结构，使不同任务间可以相互辅助
- 开发了ExpertSync模块解决遮挡带来的小目标检测和分割难题

## 主要特性

我们的系统具备以下功能：
- 多任务处理能力，可以同时进行目标检测和语义分割
- 独特的无人机专家引导先验机制，能根据无人机姿态动态调整处理策略
- 任务间特征共享和互促，通过专家引导同步提升整体性能
- 提供C++扩展以满足实时应用场景的性能需求
- 代码结构清晰模块化，方便后续开发和功能扩展

## 性能指标

在STUDataset数据集上的实验结果：

- 目标检测: 75.6% mAP@0.5:0.95
- 语义分割: 71.5% mIoU
- 在RTX 3090上可达44.0 FPS的推理速度
- 计算复杂度约为91.8 GFLOPs
- 模型参数量23.7百万

## 安装步骤

### 环境要求

需要以下环境配置：
- 操作系统: Linux, macOS 或 Windows
- Python 3.7或更高版本
- 建议使用CUDA 11.1+以获得GPU加速
- 推荐使用显存≥8GB的NVIDIA GPU，如RTX 3090

### 安装步骤

1. 克隆代码仓库

```bash
git clone https://github.com/yourusername/UESDNet.git
cd UESDNet
```

2. 创建并激活虚拟环境

```bash
# 使用conda
conda create -n uesdnet python=3.7
conda activate uesdnet

# 或使用Python内置venv
python -m venv uesdnet
source uesdnet/bin/activate  # Linux/macOS
uesdnet\Scripts\activate  # Windows
```

3. 安装所需依赖

```bash
pip install -r requirements.txt
```

4. 编译C扩展（可选，可显著提升性能）

```bash
cd uav_self
python setup.py build_ext --inplace
```

## 使用说明

### 数据准备

使用STUDataset数据集，完整版本可在以下链接获取：https://doi.org/10.57760/sciencedb.28912

### 训练模型

运行分阶段训练：

```bash
python src/training/phased_trainer.py --config configs/train_phase1.yaml
python src/training/phased_trainer.py --config configs/train_phase2.yaml
```

### 评估模型

使用训练好的模型进行评估：

```bash
python src/experiments/eval_metrics.py --config configs/inference.yaml --checkpoint path/to/model.pth
```

### 使用无人机自定位模块

无人机自定位模块使用示例：

```python
from uav_self import process_uav_image
import cv2

# 加载图像
img = cv2.imread('drone_image.jpg')

# 处理获取无人机信息
info = process_uav_image(img)
print(f"无人机位置信息: {info}")
```

## 项目结构

```
UESDNet/
├── src/            # 源代码目录
│   ├── data/       # 数据处理相关
│   ├── models/     # 模型定义
│   ├── training/   # 训练代码
│   ├── experiments/# 实验脚本
│   └── config.py   # 配置管理
├── configs/        # 配置文件
├── tests/          # 测试用例
├── uav_self/       # 无人机自定位模块
├── requirements.txt # 依赖列表
├── LICENSE         # 许可证
└── README.md       # 项目说明
```

## 文档说明

- 无人机自定位模块的详细说明请参见[uav_self/README.md](uav_self/README.md)

## 测试

运行测试确保功能正常：

```bash
python -m pytest tests/ -v
```

## 如何贡献

欢迎大家为项目做出贡献！

简单步骤：
1. Fork这个仓库
2. 创建你的特性分支
3. 提交你的代码变更
4. 推送到你自己的Fork仓库
5. 提交Pull Request给我们

## 许可证

采用MIT许可证，详情请查看[LICENSE](LICENSE)文件



## 联系方式

有问题或建议？请通过以下方式联系我们：
- GitHub Issues: 直接在仓库中提Issue