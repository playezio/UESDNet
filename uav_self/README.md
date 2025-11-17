# uav_self 模块文档

## 模块概述

`uav_self` 是一个无人机自信息处理模块，用于处理无人机图像并获取无人机的位置和姿态信息。该模块支持两种实现方式：
- **C扩展实现**：高性能的C/C++实现，处理速度快
- **Python模拟实现**：纯Python实现，用于开发和测试

## 目录结构

```
uav_self/
├── __init__.py        # Python接口定义
├── setup.py           # 编译配置文件
├── uav_self.cc        # 主C++源代码
├── common/            # 通用依赖文件
│   ├── g2d/           # 几何相关库
│   ├── homography/    # 单应性变换库
│   └── ...            # 其他C语言依赖
├── encodeSrc/         # 编码相关文件
│   ├── h264decoder.cpp
│   ├── h264encoder.cpp
│   └── ...
├── build_and_test.py  # 编译测试脚本
├── verify_python_interface.py  # Python接口验证脚本
└── README.md          # 本文档
```

## 依赖项

### Python依赖
- Python 3.6+
- numpy
- opencv-python
- pyyaml

### C++编译依赖
- C++编译器（Windows下需要Visual Studio 2015或更高版本）
- setuptools（Python包，用于编译C扩展）

## 安装和配置

### 1. 安装Python依赖

```bash
pip install numpy opencv-python pyyaml
```

### 2. 编译C扩展（可选）

**注意**：如果C扩展编译失败，模块会自动回退到Python模拟实现。

#### Windows环境下编译

```bash
# 进入uav_self目录
cd uav_self

# 编译并安装C扩展
python setup.py build_ext --inplace
```

**编译注意事项**：
- Windows用户需要安装Visual Studio 2015或更高版本
- 确保在编译前安装了所有必要的Python依赖
- 如果出现编译错误，可能需要调整setup.py中的编译参数

#### Linux/macOS环境下编译

```bash
# 进入uav_self目录
cd uav_self

# 编译并安装C扩展
python setup.py build_ext --inplace
```

## 使用方法

### 基本使用

```python
import cv2
from uav_self import UAVSelfLocalizer

# 创建定位器实例（默认使用Python模拟实现）
localizer = UAVSelfLocalizer()

# 加载图像
image = cv2.imread('drone_image.jpg')

# 处理图像
result = localizer.process_image(image)
print(f"无人机位置信息: {result}")

# 获取详细的自信息
ego_info = localizer.get_ego_information()
print(f"详细位置: {ego_info['position']}")
print(f"详细姿态: {ego_info['attitude']}")
```

### 使用C扩展实现

```python
from uav_self import UAVSelfLocalizer

# 创建配置，启用C扩展
config = {"use_extension": True}
localizer = UAVSelfLocalizer(config)

# 使用方式与上面相同
```

### 使用便捷函数

```python
import cv2
from uav_self import process_uav_image

# 加载图像
image = cv2.imread('drone_image.jpg')

# 使用便捷函数处理图像
result = process_uav_image(image)
print(f"无人机位置信息: {result}")
```

## 验证脚本

模块包含两个验证脚本：

1. **验证Python接口**：仅测试Python实现功能，不依赖C扩展
   ```bash
   python verify_python_interface.py
   ```

2. **编译测试脚本**：尝试编译C扩展并进行全面测试
   ```bash
   python build_and_test.py
   ```

## 故障排除

### C扩展编译失败

如果C扩展编译失败，模块会自动回退到Python模拟实现，功能不会受到影响，只是性能可能较低。

常见编译错误及解决方案：

1. **缺少Visual Studio编译工具**
   - 安装Visual Studio 2015或更高版本
   - 或安装Visual C++ Build Tools

2. **编译参数不兼容**
   - 修改setup.py中的编译参数，针对您的编译环境进行调整

3. **依赖文件缺失**
   - 确保common和encodeSrc目录下的所有必要文件都存在

### 模块导入失败

确保您的Python路径正确设置，可以通过以下方式解决：

```python
import sys
sys.path.insert(0, 'path/to/uav_self/parent/directory')
```

## 性能对比

- **C扩展实现**：处理速度快，适合实时应用
- **Python模拟实现**：处理速度较慢，但开发和调试更方便

## 版本历史

- v1.0: 初始版本，支持C扩展和Python模拟实现

## 许可证

请参考项目根目录的许可证文件。