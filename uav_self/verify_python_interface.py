#!/usr/bin/env python
# 简单验证脚本 - 仅测试Python接口功能
# 不依赖C扩展编译

import os
import sys

# 添加父目录到Python路径，这样可以导入uav_self模块
sys.path.insert(0, '..')

print("开始验证uav_self模块的Python接口...")
print("=" * 60)

# 尝试导入模块
print("测试模块导入...")
try:
    # 导入uav_self模块
    from uav_self import UAVSelfLocalizer, process_uav_image
    print("OK 成功导入UAVSelfLocalizer和process_uav_image")
except ImportError as e:
    print(f"ERROR 导入失败: {e}")
    print("尝试直接导入当前目录的__init__.py...")
    try:
        # 直接导入当前目录的__init__.py
        import __init__ as uav_self
        # 从__init__模块获取类和函数
        UAVSelfLocalizer = uav_self.UAVSelfLocalizer
        process_uav_image = uav_self.process_uav_image
        print("OK 成功直接导入模块内容")
    except Exception as e2:
        print(f"ERROR 直接导入也失败: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 测试创建实例
print("\n测试创建UAVSelfLocalizer实例...")
try:
    # 强制使用Python实现
    config = {"use_extension": False}
    localizer = UAVSelfLocalizer(config)
    print("OK 成功创建UAVSelfLocalizer实例")
except Exception as e:
    print(f"ERROR 创建实例失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试图像处理
print("\n测试图像处理功能...")
try:
    import numpy as np
    # 创建一个简单的测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 处理图像
    result = localizer.process_image(test_image)
    print(f"OK 成功处理测试图像")
    print(f"  结果: {result}")
    
    # 验证结果格式
    if isinstance(result, dict) and all(k in result for k in ['x', 'y', 'z', 'yaw']):
        print("OK 结果格式正确")
    else:
        print("ERROR 结果格式不正确")
except Exception as e:
    print(f"ERROR 图像处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试获取自信息
print("\n测试获取无人机自信息...")
try:
    ego_info = localizer.get_ego_information()
    print(f"OK 成功获取无人机自信息")
    print(f"  位置信息: {ego_info['position']}")
    print(f"  姿态信息: {ego_info['attitude']}")
except Exception as e:
    print(f"ERROR 获取自信息失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试便捷函数
print("\n测试便捷函数...")
try:
    result2 = process_uav_image(test_image, config)
    print(f"OK 成功使用便捷函数")
    print(f"  结果: {result2}")
except Exception as e:
    print(f"ERROR 便捷函数调用失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("Python接口验证成功!")
print("模块可以正常使用Python实现，即使C扩展编译失败。")
print("注意: 要使用C扩展功能，需要正确配置编译环境和依赖库。")
print("=" * 60)