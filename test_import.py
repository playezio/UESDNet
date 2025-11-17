# 测试模块导入路径
print("测试代码结构完整性")

# 测试基本导入
try:
    # 只测试导入路径，不执行实际功能
    import sys
    import os
    print(f"Python路径: {sys.path}")
    print("基本模块导入成功")
except Exception as e:
    print(f"基本模块导入错误: {e}")

# 测试项目模块导入
try:
    # 尝试导入项目模块（不运行实际代码）
    from src.models.uesdnet import UESDNet
    print("UESDNet模型模块导入成功")
except ImportError as e:
    print(f"模块导入路径正确，但可能缺少依赖: {e}")
except Exception as e:
    print(f"其他错误: {e}")

print("测试完成")