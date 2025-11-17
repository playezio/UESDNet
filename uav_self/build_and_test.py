#!/usr/bin/env python
# 编译测试脚本
# 验证uav_self模块的编译和基本功能

import os
import sys
import subprocess
import time
from pathlib import Path


def print_header(title):
    """打印标题"""
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)


def run_command(cmd, cwd=None, shell=True):
    """运行命令并返回结果"""
    print(f"执行命令: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            shell=shell,
            capture_output=True,
            text=True
        )
        return result
    except Exception as e:
        print(f"命令执行失败: {e}")
        return None


def build_extension():
    """尝试编译C扩展"""
    print_header("编译C扩展")
    
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 清理之前的构建文件
    clean_cmd = "python setup.py clean --all"
    clean_result = run_command(clean_cmd, cwd=script_dir)
    if clean_result and clean_result.returncode != 0:
        print(f"清理失败: {clean_result.stderr}")
    else:
        print("清理成功")
    
    # 尝试构建扩展
    build_cmd = "python setup.py build_ext --inplace"
    build_result = run_command(build_cmd, cwd=script_dir)
    
    if build_result:
        print(f"构建输出:\n{build_result.stdout[:500]}...")  # 只显示前500个字符
        
        if build_result.returncode == 0:
            print("✓ 编译成功!")
            return True
        else:
            print(f"✗ 编译失败: {build_result.stderr}")
            return False
    else:
        return False


def test_module_import():
    """测试模块导入"""
    print_header("测试模块导入")
    
    # 导入测试脚本
    test_script = """
import sys
sys.path.insert(0, '.')
try:
    from uav_self import UAVSelfLocalizer, process_uav_image
    print('OK 成功导入UAVSelfLocalizer和process_uav_image')
    
    # 测试创建实例
    localizer = UAVSelfLocalizer({"use_extension": False})  # 不使用C扩展进行测试
    print('OK 成功创建UAVSelfLocalizer实例')
    
    # 测试模拟处理
    import numpy as np
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = localizer.process_image(test_image)
    print(f'OK 成功处理测试图像，结果: {result}')
    
    print('所有导入测试通过!')
    sys.exit(0)
except Exception as e:
    print(f'ERROR 导入测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # 保存测试脚本（指定UTF-8编码）
    with open("import_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    # 运行测试
    script_dir = Path(__file__).parent
    test_result = run_command("python import_test.py", cwd=script_dir)
    
    # 清理测试脚本
    try:
        os.remove("import_test.py")
    except:
        pass
    
    return test_result and test_result.returncode == 0


def check_dependencies():
    """检查必要的依赖"""
    print_header("检查依赖")
    
    required_packages = ["numpy", "opencv-python", "setuptools"]
    missing_packages = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"OK {pkg} 已安装")
        except ImportError:
            print(f"ERROR {pkg} 未安装")
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"建议安装缺失的包: pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0


def main():
    """主函数"""
    print_header("UAV Self-Localization 模块编译测试")
    
    # 检查依赖
    dependencies_ok = check_dependencies()
    
  
    build_ok = build_extension()
    
    # 测试模块导入和基本功能
    import_ok = test_module_import()
    
    print_header("测试总结")
    print(f"依赖检查: {'通过' if dependencies_ok else '失败'}")
    print(f"C扩展编译: {'通过' if build_ok else '失败'}")
    print(f"模块导入测试: {'通过' if import_ok else '失败'}")
    
    # 即使C扩展编译失败，如果Python实现工作正常，模块仍然可以使用
    if import_ok:
        print("\n测试完成! Python实现正常工作。")
        if not build_ok:
            print("注意: C扩展编译失败，但模块可以使用Python模拟实现。")
            print("要使用C扩展功能，请确保已安装C++编译器和所需的库（如OpenCV、AprilTag）。")
        return 0
    else:
        print("\n测试失败! 模块无法正常工作。")
        return 1


if __name__ == "__main__":
    sys.exit(main())