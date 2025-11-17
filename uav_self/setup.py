import os
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

# 获取OpenCV和AprilTag的路径（根据实际情况修改）
include_dirs = ['.', 'common']
library_dirs = []
libraries = ['opencv_core', 'opencv_highgui', 'opencv_imgproc', 'opencv_videoio', 'apriltag']

defines = []

# 收集所有源文件
sources = ['uav_self.cc']
# 添加common文件夹中的.c文件
sources.extend(glob.glob('common/*.c'))
# 添加encodeSrc文件夹中的.cpp文件
sources.extend(glob.glob('encodeSrc/*.cpp'))

# 根据操作系统设置不同的编译参数
extra_compile_args = []
extra_link_args = []
if sys.platform.startswith('win'):
    # Windows平台 (Visual Studio编译器) - 使用更基础的选项
    extra_compile_args = ['/EHsc']
    extra_link_args = []
else:
    # Linux/macOS平台
    extra_compile_args = ['-std=c++11', '-fPIC']
    extra_link_args = ['-shared']

# 创建扩展模块
extension = Extension(
    name='_uav_self',
    sources=sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=defines,
    language='c++',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
)

# 设置setup
setup(
    name='uav_self',
    version='0.1.0',
    description='UAV Self-Localization Module',
    ext_modules=[extension],
    packages=['uav_self'],
    package_dir={'uav_self': '.'},
    package_data={'uav_self': ['*.h', '*.cc', 'common/*.h', 'common/*.c', 'encodeSrc/*.h', 'encodeSrc/*.cpp']},
)