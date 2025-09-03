"""
Point Cloud Style Transfer项目安装脚本
基于Diffusion模型的LiDAR点云风格迁移
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """读取requirements.txt文件"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(requirements_path):
        return []
    
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith("#"):
                # 移除inline注释
                if "#" in line:
                    line = line[:line.index("#")].strip()
                if line:
                    requirements.append(line)
    
    return requirements

def read_file(filename):
    """读取文件内容"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# 项目元信息
__version__ = "1.0.0"
__author__ = "Point Cloud Research Team"
__email__ = "pointcloud@research.com"

setup(
    name="pointcloud-style-transfer",
    version=__version__,
    author=__author__,
    author_email=__email__,
    description="基于扩散模型的点云风格转换系统",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/pointcloud/style-transfer",
    
    # 包发现
    packages=find_packages(exclude=["tests*", "docs*", "experiments*"]),
    
    # Python版本要求
    python_requires=">=3.10",
    
    # 依赖管理
    install_requires=read_requirements(),
    
    # 额外依赖组
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "isort>=5.13.2",
            "mypy>=1.7.1",
            "pre-commit>=3.0.0",
        ],
        "viz": [
            "mayavi>=4.8.0",
            "vtk>=9.2.0",
            "napari>=0.4.17",
        ],
        "optim": [
            "optuna>=3.4.0",
            "ray[tune]>=2.8.0",
            "hyperopt>=0.2.7",
        ],
        "export": [
            "onnx>=1.15.0", 
            "tensorrt>=8.6.0",
            "openvino>=2023.2.0",
        ]
    },
    
    # 项目分类
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 12.5",
    ],
    
    # 关键词
    keywords=[
        "point-cloud", "style-transfer", "diffusion-model", 
        "lidar", "3d-vision", "deep-learning", "pytorch",
        "computer-vision", "generative-model", "pointnet"
    ],
    
    # 控制台脚本入口点
    entry_points={
        "console_scripts": [
            "pc-preprocess=scripts.preprocess_data:main",
            "pc-train=scripts.train:main", 
            "pc-test=scripts.test:main",
            "pc-inference=scripts.inference:main",
            "pc-visualize=scripts.visualize_results:main",
        ],
    },
    
    # 包含非Python文件
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "config": ["*.yaml", "*.yml"],
        "docker": ["*.yml", "requirements.txt", "Dockerfile"],
    },
    
    # 项目URLs
    project_urls={
        "Bug Reports": "https://github.com/pointcloud/style-transfer/issues",
        "Source": "https://github.com/pointcloud/style-transfer",
        "Documentation": "https://pointcloud-style-transfer.readthedocs.io/",
        "Funding": "https://github.com/sponsors/pointcloud",
    },
    
    # 测试套件
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0", 
        "pytest-mock>=3.12.0",
    ],
    
    # ZIP安全
    zip_safe=False,
    
    # 数据文件
    data_files=[
        ('config', ['config/config.py']),
        ('docker', ['docker/requirements.txt', 'docker/docker-compose.yml']),
    ],
    
    # 平台特定配置
    platforms=["linux", "windows", "macos"],
    
    # 许可证
    license="MIT",
    
    # 依赖链接（如果需要从特定源安装）
    dependency_links=[
        "https://download.pytorch.org/whl/cu124",
    ],
)