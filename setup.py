#!/usr/bin/env python3

from setuptools import setup, find_packages
import os


def read_requirements():
    """读取依赖文件"""
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def read_readme():
    """读取README文件"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name="pointcloud-style-transfer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Point Cloud Style Transfer using PointNet++ and GAN",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pointcloud-style-transfer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.910",
        ],
        "viz": [
            "open3d>=0.15.0",
            "plotly>=5.0.0",
            "mayavi>=4.7.0",
        ],
        "all": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.910",
            "open3d>=0.15.0",
            "plotly>=5.0.0",
            "mayavi>=4.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "pc-style-train=scripts.train:main",
            "pc-style-test=scripts.test:main",
            "pc-style-infer=scripts.inference:main",
            "pc-style-visualize=scripts.visualize:main",
            "pc-style-preprocess=data.preprocess:preprocess_dataset",
            "pc-style-convert=scripts.convert_data:main",
            "pc-style-check=scripts.check_data_quality:main",
            "pc-style-benchmark=scripts.benchmark:main",
        ],
    },
    package_data={
        "pointcloud_style_transfer": [
            "config/*.py",
            "data/*.py",
            "models/*.py",
            "training/*.py",
            "evaluation/*.py",
            "visualization/*.py",
            "scripts/*.py",
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "point cloud",
        "style transfer",
        "pointnet",
        "gan",
        "deep learning",
        "computer vision",
        "3d processing",
        "neural networks",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pointcloud-style-transfer/issues",
        "Source": "https://github.com/yourusername/pointcloud-style-transfer",
        "Documentation": "https://github.com/yourusername/pointcloud-style-transfer/blob/main/README.md",
    },
)