"""
项目安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("docker/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pointcloud-style-transfer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Point Cloud Style Transfer using Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pointcloud-style-transfer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pc-preprocess=scripts.preprocess_data:main",
            "pc-train=scripts.train:main",
            "pc-test=scripts.test:main",
            "pc-inference=scripts.inference:main",
            "pc-visualize=scripts.visualize_results:main",
        ],
    },
)

