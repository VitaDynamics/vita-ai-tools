from setuptools import setup, find_packages

setup(
    name="vita-tools",
    version="0.1.0",
    description="Computer vision toolkit for 3D point cloud processing and visualization",
    author="DylanLi",
    author_email="dylan.h.li@outlook.com",
    packages=find_packages(include=["vita_toolkit", "vita_toolkit.*"]),
    python_requires=">=3.11",
    install_requires=[
        "lmdb>=1.6.2",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "open3d>=0.17.0",
        "rerun-sdk>=0.24.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
