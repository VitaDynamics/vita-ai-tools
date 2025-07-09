# VITA Tools

A computer vision toolkit for 3D point cloud processing and visualization. This package provides utilities for converting depth and RGB images to point clouds, transforming point clouds to bird's eye view representations, and visualizing 3D data.

## Features

- **Image to Point Cloud**: Convert depth and RGB images to 3D point clouds.
- **Depth-LiDAR Alignment**: Align point clouds generated from depth sensors with LiDAR data.
- **Point Cloud Processing**: Filter, transform, and process point cloud data.
- **Bird's Eye View**: Generate BEV representations from point clouds.
- **Visualization**: Tools for visualizing point clouds and BEV maps using `rerun`.

## Design

- For visualization, this repo is optimized for headless server. So we use rerun and matplotlib to viz the data. Always prioritize Rerun.

## Installation

### Using uv (recommended)

```bash
# Clone the repository first
git clone # This project
cd vita-ai-tools

# Basic installation (core dependencies only)
uv sync

# Install with GPU dependencies
uv sync --extra gpu

# Install with development dependencies
uv sync --extra dev

# Install with all dependencies (GPU + dev)
uv sync --all-extras

# Production installation (no dev dependencies)
uv sync --no-dev

# editable usage 
u sync --editable
```

### Alternative: Using pip

```bash
# General mode (lightweight, no GPU dependencies)
pip install vita-tools

# GPU mode (includes torch, torchvision, transformers)
pip install vita-tools[gpu]

# Development mode
pip install vita-tools[dev]

# Full development setup
pip install vita-tools[gpu,dev]
```

## Quick Start

```python
import numpy as np
from vita_toolkit import (
    depth_rgb_to_pcd,
    visualize_point_cloud,
    visualize_bev,
    points_to_voxel_batch,
)

# Convert depth image to point cloud
depth = np.random.rand(480, 640).astype(np.float32)
intrinsic = {
    'fx': 525.0, 'fy': 525.0,
    'cx': 320.0, 'cy': 240.0
}
extrinsic = {
    'rotation': np.eye(3),
    'translation': np.zeros(3)
}

pcd = depth_rgb_to_pcd(depth, intrinsic, extrinsic)

# Visualize point cloud
visualize_point_cloud(pcd)

# Convert to voxel representation
voxel_size = np.array([0.1, 0.1, 0.1])
coors_range = np.array([-50, -50, -3, 50, 50, 1])
voxels, coords, num_points = points_to_voxel_batch(pcd, voxel_size, coors_range)
```
- All template usage is in notebooks folder.

## Directory Structure

```
vita_toolkit/
    filesystem_reader.py  # Reads data from filesystem
    lmdb_reader.py        # Reads data from LMDB databases
    point_cloud/          # Point cloud processing functionalities
        __init__.py
        adaptive_scaling.py     # Adaptive scaling for depth-lidar alignment
        depth_lidar_matching.py # Align depth and LiDAR data
        img_to_pc.py            # Convert images/depth to point clouds
        pc_to_bev.py            # Project point clouds to Bird's Eye View
        solvers.py              # Solvers for alignment tasks
        viz.py                  # Visualization utilities (e.g., using rerun)
        README.md               # Detailed documentation for this module
    __init__.py           # Makes vita_toolkit a package
notebooks/
    point_cloud.ipynb     # Example usage of the point_cloud module
README.md                 # This file
.gitignore
pyproject.toml            # Project metadata and dependencies
uv.lock                   # Lockfile for uv
RULES.md                  # Contributor guidelines or rules (if present)
```

## `vita_toolkit.point_cloud` Module

The `vita_toolkit.point_cloud` directory contains a comprehensive suite of tools for 3D point cloud processing. Key capabilities include:

*   **Point Cloud Generation**: Creating 3D point clouds from depth maps and RGB images (`img_to_pc.py`).
*   **Depth-LiDAR Alignment**: Sophisticated methods to align point clouds derived from depth sensors with those from LiDAR, including various solvers and adaptive scaling techniques (`depth_lidar_matching.py`, `solvers.py`, `adaptive_scaling.py`).
*   **Bird's Eye View (BEV) Projection**: Transforming 3D point clouds into 2D BEV representations, commonly used in robotics and autonomous systems (`pc_to_bev.py`).
*   **Visualization**: Utilities to visualize point clouds and BEV maps using the `rerun` library (`viz.py`).

For detailed information on how to use this module, including code examples and descriptions of each sub-component, please refer to its dedicated README: [`vita_toolkit/point_cloud/README.md`](vita_toolkit/point_cloud/README.md).

## Development

```bash
# Setup development environment
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black vita_toolkit/
uv run isort vita_toolkit/

# Type checking
uv run mypy vita_toolkit/

# Run linting
uv run ruff check vita_toolkit/
uv run ruff format vita_toolkit/
```

## Requirements

### Core Dependencies

- Python >= 3.11
- NumPy
- OpenCV
- Pillow
- Matplotlib
- SciPy
- LMDB

### GPU Dependencies (optional)

- PyTorch
- Torchvision
- Open3D
- Transformers
- Accelerate

## License

MIT License
