"""
VITA Toolkit - Computer vision toolkit for 3D point cloud processing and visualization.

This package provides utilities for:
- Converting depth and RGB images to point clouds
- Point cloud to bird's eye view transformations
- 3D visualization tools
"""

__version__ = "0.1.0"

# Import main functions from submodules
from .img_to_pc import (
    align_size,
    depth_rgb_to_pcd,
)

from .pc_to_bev import (
    points_to_voxel_batch,
    filter_pcd,
    project_pc_to_bev,
)

from .viz import (
    visualize_point_cloud,
    visualize_bev,
)

# Define what gets imported with "from vita_toolkit import *"
__all__ = [
    # Image to point cloud functions
    "align_size",
    "depth_rgb_to_pcd",
    # Point cloud to BEV functions
    "points_to_voxel_batch",
    "filter_pcd",
    "project_pc_to_bev",
    # Visualization functions
    "visualize_point_cloud",
    "visualize_bev",
]