# Point Cloud Processing Module (`vita_toolkit.point_cloud`)

This module provides a suite of tools for processing 3D point cloud data, focusing on generation, alignment between different sensor modalities (e.g., depth cameras and LiDAR), and transformation into common representations like Bird's Eye View (BEV).

## Core Functionalities

The module is organized into several Python files, each responsible for specific aspects of point cloud manipulation:

*   **`img_to_pc.py`**:
    *   **Purpose**: Converts 2D image data (depth maps, RGB images) into 3D point clouds.
    *   **Key Functions**:
        *   `depth_rgb_to_pcd()`: Generates a 3D point cloud from a depth map and an optional RGB image, using camera intrinsic and extrinsic parameters. Leverages Open3D for the conversion.
        *   `pcd_to_camera_coordinate()`: Transforms point clouds from a source coordinate system (e.g., LiDAR) to the camera's coordinate system using extrinsic transformation matrices.
        *   `align_size()`: Utility to resize depth and RGB images, ensuring their dimensions are compatible.

*   **`depth_lidar_matching.py`**:
    *   **Purpose**: Aligns depth data (often from cameras) with point clouds from LiDAR sensors. This is crucial for fusing data from different sensors.
    *   **Key Functions**:
        *   `align_depth_lidar()`: The primary function for performing alignment. It estimates an optimal scale factor to apply to the depth data to match the LiDAR point cloud. This function utilizes the `DepthLidarSolver` for robust estimation.
        *   `project_points_to_image_plane()`: A utility to project 3D points onto a 2D image plane given camera intrinsic parameters.

*   **`pc_to_bev.py`**:
    *   **Purpose**: Transforms 3D point clouds into a 2D Bird's Eye View (BEV) representation, which is commonly used in autonomous driving and robotics applications.
    *   **Key Functions**:
        *   `project_pc_to_bev()`: Converts a point cloud into a BEV grid by first voxelizing the point cloud and then scattering these voxels onto a 2D plane.
        *   `points_to_voxel_batch()`: Efficiently converts raw point cloud data into a voxel representation.
        *   `filter_pcd()`: Pre-processes point clouds by removing points belonging to the ego-vehicle and those outside a specified range.
        *   `RadScatter`: A NumPy-based class for efficiently scattering voxel features into a BEV grid.

*   **`solvers.py`**:
    *   **Purpose**: Provides advanced and robust algorithms for solving the depth-Lidar alignment problem.
    *   **Key Components**:
        *   `DepthLidarSolver`: A comprehensive solver offering multiple optimization strategies:
            *   Standard Least Squares
            *   Robust Least Squares (with various loss functions like Huber, Soft L1)
            *   Median Ratio (robust to outliers)
            *   RANSAC (Random Sample Consensus)
            *   Adaptive Neighborhood Scaling (see `adaptive_scaling.py`)
        *   `AlignmentResult`: A data structure (dataclass) that encapsulates the results of an alignment process, including the scale factor, error metrics (RMSE, mean/median error), number of correspondences, outlier ratio, etc.
        *   Includes methods for evaluating alignment quality and applying adaptive scaling results to a full depth map.

*   **`adaptive_scaling.py`**:
    *   **Purpose**: Implements adaptive scaling techniques for depth-Lidar alignment. This method adjusts scaling locally based on neighborhood information while maintaining global consistency.
    *   **Key Components**:
        *   `AdaptiveNeighborhoodScaler`: The core class that computes adaptive scale factors for each correspondence point between depth and LiDAR data.
        *   `AdaptiveScalingConfig`: Configuration dataclass for the `AdaptiveNeighborhoodScaler`.

*   **`viz.py`**:
    *   **Purpose**: Offers utilities for visualizing point clouds and BEV maps.
    *   **Key Functions**:
        *   `visualize_point_cloud()`: Displays 3D point clouds, optionally with color information.
        *   `visualize_bev()`: Displays 2D BEV maps.
    *   **Backend**: Currently uses the [`rerun`](https://www.rerun.io/) library for visualization.

## High-Level Usage Example

The following example outlines a typical workflow using this module, inspired by `notebooks/point_cloud.ipynb`.

```python
import numpy as np
from vita_toolkit.point_cloud import (
    depth_rgb_to_pcd,
    align_depth_lidar,
    project_pc_to_bev,
    DepthLidarSolver # For more control
)
from vita_toolkit.point_cloud.viz import visualize_point_cloud, visualize_bev
# Assume 'filesystem_reader' is used to load data as in the notebook
# from vita_toolkit.filesystem_reader import FilesystemReader

# --- 1. Load Data (Conceptual) ---
# reader = FilesystemReader("path/to/your/data")
# frame_data = next(reader.iterate_frames()) # Get a sample frame

# Example data (replace with actual loaded data)
rgb_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
# Depth map typically comes from a depth sensor or a model (e.g., DepthPro)
# For this example, let's assume depth is in meters.
predicted_depth_map_meters = np.random.rand(480, 640).astype(np.float32) * 10 + 1
# LiDAR points in LiDAR coordinate system (N, 3)
lidar_points_lidar_coords = np.random.rand(10000, 3) * 50 - 25

# Camera parameters (example values)
intrinsic_params = {"fx": 720.0, "fy": 720.0, "cx": 320.0, "cy": 240.0}
# Extrinsic matrix: LiDAR to Camera transformation (4x4)
extrinsic_matrix_lidar_to_camera = np.eye(4) # Replace with actual matrix
extrinsic_params = {"data": extrinsic_matrix_lidar_to_camera}

# --- 2. Generate Point Cloud from Depth ---
# Convert depth from meters to millimeters if Open3D expects it (check specific function docs)
# depth_rgb_to_pcd in this toolkit handles depth_trunc in meters if rgb is provided,
# or in mm (via depth_trunc*1000) if rgb is None.
# Let's assume depth_rgb_to_pcd takes depth in meters when RGB is provided.
points_from_depth, colors_from_rgb = depth_rgb_to_pcd(
    depth=predicted_depth_map_meters,
    intrinsic=intrinsic_params,
    extrinsic={"data": np.eye(4)}, # Assuming depth is already in camera coordinates
    rgb=rgb_image,
    depth_trunc=20.0 # Max depth to consider, in meters
)
print(f"Generated {points_from_depth.shape[0]} points from depth map.")

# --- 3. Prepare LiDAR Data (Transform to Camera Coordinates) ---
# If your LiDAR points are not yet in camera coordinates, transform them:
# from vita_toolkit.point_cloud.img_to_pc import pcd_to_camera_coordinate
# lidar_points_camera_coords = pcd_to_camera_coordinate(lidar_points_lidar_coords, extrinsic_params)
# For this example, let's assume lidar_points_lidar_coords are already what we need for alignment
# but the Z-axis (depth) is the one we want to align.
# The align_depth_lidar function expects lidar_points in camera coordinates.
# Let's use a placeholder for lidar points already in camera frame for simplicity here.
lidar_points_camera_frame = np.random.rand(5000, 3) * np.array([20, 20, 30]) - np.array([10,10,0])


# --- 4. Align Depth-Generated Point Cloud with LiDAR Point Cloud ---
# Use the DepthLidarSolver for more control and access to detailed results
solver = DepthLidarSolver(outlier_threshold=0.2, max_iterations=100)
alignment_result = solver.solve(
    depth_map=predicted_depth_map_meters, # Original depth map in meters
    lidar_points=lidar_points_camera_frame, # LiDAR points in camera coordinates
    intrinsic_params=intrinsic_params,
    method="robust_least_squares" # Or "adaptive_neighborhood", "ransac", etc.
)

scale_factor = alignment_result.scale_factor
print(f"Estimated scale factor: {scale_factor:.4f}, RMSE: {alignment_result.rmse:.4f}")

# Apply scale factor to the Z-coordinate of the depth-generated points if needed,
# or directly use the scaled depth map if the solver provides it.
# The solver primarily gives a scale for the *depth_map* values.
scaled_depth_map = predicted_depth_map_meters * scale_factor
# Re-generate point cloud with scaled depth if you want scaled points_from_depth
# points_from_scaled_depth, _ = depth_rgb_to_pcd(...)

# --- 5. Project to Bird's Eye View (BEV) ---
# Using the point cloud generated from (scaled) depth
pc_range = np.array([-10.0, -10.0, -2.0, 10.0, 10.0, 3.0]) # x_min, y_min, z_min, x_max, y_max, z_max
voxel_size = np.array([0.1, 0.1, 0.5]) # dx, dy, dz

# Ensure points_from_depth are suitable for BEV (e.g., in a consistent coordinate frame)
bev_representation = project_pc_to_bev(
    points=points_from_depth, # Or points_from_scaled_depth
    pc_range=pc_range,
    voxel_size=voxel_size,
    coors_range=pc_range # Typically same as pc_range for BEV
)
print(f"BEV representation shape: {bev_representation.shape}") # (batch_size, nz, ny, nx) or (batch_size, nz, nx, ny)

# --- 6. Visualize Results (using rerun) ---
# visualize_point_cloud(points_from_depth, colors=colors_from_rgb)
# visualize_point_cloud(lidar_points_camera_frame, colors=np.array([[1,0,0]]*len(lidar_points_camera_frame))) # LiDAR in red

# To visualize BEV, you might sum over the Z dimension if it's an occupancy grid
# bev_map_2d = np.sum(bev_representation[0], axis=0) # Assuming first axis is Z after batch
# visualize_bev(bev_map_2d)

# Note: For actual visualization, ensure rerun is initialized and connected if needed,
# as shown in the notebooks/point_cloud.ipynb example.
# import rerun as rr
# rr.init("my_point_cloud_app", spawn=True)
# ... then call visualization functions ...

```

## Key Dependencies

*   **Open3D**: Used for some core point cloud operations like creating point clouds from depth images.
*   **NumPy**: For numerical operations and array manipulations.
*   **SciPy**: Used for optimization routines (e.g., in `solvers.py`) and spatial data structures (e.g., KDTree in `adaptive_scaling.py`).
*   **rerun**: For 3D point cloud and 2D image/BEV visualization.

Make sure these dependencies are installed in your environment to use the full functionality of this module. Refer to the main project's `pyproject.toml` or `requirements.txt` for specific versions.
