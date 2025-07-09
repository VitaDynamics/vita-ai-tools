import numpy as np
from typing import Dict, Tuple
from .solvers import DepthLidarSolver

def project_points_to_image_plane(points_3d: np.ndarray, intrinsic_params: Dict[str, float]) -> np.ndarray:
    """
    Project 3D points to 2D image plane using camera intrinsic parameters.
    
    Args:
        points_3d: 3D points in camera coordinate system, shape (N, 3) where columns are [X, Y, Z]
        intrinsic_params: Dictionary containing camera intrinsic parameters
                         {"fx": focal_length_x, "fy": focal_length_y, "cx": principal_point_x, "cy": principal_point_y}
    
    Returns:
        2D image coordinates, shape (N, 2) where columns are [u, v]
    """
    if points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")
    
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    # Avoid division by zero
    valid_mask = Z > 1e-6
    
    fx, fy = intrinsic_params["fx"], intrinsic_params["fy"]
    cx, cy = intrinsic_params["cx"], intrinsic_params["cy"]
    
    # Project to image plane using pinhole camera model
    u = np.full_like(X, -1, dtype=np.float32)
    v = np.full_like(Y, -1, dtype=np.float32)
    
    u[valid_mask] = fx * X[valid_mask] / Z[valid_mask] + cx
    v[valid_mask] = fy * Y[valid_mask] / Z[valid_mask] + cy
    
    return np.column_stack([u, v])


def align_depth_lidar(
    depth_map: np.ndarray,
    lidar_points: np.ndarray,
    intrinsic_params: Dict[str, float],
    method: str = "least_squares",
    use_solver: bool = True,
    **solver_kwargs
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Align depth and lidar data by finding optimal scale factor.
    
    Args:
        depth_map: Estimated depth map, shape (H, W)
        lidar_points: 3D lidar points in camera coordinate system, shape (N, 3)
        intrinsic_params: Camera intrinsic parameters
        method: "least_squares", "median_ratio", "robust_least_squares", "ransac", or "adaptive_neighborhood"
        use_solver: Whether to use the DepthLidarSolver (recommended)
        **solver_kwargs: Additional arguments for DepthLidarSolver
    
    Returns:
        Tuple of (scale_factor, valid_depth_pairs, valid_lidar_depths)
        - scale_factor: Optimal scale to apply to depth_map
        - valid_depth_pairs: Array of (depth_estimated, depth_lidar) pairs used
        - valid_lidar_depths: Corresponding lidar depths
    """
    if use_solver:
        # Project lidar points to image plane
        image_coords_all = project_points_to_image_plane(lidar_points, intrinsic_params)
        lidar_depths_all = lidar_points[:, 2]

        # Use the comprehensive solver
        solver = DepthLidarSolver(**solver_kwargs)
        result = solver.solve(
            depth_map,
            image_coords_all,
            lidar_depths_all,
            method,
            robust_loss=solver_kwargs.get("robust_loss", "huber"), # Pass relevant kwargs if needed by solve
            initial_scale=solver_kwargs.get("initial_scale")
        )
        
        # Extract lidar depths from depth pairs for backward compatibility
        valid_lidar_depths = result.depth_pairs[:, 1] # This remains the same as AlignmentResult structure is unchanged
        
        return result.scale_factor, result.depth_pairs, valid_lidar_depths
    
    # Legacy implementation for backward compatibility
    if method not in ["least_squares", "median_ratio"]:
        raise ValueError("method must be 'least_squares' or 'median_ratio'")
    
    # Project lidar points to image plane
    image_coords = project_points_to_image_plane(lidar_points, intrinsic_params)
    
    # Extract depths from lidar points (Z coordinate)
    lidar_depths = lidar_points[:, 2]
    
    # Find valid projections (within image bounds and positive depth)
    h, w = depth_map.shape
    u, v = image_coords[:, 0], image_coords[:, 1]
    
    valid_mask = (
        (u >= 0) & (u < w) & 
        (v >= 0) & (v < h) & 
        (lidar_depths > 0)
    )
    
    if np.sum(valid_mask) < 3:
        raise ValueError("Not enough valid correspondences found")
    
    # Get valid coordinates and depths
    valid_u = u[valid_mask].astype(int)
    valid_v = v[valid_mask].astype(int)
    valid_lidar_depths = lidar_depths[valid_mask]
    
    # Sample depth values from depth map at projected coordinates
    estimated_depths = depth_map[valid_v, valid_u]
    
    # Remove pairs where estimated depth is zero or invalid
    depth_valid_mask = estimated_depths > 1e-6
    if np.sum(depth_valid_mask) < 3:
        raise ValueError("Not enough valid depth correspondences found")
    
    final_estimated_depths = estimated_depths[depth_valid_mask]
    final_lidar_depths = valid_lidar_depths[depth_valid_mask]
    
    # Calculate scale factor
    if method == "least_squares":
        # s = sum(d_radar * d_est) / sum(d_est^2)
        numerator = np.sum(final_lidar_depths * final_estimated_depths)
        denominator = np.sum(final_estimated_depths ** 2)
        scale_factor = float(numerator / denominator)
    else:  # median_ratio
        # s = median(d_radar / d_est)
        ratios = final_lidar_depths / final_estimated_depths
        scale_factor = float(np.median(ratios))
    
    # Create pairs for output
    valid_depth_pairs = np.column_stack([final_estimated_depths, final_lidar_depths])
    
    return scale_factor, valid_depth_pairs, final_lidar_depths


