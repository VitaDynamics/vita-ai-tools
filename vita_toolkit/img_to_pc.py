import open3d as o3d
import numpy as np
from typing import Dict, Optional, Any 

def align_size(depth: np.ndarray, rgb: Optional[np.ndarray] = None, factor: Optional[float] = None):
    """Align depth and rgb image sizes."""
    if rgb is None:
        if factor is not None:
            h, w = depth.shape[:2]
            new_h, new_w = int(h * factor), int(w * factor)
            depth = np.resize(depth, (new_h, new_w))
        return depth, None
    else:
        return depth, rgb

def depth_rgb_to_pcd(
    depth: np.ndarray,
    intrinsic: Dict[str, Any],
    extrinsic: Dict[str, Any],
    rgb: Optional[np.ndarray] = None,
    depth_rgb_scale: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate pcd from depthmap.
    Args:
        depth: input depth data.
        camera: camera param. If rgb provided, this camera should be rgb camera.
        rgb: input rgb data, used as pointcloud color.
        depth_rgb_scale: depth and rgb scale factor, only used when rgb is None.
    
    Returns:
        tuple: (points, colors) where points is np.ndarray of shape (N, 3) and 
               colors is np.ndarray of shape (N, 3) or None if no RGB provided.
    """
    # Process depth data only when no RGB data is provided
    if rgb is None:
        # Ensure depth_rgb_scale is provided
        assert(depth_rgb_scale is not None)
        # Align depth data size, no RGB data needed
        depth, _ = align_size(depth, None, factor=depth_rgb_scale)
        # Get depth image dimensions
        height, width = depth.shape[:2]
        # Convert camera intrinsic parameters to Open3D format
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic["data"])
        # Convert depth data to Open3D image format
        depth_map = o3d.geometry.Image(np.asarray(depth, order="C"))
        # Generate point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_map, intrinsic_o3d, extrinsic["data"])
        # Return point cloud coordinates, no color information
        return np.asarray(pcd.points), None
    else:
        # Align depth and RGB data sizes
        depth, rgb = align_size(depth, rgb)
        # Get depth image dimensions
        height, width = depth.shape[:2]
        # Convert camera intrinsic parameters to Open3D format
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic["fx"], intrinsic["fy"], intrinsic["cx"], intrinsic["cy"])
        # Convert depth data to Open3D image format
        depth_map = o3d.geometry.Image(np.asarray(depth, order="C"))
        # Convert RGB data to Open3D image format
        rgb_map = o3d.geometry.Image(rgb)
        # Combine RGB and depth images
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_map, depth_map, convert_rgb_to_intensity=False)
        # Generate point cloud from RGB-D image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic_o3d, extrinsic["data"])
        # Return point cloud coordinates and color information
        return np.asarray(pcd.points), np.asarray(pcd.colors)