import open3d as o3d
import numpy as np
import cv2
from typing import Dict, Optional, Any


def align_size(
    depth: np.ndarray, rgb: Optional[np.ndarray] = None, factor: Optional[float] = None
):
    """Align depth (and RGB) sizes with optional scaling.

    Args:
        depth: Depth image array.
        rgb: Optional RGB image. Size is unchanged when provided.
        factor: Scale factor applied to ``depth`` using ``cv2.resize`` with
            ``cv2.INTER_NEAREST`` interpolation.

    Returns:
        Tuple of resized depth and RGB image (if given).
    """
    if rgb is None:
        if factor is not None:
            depth = cv2.resize(
                depth,
                None,
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_NEAREST,
            )
        return depth, None
    else:
        return depth, rgb


def pcd_to_camera_coordinate(
    point_cloud: np.ndarray,
    extrinsic: Dict[str, Any],
):
    """
    This function converts point cloud from lidar coordinate system to camera coordinate system
    Args:
        point_cloud: Point cloud in lidar coordinate system (N, 3)
        extrinsic: Extrinsic parameters containing 4x4 transformation matrix

    Returns:
        np.ndarray: Point cloud in camera coordinate system (N, 3)
    """
    # Get the 4x4 transformation matrix from extrinsic parameters
    extrinsic_matrix = np.array(extrinsic["data"])

    # Ensure it's a 4x4 matrix
    if extrinsic_matrix.shape != (4, 4):
        raise ValueError(f"Extrinsic matrix must be 4x4, got {extrinsic_matrix.shape}")

    # Add homogeneous coordinates (convert from (N, 3) to (N, 4))
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack([point_cloud, ones])

    # Apply transformation matrix
    camera_points = homogeneous_points @ extrinsic_matrix.T

    # Return only xyz coordinates (drop homogeneous coordinate)
    return camera_points[:, :3]


def depth_rgb_to_pcd(
    depth: np.ndarray,
    intrinsic: Dict[str, Any],
    extrinsic: Dict[str, Any],
    rgb: Optional[np.ndarray] = None,
    depth_rgb_scale: Optional[float] = None,
    depth_trunc: float = 10.0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate pcd from depthmap.
    Args:
        depth: input depth data.
        intrinsic: camera intrinsic param. If rgb provided, this camera should be rgb camera.
        extrinsic: camera extrinsic param. If rgb provided, this camera should be rgb camera.
        rgb: input rgb data, used as pointcloud color.
        depth_rgb_scale: depth and rgb scale factor, only used when rgb is None.
        depth_trunc: truncate depth value. Unit is meter.

    Returns:
        tuple: (points, colors) where points is np.ndarray of shape (N, 3) and
               colors is np.ndarray of shape (N, 3) or None if no RGB provided.
    """
    # Process depth data only when no RGB data is provided
    if rgb is None:
        # Ensure depth_rgb_scale is provided
        assert depth_rgb_scale is not None
        # Align depth data size, no RGB data needed
        depth, _ = align_size(depth, None, factor=depth_rgb_scale)
        # Get depth image dimensions
        height, width = depth.shape[:2]
        # Convert camera intrinsic parameters to Open3D format
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic["fx"],
            intrinsic["fy"],
            intrinsic["cx"],
            intrinsic["cy"],
        )
        # Convert depth data to Open3D image format
        depth_map = o3d.geometry.Image(np.asarray(depth))
        # Generate point cloud from depth image
        # NOTE: In open3d, create_from_depth_image function support depth_trunc use a millimeter unit. See https://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.create_point_cloud_from_depth_image.html.
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_map, intrinsic_o3d, extrinsic["data"], depth_trunc=depth_trunc * 1000
        )
        # Return point cloud coordinates, no color information
        return np.asarray(pcd.points), None
    else:
        # Align depth and RGB data sizes
        depth, rgb = align_size(depth, rgb)
        # Get depth image dimensions
        height, width = depth.shape[:2]
        # Convert camera intrinsic parameters to Open3D format
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic["fx"],
            intrinsic["fy"],
            intrinsic["cx"],
            intrinsic["cy"],
        )
        # Convert depth data to Open3D image format
        depth_map = o3d.geometry.Image(np.asarray(depth))
        # Convert RGB data to Open3D image format
        rgb_map = o3d.geometry.Image(rgb)
        # Combine RGB and depth images
        # NOTE: In open3d, create_from_color_and_depth function support depth_trunc use a meter unit. See https://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.create_rgbd_image_from_color_and_depth.html.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_map, depth_map, convert_rgb_to_intensity=False, depth_trunc=depth_trunc
        )
        # Generate point cloud from RGB-D image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic_o3d, extrinsic["data"]
        )
        # Return point cloud coordinates and color information        return np.asarray(pcd.points), np.asarray(pcd.colors)
