import numpy as np 
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R

def points_to_voxel_batch(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    max_points: int = 35,
    max_voxels: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy batch processing version for maximum vectorization."""

    points = np.asarray(points, dtype=np.float32)
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    coors_range = np.asarray(coors_range, dtype=np.float32)

    # Batch coordinate calculation
    shifted_points = points[:, :3] - coors_range[:3]
    voxel_coords = (shifted_points / voxel_size).astype(np.int32)
    grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(np.int32)

    # Batch bounds checking
    valid_mask = np.logical_and.reduce([
        voxel_coords[:, i] >= 0 for i in range(3)
    ] + [
        voxel_coords[:, i] < grid_size[i] for i in range(3)
    ])

    if not valid_mask.any():
        return (np.zeros((0, max_points, points.shape[-1]), dtype=points.dtype),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    valid_points = points[valid_mask]
    valid_coords = voxel_coords[valid_mask]

    # Efficient linear indexing
    coord_indices = (valid_coords[:, 0] * grid_size[1] * grid_size[2] +
                    valid_coords[:, 1] * grid_size[2] +
                    valid_coords[:, 2])

    # Group by voxel using lexsort for stability
    sort_indices = np.argsort(coord_indices)
    coord_indices_sorted = coord_indices[sort_indices]
    valid_points_sorted = valid_points[sort_indices]

    # Find unique voxels and their boundaries
    unique_indices, unique_starts = np.unique(coord_indices_sorted, return_index=True)

    if len(unique_indices) > max_voxels:
        unique_indices = unique_indices[:max_voxels]
        unique_starts = unique_starts[:max_voxels + 1]
    else:
        unique_starts = np.append(unique_starts, len(coord_indices_sorted))

    num_voxels = len(unique_indices)

    # Convert back to 3D coordinates
    unique_coords_3d = np.column_stack([
        unique_indices // (grid_size[1] * grid_size[2]),
        (unique_indices // grid_size[2]) % grid_size[1],
        unique_indices % grid_size[2]
    ]).astype(np.int32)

    # Initialize output
    voxels = np.zeros((num_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    num_points_per_voxel = np.zeros(num_voxels, dtype=np.int32)

    # Batch assignment
    for i in range(num_voxels):
        start_idx = unique_starts[i]
        end_idx = unique_starts[i + 1] if i + 1 < len(unique_starts) else len(valid_points_sorted)
        n_points = min(end_idx - start_idx, max_points)

        voxels[i, :n_points] = valid_points_sorted[start_idx:start_idx + n_points]
        num_points_per_voxel[i] = n_points

    return voxels, unique_coords_3d, num_points_per_voxel

# TODO: we can vectorize ths implementation as following I think
"""
Key Vectorization Strategy

Replace the batch loop with vectorized operations:

1. Process all batches at once instead of iterating through each batch_itt
2. Use advanced indexing to scatter values directly to the final tensor shape
3. Leverage numpy's broadcasting for coordinate transformations

Specific Approach

# Instead of the current for loop, use:
if coords is not None and len(coords) > 0:
    # Extract coordinates for all batches at once
    batch_idx = coords[:, 0].astype(np.int64)
    z_coords = np.clip(coords[:, 1], 0, self.nz - 1)
    y_coords = np.clip(coords[:, 2], 0, self.ny - 1)
    x_coords = np.clip(coords[:, 3], 0, self.nx - 1)

    # Create final tensor directly
    if self.is_vcs:
        batch_canvas = np.zeros((batch_size, self.nz, self.nx, self.ny), dtype=dtype)
        batch_canvas[batch_idx, z_coords, self.nx - x_coords - 1, self.ny - y_coords - 1] = 1
    else:
        batch_canvas = np.zeros((batch_size, self.nz, self.ny, self.nx), dtype=dtype)
        batch_canvas[batch_idx, z_coords, y_coords, x_coords] = 1
else:
    # Handle empty coords case
    shape = (batch_size, self.nz, self.nx, self.ny) if self.is_vcs else (batch_size, self.nz, self.ny, self.nx)
    batch_canvas = np.zeros(shape, dtype=dtype)
"""
class RadScatter:
    """
    Numpy-based implementation of RadScatter for voxel grid creation.

    This class performs scattering operations to create voxel grid representations
    from coordinate data, replacing the original PyTorch implementation with numpy.
    """

    def __init__(self, is_vcs: bool = False):
        """
        Initialize RadScatter.

        Args:
            is_vcs: Whether to use VCS (Vehicle Coordinate System) indexing
        """
        self.is_vcs = is_vcs

    def __call__(
        self,
        batch_size: int,
        input_shape: np.ndarray,
        voxel_features: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform the scattering operation to create voxel grids.

        Args:
            batch_size: Number of samples in the batch
            input_shape: Shape of the voxel grid [nx, ny, nz]
            voxel_features: Feature array (used for dtype reference)
            coords: Coordinate array with shape [N, 4] where columns are [batch_idx, z, y, x]

        Returns:
            Numpy array of shape (batch_size, nz, ny, nx) or (batch_size, nz, nx, ny) depending on is_vcs
        """
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.nz = input_shape[2]

        # Determine dtype from voxel_features or default to float32
        dtype = voxel_features.dtype if voxel_features is not None else np.float32

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = np.zeros(self.nx * self.ny * self.nz, dtype=dtype)

            # Only include non-empty pillars if coords are provided
            if coords is not None:
                batch_mask = coords[:, 0] == batch_itt
                this_coords = coords[batch_mask, :]
            else:
                this_coords = np.array([])

            if len(this_coords) > 0:  # Only process if there are coordinates for this batch
                # Clamp coordinates to valid bounds
                z_coords = np.clip(this_coords[:, 1], 0, self.nz - 1)
                y_coords = np.clip(this_coords[:, 2], 0, self.ny - 1)
                x_coords = np.clip(this_coords[:, 3], 0, self.nx - 1)
                
                if self.is_vcs:
                    indices = (
                        z_coords * self.nx * self.ny
                        + (self.nx - x_coords - 1) * self.ny
                        + (self.ny - y_coords - 1)
                    )
                else:
                    indices = (
                        z_coords * self.nx * self.ny
                        + y_coords * self.nx
                        + x_coords
                    )
                indices = indices.astype(np.int64)
                
                # Additional bounds check for safety
                valid_indices = (indices >= 0) & (indices < len(canvas))
                indices = indices[valid_indices]

                # Now scatter the blob back to the canvas.
                if len(indices) > 0:
                    canvas[indices] = 1

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim array (batch-size, nchannels*nrows*ncols)
        batch_canvas = np.stack(batch_canvas, axis=0)

        # Undo the column stacking to final 4-dim array
        if self.is_vcs:
            batch_canvas = batch_canvas.reshape(batch_size, self.nz, self.nx, self.ny)
        else:
            batch_canvas = batch_canvas.reshape(batch_size, self.nz, self.ny, self.nx)
        return batch_canvas


def filter_pcd(point_cloud: np.ndarray, pc_range: np.ndarray) -> np.ndarray:
    """
    Filter point cloud data by removing ego vehicle points and points outside the specified range.

    Parameters:
    point_cloud (np.ndarray): The input point cloud data with shape (N, 3), where N is the number of points.
    pc_range (np.ndarray): The point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
    np.ndarray: The filtered point cloud data.
    """
    # Expand point cloud data with homogeneous coordinates
    point_cloud = np.concatenate([point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))], axis=1)
    xyz = np.array([0, 0, 0, 1])  # lidar state
    rpy = np.array([0, 0, 180])
    rotation_matrix = R.from_euler('xyz', rpy, degrees=True).as_matrix()

    # Create transformation matrix
    Tr = np.eye(4)  # Start with 4x4 identity matrix
    Tr[:3, :3] = rotation_matrix  # Set rotation part
    Tr[:3, 3] = xyz[:3]  # Set translation part

    # Apply inverse transformation to the point cloud
    point_cloud = (np.linalg.inv(Tr) @ point_cloud.T).T
        
    # Remove points that belong to the ego vehicle
    ego_point_mask = (point_cloud[:, 0] < 0.1) & (point_cloud[:, 0] > -1) & (point_cloud[:, 1] > -0.2) & (point_cloud[:, 1] < 0.2)
    point_cloud = point_cloud[~ego_point_mask]

    # Filter points based on the specified range
    filter_mask = (point_cloud[:, 0] > pc_range[0]) & (point_cloud[:, 0] < pc_range[3]) & \
                    (point_cloud[:, 1] > pc_range[1]) & (point_cloud[:, 1] < pc_range[4]) & \
                    (point_cloud[:, 2] > pc_range[2]) & (point_cloud[:, 2] < pc_range[5])
    point_cloud = point_cloud[filter_mask]
    return point_cloud

def project_pc_to_bev(
    points: np.ndarray,
    pc_range: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    max_points: int = 35,
    max_voxels: int = 20000,
    batch_size: int = 1,
    is_vcs: bool = False
) -> np.ndarray:
    """
    Project point cloud to BEV (Bird's Eye View) using voxel generation and RadScatter.
    
    Args:
        points: Point cloud data with shape [N, 3+] where first 3 columns are x, y, z
        pc_range: Range of point cloud [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: Size of each voxel [dx, dy, dz]
        coors_range: Coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
        max_points: Maximum points per voxel
        max_voxels: Maximum number of voxels
        batch_size: Number of samples in batch
        is_vcs: Whether to use VCS coordinate system
        
    Returns:
        BEV representation with shape (batch_size, nz, ny, nx) or (batch_size, nz, nx, ny)
    """
    # Generate voxels from point cloud
    
    filtered_points = filter_pcd(points, pc_range)
    voxels, coords, num_points_per_voxel = points_to_voxel_batch(
        filtered_points, voxel_size, coors_range, max_points, max_voxels
    )
    
    # Calculate grid dimensions
    grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(np.int32)
    
    # Add batch dimension to coordinates (assuming single batch for now)
    if len(coords) > 0:
        batch_coords = np.column_stack([
            np.zeros(len(coords), dtype=np.int32),  # batch index
            coords[:, 2],  # z
            coords[:, 1],  # y  
            coords[:, 0]   # x
        ])
    else:
        batch_coords = np.array([]).reshape(0, 4)
    
    # Use RadScatter to create BEV representation
    scatter = RadScatter(is_vcs=is_vcs)
    bev_features = scatter(
        batch_size=batch_size,
        input_shape=grid_size,
        voxel_features=voxels,
        coords=batch_coords
    )
    
    return bev_features