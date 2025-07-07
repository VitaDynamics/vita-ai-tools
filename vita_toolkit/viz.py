import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Optional, Tuple

def visualize_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud Visualization",
    show_axes: bool = True
) -> None:
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: Point coordinates array of shape (N, 3)
        colors: Optional color array of shape (N, 3) with values in [0, 1]
        title: Window title for visualization
        show_axes: Whether to show coordinate axes
    """
    if len(points) == 0:
        print("Warning: Empty point cloud provided")
        return
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0  # Convert from [0, 255] to [0, 1] if needed
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Set up visualization using draw_geometries for simplicity
    geometries = [pcd]
    
    if show_axes:
        # Add coordinate frame
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        geometries.append(axes)
    
    # Use simple visualization
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        point_show_normal=False
    )

# NOTE: Currently we support dynamic input channel to visualize BEVs.
def visualize_bev(
    bev_tensor: np.ndarray,
    title: str = "BEV Visualization",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_colorbar: bool = True
) -> None:
    """
    Visualize BEV (Bird's Eye View) representation.
    
    Args:
        bev_tensor: BEV tensor of shape (batch_size, nz, ny, nx) or (batch_size, nz, nx, ny)
        title: Title for the visualization
        cmap: Colormap to use for visualization
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        show_colorbar: Whether to show colorbar
    """
    if len(bev_tensor.shape) != 4:
        raise ValueError(f"Expected 4D BEV tensor, got shape {bev_tensor.shape}")
    
    batch_size, nz, _, _ = bev_tensor.shape
    
    # If batch size > 1, show first batch item
    if batch_size > 1:
        print(f"Visualizing first item from batch of size {batch_size}")
    
    bev_sample = bev_tensor[0]  # Shape: (nz, dim1, dim2)
    
    # Handle different channel configurations
    if nz == 1:
        # Single channel - show as 2D heatmap
        bev_2d = bev_sample[0]
        
        plt.figure(figsize=figsize)
        im = plt.imshow(bev_2d, cmap=cmap, origin='lower')
        plt.title(f"{title} (Single Channel)")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        if show_colorbar:
            plt.colorbar(im, label="Intensity")
            
    elif nz <= 3:
        # Multi-channel - show each channel separately
        _, axes = plt.subplots(1, nz, figsize=(figsize[0] * nz, figsize[1]))
        if nz == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        for i in range(nz):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            im = ax.imshow(bev_sample[i], cmap=cmap, origin='lower')
            ax.set_title(f"{title} - Channel {i}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            if show_colorbar:
                plt.colorbar(im, ax=ax, label="Intensity")
                
    else:
        # Many channels - show sum/max projection
        bev_projection = np.max(bev_sample, axis=0)  # Max projection across channels
        
        plt.figure(figsize=figsize)
        im = plt.imshow(bev_projection, cmap=cmap, origin='lower')
        plt.title(f"{title} (Max Projection of {nz} channels)")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        if show_colorbar:
            plt.colorbar(im, label="Max Intensity")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
