import numpy as np
from typing import Tuple, Optional
from scipy.spatial import KDTree
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class AdaptiveScalingConfig:
    """Configuration for adaptive neighborhood scaling."""
    k_neighbors: int = 10
    sigma_spatial: float = 50.0
    regularization_weight: float = 0.1
    global_weight: float = 0.3
    max_iterations: int = 100
    convergence_threshold: float = 1e-6


class AdaptiveNeighborhoodScaler:
    """
    Adaptive neighborhood scaling for depth-lidar alignment.
    
    Implements local scale factor estimation with multi-scale consistency
    and regularization to balance local adaptation with global consistency.
    """
    
    def __init__(self, config: AdaptiveScalingConfig = None):
        self.config = config or AdaptiveScalingConfig()
        
    def compute_adaptive_scales(
        self,
        image_coords: np.ndarray,
        estimated_depths: np.ndarray,
        lidar_depths: np.ndarray,
        initial_global_scale: Optional[float] = None
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Compute adaptive scale factors for each correspondence.
        
        Args:
            image_coords: Image coordinates (N, 2) - [u, v]
            estimated_depths: Estimated depths (N,)
            lidar_depths: LiDAR depths (N,)
            initial_global_scale: Initial global scale factor
            
        Returns:
            Tuple of (local_scales, global_scale, metrics)
        """
        n_points = len(image_coords)
        
        if initial_global_scale is None:
            # Compute initial global scale
            initial_global_scale = np.sum(lidar_depths * estimated_depths) / np.sum(estimated_depths ** 2)
        
        # Build spatial index for fast neighbor search
        spatial_tree = KDTree(image_coords)
        
        # Find k-nearest neighbors for each point
        neighbor_distances, neighbor_indices = spatial_tree.query(
            image_coords, k=min(self.config.k_neighbors + 1, n_points)
        )
        
        # Remove self from neighbors (first column)
        neighbor_distances = neighbor_distances[:, 1:]
        neighbor_indices = neighbor_indices[:, 1:]
        
        # Compute local scale factors using weighted averages
        local_scales = self._compute_local_scales(
            neighbor_indices, neighbor_distances, estimated_depths, lidar_depths
        )
        
        # Optimize with multi-scale consistency
        optimized_scales, final_global_scale = self._optimize_multiscale_consistency(
            local_scales, estimated_depths, lidar_depths, initial_global_scale
        )
        
        # Compute metrics
        metrics = self._compute_metrics(
            optimized_scales, final_global_scale, estimated_depths, lidar_depths
        )
        
        return optimized_scales, final_global_scale, metrics
    
    def _compute_local_scales(
        self,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        estimated_depths: np.ndarray,
        lidar_depths: np.ndarray
    ) -> np.ndarray:
        """Compute local scale factors using weighted k-nearest neighbors."""
        n_points = len(estimated_depths)
        local_scales = np.zeros(n_points)
        
        for i in range(n_points):
            # Get neighbors and their distances
            neighbors = neighbor_indices[i]
            distances = neighbor_distances[i]
            
            # Filter out invalid neighbors (in case of duplicates or boundary effects)
            valid_neighbors = neighbors[neighbors < n_points]
            valid_distances = distances[:len(valid_neighbors)]
            
            if len(valid_neighbors) == 0:
                # Fallback to global scale for isolated points
                local_scales[i] = np.sum(lidar_depths * estimated_depths) / np.sum(estimated_depths ** 2)
                continue
            
            # Compute distance-based weights using Gaussian kernel
            weights = np.exp(-valid_distances ** 2 / (2 * self.config.sigma_spatial ** 2))
            weights = weights / np.sum(weights)  # Normalize
            
            # Include current point in computation
            all_indices = np.concatenate([[i], valid_neighbors])
            all_weights = np.concatenate([[1.0], weights])
            all_weights = all_weights / np.sum(all_weights)  # Renormalize
            
            # Compute weighted local scale factor
            numerator = np.sum(all_weights * lidar_depths[all_indices] * estimated_depths[all_indices])
            denominator = np.sum(all_weights * estimated_depths[all_indices] ** 2)
            
            if denominator > 1e-10:
                local_scales[i] = numerator / denominator
            else:
                local_scales[i] = 1.0
        
        return local_scales
    
    def _optimize_multiscale_consistency(
        self,
        initial_local_scales: np.ndarray,
        estimated_depths: np.ndarray,
        lidar_depths: np.ndarray,
        initial_global_scale: float
    ) -> Tuple[np.ndarray, float]:
        """Optimize scale factors with multi-scale consistency regularization."""
        n_points = len(initial_local_scales)
        
        # Initial parameters: [local_scales..., global_scale]
        initial_params = np.concatenate([initial_local_scales, [initial_global_scale]])
        
        def objective(params):
            local_scales = params[:n_points]
            global_scale = params[n_points]
            
            # Data fitting term (local scales)
            local_residuals = local_scales * estimated_depths - lidar_depths
            local_loss = np.sum(local_residuals ** 2)
            
            # Global consistency term
            global_residuals = global_scale * estimated_depths - lidar_depths
            global_loss = np.sum(global_residuals ** 2)
            
            # Regularization term - encourage local scales to be close to global
            regularization = np.sum((local_scales - global_scale) ** 2)
            
            # Combined objective
            total_loss = (
                (1 - self.config.global_weight) * local_loss +
                self.config.global_weight * global_loss +
                self.config.regularization_weight * regularization
            )
            
            return total_loss
        
        # Optimize using L-BFGS-B
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.config.max_iterations}
        )
        
        optimized_params = result.x
        optimized_local_scales = optimized_params[:n_points]
        optimized_global_scale = optimized_params[n_points]
        
        return optimized_local_scales, optimized_global_scale
    
    def _compute_metrics(
        self,
        local_scales: np.ndarray,
        global_scale: float,
        estimated_depths: np.ndarray,
        lidar_depths: np.ndarray
    ) -> dict:
        """Compute evaluation metrics for adaptive scaling."""
        
        # Local scale residuals
        local_residuals = local_scales * estimated_depths - lidar_depths
        local_rmse = np.sqrt(np.mean(local_residuals ** 2))
        
        # Global scale residuals
        global_residuals = global_scale * estimated_depths - lidar_depths
        global_rmse = np.sqrt(np.mean(global_residuals ** 2))
        
        # Scale variation metrics
        scale_std = np.std(local_scales)
        scale_range = np.max(local_scales) - np.min(local_scales)
        
        # Consistency metrics
        consistency_error = np.mean(np.abs(local_scales - global_scale))
        
        return {
            'local_rmse': float(local_rmse),
            'global_rmse': float(global_rmse),
            'scale_std': float(scale_std),
            'scale_range': float(scale_range),
            'consistency_error': float(consistency_error),
            'mean_local_scale': float(np.mean(local_scales)),
            'global_scale': float(global_scale),
            'improvement_ratio': float(global_rmse / local_rmse) if local_rmse > 0 else 1.0
        }
    
    def apply_adaptive_scaling(
        self,
        depth_map: np.ndarray,
        local_scales: np.ndarray,
        global_scale: float,
        image_coords: np.ndarray,
        blend_factor: float = 0.7
    ) -> np.ndarray:
        """
        Apply adaptive scaling to depth map using spatial interpolation.
        
        Args:
            depth_map: Original depth map (H, W)
            local_scales: Local scale factors for correspondence points
            global_scale: Global scale factor
            image_coords: Image coordinates of correspondence points
            blend_factor: Blending factor between local and global scaling
            
        Returns:
            Scaled depth map
        """
        h, w = depth_map.shape
        
        # Create coordinate grids
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        query_coords = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
        
        # Interpolate local scales to full image
        if len(local_scales) > 0:
            # Use spatial interpolation for smooth scale transitions
            scale_tree = KDTree(image_coords)
            distances, indices = scale_tree.query(query_coords, k=min(5, len(local_scales)))
            
            # Handle case where k=1 (distances and indices are 1D)
            if distances.ndim == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)
            
            # Distance-based weighting
            weights = 1.0 / (distances + 1e-10)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            # Interpolate local scales
            interpolated_scales = np.sum(weights * local_scales[indices], axis=1)
        else:
            interpolated_scales = np.full(len(query_coords), global_scale)
        
        # Blend local and global scales
        final_scales = (
            blend_factor * interpolated_scales +
            (1 - blend_factor) * global_scale
        )
        
        # Reshape back to image dimensions
        scale_map = final_scales.reshape(h, w)
        
        # Apply scaling
        scaled_depth_map = depth_map * scale_map
        
        return scaled_depth_map