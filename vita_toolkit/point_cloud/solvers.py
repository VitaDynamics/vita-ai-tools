
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares
from .adaptive_scaling import AdaptiveNeighborhoodScaler, AdaptiveScalingConfig



@dataclass
class AlignmentResult:
    """Results from depth-lidar alignment."""
    scale_factor: float
    rmse: float
    mean_error: float
    median_error: float
    std_error: float
    num_correspondences: int
    outlier_ratio: float
    depth_pairs: np.ndarray
    residuals: np.ndarray
    converged: bool
    iterations: int = 0
    local_scales: Optional[np.ndarray] = None
    image_coords: Optional[np.ndarray] = None
    adaptive_metrics: Optional[Dict] = None
    

class DepthLidarSolver:
    """
    Comprehensive solver for depth-lidar alignment with multiple optimization methods.
    
    Based on research findings from:
    - Robust estimation techniques for LiDAR-camera calibration
    - Bundle adjustment methods for sensor fusion
    - Error metrics from KITTI benchmark and calibration literature
    """
    
    def __init__(self, outlier_threshold: float = 0.1, max_iterations: int = 100):
        """
        Initialize the solver.
        
        Args:
            outlier_threshold: Threshold for outlier detection (in meters)
            max_iterations: Maximum iterations for optimization
        """
        self.outlier_threshold = outlier_threshold
        self.max_iterations = max_iterations
        
    def solve(
        self,
        depth_map: np.ndarray,
        lidar_points: np.ndarray,
        intrinsic_params: Dict[str, float],
        method: str = "robust_least_squares",
        robust_loss: str = "huber",
        initial_scale: Optional[float] = None
    ) -> AlignmentResult:
        """
        Solve for optimal depth-lidar alignment.
        
        Args:
            depth_map: Estimated depth map, shape (H, W)
            lidar_points: 3D lidar points in camera coordinate system, shape (N, 3)
            intrinsic_params: Camera intrinsic parameters
            method: "least_squares", "robust_least_squares", "median_ratio", "ransac", "adaptive_neighborhood"
            robust_loss: "linear", "huber", "soft_l1", "cauchy", "arctan"
            initial_scale: Initial guess for scale factor
            
        Returns:
            AlignmentResult with comprehensive metrics
        """
        # Get correspondences
        image_coords = project_points_to_image_plane(lidar_points, intrinsic_params)
        lidar_depths = lidar_points[:, 2]
        
        # Find valid projections
        h, w = depth_map.shape
        u, v = image_coords[:, 0], image_coords[:, 1]
        
        valid_mask = (
            (u >= 0) & (u < w) & 
            (v >= 0) & (v < h) & 
            (lidar_depths > 0)
        )
        
        if np.sum(valid_mask) < 10:
            raise ValueError("Not enough valid correspondences found")
        
        # Get valid coordinates and depths
        valid_u = u[valid_mask].astype(int)
        valid_v = v[valid_mask].astype(int)
        valid_lidar_depths = lidar_depths[valid_mask]
        
        # Sample depth values from depth map
        estimated_depths = depth_map[valid_v, valid_u]
        
        # Remove pairs where estimated depth is invalid
        depth_valid_mask = estimated_depths > 1e-6
        if np.sum(depth_valid_mask) < 10:
            raise ValueError("Not enough valid depth correspondences found")
        
        final_estimated_depths = estimated_depths[depth_valid_mask]
        final_lidar_depths = valid_lidar_depths[depth_valid_mask]
        
        # Solve based on method
        if method == "least_squares":
            result = self._solve_least_squares(final_estimated_depths, final_lidar_depths)
        elif method == "robust_least_squares":
            result = self._solve_robust_least_squares(
                final_estimated_depths, final_lidar_depths, robust_loss, initial_scale
            )
        elif method == "median_ratio":
            result = self._solve_median_ratio(final_estimated_depths, final_lidar_depths)
        elif method == "ransac":
            result = self._solve_ransac(final_estimated_depths, final_lidar_depths)
        elif method == "adaptive_neighborhood":
            # Need image coordinates for adaptive scaling
            valid_image_coords = image_coords[valid_mask][depth_valid_mask]
            result = self._solve_adaptive_neighborhood(
                final_estimated_depths, final_lidar_depths, valid_image_coords, initial_scale
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return result
    
    def _solve_least_squares(self, estimated_depths: np.ndarray, lidar_depths: np.ndarray) -> AlignmentResult:
        """Standard least squares solution."""
        # s = sum(d_radar * d_est) / sum(d_est^2)
        numerator = np.sum(lidar_depths * estimated_depths)
        denominator = np.sum(estimated_depths ** 2)
        scale_factor = float(numerator / denominator)
        
        # Calculate residuals and metrics
        residuals = scale_factor * estimated_depths - lidar_depths
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        depth_pairs = np.column_stack([estimated_depths, lidar_depths])
        
        return AlignmentResult(
            scale_factor=scale_factor,
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            std_error=float(np.std(residuals)),
            num_correspondences=len(estimated_depths),
            outlier_ratio=0.0,
            depth_pairs=depth_pairs,
            residuals=residuals,
            converged=True
        )
    
    def _solve_robust_least_squares(
        self, 
        estimated_depths: np.ndarray, 
        lidar_depths: np.ndarray,
        robust_loss: str,
        initial_scale: Optional[float]
    ) -> AlignmentResult:
        """Robust least squares using scipy.optimize."""
        
        if initial_scale is None:
            # Use standard least squares as initial guess
            initial_scale = np.sum(lidar_depths * estimated_depths) / np.sum(estimated_depths ** 2)
        
        def residual_function(params):
            scale = params[0]
            return scale * estimated_depths - lidar_depths
        
        # Robust optimization
        result = least_squares(
            residual_function,
            x0=[initial_scale],
            loss=robust_loss,
            max_nfev=self.max_iterations
        )
        
        scale_factor = float(result.x[0])
        residuals = result.fun
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        # Detect outliers
        outlier_mask = np.abs(residuals) > self.outlier_threshold
        outlier_ratio = float(np.sum(outlier_mask) / len(residuals))
        
        depth_pairs = np.column_stack([estimated_depths, lidar_depths])
        
        return AlignmentResult(
            scale_factor=scale_factor,
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            std_error=float(np.std(residuals)),
            num_correspondences=len(estimated_depths),
            outlier_ratio=outlier_ratio,
            depth_pairs=depth_pairs,
            residuals=residuals,
            converged=result.success,
            iterations=result.nfev
        )
    
    def _solve_median_ratio(self, estimated_depths: np.ndarray, lidar_depths: np.ndarray) -> AlignmentResult:
        """Median ratio method - robust to outliers."""
        ratios = lidar_depths / estimated_depths
        scale_factor = float(np.median(ratios))
        
        # Calculate residuals and metrics
        residuals = scale_factor * estimated_depths - lidar_depths
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        # Detect outliers based on ratio deviation
        median_ratio = np.median(ratios)
        mad = np.median(np.abs(ratios - median_ratio))
        outlier_mask = np.abs(ratios - median_ratio) > 2.5 * mad
        outlier_ratio = float(np.sum(outlier_mask) / len(ratios))
        
        depth_pairs = np.column_stack([estimated_depths, lidar_depths])
        
        return AlignmentResult(
            scale_factor=scale_factor,
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            std_error=float(np.std(residuals)),
            num_correspondences=len(estimated_depths),
            outlier_ratio=outlier_ratio,
            depth_pairs=depth_pairs,
            residuals=residuals,
            converged=True
        )
    
    def _solve_ransac(self, estimated_depths: np.ndarray, lidar_depths: np.ndarray) -> AlignmentResult:
        """RANSAC-based robust estimation."""
        best_scale = None
        best_inliers = None
        best_score = -1
        
        n_samples = len(estimated_depths)
        n_iterations = min(self.max_iterations, 1000)
        
        for _ in range(n_iterations):
            # Sample random subset
            sample_size = max(10, n_samples // 10)
            sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
            
            sample_est = estimated_depths[sample_indices]
            sample_lidar = lidar_depths[sample_indices]
            
            # Compute scale for this sample
            scale = np.sum(sample_lidar * sample_est) / np.sum(sample_est ** 2)
            
            # Test on all data
            residuals = scale * estimated_depths - lidar_depths
            inlier_mask = np.abs(residuals) < self.outlier_threshold
            inlier_count = np.sum(inlier_mask)
            
            if inlier_count > best_score:
                best_score = inlier_count
                best_scale = scale
                best_inliers = inlier_mask
        
        if best_scale is None:
            raise ValueError("RANSAC failed to find a solution")
        
        # Refine with inliers only
        inlier_est = estimated_depths[best_inliers]
        inlier_lidar = lidar_depths[best_inliers]
        
        if len(inlier_est) > 0:
            refined_scale = np.sum(inlier_lidar * inlier_est) / np.sum(inlier_est ** 2)
        else:
            refined_scale = best_scale
        
        # Calculate final metrics
        residuals = refined_scale * estimated_depths - lidar_depths
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        outlier_ratio = float(1.0 - best_score / n_samples)
        
        depth_pairs = np.column_stack([estimated_depths, lidar_depths])
        
        return AlignmentResult(
            scale_factor=float(refined_scale),
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            std_error=float(np.std(residuals)),
            num_correspondences=len(estimated_depths),
            outlier_ratio=outlier_ratio,
            depth_pairs=depth_pairs,
            residuals=residuals,
            converged=True,
            iterations=n_iterations
        )
    
    def _solve_adaptive_neighborhood(
        self,
        estimated_depths: np.ndarray,
        lidar_depths: np.ndarray,
        image_coords: np.ndarray,
        initial_scale: Optional[float] = None
    ) -> AlignmentResult:
        """Adaptive neighborhood scaling with multi-scale consistency."""
        
        # Initialize adaptive scaler
        config = AdaptiveScalingConfig(
            k_neighbors=min(10, len(estimated_depths) // 2),
            sigma_spatial=50.0,
            regularization_weight=0.1,
            global_weight=0.3
        )
        scaler = AdaptiveNeighborhoodScaler(config)
        
        # Compute adaptive scales
        local_scales, global_scale, adaptive_metrics = scaler.compute_adaptive_scales(
            image_coords, estimated_depths, lidar_depths, initial_scale
        )
        
        # Calculate residuals using local scales
        residuals = local_scales * estimated_depths - lidar_depths
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        # Detect outliers based on local residuals
        outlier_mask = np.abs(residuals) > self.outlier_threshold
        outlier_ratio = float(np.sum(outlier_mask) / len(residuals))
        
        depth_pairs = np.column_stack([estimated_depths, lidar_depths])
        
        return AlignmentResult(
            scale_factor=global_scale,
            rmse=rmse,
            mean_error=float(np.mean(residuals)),
            median_error=float(np.median(residuals)),
            std_error=float(np.std(residuals)),
            num_correspondences=len(estimated_depths),
            outlier_ratio=outlier_ratio,
            depth_pairs=depth_pairs,
            residuals=residuals,
            converged=True,
            local_scales=local_scales,
            image_coords=image_coords,
            adaptive_metrics=adaptive_metrics
        )
    
    def evaluate_alignment(self, result: AlignmentResult) -> Dict[str, float]:
        """
        Evaluate alignment quality with comprehensive metrics.
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "rmse": result.rmse,
            "mean_absolute_error": float(np.mean(np.abs(result.residuals))),
            "median_absolute_error": float(np.median(np.abs(result.residuals))),
            "r2_score": self._calculate_r2(result.depth_pairs),
            "outlier_ratio": result.outlier_ratio,
            "num_correspondences": result.num_correspondences,
            "scale_factor": result.scale_factor
        }
        
        # Add adaptive scaling metrics if available
        if result.adaptive_metrics is not None:
            metrics.update(result.adaptive_metrics)
        
        return metrics
    
    def _calculate_r2(self, depth_pairs: np.ndarray) -> float:
        """Calculate R² score for depth alignment."""
        estimated_depths = depth_pairs[:, 0]
        lidar_depths = depth_pairs[:, 1]
        
        ss_res = np.sum((lidar_depths - estimated_depths) ** 2)
        ss_tot = np.sum((lidar_depths - np.mean(lidar_depths)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    def apply_adaptive_scaling_to_depth_map(
        self,
        depth_map: np.ndarray,
        result: AlignmentResult,
        blend_factor: float = 0.7
    ) -> np.ndarray:
        """
        Apply adaptive scaling to a full depth map using alignment result.
        
        Args:
            depth_map: Original depth map to scale
            result: AlignmentResult from adaptive_neighborhood method
            blend_factor: Blending factor between local and global scaling (0-1)
            
        Returns:
            Scaled depth map
        """
        if result.local_scales is None or result.image_coords is None:
            # Fallback to global scaling
            return depth_map * result.scale_factor
        
        # Use the adaptive scaler to apply scaling
        scaler = AdaptiveNeighborhoodScaler()
        return scaler.apply_adaptive_scaling(
            depth_map, 
            result.local_scales, 
            result.scale_factor, 
            result.image_coords,
            blend_factor
        )