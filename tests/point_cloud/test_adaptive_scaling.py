import numpy as np
import pytest
from scipy.spatial import KDTree # Used by the module, good to have for context
from scipy.optimize import OptimizeResult # For mocking minimize result
from vita_toolkit.point_cloud.adaptive_scaling import AdaptiveScalingConfig, AdaptiveNeighborhoodScaler
from unittest.mock import patch, MagicMock

# Fixtures
@pytest.fixture
def simple_image_coords():
    return np.array([[10,10], [20,10], [30,10], [40,10], [50,10]], dtype=np.float32)

@pytest.fixture
def simple_estimated_depths():
    return np.array([1.0, 1.1, 1.0, 0.9, 1.0], dtype=np.float32)

@pytest.fixture
def simple_lidar_depths():
    return np.array([2.0, 2.2, 2.0, 1.8, 2.0], dtype=np.float32) # Exactly 2x estimated

@pytest.fixture
def default_config():
    return AdaptiveScalingConfig(k_neighbors=2, sigma_spatial=10.0) # Using smaller k for easier test setup

@pytest.fixture
def small_depth_map():
    return np.ones((20,30), dtype=np.float32) * 5.0


# Initialization Test
def test_adaptive_scaler_init(default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    assert scaler.config == default_config
    scaler_no_config = AdaptiveNeighborhoodScaler()
    assert isinstance(scaler_no_config.config, AdaptiveScalingConfig)

# Test _compute_local_scales
def test_compute_local_scales_basic(simple_image_coords, simple_estimated_depths, simple_lidar_depths, default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config) # k_neighbors=2
    n_points = len(simple_image_coords)
    kdtree = KDTree(simple_image_coords)
    # Query for k_neighbors + 1 (2+1=3) to account for self. Max neighbors is n_points.
    k_query = min(default_config.k_neighbors + 1, n_points)
    distances, indices = kdtree.query(simple_image_coords, k=k_query)

    neighbor_distances = distances[:, 1:] if k_query > 1 else np.empty((n_points,0))
    neighbor_indices = indices[:, 1:] if k_query > 1 else np.empty((n_points,0), dtype=int)

    local_scales = scaler._compute_local_scales(neighbor_indices, neighbor_distances, simple_estimated_depths, simple_lidar_depths)
    assert local_scales.shape == (n_points,)
    assert np.allclose(local_scales, 2.0, rtol=0.1) # Expect scales around 2.0

def test_compute_local_scales_isolated_point(default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    img_coords_iso = np.array([[10,10]], dtype=np.float32)
    est_d_iso = np.array([1.0]); lid_d_iso = np.array([3.0])

    neighbor_indices_empty = np.empty((1,0), dtype=int) # What happens with 1 point, k_neighbors > 0
    neighbor_distances_empty = np.empty((1,0), dtype=float)
    local_scales = scaler._compute_local_scales(neighbor_indices_empty, neighbor_distances_empty, est_d_iso, lid_d_iso)
    expected_fallback_scale = (3.0*1.0) / (1.0*1.0)
    assert np.isclose(local_scales[0], expected_fallback_scale)

# Test _optimize_multiscale_consistency
@patch('vita_toolkit.point_cloud.adaptive_scaling.minimize')
def test_optimize_multiscale_consistency(mock_minimize, default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    n_pts = 5
    initial_ls = np.full(n_pts, 1.8); est_d = np.ones(n_pts); lid_d = np.ones(n_pts)*2.0; initial_gs = 1.9

    mock_res = OptimizeResult(x=np.concatenate([np.full(n_pts, 2.0), [2.0]]), success=True)
    mock_minimize.return_value = mock_res

    opt_ls, opt_gs = scaler._optimize_multiscale_consistency(initial_ls, est_d, lid_d, initial_gs)
    mock_minimize.assert_called_once()
    assert np.allclose(opt_ls, 2.0) and np.isclose(opt_gs, 2.0)

    obj_func = mock_minimize.call_args[0][0]
    assert np.isclose(obj_func(mock_res.x), 0.0) # Loss for perfect scales
    # Test with non-perfect scales (from fixture example)
    # local_scales=1.5, global_scale=2.5. est=1, lid=2. GlobalWeight=0.3, RegularizationWeight=0.1
    # Expected loss: (0.7 * 5*(1.5-2)^2) + (0.3 * 5*(2.5-2)^2) + (0.1 * 5*(1.5-2.5)^2)
    # = (0.7 * 5*0.25) + (0.3 * 5*0.25) + (0.1 * 5*1) = 0.7*1.25 + 0.3*1.25 + 0.5 = 1.25 + 0.5 = 1.75
    params_off = np.concatenate([np.full(n_pts, 1.5), [2.5]])
    assert np.isclose(obj_func(params_off), 1.75)


# Test _compute_metrics
def test_compute_metrics_basic():
    scaler = AdaptiveNeighborhoodScaler()
    ls = np.array([1.8,2.0,2.2]); gs = 2.0; est_d = np.ones(3); lid_d = np.ones(3)*2.0
    metrics = scaler._compute_metrics(ls, gs, est_d, lid_d)
    assert np.isclose(metrics['local_rmse'], np.sqrt( (0.04+0+0.04)/3 ))
    assert np.isclose(metrics['global_rmse'], 0.0)
    assert np.isclose(metrics['scale_std'], np.std(ls))
    assert np.isclose(metrics['consistency_error'], np.mean(np.abs(ls-gs)))

# Test compute_adaptive_scales (integration)
@patch.object(AdaptiveNeighborhoodScaler, '_compute_local_scales')
@patch.object(AdaptiveNeighborhoodScaler, '_optimize_multiscale_consistency')
@patch.object(AdaptiveNeighborhoodScaler, '_compute_metrics')
def test_compute_adaptive_scales_flow(mock_metrics_func, mock_optimize_func, mock_local_scales_func,
                                      simple_image_coords, simple_estimated_depths, simple_lidar_depths, default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    n_pts = len(simple_image_coords)
    # Setup mock return values
    mock_ls_val = np.full(n_pts,1.9); mock_opt_ls_val = np.full(n_pts,2.0); mock_opt_gs_val = 2.0
    mock_metrics_val = {"local_rmse": 0.01}
    mock_local_scales_func.return_value = mock_ls_val
    mock_optimize_func.return_value = (mock_opt_ls_val, mock_opt_gs_val)
    mock_metrics_func.return_value = mock_metrics_val

    # Test with initial_global_scale provided
    scales, gs, metrics = scaler.compute_adaptive_scales(simple_image_coords, simple_estimated_depths, simple_lidar_depths, initial_global_scale=2.1)
    mock_local_scales_func.assert_called_once()
    mock_optimize_func.assert_called_once_with(mock_ls_val, simple_estimated_depths, simple_lidar_depths, 2.1)
    mock_metrics_func.assert_called_once_with(mock_opt_ls_val, mock_opt_gs_val, simple_estimated_depths, simple_lidar_depths)
    assert np.array_equal(scales, mock_opt_ls_val) and gs == mock_opt_gs_val and metrics == mock_metrics_val

    # Test with initial_global_scale=None
    mock_local_scales_func.reset_mock(); mock_optimize_func.reset_mock(); mock_metrics_func.reset_mock()
    expected_initial_gs = np.sum(simple_lidar_depths*simple_estimated_depths) / np.sum(simple_estimated_depths**2)
    scaler.compute_adaptive_scales(simple_image_coords, simple_estimated_depths, simple_lidar_depths, initial_global_scale=None)
    assert np.isclose(mock_optimize_func.call_args[0][3], expected_initial_gs)


# Test apply_adaptive_scaling
def test_apply_adaptive_scaling(small_depth_map, default_config): # small_depth_map is 20x30, all 5.0
    scaler = AdaptiveNeighborhoodScaler(config=default_config) # k_neighbors=2
    img_coords = np.array([[5,5], [10,15], [15,25]], dtype=np.float32) # (u,v) or (x,y)
    ls = np.array([1.5, 2.0, 2.5]); gs = 1.8; blend = 0.7

    scaled_map = scaler.apply_adaptive_scaling(small_depth_map, ls, gs, img_coords, blend)
    assert scaled_map.shape == small_depth_map.shape

    # Check scale at one of the correspondence points (e.g., img_coords[0] -> map access [5,5])
    # Interpolated scale at (5,5) should be 1.5. Final scale = 0.7*1.5 + 0.3*1.8 = 1.05 + 0.54 = 1.59
    assert np.isclose(scaled_map[img_coords[0,1].astype(int), img_coords[0,0].astype(int)], 5.0 * 1.59)

    # Check general properties: positive, within expected range
    min_s = blend*np.min(ls) + (1-blend)*gs; max_s = blend*np.max(ls) + (1-blend)*gs
    assert np.all(scaled_map > 0)
    assert np.all(scaled_map >= (5.0 * min_s) - 1e-5)
    assert np.all(scaled_map <= (5.0 * max_s) + 1e-5)

def test_apply_adaptive_scaling_no_local_scales(small_depth_map, default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    scaled_map = scaler.apply_adaptive_scaling(small_depth_map, np.empty(0), 1.5, np.empty((0,2)))
    assert np.allclose(scaled_map, small_depth_map * 1.5)

def test_apply_adaptive_scaling_k_is_1_for_query(small_depth_map, default_config):
    scaler = AdaptiveNeighborhoodScaler(config=default_config)
    # Only one correspondence point, so KDTree query for neighbors will use k=1
    img_coords_one = np.array([[10,10]], dtype=np.float32)
    ls_one = np.array([1.5]); gs = 1.2; blend = 1.0 # Blend=1 means final_scale = interpolated_local_scale

    scaled_map = scaler.apply_adaptive_scaling(small_depth_map, ls_one, gs, img_coords_one, blend_factor=blend)
    # With one correspondence, interpolated_scales should be that local_scale (1.5) everywhere.
    # Original depth is 5.0. Scaled map should be 5.0 * 1.5 = 7.5 everywhere.
    assert np.allclose(scaled_map, 5.0 * 1.5)
