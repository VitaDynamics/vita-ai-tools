import numpy as np
import pytest
from vita_toolkit.point_cloud.depth_lidar_matching import project_points_to_image_plane, align_depth_lidar
from vita_toolkit.point_cloud.solvers import DepthLidarSolver, AlignmentResult # For mocking
from unittest.mock import patch, MagicMock

# Fixtures
@pytest.fixture
def sample_intrinsic_params():
    return {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

@pytest.fixture
def sample_3d_points():
    # Points: [X, Y, Z]
    return np.array([
        [10, 5, 20],   # In front, projects within typical image
        [-5, -2, 10],  # In front, projects within typical image
        [0, 0, 1],     # Close to camera center
        [10, 5, 0.001],# Very close, positive Z
        [10, 5, -10],  # Behind camera
        [1000, 500, 100] # Far away, likely outside typical FoV if cx/cy are central
    ], dtype=np.float32)

@pytest.fixture
def sample_depth_map():
    # A 480x640 depth map
    depth = np.ones((480, 640), dtype=np.float32) * 10.0 # Default depth of 10m
    # Add some variation
    depth[100:200, 100:200] = 5.0
    depth[240, 320] = 1.0 # Corresponds to cx, cy for one point
    return depth

@pytest.fixture
def sample_lidar_points_cam_coords(sample_intrinsic_params, sample_depth_map):
    # Create some LiDAR points that would project into the depth map
    # and have corresponding depths similar to those in sample_depth_map
    # For simplicity, let's create points whose 2D projection we can easily determine
    # and whose Z value (Lidar depth) we can set.

    fx = sample_intrinsic_params["fx"]
    fy = sample_intrinsic_params["fy"]
    cx = sample_intrinsic_params["cx"]
    cy = sample_intrinsic_params["cy"]

    points = []

    # Point 1: projects to (cx, cy), depth map has 1.0 there. Let lidar depth be 1.0.
    # u = fx * X/Z + cx  => cx = fx * X/Z + cx => X/Z = 0 => X = 0
    # v = fy * Y/Z + cy  => cy = fy * Y/Z + cy => Y/Z = 0 => Y = 0
    # Z = 1.0
    points.append([0, 0, 1.0]) # Projects to (cx, cy) = (320, 240)

    # Point 2: projects to u=cx+100, v=cy+50. Depth map has ~10.0 there. Let lidar depth be 10.0
    # u = 420, v = 290.
    # 420 = 500 * X/Z + 320 => 100 = 500 * X/Z => X/Z = 100/500 = 0.2
    # 290 = 500 * Y/Z + 240 => 50 = 500 * Y/Z => Y/Z = 50/500 = 0.1
    # Let Z = 10.0. Then X = 2.0, Y = 1.0.
    points.append([2.0, 1.0, 10.0]) # Projects to (420, 290)

    # Point 3: projects to u=cx-50, v=cy-20. Depth map has ~10.0 there. Let lidar depth be 9.0 (slight diff)
    # u = 270, v = 220
    # 270 = 500 * X/Z + 320 => -50 = 500 * X/Z => X/Z = -0.1
    # 220 = 500 * Y/Z + 240 => -20 = 500 * Y/Z => Y/Z = -0.04
    # Let Z = 9.0. Then X = -0.9, Y = -0.36
    points.append([-0.9, -0.36, 9.0]) # Projects to (270, 220)

    # Point 4: projects to an area with depth 5.0 in depth_map (e.g., 150,150)
    # u = 150, v = 150
    # 150 = 500 * X/Z + 320 => -170 = 500 * X/Z => X/Z = -170/500 = -0.34
    # 150 = 500 * Y/Z + 240 => -90 = 500 * Y/Z => Y/Z = -90/500 = -0.18
    # Let Z = 5.0. Then X = -1.7, Y = -0.9
    points.append([-1.7, -0.9, 5.0]) # Projects to (150, 150)

    # Add a point that projects outside image bounds
    points.append([100, 100, 20]) # X/Z = 5, Y/Z = 5. u = 500*5+320 = 2820 (way outside)

    # Add a point with Lidar depth <= 0
    points.append([0.1, 0.1, -2.0])

    # Add a point whose estimated depth from map will be 0 or invalid (if map had zeros)
    # For this, we need to ensure sample_depth_map has a zero. Let's add one.
    # sample_depth_map[10,10] = 0. This needs to be done if we test this case.
    # For now, assume all sampled depths are > 0.

    return np.array(points, dtype=np.float32)


# Tests for project_points_to_image_plane
def test_project_points_to_image_plane_valid(sample_3d_points, sample_intrinsic_params):
    points_3d = sample_3d_points
    intrinsic = sample_intrinsic_params

    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]

    projected_coords = project_points_to_image_plane(points_3d, intrinsic)

    assert projected_coords.shape == (points_3d.shape[0], 2)

    # Expected for point [10, 5, 20]
    # X=10, Y=5, Z=20
    # u = 500 * 10/20 + 320 = 250 + 320 = 570
    # v = 500 * 5/20 + 240 = 125 + 240 = 365
    assert np.allclose(projected_coords[0], [570, 365])

    # Expected for point [0, 0, 1]
    # X=0, Y=0, Z=1
    # u = 500 * 0/1 + 320 = 320
    # v = 500 * 0/1 + 240 = 240
    assert np.allclose(projected_coords[2], [cx, cy])

    # Point with Z very close to 0 but positive
    # X=10, Y=5, Z=0.001
    # u = 500 * 10/0.001 + 320 = 5000000 + 320 = 5000320
    # v = 500 * 5/0.001 + 240 = 2500000 + 240 = 2500240
    assert np.allclose(projected_coords[3], [5000320, 2500240])

    # Point behind camera [10, 5, -10] -> Z < 0, should result in invalid coords (-1, -1)
    assert np.allclose(projected_coords[4], [-1, -1])

def test_project_points_to_image_plane_zero_z(sample_intrinsic_params):
    points_zero_z = np.array([[10, 5, 0.0]], dtype=np.float32) # Z = 0
    projected = project_points_to_image_plane(points_zero_z, sample_intrinsic_params)
    assert np.allclose(projected[0], [-1,-1]) # Expect invalid marker

def test_project_points_to_image_plane_invalid_shape():
    points_invalid_shape = np.array([[1, 2, 3, 4]], dtype=np.float32) # (N,4) instead of (N,3)
    with pytest.raises(ValueError) as excinfo:
        project_points_to_image_plane(points_invalid_shape, {"fx":1, "fy":1, "cx":0, "cy":0})
    assert "points_3d must have shape (N, 3)" in str(excinfo.value)

def test_project_points_to_image_plane_empty_input(sample_intrinsic_params):
    empty_points = np.empty((0,3), dtype=np.float32)
    projected = project_points_to_image_plane(empty_points, sample_intrinsic_params)
    assert projected.shape == (0,2)


# Tests for align_depth_lidar (legacy path: use_solver=False)
def test_align_depth_lidar_legacy_least_squares(sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params):
    # Modify depth map so estimated depths are not all identical to lidar depths for a more realistic test
    depth_map_varied = sample_depth_map.copy()
    # Let's say estimated depths are generally 0.9 times the lidar depths
    # For point [0,0,1.0] -> projects to (320,240), map has 1.0. Let's make it 0.9
    depth_map_varied[240, 320] = 0.9 * 1.0
    # For point [2.0,1.0,10.0] -> projects to (420,290), map has 10.0. Let's make it 0.9 * 10 = 9.0
    depth_map_varied[290, 420] = 0.9 * 10.0
    # For point [-0.9,-0.36,9.0] -> projects to (270,220), map has 10.0. Let's make it 0.9 * 9.0 = 8.1
    depth_map_varied[220, 270] = 0.9 * 9.0
    # For point [-1.7,-0.9,5.0] -> projects to (150,150), map has 5.0. Let's make it 0.9 * 5.0 = 4.5
    depth_map_varied[150, 150] = 0.9 * 5.0

    scale_factor, valid_pairs, _ = align_depth_lidar(
        depth_map_varied, sample_lidar_points_cam_coords, sample_intrinsic_params,
        method="least_squares", use_solver=False
    )

    # Expected valid lidar depths: 1.0, 10.0, 9.0, 5.0 (4 points)
    # Expected estimated depths: 0.9, 9.0, 8.1, 4.5
    # d_est = [0.9, 9.0, 8.1, 4.5]
    # d_lid = [1.0, 10.0, 9.0, 5.0]
    # scale = sum(d_lid * d_est) / sum(d_est^2)
    # sum(d_lid*d_est) = 1*0.9 + 10*9 + 9*8.1 + 5*4.5 = 0.9 + 90 + 72.9 + 22.5 = 186.3
    # sum(d_est^2) = 0.9^2 + 9^2 + 8.1^2 + 4.5^2 = 0.81 + 81 + 65.61 + 20.25 = 167.67
    # expected_scale = 186.3 / 167.67 = ~1.11111... (or 1 / 0.9)

    assert np.isclose(scale_factor, 1.0 / 0.9)
    assert valid_pairs.shape[0] == 4 # 4 points should be valid
    assert valid_pairs.shape[1] == 2

    # Check contents of valid_pairs (order might vary depending on internal filtering)
    expected_est_depths = np.array([0.9, 9.0, 8.1, 4.5])
    expected_lid_depths = np.array([1.0, 10.0, 9.0, 5.0])

    # Sort pairs by estimated depth for comparison
    sorted_indices_result = np.argsort(valid_pairs[:,0])
    sorted_indices_expected = np.argsort(expected_est_depths)

    assert np.allclose(valid_pairs[sorted_indices_result, 0], expected_est_depths[sorted_indices_expected])
    assert np.allclose(valid_pairs[sorted_indices_result, 1], expected_lid_depths[sorted_indices_expected])


def test_align_depth_lidar_legacy_median_ratio(sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params):
    depth_map_ratios = sample_depth_map.copy()
    # d_lidar / d_estimated ratios:
    # Point 1: Lidar 1.0. Est map[240,320]=1.0. Ratio = 1.0/1.0 = 1.0
    # Point 2: Lidar 10.0. Est map[290,420]=10.0. Ratio = 10.0/10.0 = 1.0
    # Point 3: Lidar 9.0. Est map[220,270]=10.0. Ratio = 9.0/10.0 = 0.9
    # Point 4: Lidar 5.0. Est map[150,150]=5.0. Ratio = 5.0/5.0 = 1.0
    # Ratios: [1.0, 1.0, 0.9, 1.0]. Sorted: [0.9, 1.0, 1.0, 1.0]. Median is 1.0.
    # Let's change one more to make median different
    # Point 1: Est map[240,320]=0.8. Ratio = 1.0/0.8 = 1.25
    depth_map_ratios[240,320] = 0.8
    # Ratios: [1.25, 1.0, 0.9, 1.0]. Sorted: [0.9, 1.0, 1.0, 1.25]. Median is 1.0. Still.
    # Let's make Point 2 ratio also different:
    # Point 2: Est map[290,420]=8.0. Ratio = 10.0/8.0 = 1.25
    depth_map_ratios[290,420] = 8.0
    # Ratios: [1.25, 1.25, 0.9, 1.0]. Sorted: [0.9, 1.0, 1.25, 1.25]. Median is (1.0+1.25)/2 = 1.125

    scale_factor, _, _ = align_depth_lidar(
        depth_map_ratios, sample_lidar_points_cam_coords, sample_intrinsic_params,
        method="median_ratio", use_solver=False
    )
    assert np.isclose(scale_factor, 1.125)

def test_align_depth_lidar_legacy_not_enough_correspondences(sample_intrinsic_params):
    # Scenario 1: Enough initial correspondences, but all estimated depths are zero
    intrinsic_100x100 = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
    zero_depth_map = np.zeros((100, 100), dtype=np.float32)
    lidar_pts_corrected = np.array([
        [0, 0, 1],       # Projects to u=50, v=50
        [10, 10, 100],   # X/Z=0.1, Y/Z=0.1. Projects to u=100*0.1+50=60, v=60
        [-10, 10, 100],  # X/Z=-0.1, Y/Z=0.1. Projects to u=100*-0.1+50=40, v=60
        [10, -10, 100]   # X/Z=0.1, Y/Z=-0.1. Projects to u=60, v=40
    ], dtype=np.float32)
    # All 4 points should project inside 100x100 map. sum(valid_mask) should be 4.
    # Then, estimated_depths from zero_depth_map will all be 0.
    # So, sum(depth_valid_mask) will be 0.
    # This should raise "Not enough valid depth correspondences found".
    with pytest.raises(ValueError) as excinfo1:
        align_depth_lidar(zero_depth_map, lidar_pts_corrected, intrinsic_100x100, use_solver=False)
    assert "Not enough valid depth correspondences found" in str(excinfo1.value)

    # Scenario 2: Not enough initial lidar projections (e.g., all points project outside map)
    small_depth_map = np.ones((10, 10)) # w=10, h=10
    # Use sample_intrinsic_params (cx=320, cy=240) which will ensure points project outside 10x10 map
    bad_lidar_pts = np.array([[1,1,10], [2,2,5], [-1,-1,20]], dtype=np.float32)
    # Example: (1,1,10) with sample_intrinsic_params -> u = 500*1/10+320 = 50+320 = 370. This is >> w=10.
    # All points will be out of bounds. sum(valid_mask) will be 0.
    # This should raise "Not enough valid correspondences found".
    with pytest.raises(ValueError) as excinfo2:
        align_depth_lidar(small_depth_map, bad_lidar_pts, sample_intrinsic_params, use_solver=False)
    assert "Not enough valid correspondences found" in str(excinfo2.value)


def test_align_depth_lidar_legacy_invalid_method(sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params):
    with pytest.raises(ValueError) as excinfo:
        align_depth_lidar(sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params,
                          method="unknown_method", use_solver=False)
    assert "method must be 'least_squares' or 'median_ratio'" in str(excinfo.value)


# Tests for align_depth_lidar (modern path: use_solver=True)
@patch('vita_toolkit.point_cloud.depth_lidar_matching.DepthLidarSolver')
def test_align_depth_lidar_use_solver_true(MockDepthLidarSolver, sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params):
    # Configure the mock solver instance and its solve method
    mock_solver_instance = MagicMock(spec=DepthLidarSolver)
    mock_alignment_result = MagicMock(spec=AlignmentResult)
    mock_alignment_result.scale_factor = 1.23
    # The function expects depth_pairs to be (N,2)
    mock_alignment_result.depth_pairs = np.array([[1.0, 1.23], [2.0, 2.46]])

    mock_solver_instance.solve.return_value = mock_alignment_result
    MockDepthLidarSolver.return_value = mock_solver_instance

    solver_kwargs = {"outlier_threshold": 0.5, "max_iterations": 50}
    method_to_use = "robust_least_squares"

    scale, pairs, lidar_depths_ret = align_depth_lidar(
        sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params,
        method=method_to_use, use_solver=True, **solver_kwargs
    )

    # Check DepthLidarSolver was initialized with solver_kwargs
    MockDepthLidarSolver.assert_called_once_with(**solver_kwargs)

    # Check solver.solve was called with correct parameters
    # Need to compute what image_coords_all and lidar_depths_all would be
    from vita_toolkit.point_cloud.depth_lidar_matching import project_points_to_image_plane as actual_project_func
    expected_image_coords = actual_project_func(sample_lidar_points_cam_coords, sample_intrinsic_params)
    expected_lidar_depths = sample_lidar_points_cam_coords[:, 2]

    # The solver_kwargs are used for DepthLidarSolver initialization,
    # but specific ones like robust_loss, initial_scale are also passed to solve method itself.

    mock_solver_instance.solve.assert_called_once() # Check it was called once
    called_args, called_kwargs = mock_solver_instance.solve.call_args

    assert np.array_equal(called_args[0], sample_depth_map)
    assert np.array_equal(called_args[1], expected_image_coords)
    assert np.array_equal(called_args[2], expected_lidar_depths)
    assert called_args[3] == method_to_use

    assert called_kwargs.get("robust_loss") == solver_kwargs.get("robust_loss", "huber")
    assert called_kwargs.get("initial_scale") == solver_kwargs.get("initial_scale")

    # Check results are from the mocked solver
    assert scale == mock_alignment_result.scale_factor
    assert np.array_equal(pairs, mock_alignment_result.depth_pairs)
    assert np.array_equal(lidar_depths_ret, mock_alignment_result.depth_pairs[:, 1])

# Test case for when the solver itself raises an error (e.g. not enough points)
@patch('vita_toolkit.point_cloud.depth_lidar_matching.DepthLidarSolver')
def test_align_depth_lidar_use_solver_handles_solver_exception(MockDepthLidarSolver, sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params):
    mock_solver_instance = MagicMock(spec=DepthLidarSolver)
    mock_solver_instance.solve.side_effect = ValueError("Solver internal error")
    MockDepthLidarSolver.return_value = mock_solver_instance

    with pytest.raises(ValueError, match="Solver internal error"):
        align_depth_lidar(
            sample_depth_map, sample_lidar_points_cam_coords, sample_intrinsic_params,
            method="ransac", use_solver=True
        )

# Fixture for sample_lidar_points_cam_coords needs sample_depth_map to be initialized with a 0
# if we want to test the case where estimated_depth is 0.
# The current sample_depth_map is all non-zero.
# The test for "Not enough valid depth correspondences found" in legacy path covers cases
# where estimated_depths are zero OR if the mask `depth_valid_mask` results in <3 points.
# For the `sample_lidar_points_cam_coords` fixture, it's okay as is, it provides points for "happy path" tests.
# Edge cases for zero depth from map are implicitly handled by `depth_valid_mask = estimated_depths > 1e-6`.
# The `test_align_depth_lidar_legacy_not_enough_correspondences` already tests this by providing a zero_depth_map.

# One small detail: sample_intrinsic_params cx, cy are 320, 240.
# sample_depth_map is 480x640.
# A point projecting to (cx,cy) would be depth_map[cy,cx] = depth_map[240,320]. This is correct.
# A point projecting to (420,290) is depth_map[290,420]. This is correct.
# A point projecting to (270,220) is depth_map[220,270]. This is correct.
# A point projecting to (150,150) is depth_map[150,150]. This is correct.
# All these indices are within the 480x640 bounds.

# The `valid_mask` in `align_depth_lidar` checks:
# (u >= 0) & (u < w) & (v >= 0) & (v < h) & (lidar_depths > 0)
# My `sample_lidar_points_cam_coords` includes:
# - A point projecting outside bounds: `[100, 100, 20]`. This will be filtered by `u < w` or `v < h`.
# - A point with Lidar depth <= 0: `[0.1, 0.1, -2.0]`. This will be filtered by `lidar_depths > 0`.
# So, from 6 points in `sample_lidar_points_cam_coords`, 2 are filtered out by `valid_mask`.
# This leaves 4 points for `estimated_depths = depth_map[valid_v, valid_u]`.
# Then `depth_valid_mask = estimated_depths > 1e-6`. If all 4 sampled depths are > 1e-6, we have 4 points.
# This is >= 3, so it passes the "Not enough valid depth correspondences found" check.
# This seems fine for the tests.

# # Plan items for this step:
# # - Create tests/point_cloud/test_depth_lidar_matching.py. (Done by creating file)
# # - Test project_points_to_image_plane(): (Done)
# #   - With valid 3D points and intrinsic parameters.
# #   - With points having Z=0 or Z<0.
# #   - With invalid input shapes.
# # - Test align_depth_lidar(): (Done)
# #   - Test the legacy path (use_solver=False) for "least_squares" and "median_ratio".
# #   - Test conditions that raise ValueErrors (not enough correspondences).
# #   - Mock DepthLidarSolver for use_solver=True and verify it's called correctly.

# # All items seem covered.I've written the tests for `depth_lidar_matching.py` and saved them into `tests/point_cloud/test_depth_lidar_matching.py`.

# The tests cover:
# *   `project_points_to_image_plane`:
#     *   Valid projections.
#     *   Points with Z=0 or Z<0 (behind the camera or on the image plane).
#     *   Invalid input shapes and empty inputs.
# *   `align_depth_lidar`:
#     *   **Legacy Path (`use_solver=False`)**:
#         *   "least_squares" method with calculations to verify the scale factor.
#         *   "median_ratio" method with calculations to verify the scale factor.
#         *   Cases where not enough valid correspondences are found, leading to `ValueError`.
#         *   Invalid method name.
#     *   **Modern Path (`use_solver=True`)**:
#         *   Mocking `DepthLidarSolver` to ensure it's initialized and its `solve` method is called with the correct arguments.
#         *   Verifying that the results from the mocked solver are correctly returned.
#         *   Testing that exceptions raised by the solver are propagated.

# Fixtures are used to provide sample data like intrinsic parameters, 3D points, depth maps, and LiDAR points in camera coordinates.

# Here's the content of `tests/point_cloud/test_depth_lidar_matching.py`:
