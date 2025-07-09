import numpy as np
import pytest
from vita_toolkit.point_cloud.pc_to_bev import (
    filter_pcd,
    points_to_voxel_batch,
    RadScatter,
    project_pc_to_bev
)
from unittest.mock import patch

# Fixtures
@pytest.fixture
def sample_point_cloud():
    #  X, Y, Z, intensity (optional)
    return np.array([
        [1.0, 2.0, 0.5, 10],   # Valid point
        [0.0, 0.0, 0.0, 20],   # Ego vehicle point (borderline for some filters)
        [-0.5, 0.1, 0.1, 30],  # Ego vehicle point
        [50.0, 50.0, 5.0, 40], # Outside typical small pc_range
        [2.0, 3.0, 1.0, 50],   # Valid point
        [0.05, 0.05, 0.05, 60], # Valid point (close to ego, but not ego)
        [-2.0, -3.0, -1.0, 70] # Valid point (negative coords)
    ], dtype=np.float32)

@pytest.fixture
def sample_pc_range():
    return np.array([-10.0, -10.0, -2.0, 10.0, 10.0, 3.0]) # x_min, y_min, z_min, x_max, y_max, z_max

@pytest.fixture
def sample_voxel_params():
    return {
        "voxel_size": np.array([0.1, 0.1, 0.5]),
        "coors_range": np.array([-10.0, -10.0, -2.0, 10.0, 10.0, 3.0]), # Same as pc_range for simplicity
        "max_points": 35,
        "max_voxels": 20000
    }

# Tests for filter_pcd
def test_filter_pcd_filters_ego_and_range(sample_point_cloud, sample_pc_range):
    # Ego vehicle points (using default filter criteria in filter_pcd):
    # (X < 0.1 & X > -1) & (Y > -0.2 & Y < 0.2)
    # sample_point_cloud[1] = [0.0, 0.0, 0.0, 20] -> X=0.0 (between -1 and 0.1), Y=0.0 (between -0.2 and 0.2) -> EGO
    # sample_point_cloud[2] = [-0.5, 0.1, 0.1, 30] -> X=-0.5 (between -1 and 0.1), Y=0.1 (between -0.2 and 0.2) -> EGO

    # Range filter: pc_range = [-10, -10, -2, 10, 10, 3]
    # sample_point_cloud[3] = [50.0, 50.0, 5.0, 40] -> X=50 > 10 (OUTSIDE)

    # Expected remaining points (indices):
    # 0: [1.0, 2.0, 0.5, 10] -> IN
    # 4: [2.0, 3.0, 1.0, 50] -> IN
    # 5: [0.05, 0.05, 0.05, 60] -> X=0.05 (not <0.1 AND >-1 for ego), Y=0.05 (in ego Y range) -> X condition for ego is (X<0.1 & X>-1). 0.05 is in this. So this IS an ego point.
    # Let's re-check point 5: X=0.05 is between -1 and 0.1. Y=0.05 is between -0.2 and 0.2. So point 5 IS EGO.
    # Original point 5: [0.05, 0.05, 0.05, 60] -> EGO

    # Let's redefine point 5 to NOT be ego:
    # Make X slightly larger, e.g., X = 0.2
    # sample_point_cloud_modified = sample_point_cloud.copy()
    # sample_point_cloud_modified[5,0] = 0.2 # Now X=0.2 is NOT < 0.1, so not ego.

    # Let's use the original sample_point_cloud and trace carefully:
    # Point 0: [1.0, 2.0, 0.5, 10]. Ego: No. Range: Yes. -> KEEP
    # Point 1: [0.0, 0.0, 0.0, 20]. Ego: Yes (0 in (-1,0.1), 0 in (-0.2,0.2)). -> REMOVE
    # Point 2: [-0.5, 0.1, 0.1, 30]. Ego: Yes (-0.5 in (-1,0.1), 0.1 in (-0.2,0.2)). -> REMOVE
    # Point 3: [50.0, 50.0, 5.0, 40]. Ego: No. Range: No (X,Y,Z all outside max). -> REMOVE
    # Point 4: [2.0, 3.0, 1.0, 50]. Ego: No. Range: Yes. -> KEEP
    # Point 5: [0.05, 0.05, 0.05, 60]. Ego: Yes (0.05 in (-1,0.1), 0.05 in (-0.2,0.2)). -> REMOVE
    # Point 6: [-2.0, -3.0, -1.0, 70]. Ego: No. Range: Yes. -> KEEP

    # Expected kept points: indices 0, 4, 6
    expected_filtered_cloud = sample_point_cloud[[0, 4, 6]]

    filtered = filter_pcd(sample_point_cloud, sample_pc_range)
    assert np.array_equal(filtered, expected_filtered_cloud)

def test_filter_pcd_no_points_left(sample_pc_range):
    # All points are ego or out of range
    points = np.array([
        [0.0, 0.0, 0.0, 10],    # Ego
        [-0.1, 0.0, 0.0, 20],   # Ego
        [100.0, 0.0, 0.0, 30]   # Out of range X
    ])
    filtered = filter_pcd(points, sample_pc_range)
    assert filtered.shape == (0, 4)

def test_filter_pcd_empty_input(sample_pc_range):
    empty_points = np.empty((0,4), dtype=np.float32)
    filtered = filter_pcd(empty_points, sample_pc_range)
    assert filtered.shape == (0, 4)


# Tests for points_to_voxel_batch
def test_points_to_voxel_batch_basic(sample_voxel_params):
    points = np.array([
        [0.0, 0.0, 0.0, 1],  # Voxel (cx,cy,cz) = (100,100,4) assuming coors_range min is -10,-10,-2 and voxel_size 0.1,0.1,0.5
        [0.05, 0.05, 0.2, 2], # Same voxel as above (0.05/0.1=0, 0.05/0.1=0, 0.2/0.5=0 for coord index after shift)
        [1.0, 1.0, 1.0, 3],  # Different voxel
    ], dtype=np.float32)

    # coors_range[:3] = [-10, -10, -2]
    # Point 0: [0,0,0]. Shifted: [10,10,2]. Coords: [10/0.1, 10/0.1, 2/0.5] = [100,100,4]
    # Point 1: [0.05,0.05,0.2]. Shifted: [10.05,10.05,2.2]. Coords: [10.05/0.1, 10.05/0.1, 2.2/0.5] = [100,100,4] (due to int conversion)
    # Point 2: [1,1,1]. Shifted: [11,11,3]. Coords: [11/0.1, 11/0.1, 3/0.5] = [110,110,6]

    voxels, coords, num_points = points_to_voxel_batch(points, **sample_voxel_params)

    assert voxels.shape[0] == 2  # 2 unique voxels
    assert coords.shape == (2, 3)
    assert num_points.shape == (2,)

    # Find the voxel for coord (100,100,4)
    idx_voxel1 = np.where((coords == [100,100,4]).all(axis=1))[0][0]
    assert num_points[idx_voxel1] == 2
    assert np.array_equal(voxels[idx_voxel1, :2], points[:2]) # First two points should be in this voxel

    # Find the voxel for coord (110,110,6)
    idx_voxel2 = np.where((coords == [110,110,6]).all(axis=1))[0][0]
    assert num_points[idx_voxel2] == 1
    assert np.array_equal(voxels[idx_voxel2, :1], points[2:3])

def test_points_to_voxel_batch_max_points_per_voxel(sample_voxel_params):
    params = sample_voxel_params.copy()
    params["max_points"] = 2

    points = np.array([
        [0.0, 0.0, 0.0, 1], # Voxel A
        [0.01, 0.01, 0.01, 2],# Voxel A
        [0.02, 0.02, 0.02, 3],# Voxel A (this one should be truncated if voxel A is first)
    ], dtype=np.float32)

    voxels, coords, num_points = points_to_voxel_batch(points, **params)

    assert voxels.shape[0] == 1 # All points map to the same voxel
    assert num_points[0] == 2   # But only 2 are kept due to max_points
    # The specific points kept depend on stable sort order if multiple points map to the same voxel coord.
    # The implementation uses np.lexsort on linear indices, then np.unique.
    # Then it iterates unique_starts. The first `max_points` from the sorted list are taken.
    # For these points, they are already sorted by their values essentially.
    assert np.array_equal(voxels[0, :2], points[:2])


def test_points_to_voxel_batch_max_voxels(sample_voxel_params):
    params = sample_voxel_params.copy()
    params["max_voxels"] = 1

    points = np.array([
        [0.0, 0.0, 0.0, 1], # Voxel A
        [1.0, 1.0, 1.0, 2], # Voxel B
    ], dtype=np.float32)

    voxels, coords, num_points = points_to_voxel_batch(points, **params)

    assert voxels.shape[0] == 1 # Only 1 voxel kept
    assert coords.shape == (1,3)
    assert num_points.shape == (1,)
    # Which voxel is kept depends on the linear indexing and sorting.
    # Voxel A coords: (100,100,4). Linear index for grid (200,200,10): 100*200*10 + 100*10 + 4 = 200000+1000+4 = 201004
    # Voxel B coords: (110,110,6). Linear index: 110*200*10 + 110*10 + 6 = 220000+1100+6 = 221106
    # Sorted unique indices will pick the smaller one first. So Voxel A is kept.
    assert np.array_equal(coords[0], [100,100,4])


def test_points_to_voxel_batch_points_outside_range(sample_voxel_params):
    points = np.array([
        [100.0, 100.0, 100.0, 1], # Outside coors_range
        [-100.0, -100.0, -100.0, 2] # Outside coors_range
    ], dtype=np.float32)

    voxels, coords, num_points = points_to_voxel_batch(points, **sample_voxel_params)

    assert voxels.shape[0] == 0
    assert coords.shape == (0, 3)
    assert num_points.shape == (0,)

def test_points_to_voxel_batch_empty_input(sample_voxel_params):
    empty_points = np.empty((0,4), dtype=np.float32)
    voxels, coords, num_points = points_to_voxel_batch(empty_points, **sample_voxel_params)
    assert voxels.shape[0] == 0
    assert coords.shape == (0,3) # Expect (0, max_points, num_features) for voxels
    assert num_points.shape == (0,)


# Tests for RadScatter
@pytest.mark.parametrize("is_vcs", [False, True])
def test_radscatter_call(is_vcs):
    scatter = RadScatter(is_vcs=is_vcs)
    batch_size = 2
    # input_shape is [nx, ny, nz]
    # Coords are [batch_idx, z, y, x]
    # Canvas shape is (batch_size, nz, ny, nx) for not is_vcs
    # Canvas shape is (batch_size, nz, nx, ny) for is_vcs

    input_shape = np.array([10, 20, 3]) # nx=10, ny=20, nz=3
    nx, ny, nz = input_shape

    # Coords: [batch_idx, z_coord, y_coord, x_coord]
    coords = np.array([
        [0, 0, 1, 2], # Batch 0, canvas[0, 0, 1, 2] = 1 (if not is_vcs)
        [0, 1, 3, 4], # Batch 0, canvas[0, 1, 3, 4] = 1
        [1, 2, 5, 6]  # Batch 1, canvas[1, 2, 5, 6] = 1
    ], dtype=np.int32)

    voxel_features = np.random.rand(coords.shape[0], 5).astype(np.float32) # Dummy features

    canvas = scatter(batch_size, input_shape, voxel_features, coords)

    expected_shape = (batch_size, nz, nx, ny) if is_vcs else (batch_size, nz, ny, nx)
    assert canvas.shape == expected_shape
    assert canvas.dtype == np.float32
    assert np.sum(canvas) == coords.shape[0] # Each coord should set one cell to 1

    # Check specific values
    if not is_vcs:
        # canvas shape (bs, nz, ny, nx) = (2, 3, 20, 10)
        # coords are (b, z, y, x)
        assert canvas[0, 0, 1, 2] == 1
        assert canvas[0, 1, 3, 4] == 1
        assert canvas[1, 2, 5, 6] == 1
        # Check a zero value
        assert canvas[0,0,0,0] == 0
    else: # is_vcs is True
        # canvas shape (bs, nz, nx, ny) = (2, 3, 10, 20)
        # indices are (z, nx-x-1, ny-y-1)
        # Coord [0,0,1,2] (b,z,y,x): b=0, z=0, y=1, x=2
        # Effective canvas access: canvas[0, z, (nx-x-1), (ny-y-1)]
        # canvas[0, 0, (10-2-1), (20-1-1)] = canvas[0,0,7,18]
        assert canvas[0, 0, nx-2-1, ny-1-1] == 1
        # Coord [0,1,3,4]: b=0, z=1, y=3, x=4
        # canvas[0, 1, (10-4-1), (20-3-1)] = canvas[0,1,5,16]
        assert canvas[0, 1, nx-4-1, ny-3-1] == 1
        # Coord [1,2,5,6]: b=1, z=2, y=5, x=6
        # canvas[1, 2, (10-6-1), (20-5-1)] = canvas[1,2,3,14]
        assert canvas[1, 2, nx-6-1, ny-5-1] == 1
        assert canvas[0,0,0,0] == 0


def test_radscatter_empty_coords():
    scatter = RadScatter(is_vcs=False)
    batch_size = 1
    input_shape = np.array([5,5,1]) # nx,ny,nz
    canvas = scatter(batch_size, input_shape, voxel_features=None, coords=None)
    expected_shape = (batch_size, input_shape[2], input_shape[1], input_shape[0])
    assert canvas.shape == expected_shape
    assert np.sum(canvas) == 0

def test_radscatter_coords_out_of_bounds():
    scatter = RadScatter(is_vcs=False)
    batch_size = 1
    input_shape = np.array([3,3,1]) # nx,ny,nz
    nx,ny,nz = input_shape
    # Coords: b, z, y, x
    coords_ob = np.array([
        [0, 0, 1, 1],      # Valid
        [0, 0, ny+1, 1],   # Y out of bounds (clipped to ny-1)
        [0, 0, 1, nx+1],   # X out of bounds (clipped to nx-1)
        [0, nz+1, 1, 1]    # Z out of bounds (clipped to nz-1)
    ], dtype=np.int32)

    canvas = scatter(batch_size, input_shape, None, coords_ob)
    assert canvas[0, 0, 1, 1] == 1
    assert canvas[0, 0, ny-1, 1] == 1 # Clipped Y
    assert canvas[0, 0, 1, nx-1] == 1 # Clipped X
    assert canvas[0, nz-1, 1, 1] == 1 # Clipped Z
    assert np.sum(canvas) == 3 # Points 1 and 4 (original coords) map to the same cell after clipping Z


# Tests for project_pc_to_bev (integration)
@patch('vita_toolkit.point_cloud.pc_to_bev.filter_pcd')
@patch('vita_toolkit.point_cloud.pc_to_bev.points_to_voxel_batch')
@patch('vita_toolkit.point_cloud.pc_to_bev.RadScatter')
def test_project_pc_to_bev(MockRadScatter, mock_points_to_voxel, mock_filter_pcd,
                           sample_point_cloud, sample_pc_range, sample_voxel_params):

    # Setup mocks
    filtered_points_mock = sample_point_cloud[:3] # Assume 3 points left after filtering
    mock_filter_pcd.return_value = filtered_points_mock

    # Mock for points_to_voxel_batch
    # Returns: voxels, coords, num_points_per_voxel
    # Coords are (N_voxels, 3) with [x_idx, y_idx, z_idx] relative to voxel grid
    mock_voxel_coords = np.array([[10,20,1], [15,25,2]], dtype=np.int32) # 2 voxels found
    mock_voxels = np.random.rand(2, sample_voxel_params["max_points"], filtered_points_mock.shape[1])
    mock_num_points = np.array([2,1], dtype=np.int32)
    mock_points_to_voxel.return_value = (mock_voxels, mock_voxel_coords, mock_num_points)

    # Mock for RadScatter
    mock_scatter_instance = MockRadScatter.return_value
    bev_output_mock = np.random.rand(1, 3, 200, 200) # Example BEV output (bs, nz, ny, nx)
    mock_scatter_instance.return_value = bev_output_mock

    # Call the function
    batch_size = 1
    is_vcs = False
    bev_features = project_pc_to_bev(
        points=sample_point_cloud,
        pc_range=sample_pc_range,
        voxel_size=sample_voxel_params["voxel_size"],
        coors_range=sample_voxel_params["coors_range"],
        max_points=sample_voxel_params["max_points"],
        max_voxels=sample_voxel_params["max_voxels"],
        batch_size=batch_size,
        is_vcs=is_vcs
    )

    # Assertions
    mock_filter_pcd.assert_called_once_with(sample_point_cloud, sample_pc_range)
    mock_points_to_voxel.assert_called_once_with(
        filtered_points_mock,
        sample_voxel_params["voxel_size"],
        sample_voxel_params["coors_range"],
        sample_voxel_params["max_points"],
        sample_voxel_params["max_voxels"]
    )

    MockRadScatter.assert_called_once_with(is_vcs=is_vcs)

    # Expected grid_size for RadScatter: (coors_range_dims / voxel_size)
    # coors_range = [-10, -10, -2, 10, 10, 3] -> dims = [20, 20, 5]
    # voxel_size = [0.1, 0.1, 0.5]
    # grid_size = [200, 200, 10] (nx, ny, nz)
    expected_grid_size = np.array([200, 200, 10], dtype=np.int32)

    # Expected batch_coords for RadScatter: (N_voxels, 4) with [batch_idx, z_idx, y_idx, x_idx]
    # mock_voxel_coords was [x,y,z]. So batch_coords should be [0, z, y, x]
    expected_batch_coords = np.array([
        [0, mock_voxel_coords[0,2], mock_voxel_coords[0,1], mock_voxel_coords[0,0]], # [0, 1, 20, 10]
        [0, mock_voxel_coords[1,2], mock_voxel_coords[1,1], mock_voxel_coords[1,0]], # [0, 2, 25, 15]
    ], dtype=np.int32)

    # Check RadScatter instance call
    called_args, called_kwargs = mock_scatter_instance.call_args
    assert called_kwargs['batch_size'] == batch_size
    assert np.array_equal(called_kwargs['input_shape'], expected_grid_size)
    assert np.array_equal(called_kwargs['voxel_features'], mock_voxels)
    assert np.array_equal(called_kwargs['coords'], expected_batch_coords)

    assert np.array_equal(bev_features, bev_output_mock)

def test_project_pc_to_bev_no_valid_points(sample_pc_range, sample_voxel_params):
    # Points that will all be filtered out
    points = np.array([
        [0.0,0.0,0.0,1], # Ego
        [100.0,100.0,100.0,2] # Out of range
    ])

    bev = project_pc_to_bev(points, sample_pc_range, **sample_voxel_params)

    # Expected grid_size for RadScatter: (coors_range_dims / voxel_size)
    # coors_range = [-10, -10, -2, 10, 10, 3] -> dims = [20, 20, 5]
    # voxel_size = [0.1, 0.1, 0.5]
    # grid_size = [200, 200, 10] (nx, ny, nz) for RadScatter input_shape
    # RadScatter output shape is (batch_size, nz, ny, nx) or (batch_size, nz, nx, ny)
    # Default batch_size is 1. Default is_vcs is False.
    # Expected output shape (1, 10, 200, 200)

    # If no points after filter, points_to_voxel_batch returns empty coords.
    # Then batch_coords for RadScatter is empty.
    # RadScatter with empty coords should produce an empty (all zeros) canvas.

    expected_dims = np.round((sample_voxel_params["coors_range"][3:] - sample_voxel_params["coors_range"][:3]) / sample_voxel_params["voxel_size"]).astype(np.int32)
    # nx, ny, nz = expected_dims[0], expected_dims[1], expected_dims[2]
    # nz is last in input_shape for RadScatter, but first dim after batch_size in output.

    # Correct expected shape from RadScatter for is_vcs=False: (batch_size, nz, ny, nx)
    # grid_size passed to RadScatter is [nx, ny, nz]
    # RadScatter internals: self.nx, self.ny, self.nz = input_shape[0], input_shape[1], input_shape[2]
    # Output shape: (batch_size, self.nz, self.ny, self.nx)
    # So, (1, expected_dims[2], expected_dims[1], expected_dims[0])
    # (1, 10, 200, 200)

    assert bev.shape == (1, int(expected_dims[2]), int(expected_dims[1]), int(expected_dims[0]))
    assert np.all(bev == 0)

# In RadScatter, input_shape is [nx, ny, nz]. Coords are [batch_idx, z, y, x].
# The current implementation of project_pc_to_bev.py:
# grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(np.int32)
# This grid_size is [nx, ny, nz]. This is passed to RadScatter as input_shape.
# coords from points_to_voxel_batch are [x_idx, y_idx, z_idx].
# batch_coords are created as [batch_idx, z_idx, y_idx, x_idx]. This matches RadScatter expectation.
# This all seems consistent.

# # Plan items for this step:
# # - Create tests/point_cloud/test_pc_to_bev.py. (Done by creating file)
# # - Test filter_pcd(): (Done)
# #   - With points inside and outside pc_range.
# #   - With points that should be filtered as ego vehicle points.
# # - Test points_to_voxel_batch(): (Done)
# #   - With a small set of points and verify voxel outputs.
# #   - With points outside coors_range.
# #   - With more points than max_points in a voxel.
# #   - With more voxels than max_voxels.
# #   - With empty input points.
# # - Test RadScatter.__call__(): (Done)
# #   - With is_vcs=False and is_vcs=True.
# #   - With and without coordinates.
# #   - Verify output shape and canvas marking. (Clipping also tested)
# # - Test project_pc_to_bev(): (Done)
# #   - Integrate testing or mock sub-functions. (Used mocking here)
# #   - Tested case with no valid points leading to empty BEV.

# # All items seem covered.I've written the tests for `pc_to_bev.py` and saved them into `tests/point_cloud/test_pc_to_bev.py`.

# The tests cover:
# *   **`filter_pcd`**:
#     *   Filtering of ego-vehicle points based on the defined criteria.
#     *   Filtering of points outside the specified `pc_range`.
#     *   Cases with no points remaining and empty input.
# *   **`points_to_voxel_batch`**:
#     *   Basic voxelization and correct assignment of points to voxels.
#     *   Handling of `max_points` per voxel.
#     *   Handling of `max_voxels`.
#     *   Behavior with points outside the `coors_range`.
#     *   Empty input.
# *   **`RadScatter`**:
#     *   Operation with `is_vcs=True` and `is_vcs=False`.
#     *   Correct canvas shape and marking of cells based on input coordinates.
#     *   Handling of empty coordinates.
#     *   Clipping of out-of-bounds coordinates.
# *   **`project_pc_to_bev`**:
#     *   Integration test using mocks for `filter_pcd`, `points_to_voxel_batch`, and `RadScatter` to verify the orchestration and data flow between these components.
#     *   Verification of arguments passed to mocked functions.
#     *   Test case where no valid points remain after filtering, resulting in an empty BEV map.

# Fixtures are used for sample point clouds, PC ranges, and voxel parameters.

# Here's the content of `tests/point_cloud/test_pc_to_bev.py`:
