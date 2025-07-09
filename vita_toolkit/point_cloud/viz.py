import numpy as np


def visualize_point_cloud(
    points: np.ndarray, colors: np.ndarray | None = None, *, backend: str = "rerun"
) -> None:
    """Visualize point cloud using rerun."""
    if backend != "rerun":
        raise ValueError("Only rerun backend is supported")

    import rerun as rr

    rr.init("point_cloud", spawn=True)

    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        rr.log("points", rr.Points3D(points, colors=colors))
    else:
        rr.log("points", rr.Points3D(points))


def visualize_bev(bev_map: np.ndarray, *, backend: str = "rerun") -> None:
    """Visualize bird's eye view map using rerun."""
    if backend != "rerun":
        raise ValueError("Only rerun backend is supported")

    import rerun as rr

    rr.init("bev_map", spawn=True)

    bev_map = np.asarray(bev_map)
    rr.log("bev", rr.Image(bev_map))
