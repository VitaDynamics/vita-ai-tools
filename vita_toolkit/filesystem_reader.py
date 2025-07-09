#!/usr/bin/env python3
"""
Filesystem Reader - Read multi-modal sensor data from filesystem


This module provides functionality to read aligned multi-modal data including:
- Point clouds (.pcd files)
- RGB images (left/right cameras, .jpeg files)
- Depth images (.npy files)

Files are organized by timestamp and automatically aligned.
"""

import re
import numpy as np
import open3d as o3d
from PIL import Image
from typing import Dict, List, Optional, Generator, Any
from pathlib import Path


class FilesystemReader:
    """
    Reader for multi-modal sensor data organized by timestamp in filesystem.

    Expected file structure:
    data_folder/
     123.pcd
     123_depth.npy
     123_left.jpeg
     123_right.jpeg
     456.pcd
     456_depth.npy
     456_left.jpeg
     456_right.jpeg

    Where 123, 456 etc. are timestamps.
    """

    def __init__(self, data_folder: str, data_types: Optional[List[str]] = None):
        """
        Initialize the filesystem reader.

        Args:
            data_folder: Path to folder containing timestamped sensor data
            data_types: List of data types to read. Available: 'point_clouds', 'depth', 'left_image', 'right_image'
                       If None, reads all available types.
        """
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {data_folder}")

        self.available_types = ["point_clouds", "depth", "left_image", "right_image"]
        if data_types is None:
            self.data_types = self.available_types
        else:
            invalid_types = set(data_types) - set(self.available_types)
            if invalid_types:
                raise ValueError(
                    f"Invalid data types: {invalid_types}. Available: {self.available_types}"
                )
            self.data_types = data_types
            
        self.patterns = {
            "point_clouds": r"(\d+)\.pcd$",
            "depth": r"(\d+)_depth\.npy$",
            "left_image": r"(\d+)\.jpeg$",
            "right_image": r"(\d+)_right\.jpeg$",
        }

        # Discover all available timestamps
        self.timestamps = self._discover_timestamps()
        print(f"Found {len(self.timestamps)} timestamps in {data_folder}")

    def _discover_timestamps(self) -> List[str]:
        """
        Discover all timestamps that have at least one data file.

        Returns:
            Sorted list of timestamps as strings
        """
        timestamps = set()

        for data_type in self.data_types:
            if data_type in self.patterns:
                pattern = self.patterns[data_type]
                for file_path in self.data_folder.glob("*"):
                    match = re.search(pattern, file_path.name)
                    if match:
                        timestamps.add(match.group(1))

        return sorted(list(timestamps))

    def _get_file_path(self, timestamp: str, data_type: str) -> Optional[Path]:
        """
        Get file path for specific timestamp and data type.

        Args:
            timestamp: Timestamp string
            data_type: Type of data ('point_clouds', 'depth', 'left_image', 'right_image')

        Returns:
            Path to file or None if not found
        """
        # TODO: we need a more elegant way to do this intead of use regex
        def pattern_to_filename(pattern: str, timestamp: str) -> str:
            # Replace (\d+) with the timestamp, and remove any regex-specific characters
            # e.g., r"(\d+)_right\.jpeg$" -> f"{timestamp}_right.jpeg"
            filename = re.sub(r"\(\\d\+\)", timestamp, pattern)
            filename = filename.replace("\\.", ".").replace("$", "")
            return filename

        if data_type not in self.patterns:
            return None

        filename = pattern_to_filename(self.patterns[data_type], timestamp)
        file_path = self.data_folder / filename
        return file_path if file_path.exists() else None

    def _load_point_cloud(self, file_path: Path) -> Optional[np.ndarray]:
        """Load point cloud from .pcd file."""
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            return np.asarray(pcd.points)
        except Exception as e:
            print(f"Failed to load point cloud {file_path}: {e}")
            return None

    def _load_depth(self, file_path: Path) -> Optional[np.ndarray]:
        """Load depth data from .npy file."""
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Failed to load depth {file_path}: {e}")
            return None

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Load image from .jpeg file."""
        try:
            image = Image.open(file_path)
            return np.array(image)
        except Exception as e:
            print(f"Failed to load image {file_path}: {e}")
            return None

    def read_frame(self, timestamp: str) -> Dict[str, Any]:
        """
        Read all available data for a specific timestamp.

        Args:
            timestamp: Timestamp string

        Returns:
            Dictionary with 'timestamp' and requested data types as numpy arrays
        """
        frame_data: Dict[str, Any] = {"timestamp": timestamp}

        loaders = {
            "point_clouds": self._load_point_cloud,
            "depth": self._load_depth,
            "left_image": self._load_image,
            "right_image": self._load_image,
        }

        for data_type in self.data_types:
            file_path = self._get_file_path(timestamp, data_type)
            if file_path and data_type in loaders:
                data = loaders[data_type](file_path)
                frame_data[data_type] = data
            else:
                frame_data[data_type] = None

        return frame_data

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        yield from self.iterate_frames()
    def iterate_frames(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields frame data for all timestamps.

        Yields:
            Dictionary containing timestamp and all requested data types
        """
        for timestamp in self.timestamps:
            yield self.read_frame(timestamp)

    def get_available_timestamps(self) -> List[str]:
        """Get list of all available timestamps."""
        return self.timestamps.copy()

    def get_frame_count(self) -> int:
        """Get total number of available frames."""
        return len(self.timestamps)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_frames": len(self.timestamps),
            "data_folder": str(self.data_folder),
            "requested_data_types": self.data_types,
            "available_files_per_type": {},
        }

        # Count available files for each data type
        for data_type in self.data_types:
            count = 0
            for timestamp in self.timestamps:
                if self._get_file_path(timestamp, data_type):
                    count += 1
            stats["available_files_per_type"][data_type] = count

        return stats


def read_sensor_data(
    data_folder: str, data_types: Optional[List[str]] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Convenience function to read multi-modal sensor data from filesystem.

    Args:
        data_folder: Path to folder containing timestamped sensor data
        data_types: List of data types to read. If None, reads all available types.

    Yields:
        Dictionary containing timestamp and sensor data as numpy arrays

    Example:
        >>> for frame in read_sensor_data('/path/to/data', ['point_clouds', 'left_image']):
        ...     timestamp = frame['timestamp']
        ...     points = frame['point_clouds']  # numpy array (N, 3)
        ...     image = frame['left_image']     # numpy array (H, W, 3)
    """
    reader = FilesystemReader(data_folder, data_types)
    yield from reader.iterate_frames()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filesystem_reader.py <data_folder> [data_types...]")
        print(
            "Example: python filesystem_reader.py /path/to/data point_clouds left_image"
        )
        print("Available data types: point_clouds, depth, left_image, right_image")
        sys.exit(1)

    data_folder = sys.argv[1]
    data_types = sys.argv[2:] if len(sys.argv) > 2 else None

    # Create reader and show statistics
    reader = FilesystemReader(data_folder, data_types)
    stats = reader.get_statistics()

    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Read first few frames
    print(f"\nReading first 3 frames:")
    for i, frame in enumerate(reader.iterate_frames()):
        if i >= 3:
            break

        print(f"\nFrame {i + 1} - Timestamp: {frame['timestamp']}")
        for data_type, data in frame.items():
            if data_type == "timestamp":
                continue
            if data is not None:
                print(f"  {data_type}: shape {data.shape}, dtype {data.dtype}")
            else:
                print(f"  {data_type}: None (file not found)")
