#!/usr/bin/env python3
"""
Pure LMDB Reader - Read data from LMDB databases without preprocessing or PyTorch Dataset

This module provides simple functions to read data from LMDB databases used in the Vita-VLN project.
It supports reading meta, lidar, and camera data without any preprocessing or PyTorch dependencies.
"""

import os
import lmdb
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class PureLMDBReader:
    """
    A simple LMDB reader that provides direct access to stored data without preprocessing.
    """
    
    def __init__(self, data_folder: str, data_types: List[str] = ["meta", "lidar", "camera"]):
        """
        Initialize the LMDB reader.
        
        Args:
            data_folder: Path to the folder containing LMDB databases
            data_types: List of data types to read (meta, lidar, camera)
        """
        self.data_folder = data_folder
        self.data_types = data_types
        self.db_paths = {}
        self.envs = {}
        
        # Set up database paths
        for data_type in data_types:
            db_path = os.path.join(data_folder, data_type)
            if not os.path.exists(db_path):
                print(f"Warning: {data_type} LMDB database does not exist: {db_path}")
                continue
            self.db_paths[data_type] = db_path
        
        # Open LMDB environments
        self._open_environments()
        
        # Get all available keys
        self.all_keys = self._get_all_keys()
        print(f"Found {len(self.all_keys)} keys in the database")
    
    def _open_environments(self):
        """Open LMDB environments for reading."""
        for data_type, db_path in self.db_paths.items():
            try:
                self.envs[data_type] = lmdb.open(db_path, readonly=True, max_readers=1024)
                print(f"Opened {data_type} database: {db_path}")
            except Exception as e:
                print(f"Failed to open {data_type} database: {e}")
    
    def _get_all_keys(self) -> List[str]:
        """Get all keys from the main database."""
        # Use the first available database to get keys
        main_db_type = None
        for db_type in ["meta", "lidar", "camera"]:
            if db_type in self.envs:
                main_db_type = db_type
                break
        
        if main_db_type is None:
            print("No valid database found")
            return []
        
        all_keys = []
        with self.envs[main_db_type].begin() as txn:
            cursor = txn.cursor()
            all_keys = [key.decode() for key in cursor.iternext(keys=True, values=False)]
        
        # Sort by timestamp (assuming keys end with timestamp)
        all_keys.sort(key=lambda x: x.split("_")[-1])
        return all_keys
    
    def get_keys_by_prefix(self, prefix: str) -> List[str]:
        """
        Get keys that start with a specific prefix.
        
        Args:
            prefix: Prefix to filter keys
            
        Returns:
            List of keys matching the prefix
        """
        return [key for key in self.all_keys if key.startswith(prefix)]
    
    def read_meta_data(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read metadata for a specific key.
        
        Args:
            key: The key to read
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        if "meta" not in self.envs:
            print("Meta database not available")
            return None
        
        with self.envs["meta"].begin() as txn:
            data_bytes = txn.get(key.encode())
            if data_bytes:
                return pickle.loads(data_bytes)
            else:
                print(f"Key {key} not found in meta database")
                return None
    
    def read_lidar_data(self, key: str) -> Optional[np.ndarray]:
        """
        Read lidar point cloud data for a specific key.
        
        Args:
            key: The key to read
            
        Returns:
            Numpy array containing point cloud data or None if not found
        """
        if "lidar" not in self.envs:
            print("Lidar database not available")
            return None
        
        with self.envs["lidar"].begin() as txn:
            data_bytes = txn.get(key.encode())
            if data_bytes:
                lidar_data = pickle.loads(data_bytes)
                return lidar_data["point_cloud"]
            else:
                print(f"Key {key} not found in lidar database")
                return None
    
    def read_camera_data(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read camera image data for a specific key.
        
        Args:
            key: The key to read
            
        Returns:
            Dictionary containing image data or None if not found
        """
        if "camera" not in self.envs:
            print("Camera database not available")
            return None
        
        with self.envs["camera"].begin() as txn:
            data_bytes = txn.get(key.encode())
            if data_bytes:
                return pickle.loads(data_bytes)
            else:
                print(f"Key {key} not found in camera database")
                return None
    
    def read_all_data(self, key: str) -> Dict[str, Any]:
        """
        Read all available data for a specific key.
        
        Args:
            key: The key to read
            
        Returns:
            Dictionary containing all available data types
        """
        result = {"key": key}
        
        # Read meta data
        meta_data = self.read_meta_data(key)
        if meta_data:
            result["meta"] = meta_data
        
        # Read lidar data
        lidar_data = self.read_lidar_data(key)
        if lidar_data is not None:
            result["lidar"] = lidar_data
        
        # Read camera data
        camera_data = self.read_camera_data(key)
        if camera_data:
            result["camera"] = camera_data
        
        return result
    
    def iterate_all_data(self):
        """
        Generator that yields all data in the database.
        
        Yields:
            Dictionary containing all data for each key
        """
        for key in self.all_keys:
            yield self.read_all_data(key)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the database.
        
        Returns:
            Dictionary containing database statistics
        """
        stats = {
            "total_keys": len(self.all_keys),
            "available_databases": list(self.envs.keys()),
            "sample_keys": self.all_keys[:5] if self.all_keys else []
        }
        
        # Get size information for each database
        for db_type, env in self.envs.items():
            with env.begin() as txn:
                stats[f"{db_type}_size"] = txn.stat()["entries"]
        
        return stats
    
    def close(self):
        """Close all LMDB environments."""
        for env in self.envs.values():
            env.close()
        self.envs.clear()
        print("All LMDB environments closed")
    
    def __del__(self):
        """Destructor to ensure environments are closed."""
        self.close()


def simple_read_example(data_folder: str):
    """
    Simple example of how to use the PureLMDBReader.
    
    Args:
        data_folder: Path to the folder containing LMDB databases
    """
    # Initialize reader
    reader = PureLMDBReader(data_folder)
    
    # Get database statistics
    stats = reader.get_data_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Read first few samples
    print("\nReading first 3 samples:")
    for i, key in enumerate(reader.all_keys[:3]):
        print(f"\nSample {i+1} - Key: {key}")
        
        # Read meta data
        meta = reader.read_meta_data(key)
        if meta:
            print(f"  Meta keys: {list(meta.keys())}")
            if "uwb_position" in meta:
                print(f"  UWB position: {meta['uwb_position']}")
        
        # Read lidar data
        lidar = reader.read_lidar_data(key)
        if lidar is not None:
            print(f"  Lidar shape: {lidar.shape}")
            print(f"  Lidar range: [{lidar.min():.3f}, {lidar.max():.3f}]")
        
        # Read camera data
        camera = reader.read_camera_data(key)
        if camera:
            print(f"  Camera keys: {list(camera.keys())}")
            if "image" in camera:
                image = camera["image"]
                print(f"  Image shape: {np.array(image).shape}")
    
    # Close reader
    reader.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pure_lmdb_reader.py <data_folder>")
        print("Example: python pure_lmdb_reader.py /path/to/lmdb/data")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    simple_read_example(data_folder)
