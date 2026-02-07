"""PointCloud2 decoder for G1 SLAM point clouds."""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PointCloud2Decoder:
    """Decode ROS2 PointCloud2 binary data into XYZ points."""

    def decode(self, binary_data: bytes, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            data_len = len(binary_data)
            if data_len == 0:
                return {"point_count": 0, "points": np.array([])}

            # G1 SLAM commonly uses 12-byte points (6 int16 values)
            if data_len % 12 == 0:
                return self._decode_int16_12byte(binary_data)

            # Try XYZI float32 (16 bytes per point)
            if data_len % 16 == 0:
                return self._decode_xyzi_float32(binary_data)

            # Try XYZ float32 (12 bytes per point)
            if data_len % 12 == 0:
                return self._decode_xyz_float32(binary_data)

            return self._decode_int16_12byte(binary_data)
        except Exception as exc:
            logger.error("PointCloud2 decode error: %s", exc, exc_info=True)
            return {"point_count": 0, "points": np.array([]), "error": str(exc)}

    def _decode_xyz_float32(self, binary_data: bytes) -> Dict[str, Any]:
        num_points = len(binary_data) // 12
        points = np.frombuffer(binary_data, dtype=np.float32).reshape((num_points, 3))
        return {"point_count": num_points, "points": points, "format": "xyz_float32"}

    def _decode_xyzi_float32(self, binary_data: bytes) -> Dict[str, Any]:
        num_points = len(binary_data) // 16
        data = np.frombuffer(binary_data, dtype=np.float32).reshape((num_points, 4))
        return {
            "point_count": num_points,
            "points": data[:, :3],
            "intensities": data[:, 3],
            "format": "xyzi_float32",
        }

    def _decode_int16_12byte(self, binary_data: bytes) -> Dict[str, Any]:
        if len(binary_data) % 12 != 0:
            return {"point_count": 0, "points": np.array([])}

        num_points = len(binary_data) // 12
        data_int16 = np.frombuffer(binary_data, dtype=np.int16).reshape((num_points, 6))
        xyz_raw = data_int16[:, :3].astype(np.float32)
        points = xyz_raw / 1000.0
        return {
            "point_count": num_points,
            "points": points,
            "format": "g1_slam_int16_12byte",
        }
