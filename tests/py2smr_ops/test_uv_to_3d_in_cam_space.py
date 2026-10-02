import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["uv", "timestamp", "camera_matrix", "left_image", "right_image"], outputs=["points"])
def traced_uv_to_3d(uv, timestamp, camera_matrix, left_image, right_image):
    return ops.uv_to_3d_in_camera_space(uv, timestamp, camera_matrix, left_image, right_image, output_name="points")


def test_uv_to_3d_in_camera_space_host():
    uv = np.array([[10, 12], [20, 30]], dtype=np.int32)
    timestamp = np.zeros((1, 4), dtype=np.int32)
    camera_matrix = np.eye(3, dtype=np.float32)
    left_image = np.zeros((2, 2, 3), dtype=np.uint8)
    right_image = np.zeros((2, 2, 3), dtype=np.uint8)
    _, verification = run_op_test(
        traced_uv_to_3d,
        {
            "uv": uv,
            "timestamp": timestamp,
            "camera_matrix": camera_matrix,
            "left_image": left_image,
            "right_image": right_image,
        },
        expected_output_name="points",
        test_device=False,
    )
    assert verification.success


def test_uv_to_3d_in_camera_space_malformed_uv_host_defaults():
    uv = np.zeros((1, 21), dtype=np.float32)
    timestamp = np.zeros((1, 4), dtype=np.int32)
    camera_matrix = np.eye(3, dtype=np.float32)
    left_image = np.zeros((2, 2, 3), dtype=np.uint8)
    right_image = np.zeros((2, 2, 3), dtype=np.uint8)

    points = ops.uv_to_3d_in_camera_space(uv, timestamp, camera_matrix, left_image, right_image)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points, 0.0)


def test_uv_to_3d_in_camera_space_device_skip():
    pytest.skip("uv_to_3d_in_camera_space relies on depth sensor data; host stub cannot match device output")


def test_uv_to_3d_in_camera_space_uses_camera_and_stereo_data():
    uv = np.array([[3, 0]], dtype=np.float32)
    timestamp = np.array([1, 0, 0, 0], dtype=np.int32)
    camera_matrix = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    left_image = np.arange(50, dtype=np.uint8).reshape(5, 10)
    right_image = np.zeros_like(left_image)
    right_image[:, 0:8] = left_image[:, 2:10]

    points = ops.uv_to_3d_in_camera_space(uv, timestamp, camera_matrix, left_image, right_image)

    np.testing.assert_allclose(points, [[0.09, 0.0, 0.3]], rtol=1e-5, atol=1e-5)
