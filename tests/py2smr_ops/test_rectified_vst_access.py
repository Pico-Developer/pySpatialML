import numpy as np
import pytest

from securemr.py2smr import ops, trace, convert, verify


@trace(inputs=[], outputs=["right", "left", "timestamp", "cam_matrix"])
def traced_camera_access():
    return ops.camera_access(output_names=["right", "left", "timestamp", "cam_matrix"])


def test_camera_access_host():
    (right, left, timestamp, cam_matrix), ctx = traced_camera_access.trace()
    spec = convert(ctx)
    expected = {
        "right": right,
        "left": left,
        "timestamp": timestamp,
        "cam_matrix": cam_matrix,
    }
    verification = verify(spec, {}, expected_outputs=expected)
    assert verification.success


def test_camera_access_device_skip():
    pytest.skip("camera_access depends on device camera stream; host stub cannot match device output")
