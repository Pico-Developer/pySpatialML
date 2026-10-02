import numpy as np
import cv2
import pytest

from securemr.py2smr import ops, trace, convert, verify
from .conftest import skip_if_no_device


@trace(inputs=["object_points", "image_points", "camera_matrix"], outputs=["rvec", "tvec"])
def traced_solve_p_n_p(object_points, image_points, camera_matrix):
    return ops.solve_p_n_p(object_points, image_points, camera_matrix, output_names=["rvec", "tvec"])


def _build_inputs():
    obj = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float32)
    rvec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    tvec = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    cam = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, cam, None)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    return obj, img_pts, cam.astype(np.float32)


def test_solve_p_n_p_host():
    obj, img, cam = _build_inputs()
    (rvec, tvec), ctx = traced_solve_p_n_p.trace(
        object_points=obj, image_points=img, camera_matrix=cam
    )
    spec = convert(ctx)
    expected = {"rvec": rvec, "tvec": tvec}
    verification = verify(spec, {"object_points": obj, "image_points": img, "camera_matrix": cam}, expected_outputs=expected)
    assert verification.success


@skip_if_no_device
def test_solve_p_n_p_device():
    obj, img, cam = _build_inputs()
    (rvec, tvec), ctx = traced_solve_p_n_p.trace(
        object_points=obj, image_points=img, camera_matrix=cam
    )
    spec = convert(ctx)
    expected = {"rvec": rvec, "tvec": tvec}
    verification = verify(
        spec,
        {"object_points": obj, "image_points": img, "camera_matrix": cam},
        expected_outputs=expected,
        device=True,
    )
    if verification.error_message == "Device verification is not available":
        pytest.skip("Python verifier device execution is not available")
    assert verification.success
