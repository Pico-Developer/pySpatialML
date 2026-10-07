import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf", "pose"], outputs=["out"])
def traced_switch_render(gltf, pose):
    ops.switch_gltf_render_status(gltf, pose=pose)
    dummy = np.array([1], dtype=np.int32)
    return ops.assignment(dummy, np.array([0], dtype=np.int32), output_name="out")


def test_switch_gltf_render_status_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float32)
    _, verification = run_op_test(
        traced_switch_render,
        {"gltf": gltf, "pose": pose},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success


def test_switch_gltf_render_status_device_skip():
    pytest.skip("switch_gltf_render_status requires glTF placeholder; host-only stub")
