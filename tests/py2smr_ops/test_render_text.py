import numpy as np
import pytest

from securemr.py2smr import convert, ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf"], outputs=["out"])
def traced_render_text(gltf):
    ops.render_text(gltf, config="bold#en-us#256#64", text="hello")
    dummy = np.array([1], dtype=np.int32)
    return ops.assignment(dummy, np.array([0], dtype=np.int32), output_name="out")


def test_render_text_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    _, verification = run_op_test(
        traced_render_text,
        {"gltf": gltf},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success


def test_render_text_serializes_only_spec_fields():
    _, ctx = traced_render_text.trace(gltf=np.zeros((1,), dtype=np.uint8))
    render_spec = next(operator for operator in convert(ctx)["operators"]
                       if operator["type"].endswith("RENDER_TEXT_PICO"))
    assert render_spec["type"] == "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"
    assert render_spec["inputs"] == [
        {"tensor": "tensor_0"}, {"tensor": "tensor_1"},
        {"tensor": "tensor_2"}, {"tensor": "gltf"},
        {"tensor": "tensor_3"}, {"tensor": "tensor_4"},
    ]
    assert render_spec["outputs"] == []
    assert render_spec["attrs"] == ["bold#en-us#256#64"]
    assert set(render_spec) == {"type", "inputs", "outputs", "attrs"}


def test_render_text_device_skip():
    pytest.skip("render_text requires glTF placeholder; host-only stub")
