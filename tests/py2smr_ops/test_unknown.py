import numpy as np
import pytest

from securemr.py2smr import ops, trace
from securemr.py2smr.verifier import run_pipeline_python
from .conftest import run_op_test


@trace(inputs=["tensor"], outputs=["out"])
def traced_unknown(tensor):
    return ops.unknown(tensor, output_name="out")


def test_unknown_host():
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises((ValueError, NotImplementedError), match="UNKNOWN|not supported"):
        run_op_test(
            traced_unknown,
            {"tensor": tensor},
            expected_output_name="out",
            test_device=False,
        )


def test_unknown_device_skip():
    pytest.skip("unknown operator type not supported on device")


def test_unknown_operator_is_rejected():
    spec = {
        "tensors": {},
        "operators": [{"type": "vendor_custom", "inputs": [], "outputs": []}],
        "inputs": [],
        "outputs": [],
    }
    with pytest.raises(ValueError, match="unknown operator type"):
        run_pipeline_python(spec, {})
