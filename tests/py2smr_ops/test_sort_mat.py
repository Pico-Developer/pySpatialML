import numpy as np

from securemr.py2smr import ops, trace, convert, verify
from .conftest import skip_if_no_device


@trace(inputs=["mat"], outputs=["sorted", "indices"])
def traced_sort_mat_row(mat):
    return ops.sort_mat(mat, axis="ROW", output_names=["sorted", "indices"])


@trace(inputs=["mat"], outputs=["sorted", "indices"])
def traced_sort_mat_col(mat):
    return ops.sort_mat(mat, axis="COLUMN", output_names=["sorted", "indices"])


def _run(mat, traced, device):
    (sorted_mat, indices), ctx = traced.trace(mat=mat)
    spec = convert(ctx)
    assert spec["operators"][0]["attrs"] in [["ROW"], ["COLUMN"]]
    assert "mode" not in spec["operators"][0]
    expected = {"sorted": sorted_mat, "indices": indices}
    return verify(spec, {"mat": mat}, expected_outputs=expected, device=device)


def test_sort_mat_row_host():
    mat = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]], dtype=np.float32)
    verification = _run(mat, traced_sort_mat_row, device=False)
    assert verification.success


@skip_if_no_device
def test_sort_mat_row_device():
    mat = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]], dtype=np.float32)
    verification = _run(mat, traced_sort_mat_row, device=True)
    if verification.error_message == "Device verification is not available":
        import pytest
        pytest.skip("Python verifier device execution is not available")
    assert verification.success


def test_sort_mat_col_host():
    mat = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]], dtype=np.float32)
    verification = _run(mat, traced_sort_mat_col, device=False)
    assert verification.success


@skip_if_no_device
def test_sort_mat_col_device():
    mat = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]], dtype=np.float32)
    verification = _run(mat, traced_sort_mat_col, device=True)
    if verification.error_message == "Device verification is not available":
        import pytest
        pytest.skip("Python verifier device execution is not available")
    assert verification.success
