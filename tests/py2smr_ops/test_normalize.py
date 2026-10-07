# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for normalize operator (NORMALIZE)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from .conftest import run_op_test, skip_if_no_device, DEVICE_AVAILABLE


class TestNormalizeOp:
    """Tests for the normalize operation (L2 normalization)."""

    def test_normalize_basic(self):
        """Test basic L2 normalization."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        # [3, 4] -> norm = 5 -> [0.6, 0.8]
        input_arr = np.array([[3.0, 4.0]], dtype=np.float32)
        result, verification = run_op_test(
            normalize_vec,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.array([[0.6, 0.8]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalize_serializes_default_mode(self):
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        _, ctx = normalize_vec.trace(x=np.array([[3.0, 4.0]], dtype=np.float32))
        spec = convert(ctx)

        assert "attrs" not in spec["operators"][0]

    def test_normalize_accepts_named_modes(self):
        value = np.array([[1.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(ops.normalize(value, normalize_type="L1"), value / 3.0)
        np.testing.assert_allclose(ops.normalize(value, normalize_type="INF"), value / 2.0)
        np.testing.assert_allclose(ops.normalize(value, normalize_type="MINMAX"), [[1.0, 0.0]])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_normalize_accepts_alpha_beta(self, dtype):
        value = np.array([[3.0, 4.0]], dtype=np.float32)
        alpha_beta = np.array([2.0, 7.0], dtype=dtype)

        result = ops.normalize(value, normalize_type="L2", alpha_beta=alpha_beta)

        np.testing.assert_allclose(result, [[1.2, 1.6]], rtol=1e-6)

    def test_normalize_alpha_beta_changes_minmax_range(self):
        value = np.array([[1.0, 3.0]], dtype=np.float32)

        result = ops.normalize(
            value,
            normalize_type="MINMAX",
            alpha_beta=np.array([-2.0, 2.0], dtype=np.float64),
        )

        np.testing.assert_allclose(result, [[-2.0, 2.0]], rtol=1e-6)

    @pytest.mark.parametrize(
        "alpha_beta",
        [np.array([1, 0], dtype=np.int32), np.array([1.0], dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32)],
    )
    def test_normalize_rejects_invalid_alpha_beta(self, alpha_beta):
        with pytest.raises(ValueError, match="alpha_beta"):
            ops.normalize(np.array([[1.0, 2.0]], dtype=np.float32), alpha_beta=alpha_beta)

    def test_normalize_traces_declared_alpha_beta_input(self):
        @trace(inputs=["x", "alpha_beta"], outputs=["y"])
        def normalize_vec(x, alpha_beta):
            return ops.normalize(x, alpha_beta=alpha_beta)

        _, ctx = normalize_vec.trace(
            x=np.array([[3.0, 4.0]], dtype=np.float32),
            alpha_beta=np.array([2.0, 7.0], dtype=np.float32),
        )
        spec = convert(ctx)

        assert spec["operators"][0]["inputs"] == [
            {"tensor": "x"},
            {"tensor": "alpha_beta"},
        ]

    def test_normalize_traces_constant_alpha_beta_value(self):
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x, alpha_beta=np.array([-2.0, 2.0], dtype=np.float64))

        _, ctx = normalize_vec.trace(x=np.array([[1.0, 3.0]], dtype=np.float32))
        spec = convert(ctx)

        alpha_beta_name = spec["operators"][0]["inputs"][1]["tensor"]
        assert spec["tensors"][alpha_beta_name]["data"] == [-2.0, 2.0]
        assert spec["tensors"][alpha_beta_name]["is_placeholder"] is False

    def test_normalize_multiple_rows(self):
        """Test normalization with multiple rows."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        input_arr = np.array([
            [3.0, 4.0],      # norm = 5
            [1.0, 0.0],      # norm = 1
            [0.0, 2.0],      # norm = 2
        ], dtype=np.float32)

        result, verification = run_op_test(
            normalize_vec,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.array([
            [3.0, 4.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalize_3d_tensor(self):
        """Test normalization with 3D tensor."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        input_arr = np.array([
            [[3.0, 4.0], [1.0, 0.0]],
            [[0.0, 5.0], [6.0, 8.0]],
        ], dtype=np.float32)

        result, verification = run_op_test(
            normalize_vec,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr / np.linalg.norm(input_arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalize_unit_vector(self):
        """Test normalization of already unit vector."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        input_arr = np.array([[1.0, 0.0]], dtype=np.float32)
        result, verification = run_op_test(
            normalize_vec,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, input_arr, rtol=1e-5)

    @skip_if_no_device
    def test_normalize_on_device(self):
        """Test normalize operation on device."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize_vec(x):
            return ops.normalize(x)

        input_arr = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        result, verification = run_op_test(
            normalize_vec,
            inputs={"x": input_arr},
            expected_output_name="y",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
