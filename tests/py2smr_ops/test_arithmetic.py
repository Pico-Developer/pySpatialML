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
"""Tests for arithmetic operator (ARITHMETIC_COMPOSE)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from securemr.py2smr.verifier import run_pipeline_python
from .conftest import run_op_test, skip_if_no_device, DEVICE_AVAILABLE


class TestArithmeticOp:
    """Tests for the arithmetic operation."""

    def test_arithmetic_multiply(self):
        """Test arithmetic multiplication."""
        @trace(inputs=["x"], outputs=["y"])
        def multiply(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, verification = run_op_test(
            multiply,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr * 2.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_divide(self):
        """Test arithmetic division."""
        @trace(inputs=["x"], outputs=["y"])
        def divide(x):
            return ops.arithmetic(x, "{0} / 255.0")

        input_arr = np.array([[100.0, 200.0], [50.0, 150.0]], dtype=np.float32)
        result, verification = run_op_test(
            divide,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr / 255.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_add(self):
        """Test arithmetic addition."""
        @trace(inputs=["x"], outputs=["y"])
        def add(x):
            return ops.arithmetic(x, "{0} + 10.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, verification = run_op_test(
            add,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr + 10.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_multiple_operands_host(self):
        """Test schema-v2 arithmetic expressions with multiple operands."""
        spec = {
            "tensors": {
                "a": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "b": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "y": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {"type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO", "attrs": ["({0} * {1})"], "inputs": [{"tensor": "a"}, {"tensor": "b"}] + [None] * 8, "outputs": [{"tensor": "y"}]}
            ],
            "inputs": ["a", "b"],
            "outputs": ["y"],
        }
        a = np.ones((2, 2), dtype=np.float32) * 2.0
        b = np.ones((2, 2), dtype=np.float32) * 3.0

        outputs = run_pipeline_python(spec, {"a": a, "b": b})

        np.testing.assert_allclose(outputs["y"], np.ones((2, 2), dtype=np.float32) * 6.0)

    def test_arithmetic_subtract(self):
        """Test arithmetic subtraction."""
        @trace(inputs=["x"], outputs=["y"])
        def subtract(x):
            return ops.arithmetic(x, "{0} - 1.0")

        input_arr = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result, verification = run_op_test(
            subtract,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr - 1.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_complex_expression(self):
        """Test arithmetic with complex expression."""
        @trace(inputs=["x"], outputs=["y"])
        def normalize(x):
            return ops.arithmetic(x, "{0} / 255.0 * 2.0 - 1.0")

        input_arr = np.array([[0.0, 127.5, 255.0]], dtype=np.float32)
        result, verification = run_op_test(
            normalize,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = input_arr / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_with_image(self):
        """Test arithmetic with image-like tensor."""
        @trace(inputs=["image"], outputs=["normalized"])
        def normalize_image(image):
            return ops.arithmetic(image, "{0} / 255.0")

        input_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result, verification = run_op_test(
            normalize_image,
            inputs={"image": input_arr},
            expected_output_name="normalized",
        )

        assert verification.success, verification.error_message
        expected = input_arr.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected)

    @skip_if_no_device
    def test_arithmetic_on_device(self):
        """Test arithmetic operation on device."""
        @trace(inputs=["x"], outputs=["y"])
        def multiply(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, verification = run_op_test(
            multiply,
            inputs={"x": input_arr},
            expected_output_name="y",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
