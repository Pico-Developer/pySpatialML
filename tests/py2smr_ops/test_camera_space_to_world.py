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
"""Tests for CAMERA_SPACE_TO_WORLD operator."""

import numpy as np

from securemr.py2smr import trace, ops, convert, verify
from .conftest import skip_if_no_device


def _expected_for_timestamp(timestamp: np.ndarray):
    if np.all(timestamp == 0):
        right = np.zeros((4, 4), dtype=np.float32)
        left = np.zeros((4, 4), dtype=np.float32)
        return right, left
    right = np.array(
        [
            [0.99999573, 0.0008423, 0.00279751, 0.0312826],
            [-0.00081392, 0.99994834, -0.01013152, 0.0397733],
            [-0.0028059, 0.0101292, 0.99994476, -0.03535056],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    left = np.array(
        [
            [0.99997533, 0.00673646, -0.00199215, -0.03277511],
            [-0.00674351, 0.99997094, -0.0035545, 0.03963064],
            [0.00196815, 0.00356784, 0.9999917, -0.03536495],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return right, left


class TestCameraSpaceToWorld:
    def test_host_defaults(self):
        @trace(inputs=["timestamp"], outputs=["right", "left"])
        def cam2world(timestamp):
            return ops.camera_space_to_world(timestamp, output_names=["right", "left"])

        timestamp = np.array([0, 0, 0, 0], dtype=np.int32)
        result, ctx = cam2world.trace(timestamp=timestamp)

        expected_right, expected_left = _expected_for_timestamp(timestamp)
        np.testing.assert_allclose(result[0], expected_right, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(result[1], expected_left, rtol=1e-5, atol=1e-5)

        spec = convert(ctx)
        assert spec["operators"][0]["outputs"] == [{"tensor": "right"}, {"tensor": "left"}]
        verification = verify(
            pipeline=spec,
            inputs={"timestamp": timestamp},
            expected_outputs={"right": expected_right, "left": expected_left},
            device=False,
        )

        assert verification.success, verification.error_message

    @skip_if_no_device
    def test_camera_space_to_world_on_device(self):
        @trace(inputs=["timestamp"], outputs=["right", "left"])
        def cam2world(timestamp):
            return ops.camera_space_to_world(timestamp, output_names=["right", "left"])

        timestamp = np.array([0, 0, 0, 0], dtype=np.int32)
        result, ctx = cam2world.trace(timestamp=timestamp)

        spec = convert(ctx)
        expected_right, expected_left = _expected_for_timestamp(timestamp)
        verification = verify(
            pipeline=spec,
            inputs={"timestamp": timestamp},
            expected_outputs={"right": expected_right, "left": expected_left},
            device=True,
            duration=30,
        )
        if verification.error_message == "Device verification is not available":
            import pytest

            pytest.skip("Python verifier device execution is not available")

        assert verification.success, verification.error_message
