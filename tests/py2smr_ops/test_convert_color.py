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
"""Tests for cvt_color operator (CONVERT_COLOR)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from .conftest import run_op_test, skip_if_no_device, DEVICE_AVAILABLE


# OpenCV color conversion codes
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7


class TestConvertColorOp:
    """Tests for the cvt_color operation."""

    def test_convert_bgr2rgb(self):
        """Test BGR to RGB conversion."""
        @trace(inputs=["image"], outputs=["result"])
        def bgr_to_rgb(image):
            return ops.cvt_color(image, COLOR_BGR2RGB)

        # Create BGR image
        input_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        input_arr[:, :, 0] = 255  # Blue channel
        input_arr[:, :, 1] = 128  # Green channel
        input_arr[:, :, 2] = 64   # Red channel

        result, verification = run_op_test(
            bgr_to_rgb,
            inputs={"image": input_arr},
            expected_output_name="result",
        )

        assert verification.success, verification.error_message
        # After BGR2RGB, channels should be swapped
        assert result[0, 0, 0] == 64   # Red (was Blue)
        assert result[0, 0, 1] == 128  # Green (unchanged)
        assert result[0, 0, 2] == 255  # Blue (was Red)

    def test_convert_with_random_image(self):
        """Test color conversion with random image."""
        @trace(inputs=["image"], outputs=["result"])
        def convert_image(image):
            return ops.cvt_color(image, COLOR_BGR2RGB)

        input_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result, verification = run_op_test(
            convert_image,
            inputs={"image": input_arr},
            expected_output_name="result",
        )

        assert verification.success, verification.error_message
        # Verify channel swap
        np.testing.assert_array_equal(result[:, :, 0], input_arr[:, :, 2])
        np.testing.assert_array_equal(result[:, :, 1], input_arr[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], input_arr[:, :, 0])

    def test_convert_large_image(self):
        """Test color conversion with larger image."""
        @trace(inputs=["image"], outputs=["result"])
        def convert_image(image):
            return ops.cvt_color(image, COLOR_BGR2RGB)

        input_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result, verification = run_op_test(
            convert_image,
            inputs={"image": input_arr},
            expected_output_name="result",
        )

        assert verification.success, verification.error_message

    @skip_if_no_device
    def test_cvt_color_on_device(self):
        """Test color conversion on device."""
        @trace(inputs=["image"], outputs=["result"])
        def convert_image(image):
            return ops.cvt_color(image, COLOR_BGR2RGB)

        input_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result, verification = run_op_test(
            convert_image,
            inputs={"image": input_arr},
            expected_output_name="result",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
