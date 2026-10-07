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
"""Tests for the NMS (Non-Maximum Suppression) operator."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from .conftest import skip_if_no_device


class TestNMS:
    """Tests for NMS operation."""

    @staticmethod
    def _run_nms_test(traced_func, inputs, *, test_device=False, device_duration=10):
        """Verify the helper's kept-index output, serialized in NMS slot 2."""
        result, ctx = traced_func.trace(**inputs)
        spec = convert(ctx)
        expected_indices = np.asarray(result, dtype=np.int32)
        verification = verify(
            pipeline=spec,
            inputs=inputs,
            expected_outputs={"output": expected_indices},
            device=test_device,
            duration=device_duration,
        )
        if test_device and verification.error_message == "Device verification is not available":
            pytest.skip("Python verifier device execution is not available")
        return result, verification

    def test_serializes_scores_before_boxes(self):
        """NMS JSON input order must match the v2 loader contract."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def traced_nms(scores, boxes):
            return ops.nms(scores, boxes, output_name="output")

        scores = np.array([0.9, 0.8], dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        _, ctx = traced_nms.trace(scores=scores, boxes=boxes)

        spec = convert(ctx)
        assert spec["operators"][0]["inputs"] == [{"tensor": "scores"}, {"tensor": "boxes"}]

    def test_basic_nms(self):
        """Test basic NMS with overlapping boxes."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def basic_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.5, output_name="output")

        # Create overlapping boxes
        boxes = np.array([
            [0, 0, 10, 10],    # Box 0
            [1, 1, 11, 11],    # Box 1 - overlaps with 0
            [50, 50, 60, 60],  # Box 2 - no overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = self._run_nms_test(
            basic_nms, {"scores": scores, "boxes": boxes}
        )

        assert verification.success, verification.error_message
        # Box 0 should be kept (highest score), Box 1 suppressed, Box 2 kept
        assert 0 in result
        assert 2 in result

    def test_no_overlap(self):
        """Test NMS with non-overlapping boxes."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def no_overlap_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.5, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = self._run_nms_test(
            no_overlap_nms, {"scores": scores, "boxes": boxes}
        )

        assert verification.success, verification.error_message
        # All boxes should be kept
        assert len(result) == 3

    def test_high_threshold(self):
        """Test NMS with high IoU threshold (keeps more boxes)."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def high_thresh_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.9, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # Overlaps but IoU < 0.9
            [2, 2, 12, 12],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = self._run_nms_test(
            high_thresh_nms, {"scores": scores, "boxes": boxes}
        )

        assert verification.success, verification.error_message

    def test_low_threshold(self):
        """Test NMS with low IoU threshold (suppresses more boxes)."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def low_thresh_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.1, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],  # Partial overlap
            [50, 50, 60, 60],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = self._run_nms_test(
            low_thresh_nms, {"scores": scores, "boxes": boxes}
        )

        assert verification.success, verification.error_message

    def test_single_box(self):
        """Test NMS with single box."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def single_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.5, output_name="output")

        boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        result, verification = self._run_nms_test(
            single_nms, {"scores": scores, "boxes": boxes}
        )

        assert verification.success, verification.error_message
        assert len(result) == 1
        assert result[0] == 0

    @skip_if_no_device
    def test_nms_on_device(self):
        """Test NMS operation on device."""
        @trace(inputs=["scores", "boxes"], outputs=["output"])
        def device_nms(scores, boxes):
            return ops.nms(scores, boxes, threshold=0.5, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = self._run_nms_test(
            device_nms,
            {"scores": scores, "boxes": boxes},
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message

