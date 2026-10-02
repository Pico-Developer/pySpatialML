#!/usr/bin/env python3
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
"""Unit tests for py2smr module."""

import json
import os
import tempfile

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from securemr.py2smr.tracer import TraceContext, TracedOp, TensorInfo, get_current_trace
from securemr.core.utils import convert_to_dtype
from securemr.py2smr.converter import trace_to_pipeline_spec
from securemr.py2smr.verifier import (
    compare_outputs,
    VerificationResult,
    run_pipeline_python,
    validate_pipeline_spec,
)
from securemr.core.types import EOperatorType


def test_convert_to_dtype_accepts_schema_v2_string_aliases():
    assert convert_to_dtype("int32", target="numpy") == np.int32
    assert convert_to_dtype("float32", target="numpy") == np.float32


def test_type_convert_alias_converts_host_dtype():
    pipeline = {
        "tensors": {
            "input": {
                "dimensions": [2, 1],
                "channels": 1,
                "data_type": 6,
                "usage": 6,
                "is_placeholder": True,
            },
            "output": {
                "dimensions": [2, 1],
                "channels": 1,
                "data_type": 5,
                "usage": 6,
                "is_placeholder": True,
            },
        },
        "operators": [{
            "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                "inputs": [{"tensor": "input"}],
                "outputs": [{"tensor": "output"}],
        }],
        "inputs": ["input"],
        "outputs": ["output"],
    }

    outputs = run_pipeline_python(
        pipeline,
        {"input": np.array([[1.5], [2.5]], dtype=np.float32)},
    )

    assert outputs["output"].dtype == np.int32
    np.testing.assert_array_equal(outputs["output"], [[1], [2]])


class TestTracer:
    """Tests for the tracer module."""

    def test_trace_decorator_basic(self):
        """Test basic trace decorator functionality."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, ctx = simple_func.trace(x=input_arr)

        assert isinstance(ctx, TraceContext)
        assert len(ctx.operations) == 1
        assert ctx.operations[0].op_type == EOperatorType.ARITHMETIC_COMPOSE
        assert "x" in ctx.tensors
        assert ctx.tensors["x"].is_input

    def test_trace_multiple_ops(self):
        """Test tracing multiple operations."""
        @trace(inputs=["image"], outputs=["result"])
        def preprocess(image):
            normalized = ops.arithmetic(image, "{0} / 255.0")
            scaled = ops.arithmetic(normalized, "{0} * 2.0 - 1.0")
            return scaled

        input_arr = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result, ctx = preprocess.trace(image=input_arr)

        assert len(ctx.operations) == 2
        assert ctx.operations[0].attrs == ["{0} / 255.0"]
        assert ctx.operations[1].attrs == ["{0} * 2.0 - 1.0"]

    def test_trace_multiple_outputs(self):
        """Test tracing function with multiple outputs."""
        @trace(inputs=["x"], outputs=["min_val", "max_val"])
        def minmax(x):
            min_val = ops.arithmetic(x, "{0} - 1.0")
            max_val = ops.arithmetic(x, "{0} + 1.0")
            return min_val, max_val

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (min_result, max_result), ctx = minmax.trace(x=input_arr)

        assert "min_val" in ctx.tensors
        assert "max_val" in ctx.tensors
        assert ctx.tensors["min_val"].is_output
        assert ctx.tensors["max_val"].is_output

    def test_trace_context_not_active_outside(self):
        """Test that trace context is not active outside traced function."""
        assert get_current_trace() is None

    def test_missing_input_raises_error(self):
        """Test that missing input raises ValueError."""
        @trace(inputs=["x", "y"], outputs=["z"])
        def add_func(x, y):
            return ops.arithmetic(x, "{0} + 1.0")

        with pytest.raises(ValueError, match="Missing required input"):
            add_func.trace(x=np.array([1.0]))

    def test_non_array_input_raises_error(self):
        """Test that non-array input raises TypeError."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        with pytest.raises(TypeError, match="must be a numpy array"):
            simple_func.trace(x=[1.0, 2.0])

    def test_trace_allows_none_for_declared_zero_outputs(self):
        @trace(inputs=["audio"], outputs=[])
        def play(audio):
            ops.speaker(audio, sample_rate=48000)

        result, ctx = play.trace(audio=np.zeros((16, 2), dtype=np.float32))

        assert result is None
        assert len(ctx.operations) == 1
        assert ctx.operations[0].op_type == EOperatorType.SPEAKER

    def test_trace_rejects_none_when_outputs_are_declared(self):
        @trace(inputs=["audio"], outputs=["result"])
        def play(audio):
            ops.speaker(audio)

        with pytest.raises(TypeError, match="Function must return"):
            play.trace(audio=np.zeros((16, 2), dtype=np.float32))


class TestOps:
    """Tests for the ops module."""

    def test_arithmetic_basic(self):
        """Test basic arithmetic operation."""
        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = ops.arithmetic(input_arr, "{0} * 2.0")

        expected = input_arr * 2.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_complex_expression(self):
        """Test arithmetic with complex expression."""
        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = ops.arithmetic(input_arr, "{0} / 255.0 * 2.0 - 1.0")

        expected = input_arr / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(result, expected)

    def test_elementwise_min(self):
        """Test elementwise minimum."""
        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)
        result = ops.elementwise_min(a, b)

        expected = np.minimum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_max(self):
        """Test elementwise maximum."""
        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)
        result = ops.elementwise_max(a, b)

        expected = np.maximum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_multiply(self):
        """Test elementwise multiplication."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        result = ops.elementwise_multiply(a, b)

        expected = a * b
        np.testing.assert_allclose(result, expected)

    def test_normalize(self):
        """Test L2 normalization."""
        input_arr = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        result = ops.normalize(input_arr)

        expected = input_arr / np.linalg.norm(input_arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_argmax(self):
        """Test argmax operation."""
        input_arr = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=np.float32)
        result = ops.argmax(input_arr, axis=-1)

        expected = np.array([1, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_nms_basic(self):
        """Test basic NMS operation."""
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # High overlap with first
            [50, 50, 60, 60],  # No overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result = ops.nms(scores, boxes, threshold=0.5)

        # Should keep first and third (second overlaps too much with first)
        assert 0 in result
        assert 2 in result

    def test_nms_empty(self):
        """Test NMS with empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)

        result = ops.nms(scores, boxes, threshold=0.5)
        assert len(result) == 0

    def test_ops_record_to_trace(self):
        """Test that ops record to trace context."""
        @trace(inputs=["x"], outputs=["y"])
        def traced_func(x):
            return ops.arithmetic(x, "{0} + 1.0")

        input_arr = np.array([[1.0, 2.0]], dtype=np.float32)
        _, ctx = traced_func.trace(x=input_arr)

        assert len(ctx.operations) == 1
        op = ctx.operations[0]
        assert op.op_type == EOperatorType.ARITHMETIC_COMPOSE
        assert op.attrs == ["{0} + 1.0"]


class TestConverter:
    """Tests for the converter module."""

    def test_convert_basic(self):
        """Test basic conversion to pipeline spec."""
        @trace(inputs=["input"], outputs=["output"])
        def simple_func(input):
            return ops.arithmetic(input, "{0} + 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        _, ctx = simple_func.trace(input=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        assert "tensors" in spec
        assert "operators" in spec
        assert "inputs" in spec
        assert "outputs" in spec

        assert "input" in spec["inputs"]
        assert "output" in spec["outputs"]
        assert len(spec["operators"]) == 1

    def test_convert_saves_to_file(self):
        """Test that convert saves to file."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0]], dtype=np.float32)
        _, ctx = simple_func.trace(x=input_arr)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            spec = convert(ctx, output=output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r") as f:
                loaded_spec = json.load(f)

            assert loaded_spec == spec
        finally:
            os.unlink(output_path)

    def test_convert_tensor_shapes(self):
        """Test that tensor shapes are correctly converted."""
        @trace(inputs=["image"], outputs=["result"])
        def process_image(image):
            return ops.arithmetic(image, "{0} / 255.0")

        # 3D tensor (H, W, C)
        input_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, ctx = process_image.trace(image=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        # Check input tensor spec
        input_spec = spec["tensors"]["image"]
        assert input_spec["dimensions"] == [224, 224]  # W, H
        assert input_spec["channels"] == 3

    def test_convert_operator_attrs(self):
        """Test that operator attributes are correctly converted."""
        @trace(inputs=["x"], outputs=["y"])
        def func(x):
            return ops.arithmetic(x, "{0} / 255.0 * 2.0 - 1.0")

        input_arr = np.array([[1.0]], dtype=np.float32)
        _, ctx = func.trace(x=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        op_spec = spec["operators"][0]
        assert op_spec["attrs"] == ["{0} / 255.0 * 2.0 - 1.0"]

    def test_convert_microphone_preserves_audio_config_and_stereo_layout(self):
        @trace(inputs=[], outputs=["audio"])
        def capture():
            return ops.microphone(sample_rate=48000, encoding="PCM_32BIT")

        _, ctx = capture.trace()
        spec = convert(ctx)

        assert spec["operators"][0]["attrs"] == ["48000;PCM_32BIT"]
        assert spec["operators"][0]["outputs"] == [
            {"tensor": "audio"}, None, None, None
        ]
        assert spec["tensors"]["audio"]["dimensions"] == [1, 2]
        assert spec["tensors"]["audio"]["channels"] == 1
        assert spec["tensors"]["audio"]["data_type"] == 5

    def test_convert_command_only_speaker_preserves_sample_rate(self):
        @trace(inputs=["audio"], outputs=[])
        def play(audio):
            ops.speaker(audio, sample_rate=44100)

        _, ctx = play.trace(audio=np.zeros((32, 2), dtype=np.int16))
        spec = convert(ctx)

        assert spec["operators"][0] == {
            "type": "XR_SECURE_MR_OPERATOR_TYPE_SPEAKER_PICO",
            "inputs": [{"tensor": "audio"}],
            "outputs": [],
            "attrs": ["44100"],
        }
        assert spec["outputs"] == []

    def test_convert_spatial_only_ops_use_sdk_json_fields(self):
        """Test that Spatial-only operators emit fields accepted by the SDK loader."""
        @trace(inputs=["scene", "scale"], outputs=["visible"])
        def func(scene, scale):
            visible = ops.scenegraph_visibility(scene, visible=False)
            ops.update_component(scene, scale, entity_path="/target", property="Transform.Scale")
            return visible

        _, ctx = func.trace(
            scene=np.array([[1]], dtype=np.uint8),
            scale=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        )
        spec = trace_to_pipeline_spec(ctx)

        assert spec["operators"][0] == {
            "type": "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
            "inputs": [{"tensor": "scene"}, {"tensor": "tensor_0"}],
            "outputs": [],
        }
        assert spec["operators"][1] == {
            "type": "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
            "inputs": [{"tensor": "scene"}, {"tensor": "scale"}],
            "outputs": [],
            "attrs": ["/target:Transform.Scale"],
        }

    def test_convert_xr_gltf_ops_emits_native_named_fields(self):
        @trace(inputs=["gltf", "pose", "texture", "texture_ids"], outputs=["texture_id"])
        def func(gltf, pose, texture, texture_ids):
            ops.switch_gltf_render_status(gltf, pose=pose)
            ops.update_gltf(gltf, update_type="texture")
            return ops.load_texture(gltf, texture, output_name="texture_id")

        _, ctx = func.trace(
            gltf=np.zeros((1,), dtype=np.uint8),
            pose=np.eye(4, dtype=np.float32),
            texture=np.zeros((2, 2, 3), dtype=np.uint8),
            texture_ids=np.array([0], dtype=np.uint16),
        )
        spec = trace_to_pipeline_spec(ctx)

        render, update, load = spec["operators"]
        assert render["inputs"] == [{"tensor": "gltf"}, {"tensor": "pose"}, None, None]
        assert update["attrs"] == ["texture"]
        assert update["inputs"] == [
            {"tensor": "gltf"}, {"tensor": "tensor_0"}, {"tensor": "tensor_1"}
        ]
        assert load["inputs"] == [{"tensor": "gltf"}, {"tensor": "texture"}]
        assert spec["outputs"] == ["texture_id"]


class TestVerifier:
    """Tests for the verifier module."""

    @staticmethod
    def tensor(data_type=6, usage=6, dimensions=None, channels=1, **extra):
        spec = {
            "dimensions": dimensions or [1, 1],
            "channels": channels,
            "data_type": data_type,
            "is_placeholder": True,
            "usage": usage,
        }
        if usage == 7:
            spec["is_gltf"] = True
        spec.update(extra)
        return spec

    def test_validate_pipeline_spec_rejects_legacy_operator_fields(self):
        spec = {
            "tensors": {
                "input": self.tensor(),
                "output": self.tensor(),
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": [{"tensor": "input"}] + [None] * 9,
                    "outputs": [{"tensor": "output"}],
                    "expression": "{0} * 2.0",
                }
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        with pytest.raises(ValueError, match="legacy field"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_string_and_bracket_refs(self):
        spec = {
            "tensors": {
                "input": self.tensor(dimensions=[2, 1]),
                "output": self.tensor(dimensions=[2, 1]),
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                    "inputs": ["input"],
                    "outputs": [{"tensor": "output[0:1]"}],
                }
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        with pytest.raises(ValueError, match="object form"):
            validate_pipeline_spec(spec)

        spec["operators"][0]["inputs"] = [{"tensor": "input"}]
        with pytest.raises(ValueError, match="bracket tensor reference"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_unknown_execution(self):
        spec = {
            "tensors": {},
            "operators": [
                {"type": "XR_SECURE_MR_OPERATOR_TYPE_UNKNOWN_PICO", "inputs": [], "outputs": []}
            ],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="UNKNOWN is reserved"):
            validate_pipeline_spec(spec)

    @staticmethod
    def microphone_spec(attrs):
        encoding = attrs[0].split(";", 1)[-1] if attrs else "PCM_FLOAT"
        audio_data_type = {
            "PCM_16BIT": 4,
            "PCM_32BIT": 5,
            "PCM_FLOAT": 6,
        }.get(encoding, 6)
        return {
            "tensors": {
                "stereo": TestVerifier.tensor(data_type=audio_data_type, dimensions=[128, 2]),
                "left": TestVerifier.tensor(data_type=audio_data_type, dimensions=[128, 1]),
                "right": TestVerifier.tensor(data_type=audio_data_type, dimensions=[128, 1]),
                "timestamp": TestVerifier.tensor(
                    data_type=5, usage=5, dimensions=[1], channels=4
                ),
            },
            "operators": [{
                "type": "XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO",
                "inputs": [],
                "outputs": [
                    {"tensor": "stereo"},
                    {"tensor": "left"},
                    {"tensor": "right"},
                    {"tensor": "timestamp"},
                ],
                "attrs": attrs,
            }],
            "inputs": [],
            "outputs": [],
        }

    @staticmethod
    def operator_spec(operator_name, tensors, inputs, outputs, attrs=None, model=None):
        operator = {
            "type": f"XR_SECURE_MR_OPERATOR_TYPE_{operator_name}_PICO",
            "inputs": inputs,
            "outputs": outputs,
        }
        if attrs is not None:
            operator["attrs"] = attrs
        if model is not None:
            operator["model"] = model
        return {
            "tensors": tensors,
            "operators": [operator],
            "inputs": [],
            "outputs": [],
        }

    def test_validate_pipeline_spec_rejects_all_null_optional_outputs(self):
        cases = [
            (
                "NMS",
                {
                    "scores": self.tensor(dimensions=[1, 3]),
                    "boxes": self.tensor(dimensions=[3, 4]),
                },
                [{"tensor": "scores"}, {"tensor": "boxes"}],
                [None, None, None],
                ["0.5"],
            ),
            (
                "SORT_VEC",
                {"source": self.tensor(usage=2, dimensions=[3])},
                [{"tensor": "source"}],
                [None, None],
                [],
            ),
            (
                "SORT_MAT",
                {"source": self.tensor(dimensions=[2, 3])},
                [{"tensor": "source"}],
                [None, None],
                [],
            ),
            (
                "SVD",
                {"source": self.tensor(dimensions=[2, 2])},
                [{"tensor": "source"}],
                [None, None, None],
                [],
            ),
            (
                "MICROPHONE",
                {},
                [],
                [None, None, None, None],
                ["48000;PCM_FLOAT"],
            ),
        ]
        for operator_name, tensors, inputs, outputs, attrs in cases:
            with pytest.raises(ValueError, match="at least one connected output"):
                validate_pipeline_spec(
                    self.operator_spec(operator_name, tensors, inputs, outputs, attrs)
                )

    def test_validate_pipeline_spec_accepts_spatialsdk_v2_microphone_outputs(self):
        spec = self.microphone_spec(["48000;PCM_FLOAT"])
        spec["operators"][0]["outputs"] = [
            {"tensor": "stereo"},
            {"tensor": "timestamp"},
        ]

        validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_empty_javascript_outputs_and_accepts_unequal_bindings(self):
        empty = self.operator_spec(
            "JS_SCRIPTING", {}, [], [], ["var x;"],
        )
        with pytest.raises(ValueError, match="expected 1..64 output"):
            validate_pipeline_spec(empty)

        tensors = {"x": self.tensor(), "y": self.tensor()}
        unequal = self.operator_spec(
            "JS_SCRIPTING", tensors, [{"tensor": "x"}],
            [{"tensor": "y"}, {"tensor": "x"}], ["var x;"],
        )
        validate_pipeline_spec(unequal)

    def test_validate_pipeline_spec_rejects_null_model_bindings(self):
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": "input", "shape": [1, 2], "encoding_type": "float32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "float32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE", {"output": self.tensor()},
            [None], [{"tensor": "output"}], model=model,
        )
        with pytest.raises(ValueError, match="all model input and output slots"):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("binding_name", "tensor_name"),
        [
            ("a" * 512, "input"),
            ("é" * 256, "input"),
            ("input\0suffix", "input"),
            ("input", "t" * 512),
        ],
    )
    def test_validate_pipeline_spec_rejects_unsafe_model_binding_names(self, binding_name, tensor_name):
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": binding_name, "shape": [1, 2], "encoding_type": "FP32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE",
            {tensor_name: self.tensor(dimensions=[1, 2]), "output": self.tensor(dimensions=[1, 2])},
            [{"name": binding_name, "tensor": tensor_name}],
            [{"name": "output", "tensor": "output"}],
            model=model,
        )

        with pytest.raises(ValueError, match="(binding|tensor) name"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_accepts_maximum_model_binding_name(self):
        binding_name = "a" * 511
        tensor_name = "t" * 511
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": binding_name, "shape": [1, 2], "encoding_type": "FP32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE",
            {tensor_name: self.tensor(dimensions=[1, 2]), "output": self.tensor(dimensions=[1, 2])},
            [{"name": binding_name, "tensor": tensor_name}],
            [{"name": "output", "tensor": "output"}],
            model=model,
        )

        validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_accepts_reused_model_tensor_bindings(self):
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [
                {"name": "left", "shape": [1, 2], "encoding_type": "FP32"},
                {"name": "right", "shape": [1, 2], "encoding_type": "FP32"},
            ],
            "output": [{"name": "result", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE",
            {
                "shared": self.tensor(dimensions=[1, 2]),
                "output": self.tensor(dimensions=[1, 2]),
            },
            [
                {"name": "left", "tensor": "shared"},
                {"name": "right", "tensor": "shared"},
            ],
            [{"name": "result", "tensor": "output"}],
            model=model,
        )

        validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("usage", "dimensions", "channels", "data_type", "message"),
        [
            (2, [4], 2, 6, "scalar usage"),
            (1, [4], 4, 6, "point usage"),
            (4, [4], 2, 1, "color usage"),
            (5, [1], 1, 5, "timestamp usage"),
            (3, [4], 1, 5, "slice usage"),
        ],
    )
    def test_validate_pipeline_spec_rejects_invalid_usage_layouts(
        self, usage, dimensions, channels, data_type, message
    ):
        spec = {
            "tensors": {
                "value": self.tensor(
                    usage=usage, dimensions=dimensions, channels=channels, data_type=data_type,
                    is_placeholder=False,
                )
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_non_placeholder_boundary_tensor(self):
        spec = {
            "tensors": {
                "input": self.tensor(is_placeholder=False),
            },
            "operators": [],
            "inputs": ["input"],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="requires is_placeholder=true"):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("operator_name", "data_type"),
        [("ELEMENTWISE_MIN", None), ("ELEMENTWISE_OR", 6)],
    )
    def test_validate_pipeline_spec_rejects_elementwise_tensor_contracts(self, operator_name, data_type):
        tensors = {
            "left": self.tensor(data_type=data_type or 6, dimensions=[2, 2]),
            "right": self.tensor(data_type=data_type or 6, dimensions=[2, 3]),
            "result": self.tensor(data_type=data_type or 6, dimensions=[2, 2]),
        }
        if operator_name == "ELEMENTWISE_OR":
            tensors["right"] = self.tensor(data_type=6, dimensions=[2, 2])
        spec = self.operator_spec(
            operator_name, tensors,
            [{"tensor": "left"}, {"tensor": "right"}],
            [{"tensor": "result"}],
        )
        with pytest.raises(ValueError, match="(shapes/channels must match|requires integer tensors)"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_normalize_tensor_contracts(self):
        tensors = {
            "source": self.tensor(dimensions=[2, 2]),
            "result": self.tensor(dimensions=[2, 3]),
            "alpha_beta": self.tensor(data_type=5, dimensions=[1, 2]),
        }
        mismatch = self.operator_spec(
            "NORMALIZE", tensors, [{"tensor": "source"}],
            [{"tensor": "result"}],
        )
        with pytest.raises(ValueError, match="source and result tensor types/shapes"):
            validate_pipeline_spec(mismatch)

        invalid_alpha_beta = self.operator_spec(
            "NORMALIZE", tensors,
            [{"tensor": "source"}, {"tensor": "alpha_beta"}],
            [{"tensor": "source"}],
        )
        with pytest.raises(ValueError, match="alpha_beta"):
            validate_pipeline_spec(invalid_alpha_beta)

    def test_validate_pipeline_spec_rejects_invalid_matrix_contracts(self):
        inversion_tensors = {
            "source": self.tensor(dimensions=[2, 3]),
            "result": self.tensor(dimensions=[2, 3]),
        }
        with pytest.raises(ValueError, match="square input"):
            validate_pipeline_spec(
                self.operator_spec(
                    "INVERSION", inversion_tensors,
                    [{"tensor": "source"}], [{"tensor": "result"}],
                )
            )

    def test_validate_pipeline_spec_rejects_assignment_slice_rank_mismatch(self):
        tensors = {
            "source": self.tensor(usage=1, dimensions=[1], channels=3),
            "result": self.tensor(dimensions=[1, 3]),
        }
        spec = self.operator_spec(
            "ASSIGNMENT", tensors,
            [{"tensor": "source"}], [{"tensor": "result"}],
        )
        spec["operators"][0]["src_slices"] = [[0, 1], [0, 3]]
        with pytest.raises(ValueError, match="src_slices rank 2 does not match tensor 'source' rank 1"):
            validate_pipeline_spec(spec)

        transform_tensors = {
            "rotation": self.tensor(dimensions=[2, 2]),
            "translation": self.tensor(dimensions=[3, 1]),
            "transform": self.tensor(dimensions=[4, 4]),
        }
        with pytest.raises(ValueError, match="3-value"):
            validate_pipeline_spec(
                self.operator_spec(
                    "MAKE_TRANSFORM_MAT", transform_tensors,
                    [{"tensor": "rotation"}, {"tensor": "translation"}, None],
                    [{"tensor": "transform"}],
                )
            )

    @pytest.mark.parametrize(
        "attrs", [["16000"], ["8000"], ["96000"], ["7999"], ["96001"], ["16000x"], ["0"]]
    )
    def test_validate_pipeline_spec_rejects_invalid_speaker_sample_rate(self, attrs):
        spec = self.operator_spec(
            "SPEAKER", {"audio": self.tensor()},
            [{"tensor": "audio"}], [], attrs,
        )
        if attrs in (["8000"], ["16000"], ["96000"]):
            validate_pipeline_spec(spec)
        else:
            with pytest.raises(ValueError, match="between 8000 and 96000"):
                validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("tensor", "message"),
        [
            ({"dimensions": [32, 1], "channels": 3, "data_type": 6}, "one or two channels"),
            ({"dimensions": [32, 1], "channels": 1, "data_type": 1}, "UINT16, INT16, INT32, or FLOAT32"),
        ],
    )
    def test_validate_pipeline_spec_rejects_invalid_speaker_tensor(self, tensor, message):
        audio = self.tensor(**tensor)
        spec = self.operator_spec(
            "SPEAKER", {"audio": audio}, [{"tensor": "audio"}], [], ["16000"]
        )

        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_invalid_nms_result_type(self):
        tensors = {
            "scores": self.tensor(dimensions=[1, 3]),
            "boxes": self.tensor(dimensions=[3, 4]),
            "result_scores": self.tensor(dimensions=[3, 1]),
            "result_boxes": self.tensor(dimensions=[3, 4]),
            "indices": self.tensor(data_type=6, dimensions=[3, 1]),
        }
        spec = self.operator_spec(
            "NMS", tensors,
            [{"tensor": "scores"}, {"tensor": "boxes"}],
            [{"tensor": "result_scores"}, {"tensor": "result_boxes"}, {"tensor": "indices"}],
            ["0.5"],
        )
        with pytest.raises(ValueError, match="index result must be integer"):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize("operator_name", ["SORT_VEC", "SORT_MAT"])
    def test_validate_pipeline_spec_rejects_invalid_sort_contracts(self, operator_name):
        if operator_name == "SORT_VEC":
            tensors = {
                "source": self.tensor(dimensions=[2, 2]),
                "values": self.tensor(dimensions=[2, 2]),
                "indices": self.tensor(data_type=6, dimensions=[2, 2]),
            }
            message = "one-dimensional"
        else:
            tensors = {
                "source": self.tensor(dimensions=[2, 2], channels=2),
                "values": self.tensor(dimensions=[2, 2]),
                "indices": self.tensor(data_type=6, dimensions=[2, 2]),
            }
            message = "one-channel 2D"
        bad_input = self.operator_spec(
            operator_name, tensors, [{"tensor": "source"}],
            [{"tensor": "values"}, {"tensor": "indices"}],
        )
        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(bad_input)

        if operator_name == "SORT_VEC":
            tensors["source"] = self.tensor(usage=2, dimensions=[2])
            tensors["values"] = self.tensor(usage=2, dimensions=[2])
            tensors["indices"] = self.tensor(usage=2, data_type=6, dimensions=[2])
        else:
            tensors["source"] = self.tensor(dimensions=[2, 2])
            tensors["values"] = self.tensor(dimensions=[2, 2])
            tensors["indices"] = self.tensor(data_type=6, dimensions=[2, 2])
        bad_index = self.operator_spec(
            operator_name, tensors, [{"tensor": "source"}],
            [{"tensor": "values"}, {"tensor": "indices"}],
        )
        with pytest.raises(ValueError, match="index result must be integer"):
            validate_pipeline_spec(bad_index)

    @pytest.mark.parametrize(
        "attrs",
        [["8000;PCM_16BIT"], ["96000;PCM_32BIT"], ["48000;PCM_FLOAT"]],
    )
    def test_validate_pipeline_spec_accepts_valid_microphone_attrs(self, attrs):
        validate_pipeline_spec(self.microphone_spec(attrs))

    @pytest.mark.parametrize(
        "attrs",
        [
            [],
            ["48000;PCM_FLOAT", "16000;PCM_16BIT"],
            ["7999;PCM_16BIT"],
            ["96001;PCM_16BIT"],
            ["0;PCM_16BIT"],
            ["48000.0;PCM_16BIT"],
            ["48000PCM_16BIT"],
            ["48000;PCM_8BIT"],
            ["48000;PCM_16BIT;extra"],
        ],
    )
    def test_validate_pipeline_spec_rejects_invalid_microphone_attrs(self, attrs):
        with pytest.raises(ValueError):
            validate_pipeline_spec(self.microphone_spec(attrs))

    @pytest.mark.parametrize(
        ("output_index", "tensor", "attrs", "message"),
        [
            (0, {"dimensions": [128, 1], "channels": 1, "data_type": 6}, ["48000;PCM_FLOAT"], "stereo output"),
            (1, {"dimensions": [128, 1], "channels": 2, "data_type": 6}, ["48000;PCM_FLOAT"], "one channel"),
            (2, {"dimensions": [128, 1], "channels": 1, "data_type": 4}, ["48000;PCM_FLOAT"], "must match PCM_FLOAT"),
            (3, {"dimensions": [1], "channels": 1, "data_type": 5, "usage": 5}, ["48000;PCM_FLOAT"], "timestamp"),
        ],
    )
    def test_validate_pipeline_spec_rejects_invalid_microphone_tensor(
        self, output_index, tensor, attrs, message
    ):
        spec = self.microphone_spec(attrs)
        output_name = ("stereo", "left", "right", "timestamp")[output_index]
        spec["tensors"][output_name] = self.tensor(**tensor)

        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("operator_name", "tensors", "inputs", "outputs", "message"),
        [
            (
                "CAMERA_SPACE_TO_WORLD",
                {
                    "timestamp": tensor.__func__(data_type=5, usage=5, dimensions=[1], channels=1),
                    "world": tensor.__func__(dimensions=[4, 4]),
                },
                [{"tensor": "timestamp"}], [{"tensor": "world"}, None], "timestamp",
            ),
            (
                "DEPTH", {"depth": tensor.__func__(data_type=5, dimensions=[4, 4])},
                [], [{"tensor": "depth"}], "floating-point",
            ),
            (
                "UPLOAD_TEXTURE_TO_GLTF",
                {
                    "gltf": tensor.__func__(data_type=1, usage=7, is_gltf=True),
                    "image": tensor.__func__(data_type=6, dimensions=[4, 4], channels=3),
                    "texture": tensor.__func__(data_type=3, usage=2, dimensions=[1]),
                },
                [{"tensor": "gltf"}, {"tensor": "image"}], [{"tensor": "texture"}], "UINT8",
            ),
            (
                "SWITCH_GLTF_RENDER_STATUS",
                {
                    "gltf": tensor.__func__(data_type=1, usage=7, is_gltf=True),
                    "pose": tensor.__func__(dimensions=[3, 3]),
                },
                [{"tensor": "gltf"}, {"tensor": "pose"}, None, None], [], "4, 4",
            ),
        ],
    )
    def test_validate_pipeline_spec_rejects_camera_depth_and_gltf_contracts(
        self, operator_name, tensors, inputs, outputs, message
    ):
        spec = self.operator_spec(operator_name, tensors, inputs, outputs)

        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("data_type", "data", "message"),
        [
            (1, [0, 1, 2], "exactly 4 value"),
            (1, [0, 1, 2, 256], "out of range for UINT8"),
            (4, [0, 1, 2, 1.5], "must be an integer for INT16"),
            (6, [0.0, 1.0, 2.0, float("inf")], "must be finite"),
        ],
    )
    def test_validate_pipeline_spec_rejects_invalid_preloaded_data(self, data_type, data, message):
        spec = {
            "tensors": {
                "value": self.tensor(
                    data_type=data_type, dimensions=[2, 2], is_placeholder=False, data=data
                )
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match=message):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize("model_name", ["", "face-model", "face model", "模型"])
    def test_validate_pipeline_spec_rejects_invalid_model_name(self, model_name):
        model = {
            "bin_path": "models/model.tflite",
            "model_name": model_name,
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": "input", "shape": [1, 2], "encoding_type": "FP32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE", {"input": self.tensor(), "output": self.tensor()},
            [{"tensor": "input"}], [{"tensor": "output"}], model=model,
        )

        with pytest.raises(ValueError, match="model_name"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_defaults_omitted_model_name_to_main(self):
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": "input", "shape": [1, 2], "encoding_type": "FP32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        validate_pipeline_spec(self.operator_spec(
            "RUN_MODEL_INFERENCE", {"input": self.tensor(dimensions=[1, 2]), "output": self.tensor(dimensions=[1, 2])},
            [{"tensor": "input"}], [{"tensor": "output"}], model=model,
        ))

    def test_validate_pipeline_spec_rejects_model_binding_metadata_mismatch(self):
        model = {
            "bin_path": "models/model.tflite",
            "model_type": "tflite",
            "model_target": "cpu",
            "input": [{"name": "model_input", "shape": [1, 2], "encoding_type": "FP32"}],
            "output": [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}],
        }
        spec = self.operator_spec(
            "RUN_MODEL_INFERENCE",
            {"input": self.tensor(dimensions=[1, 2]), "output": self.tensor(dimensions=[1, 2])},
            [{"name": "wrong_input", "tensor": "input"}],
            [{"name": "output", "tensor": "output"}],
            model=model,
        )

        with pytest.raises(ValueError, match="binding name must match"):
            validate_pipeline_spec(spec)

    @pytest.mark.parametrize(
        ("operator_name", "inputs", "attrs"),
        [
            ("SCENEGRAPH_VISIBILITY", ["scene", "data"], []),
            ("UPDATE_COMPONENT", ["scene", "data"], ["/target:Transform.Scale"]),
            ("JAVASCRIPT", ["out"], ["out = in;"]),
        ],
    )
    def test_validate_pipeline_spec_accepts_schema_operator_aliases(
        self, operator_name, inputs, attrs
    ):
        tensors = {
            "scene": self.tensor(data_type=1, usage=7),
            "data": self.tensor(),
        }
        if operator_name == "JAVASCRIPT":
            tensors = {"out": self.tensor()}
        spec = {
            "tensors": tensors,
            "operators": [{
                "type": f"XR_SECURE_MR_OPERATOR_TYPE_{operator_name}_PICO",
                "inputs": [
                    None if ref is None else {"tensor": ref}
                    for ref in inputs
                ],
                "outputs": [
                    {"tensor": "out"},
                ] if operator_name == "JAVASCRIPT" else [],
                "attrs": attrs,
            }],
            "inputs": [],
            "outputs": [],
        }

        validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_rejects_native_binding_contracts(self):
        scene_spec = self.operator_spec(
            "SCENEGRAPH_VISIBILITY",
            {"scene": self.tensor(data_type=1, usage=7), "visible": self.tensor(data_type=1)},
            [{"tensor": "scene"}, None], [],
        )
        with pytest.raises(ValueError, match="requires inputs\[1\]"):
            validate_pipeline_spec(scene_spec)

        update_spec = self.operator_spec(
            "UPDATE_GLTF",
            {"gltf": self.tensor(data_type=1, usage=7)},
            [{"tensor": "gltf"}, None, None], [], ["world pose"],
        )
        validate_pipeline_spec(update_spec)

        chw_tensors = {
            "hwc": self.tensor(dimensions=[6], channels=1, usage=2),
            "chw": self.tensor(dimensions=[3, 2, 3], usage=6),
        }
        chw_spec = self.operator_spec(
            "CHW_HWC", chw_tensors,
            [{"tensor": "hwc"}], [{"tensor": "chw"}],
        )
        with pytest.raises(ValueError, match="must use MAT tensors"):
            validate_pipeline_spec(chw_spec)

    def test_run_pipeline_python_uses_attrs_and_spec_aliases(self):
        spec = {
            "tensors": {
                "input": self.tensor(dimensions=[2, 2]),
                "output": self.tensor(dimensions=[2, 2]),
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": [{"tensor": "input"}] + [None] * 9,
                    "outputs": [{"tensor": "output"}],
                    "attrs": ["{0} * 3.0"],
                }
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        outputs = run_pipeline_python(spec, {"input": np.ones((2, 2), dtype=np.float32)})

        np.testing.assert_allclose(outputs["output"], np.ones((2, 2), dtype=np.float32) * 3.0)

    def test_run_pipeline_python_preserves_nullable_output_slots(self):
        spec = {
            "tensors": {
                "scores": self.tensor(dimensions=[3, 1]),
                "boxes": self.tensor(dimensions=[3, 4]),
                "indices": self.tensor(data_type=5, dimensions=[3, 1]),
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO",
                    "inputs": [{"tensor": "scores"}, {"tensor": "boxes"}],
                    "outputs": [None, None, {"tensor": "indices"}],
                    "attrs": ["0.5"],
                }
            ],
            "inputs": ["scores", "boxes"],
            "outputs": ["indices"],
        }

        outputs = run_pipeline_python(
            spec,
            {
                "scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
                "boxes": np.array(
                    [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                    dtype=np.float32,
                ),
            },
        )

        np.testing.assert_array_equal(outputs["indices"], np.array([0, 2], dtype=np.int32))

    def test_validate_pipeline_spec_accepts_2d_matrix_tensor(self):
        """Test that MAT tensors can be declared as row/column matrices."""
        spec = {
            "tensors": {
                "row_vec": {
                    "dimensions": [1, 4],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "named_matrix": {
                    "dimensions": [4, 1],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        }

        validate_pipeline_spec(spec)

    @pytest.mark.parametrize("dimensions", [[], [4]])
    def test_validate_pipeline_spec_rejects_1d_matrix_tensor(self, dimensions):
        """Test that matrix/MAT tensors require at least two dimensions."""
        spec = {
            "tensors": {
                "bad_vec": {
                    "dimensions": dimensions,
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                }
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="(non-empty array of positive integers|matrix tensors must have at least 2 dimensions)"):
            validate_pipeline_spec(spec)

    def test_run_pipeline_python_rejects_invalid_matrix_tensor(self):
        """Test host verification catches invalid JSON before native runtime."""
        spec = {
            "tensors": {
                "bad_vec": {
                    "dimensions": [4],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                }
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="Tensor 'bad_vec'.*matrix tensors"):
            run_pipeline_python(spec, {})

    def test_validate_pipeline_spec_rejects_bad_swap_hwc_chw_shape(self):
        """Test swap_hwc_chw rejects 4D CHW tensor declarations."""
        spec = {
            "tensors": {
                "image_hwc": {
                    "dimensions": [1024, 768],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "bad_chw": {
                    "dimensions": [1, 3, 1024, 768],
                    "channels": 1,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_CHW_HWC_PICO",
                    "inputs": [{"tensor": "image_hwc"}],
                    "outputs": [{"tensor": "bad_chw"}],
                }
            ],
            "inputs": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="swap_hwc_chw"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_accepts_swap_hwc_chw_shape(self):
        """Test swap_hwc_chw accepts SecureMR channelized CHW tensor declarations."""
        spec = {
            "tensors": {
                "image_hwc": {
                    "dimensions": [1024, 768],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "image_chw": {
                    "dimensions": [3, 1024, 768],
                    "channels": 1,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_CHW_HWC_PICO",
                    "inputs": [{"tensor": "image_hwc"}],
                    "outputs": [{"tensor": "image_chw"}],
                }
            ],
            "inputs": [],
            "outputs": [],
        }

        validate_pipeline_spec(spec)

    def test_compare_outputs_success(self):
        """Test successful output comparison."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        result = compare_outputs(expected, actual)

        assert result.success
        assert result.error_message is None

    def test_compare_outputs_within_tolerance(self):
        """Test comparison within tolerance."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0001, 2.0001, 3.0001], dtype=np.float32)}

        result = compare_outputs(expected, actual, rtol=1e-3, atol=1e-3)

        assert result.success

    def test_compare_outputs_failure(self):
        """Test failed output comparison."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0, 2.0, 5.0], dtype=np.float32)}

        result = compare_outputs(expected, actual, rtol=1e-4, atol=1e-4)

        assert not result.success
        assert result.error_message is not None
        assert "out" in result.error_message

    def test_compare_outputs_missing_key(self):
        """Test comparison with missing output key."""
        expected = {"out1": np.array([1.0]), "out2": np.array([2.0])}
        actual = {"out1": np.array([1.0])}

        result = compare_outputs(expected, actual)

        assert not result.success
        assert "Missing output" in result.error_message

    def test_compare_outputs_shape_mismatch(self):
        """Test comparison with shape mismatch."""
        expected = {"out": np.array([1.0, 2.0, 3.0])}
        actual = {"out": np.array([1.0, 2.0])}

        result = compare_outputs(expected, actual)

        assert not result.success
        assert "Shape mismatch" in result.error_message


class TestIntegration:
    """Integration tests for the full py2smr workflow."""

    def test_full_workflow_arithmetic(self):
        """Test full workflow with arithmetic operations."""
        @trace(inputs=["input"], outputs=["output"])
        def normalize_image(input):
            return ops.arithmetic(input, "{0} / 255.0")

        # Create test input
        input_arr = np.array([[100, 200], [50, 150]], dtype=np.uint8)

        # Trace execution
        result, ctx = normalize_image.trace(input=input_arr)

        # Convert to pipeline spec
        spec = convert(ctx)

        # Verify spec structure
        assert spec["inputs"] == ["input"]
        assert spec["outputs"] == ["output"]
        assert len(spec["operators"]) == 1
        assert spec["operators"][0]["attrs"] == ["{0} / 255.0"]

        # Verify result
        expected = input_arr.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected)

    def test_full_workflow_multiple_ops(self):
        """Test full workflow with multiple operations."""
        @trace(inputs=["image"], outputs=["processed"])
        def preprocess(image):
            # Normalize to [0, 1]
            normalized = ops.arithmetic(image, "{0} / 255.0")
            # Scale to [-1, 1]
            scaled = ops.arithmetic(normalized, "{0} * 2.0 - 1.0")
            return scaled

        input_arr = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result, ctx = preprocess.trace(image=input_arr)

        spec = convert(ctx)

        assert len(spec["operators"]) == 2
        assert spec["operators"][0]["attrs"] == ["{0} / 255.0"]
        assert spec["operators"][1]["attrs"] == ["{0} * 2.0 - 1.0"]

        # Verify result
        expected = input_arr.astype(np.float32) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(result, expected)

    def test_full_workflow_elementwise_ops(self):
        """Test full workflow with elementwise operations."""
        @trace(inputs=["a", "b"], outputs=["result"])
        def clamp(a, b):
            min_val = ops.elementwise_min(a, b)
            max_val = ops.elementwise_max(a, b)
            return ops.elementwise_multiply(min_val, max_val)

        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)

        result, ctx = clamp.trace(a=a, b=b)

        spec = convert(ctx)

        assert len(spec["operators"]) == 3
        assert "a" in spec["inputs"]
        assert "b" in spec["inputs"]

    def test_save_and_load_pipeline(self):
        """Test saving and loading pipeline JSON."""
        @trace(inputs=["x"], outputs=["y"])
        def double(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        _, ctx = double.trace(x=input_arr)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            convert(ctx, output=output_path)

            with open(output_path, "r") as f:
                loaded_spec = json.load(f)

            # Verify loaded spec is valid
            assert "tensors" in loaded_spec
            assert "operators" in loaded_spec
            assert loaded_spec["inputs"] == ["x"]
            assert loaded_spec["outputs"] == ["y"]
        finally:
            os.unlink(output_path)


class TestPythonExecutor:
    """Tests for the pure Python pipeline executor."""

    def test_run_pipeline_python_preserves_non_square_h_w_output_shape(self):
        """Host shape reconstruction must preserve schema [H, W] order."""
        spec = {
            "tensors": {
                "output": {
                    "dimensions": [3, 5],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_DEPTH_PICO",
                    "inputs": [],
                    "outputs": [{"tensor": "output"}],
                }
            ],
            "inputs": [],
            "outputs": ["output"],
        }

        outputs = run_pipeline_python(spec, {})

        assert outputs["output"].shape == (3, 5)

    def test_run_pipeline_python_basic(self):
        """Test basic pipeline execution with pure Python."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "input": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "output": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": [{"tensor": "input"}] + [None] * 9,
                    "outputs": [{"tensor": "output"}],
                    "attrs": ["{0} * 2.0"],
                }
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        outputs = run_pipeline_python(spec, {"input": input_arr})

        expected = input_arr * 2.0
        np.testing.assert_allclose(outputs["output"], expected)

    def test_run_pipeline_python_multiple_ops(self):
        """Test pipeline with multiple operators."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "input": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "temp": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "output": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": [{"tensor": "input"}] + [None] * 9,
                    "outputs": [{"tensor": "temp"}],
                    "attrs": ["{0} / 255.0"],
                },
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": [{"tensor": "temp"}] + [None] * 9,
                    "outputs": [{"tensor": "output"}],
                    "attrs": ["{0} * 2.0 - 1.0"],
                },
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        input_arr = np.array([[100.0, 200.0], [50.0, 150.0]], dtype=np.float32)
        outputs = run_pipeline_python(spec, {"input": input_arr})

        expected = input_arr / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(outputs["output"], expected)

    def test_run_pipeline_python_preserves_supplied_rectified_vst_outputs(self):
        """User-provided VST tensors should not be overwritten by host stubs."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "vst_right_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_left_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_timestamp": {
                    "dimensions": [1],
                    "channels": 4,
                    "data_type": 5,
                    "is_placeholder": True,
                    "usage": 5,
                },
                "vst_camera_matrix": {
                    "dimensions": [3, 3],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO",
                    "inputs": [],
                    "outputs": [
                        {"tensor": "vst_right_image"},
                        {"tensor": "vst_left_image"},
                        {"tensor": "vst_timestamp"},
                        {"tensor": "vst_camera_matrix"},
                    ],
                }
            ],
            "inputs": [],
            "outputs": ["vst_right_image", "vst_left_image", "vst_timestamp", "vst_camera_matrix"],
        }
        right = np.ones((2, 2, 3), dtype=np.uint8) * 9
        left = np.ones((2, 2, 3), dtype=np.uint8) * 7

        outputs = run_pipeline_python(
            spec,
            {"vst_right_image": right, "vst_left_image": left},
        )

        np.testing.assert_array_equal(outputs["vst_right_image"], right)
        np.testing.assert_array_equal(outputs["vst_left_image"], left)

    def test_run_pipeline_python_decodes_mediapipe_face_postprocess(self):
        """Known face detector JavaScript postprocess should produce post_det."""
        from securemr.py2smr.verifier import run_pipeline_python

        coords_1 = np.zeros((512, 16), dtype=np.float32)
        coords_2 = np.zeros((384, 16), dtype=np.float32)
        scores_1 = np.full((512, 1), -10.0, dtype=np.float32)
        scores_2 = np.full((384, 1), -10.0, dtype=np.float32)
        scores_1[0, 0] = 10.0
        coords_1[0, :14] = [
            0.0,
            0.0,
            20.0,
            30.0,
            -5.0,
            -5.0,
            5.0,
            -5.0,
            0.0,
            0.0,
            -4.0,
            6.0,
            4.0,
            6.0,
        ]
        spec = {
            "tensors": {
                "box_coords_1": {
                    "dimensions": [512, 16],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "box_coords_2": {
                    "dimensions": [384, 16],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "box_scores_1": {
                    "dimensions": [512, 1],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "box_scores_2": {
                    "dimensions": [384, 1],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "post_det_template": {
                    "dimensions": [1, 21],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "post_det": {
                    "dimensions": [1, 21],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO",
                    "inputs": [
                        {"name": "box_coords_1", "tensor": "box_coords_1"},
                        {"name": "box_coords_2", "tensor": "box_coords_2"},
                        {"name": "box_scores_1", "tensor": "box_scores_1"},
                        {"name": "box_scores_2", "tensor": "box_scores_2"},
                        {"name": "post_det_template", "tensor": "post_det_template"},
                        {"name": "post_det", "tensor": "post_det"},
                    ],
                    "outputs": [
                        {"name": name, "tensor": name}
                        for name in (
                            "box_coords_1", "box_coords_2", "box_scores_1",
                            "box_scores_2", "post_det_template", "post_det",
                        )
                    ],
                    "attrs": ["function anchorFor(){} function decodeDetection(){}"],
                }
            ],
            "inputs": [],
            "outputs": ["post_det"],
        }

        outputs = run_pipeline_python(
            spec,
            {
                "box_coords_1": coords_1,
                "box_coords_2": coords_2,
                "box_scores_1": scores_1,
                "box_scores_2": scores_2,
                "post_det_template": np.zeros((1, 21), dtype=np.float32),
            },
        )

        assert outputs["post_det"].shape == (1, 21)
        assert outputs["post_det"][0, 4] > 0.99

    def test_verify_with_python_executor(self):
        """Test verify function uses pure Python executor."""
        @trace(inputs=["x"], outputs=["y"])
        def double(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, ctx = double.trace(x=input_arr)

        spec = convert(ctx)

        # Verify should work without native bindings
        verification = verify(
            pipeline=spec,
            inputs={"x": input_arr},
            expected_outputs={"y": result},
        )

        assert verification.success
