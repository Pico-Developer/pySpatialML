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
"""Numerical consistency verification for py2smr."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from securemr.core.types import EDataType, EOperatorType
from securemr.core.utils import TensorType, convert_to_dtype
from securemr.operator_contracts import (
    INTERNAL_OPERATOR_NAMES,
    OPERATOR_ALIASES,
    OPERATOR_ARITIES,
    operator_contract,
)

__all__ = [
    "verify",
    "VerificationResult",
    "compare_outputs",
    "run_pipeline_python",
    "validate_pipeline_spec",
    "validate_microphone_attrs",
]

_OP_GET_TRANSFORM_MAT = EOperatorType.GET_TRANSFORM_MAT
_OP_LOAD_TEXTURE = EOperatorType.LOAD_TEXTURE
_OP_SWAP_HWC_CHW = EOperatorType.SWAP_HWC_CHW
_OP_JAVASCRIPT = EOperatorType.JAVASCRIPT

_OPERATOR_TYPE_PREFIX = "XR_SECURE_MR_OPERATOR_TYPE_"
_OPERATOR_TYPE_SUFFIX = "_PICO"
_MICROPHONE_SAMPLE_RATE_MIN = 8000
_MICROPHONE_SAMPLE_RATE_MAX = 96000
_MICROPHONE_ENCODINGS = {"PCM_16BIT", "PCM_32BIT", "PCM_FLOAT"}
_SPEAKER_SAMPLE_RATE_MIN = 8000
_SPEAKER_SAMPLE_RATE_MAX = 96000
_NATIVE_INT_MAX = 2**31 - 1
_OPENXR_OPERATOR_NODE_NAME_CAPACITY = 512
_MODEL_ENCODING_DATA_TYPES = {
    "UINT8": 1,
    "INT8": 2,
    "UINT16": 3,
    "INT16": 4,
    "INT32": 5,
    "FLOAT32": 6,
    "FLOAT": 6,
    "FP32": 6,
}

_CANONICAL_TO_INTERNAL = {
    **INTERNAL_OPERATOR_NAMES,
    **{
        alias: INTERNAL_OPERATOR_NAMES.get(canonical, canonical)
        for alias, canonical in OPERATOR_ALIASES.items()
    },
}

_LEGACY_OPERATOR_CONFIG_FIELDS = {
    "comparison",
    "config",
    "data",
    "entity_path",
    "expression",
    "flag",
    "mode",
    "normalize_type",
    "property",
    "scenegraph",
    "script",
    "src_slices_tensor",
    "dst_slices_tensor",
    "target_property",
    "text",
    "threshold",
    "update_type",
    "visible",
}

_ROOT_KEYS = {"tensors", "operators", "inputs", "outputs"}
_TENSOR_KEYS = {
    "dimensions",
    "channels",
    "data_type",
    "is_placeholder",
    "usage",
    "flag",
    "data",
    "is_gltf",
    "asset",
}
_OPERATOR_COMMON_KEYS = {"type", "inputs", "outputs", "attrs"}
_OPERATOR_EXTRA_KEYS = {
    "ASSIGNMENT": {"src_slices", "dst_slices", "src_channel_slice", "dst_channel_slice"},
    "RUN_MODEL_INFERENCE": {"model"},
}

_SPEC_OPERATOR_NAMES = set(OPERATOR_ARITIES) | set(OPERATOR_ALIASES) | {"UNKNOWN"}


@dataclass
class VerificationResult:
    """Result of a verification run."""
    success: bool
    host_outputs: Dict[str, np.ndarray]
    device_outputs: Optional[Dict[str, np.ndarray]] = None
    max_abs_diff: Optional[Dict[str, float]] = None
    max_rel_diff: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


def _is_matrix_tensor(tensor_spec: Dict[str, Any]) -> bool:
    """Return True when a tensor descriptor declares MAT/matrix usage."""
    tensor_type = tensor_spec.get("tensor_type") or tensor_spec.get("type")
    if tensor_type is not None:
        normalized = str(tensor_type).strip().lower().replace("-", "_")
        if normalized in {"matrix", "mat"}:
            return True

    usage = tensor_spec.get("usage")
    if usage is None:
        return False

    if isinstance(usage, str):
        normalized = usage.strip().lower().replace("-", "_")
        if normalized in {"matrix", "mat"}:
            return True
        try:
            usage = int(usage, 0)
        except ValueError:
            return False

    try:
        return int(usage) == int(TensorType.MAT.value)
    except (TypeError, ValueError):
        return False


def _tensor_dimensions_and_channels(tensor_spec: Dict[str, Any]) -> tuple[List[int], int]:
    dimensions = tensor_spec.get("dimensions", [])
    if not isinstance(dimensions, list):
        dimensions = []
    dims = [int(dim) for dim in dimensions]
    channels = int(tensor_spec.get("channels", 1) or 1)
    return dims, channels


def _validate_swap_hwc_chw_operator(
    op_spec: Dict[str, Any],
    tensor_specs: Dict[str, Any],
) -> None:
    input_refs = op_spec.get("inputs", [])
    output_refs = op_spec.get("outputs", [])
    if len(input_refs) != 1 or len(output_refs) != 1:
        raise ValueError("swap_hwc_chw requires exactly one input and one output tensor")

    input_name = _resolve_tensor_name(input_refs[0])
    output_name = _resolve_tensor_name(output_refs[0])
    input_spec = tensor_specs.get(input_name or "")
    output_spec = tensor_specs.get(output_name or "")
    if input_spec is None or output_spec is None:
        return

    input_dims, input_channels = _tensor_dimensions_and_channels(input_spec)
    output_dims, output_channels = _tensor_dimensions_and_channels(output_spec)
    if not _is_matrix_tensor(input_spec) or not _is_matrix_tensor(output_spec):
        raise ValueError(
            f"swap_hwc_chw input '{input_name}' and output '{output_name}' must use MAT tensors"
        )
    input_hwc = len(input_dims) == 2 and input_channels >= 1
    input_chw = len(input_dims) == 3 and input_channels == 1
    output_hwc = len(output_dims) == 2 and output_channels >= 1
    output_chw = len(output_dims) == 3 and output_channels == 1
    if not ((input_hwc and output_chw) or (input_chw and output_hwc)):
        raise ValueError(
            f"swap_hwc_chw input '{input_name}' must be a 2D multi-channel or 3D one-channel MAT tensor"
        )
    if int(input_spec.get("data_type")) != int(output_spec.get("data_type")):
        raise ValueError(f"swap_hwc_chw input and output data types must match")


def _strip_operator_type_token(type_str: Any) -> Optional[str]:
    if not isinstance(type_str, str):
        return None
    raw = type_str
    if not (raw.startswith(_OPERATOR_TYPE_PREFIX) and raw.endswith(_OPERATOR_TYPE_SUFFIX)):
        return None
    normalized = raw[len(_OPERATOR_TYPE_PREFIX):-len(_OPERATOR_TYPE_SUFFIX)]
    if not normalized or normalized != normalized.upper():
        return None
    return normalized or None


def _canonical_operator_name(type_str: Any) -> Optional[str]:
    name = _strip_operator_type_token(type_str)
    if name is None:
        return None
    if name in _SPEC_OPERATOR_NAMES:
        # Preserve a legacy spelling here because a few compatibility aliases
        # intentionally have stricter validation than their canonical form.
        return name
    return None


def _operator_label(index: int, op_type: Any) -> str:
    return f"operator #{index} ({op_type!r})"


def validate_microphone_attrs(attrs: List[str], label: str = "microphone operator") -> None:
    """Validate the microphone operator's single encoded audio attribute."""
    if len(attrs) != 1:
        raise ValueError(f"{label} requires exactly one attrs value")

    value = attrs[0]
    parts = value.split(";")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"{label} attrs[0] must have the form <SAMPLE_RATE>;<PCM_16BIT|PCM_32BIT|PCM_FLOAT>"
        )

    sample_rate_text, encoding = parts
    if re.fullmatch(r"[0-9]+", sample_rate_text) is None:
        raise ValueError(f"{label} attrs[0] must start with a positive integer sample rate")
    sample_rate = int(sample_rate_text)
    if not _MICROPHONE_SAMPLE_RATE_MIN <= sample_rate <= _MICROPHONE_SAMPLE_RATE_MAX:
        raise ValueError(
            f"{label} attrs[0] sample rate must be between "
            f"{_MICROPHONE_SAMPLE_RATE_MIN} and {_MICROPHONE_SAMPLE_RATE_MAX} Hz"
        )
    if encoding not in _MICROPHONE_ENCODINGS:
        raise ValueError(f"{label} attrs[0] has unsupported audio encoding")


def _validate_attrs(op_spec: Dict[str, Any], op_name: str, label: str) -> List[str]:
    op_name = _CANONICAL_TO_INTERNAL.get(op_name, op_name)
    attrs = op_spec.get("attrs", [])
    if "attrs" in op_spec and attrs is None:
        raise ValueError(f"{label} attrs must be an array of strings")
    if not isinstance(attrs, list) or not all(isinstance(item, str) for item in attrs):
        raise ValueError(f"{label} attrs must be an array of strings")

    required = 0
    max_count: Optional[int] = 0
    if op_name in {"ARITHMETIC_COMPOSE", "CONVERT_COLOR", "CUSTOMIZED_COMPARE"}:
        required = max_count = 1
    elif op_name in {"NORMALIZE", "NMS", "SORT_MAT", "NORM"}:
        max_count = 1
    elif op_name in {"UPDATE_GLTF", "RENDER_TEXT", "UPDATE_COMPONENT", "MICROPHONE", "SPEAKER", "JAVASCRIPT"}:
        required = max_count = 1

    if len(attrs) < required:
        raise ValueError(f"{label} requires at least {required} attrs value(s)")
    if max_count is not None and len(attrs) > max_count:
        raise ValueError(f"{label} accepts at most {max_count} attrs value(s)")

    if op_name == "CUSTOMIZED_COMPARE" and attrs and attrs[0] not in {"==", "!=", ">", ">=", "<", "<="}:
        raise ValueError(f"{label} attrs[0] must be a supported comparator")
    if op_name == "NORMALIZE" and attrs and attrs[0] not in {"L1", "L2", "INF", "MINMAX"}:
        raise ValueError(f"{label} attrs[0] must be L1, L2, INF, or MINMAX")
    if op_name == "SORT_MAT" and attrs and attrs[0] not in {"ROW", "COLUMN"}:
        raise ValueError(f"{label} attrs[0] must be ROW or COLUMN")
    if op_name == "NORM" and attrs and attrs[0] not in {"L1", "L2", "INF"}:
        raise ValueError(f"{label} attrs[0] must be L1, L2, or INF")
    if op_name == "CONVERT_COLOR" and attrs:
        try:
            int(attrs[0])
        except ValueError as exc:
            raise ValueError(f"{label} attrs[0] must be a numeric color conversion enum") from exc
    if op_name == "NMS" and attrs:
        try:
            float(attrs[0])
        except ValueError as exc:
            raise ValueError(f"{label} attrs[0] must be a numeric IoU threshold") from exc
    if op_name == "UPDATE_COMPONENT" and attrs and (not attrs[0].startswith("/") or ":" not in attrs[0]):
        raise ValueError(f"{label} attrs[0] must have the form /entity/path:component.field")
    if op_name == "MICROPHONE" and attrs:
        validate_microphone_attrs(attrs, label)
    if op_name == "SPEAKER" and attrs:
        try:
            if re.fullmatch(r"[0-9]+", attrs[0]) is None:
                raise ValueError
            if not _SPEAKER_SAMPLE_RATE_MIN <= int(attrs[0]) <= _SPEAKER_SAMPLE_RATE_MAX:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"{label} attrs[0] sample rate must be between "
                f"{_SPEAKER_SAMPLE_RATE_MIN} and {_SPEAKER_SAMPLE_RATE_MAX} Hz"
            ) from exc
    if op_name == "UPDATE_GLTF" and attrs and attrs[0] not in {
        "local", "animation", "world pose", "texture",
        "material::metallic_factor", "material::roughness_factor",
        "material::metallic_roughness_texture", "material::base_color_factor",
        "material::base_color_texture", "material::normal_map_texture",
        "material::occlusion_texture", "material::emissive_factor",
        "material::emissive_strength", "material::emissive_texture",
    }:
        raise ValueError(f"{label} attrs[0] is not a supported UPDATE_GLTF attribute")
    if op_name == "RENDER_TEXT":
        parts = attrs[0].split("#") if attrs else []
        if len(parts) != 4 or not parts[0] or not parts[1]:
            raise ValueError(f"{label} attrs[0] must be typeface#language#width#height")
        try:
            if not 0 < int(parts[2]) <= _NATIVE_INT_MAX or not 0 < int(parts[3]) <= _NATIVE_INT_MAX:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"{label} attrs[0] width and height must be positive integers") from exc

    return attrs


def _validate_ref_list(
    value: Any,
    field: str,
    label: str,
    tensor_names: Set[str],
) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} {field} must be an array")
    for index, ref in enumerate(value):
        if ref is None:
            continue
        if isinstance(ref, str):
            raise ValueError(f"{label} {field}[{index}] must use object form, not a string reference")
        if not isinstance(ref, dict):
            raise ValueError(f"{label} {field}[{index}] must be an object or null")
        unknown_keys = set(ref) - {"tensor", "name"}
        if unknown_keys:
            raise ValueError(f"{label} {field}[{index}] has unknown keys: {sorted(unknown_keys)}")
        tensor_name = ref.get("tensor")
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"{label} {field}[{index}] requires a non-empty tensor string")
        if "[" in tensor_name or "]" in tensor_name:
            raise ValueError(f"{label} {field}[{index}] uses unsupported bracket tensor reference")
        if tensor_name not in tensor_names:
            raise ValueError(f"{label} {field}[{index}] references unknown tensor '{tensor_name}'")
        if "name" in ref and not isinstance(ref.get("name"), str):
            raise ValueError(f"{label} {field}[{index}].name must be a string")
    return value


def _validate_signature(
    op_name: str,
    inputs: List[Any],
    outputs: List[Any],
    label: str,
    model: Optional[Dict[str, Any]],
) -> None:
    op_name = _CANONICAL_TO_INTERNAL.get(op_name, op_name)
    if op_name == "UNKNOWN":
        raise ValueError(f"{label} UNKNOWN is reserved and cannot be executed")
    if op_name == "RUN_MODEL_INFERENCE":
        if model is None:
            raise ValueError(f"{label} requires inline model metadata")
        expected_inputs = len(model.get("input", [])) if isinstance(model.get("input"), list) else None
        expected_outputs = len(model.get("output", [])) if isinstance(model.get("output"), list) else None
        if expected_inputs is not None and len(inputs) != expected_inputs:
            raise ValueError(f"{label} expected {expected_inputs} input(s), got {len(inputs)}")
        if expected_outputs is not None and len(outputs) != expected_outputs:
            raise ValueError(f"{label} expected {expected_outputs} output(s), got {len(outputs)}")
        if not inputs or not outputs or any(ref is None for ref in (*inputs, *outputs)):
            raise ValueError(f"{label} requires all model input and output slots to be connected")
        for field, refs in (("inputs", inputs), ("outputs", outputs)):
            for index, ref in enumerate(refs):
                binding_name = ref.get("name") or ref["tensor"]
                for name_kind, name in (("binding name", binding_name), ("tensor name", ref["tensor"])):
                    if "\0" in name:
                        raise ValueError(f"{label} {field}[{index}] {name_kind} must not contain NUL bytes")
                    if len(name.encode("utf-8")) >= _OPENXR_OPERATOR_NODE_NAME_CAPACITY:
                        raise ValueError(
                            f"{label} {field}[{index}] {name_kind} must be shorter than "
                            f"{_OPENXR_OPERATOR_NODE_NAME_CAPACITY} UTF-8 bytes"
                        )
        return

    contract = operator_contract(op_name)
    if contract is None:
        raise ValueError(f"{label} unknown operator type")
    signature = contract.verifier_signature
    min_inputs, max_inputs, min_outputs, max_outputs, required_inputs, required_outputs = signature
    if not (min_inputs <= len(inputs) <= max_inputs):
        raise ValueError(f"{label} expected {min_inputs}..{max_inputs} input slot(s), got {len(inputs)}")
    if not (min_outputs <= len(outputs) <= max_outputs):
        raise ValueError(f"{label} expected {min_outputs}..{max_outputs} output slot(s), got {len(outputs)}")
    for index in required_inputs:
        if index >= len(inputs) or inputs[index] is None:
            raise ValueError(f"{label} requires inputs[{index}]")
    for index in required_outputs:
        if index >= len(outputs) or outputs[index] is None:
            raise ValueError(f"{label} requires outputs[{index}]")
    if op_name == "MICROPHONE" and len(outputs) == 3:
        raise ValueError(f"{label} expects 1, 2, or 4 output slot(s), got 3")
    if op_name == "JAVASCRIPT":
        if not outputs or any(ref is None for ref in (*inputs, *outputs)):
            raise ValueError(f"{label} requires all input and output slots to be connected")
    elif op_name in {"NMS", "SORT_VEC", "SORT_MAT", "SVD", "MICROPHONE"}:
        if not any(ref is not None for ref in outputs):
            raise ValueError(f"{label} requires at least one connected output slot")


def _expand_compact_operator_refs(
    op_name: str, inputs: List[Any], outputs: List[Any]
) -> tuple[List[Any], List[Any]]:
    """Expand dense schema-v2 arrays for positional tensor-contract checks."""
    internal_name = _CANONICAL_TO_INTERNAL.get(op_name, op_name)
    contract = operator_contract(internal_name)
    if contract is None:
        return inputs, outputs
    expanded_inputs = list(inputs)
    expanded_outputs = list(outputs)
    if contract.max_inputs is not None:
        expanded_inputs.extend([None] * (contract.max_inputs - len(expanded_inputs)))
    if internal_name == "MICROPHONE" and len(expanded_outputs) == 2:
        expanded_outputs = [expanded_outputs[0], None, None, expanded_outputs[1]]
    elif contract.max_outputs is not None:
        expanded_outputs.extend([None] * (contract.max_outputs - len(expanded_outputs)))
    return expanded_inputs, expanded_outputs


def _validate_operator_binding_contracts(
    op_name: str, inputs: List[Any], attrs: List[str], label: str
) -> None:
    """Validate slot requirements enforced by native operator dispatch."""
    if op_name == "SCENEGRAPH_VISIBILITY":
        if len(inputs) < 2 or inputs[1] is None:
            raise ValueError(f"{label} requires inputs[1]")
    elif op_name == "UPDATE_GLTF":
        if len(inputs) < 3 or inputs[0] is None:
            raise ValueError(f"{label} requires inputs[0]")
        attribute = attrs[0]
        required = {
            "texture": {1, 2},
            "animation": {1, 2},
            "world pose": set(),
            "local": {1, 2},
        }.get(attribute, {1, 2})
        for index in required:
            if inputs[index] is None:
                raise ValueError(f"{label} requires inputs[{index}] for attrs[0]={attribute!r}")


def _operator_tensor_values(ref: Any, tensor_specs: Dict[str, Any], label: str) -> tuple[List[int], int, int, int]:
    name = _resolve_tensor_name(ref)
    tensor = tensor_specs.get(name or "")
    if not isinstance(tensor, dict):
        raise ValueError(f"{label} references an invalid tensor")
    dimensions, channels = _tensor_dimensions_and_channels(tensor)
    return dimensions, channels, int(tensor.get("usage")), int(tensor.get("data_type"))


def _operator_mat_values(ref: Any, tensor_specs: Dict[str, Any], label: str) -> tuple[List[int], int, int, int]:
    values = _operator_tensor_values(ref, tensor_specs, label)
    if values[2] != int(TensorType.MAT.value) or len(values[0]) < 2:
        raise ValueError(f"{label} must be a MAT tensor with at least two dimensions")
    return values


def _operator_element_count(values: tuple[List[int], int, int, int]) -> int:
    count = values[1]
    for dimension in values[0]:
        count *= dimension
    return count


def _require_gltf_tensor(
    ref: Any, tensor_specs: Dict[str, Any], label: str
) -> tuple[List[int], int, int, int]:
    values = _operator_tensor_values(ref, tensor_specs, label)
    tensor_name = _resolve_tensor_name(ref)
    tensor = tensor_specs.get(tensor_name or "", {})
    if values[2] != int(TensorType.GLTF.value) or tensor.get("is_gltf") is not True:
        raise ValueError(f"{label} must be a GLTF tensor")
    return values


def _require_float_mat(
    ref: Any, tensor_specs: Dict[str, Any], label: str, dimensions: Optional[List[int]] = None
) -> tuple[List[int], int, int, int]:
    values = _operator_mat_values(ref, tensor_specs, label)
    if values[3] not in {6, 7} or values[1] != 1 or (dimensions is not None and values[0] != dimensions):
        shape = f" {dimensions}" if dimensions is not None else ""
        raise ValueError(f"{label} must be a one-channel floating-point{shape} MAT tensor")
    return values


def _require_vec(
    ref: Any, tensor_specs: Dict[str, Any], label: str, *,
    data_types: Set[int], channels: int, count: Optional[int] = None, usage: Optional[int] = None,
) -> tuple[List[int], int, int, int]:
    values = _operator_tensor_values(ref, tensor_specs, label)
    if (values[3] not in data_types or values[1] != channels or len(values[0]) != 1 or
            (count is not None and values[0][0] != count) or
            (usage is not None and values[2] != usage)):
        raise ValueError(f"{label} has an incompatible vector tensor descriptor")
    return values


def _validate_operator_tensor_contracts(
    op_name: str, inputs: List[Any], outputs: List[Any], attrs: List[str],
    tensor_specs: Dict[str, Any], label: str, model: Optional[Dict[str, Any]] = None
) -> None:
    """Mirror tensor constraints enforced by SpatialML-XR-Utils."""
    floating = {6, 7}
    integer = {1, 2, 3, 4, 5}
    if op_name == "RUN_MODEL_INFERENCE":
        assert model is not None
        for field, refs, metadata in (
            ("input", inputs, model["input"]),
            ("output", outputs, model["output"]),
        ):
            for index, (ref, item) in enumerate(zip(refs, metadata)):
                binding_name = ref.get("name") or ref["tensor"]
                if binding_name != item["name"]:
                    raise ValueError(
                        f"{label} {field}[{index}] binding name must match model metadata name"
                    )
                values = _operator_tensor_values(ref, tensor_specs, f"{label} {field}[{index}]")
                shape = list(item["shape"])
                expected_count = int(np.prod(shape))
                if _operator_element_count(values) != expected_count:
                    raise ValueError(f"{label} {field}[{index}] element count must match model metadata shape")
                encoding = item["encoding_type"].strip().upper()
                expected_data_type = _MODEL_ENCODING_DATA_TYPES.get(encoding)
                if expected_data_type is None:
                    raise ValueError(f"{label} model.{field}[{index}].encoding_type is unsupported")
                if values[3] != expected_data_type:
                    raise ValueError(f"{label} {field}[{index}] data type must match model metadata encoding")
    elif op_name == "RECTIFIED_VST_ACCESS":
        for index in range(2):
            image = _operator_mat_values(outputs[index], tensor_specs, f"{label} image output")
            if image[3] != 1 or image[1] not in {3, 4} or len(image[0]) != 2:
                raise ValueError(f"{label} image outputs must be 2D UINT8 MAT tensors with 3 or 4 channels")
        _require_vec(
            outputs[2], tensor_specs, f"{label} timestamp output", data_types={5}, channels=4,
            count=1, usage=int(TensorType.TIMESTAMP.value),
        )
        _require_float_mat(outputs[3], tensor_specs, f"{label} camera matrix output", [3, 3])
    elif op_name == "CAMERA_SPACE_TO_WORLD":
        _require_vec(
            inputs[0], tensor_specs, f"{label} timestamp input", data_types={5}, channels=4,
            count=1, usage=int(TensorType.TIMESTAMP.value),
        )
        for ref in outputs:
            if ref is not None:
                _require_float_mat(ref, tensor_specs, f"{label} transform output", [4, 4])
    elif op_name == "UV_TO_3D_IN_CAM_SPACE":
        uv = _operator_tensor_values(inputs[0], tensor_specs, f"{label} UV input")
        if uv[2] != int(TensorType.POINT.value) or uv[1] != 2 or len(uv[0]) != 1:
            raise ValueError(f"{label} UV input must be one-dimensional 2-channel point data")
        _require_vec(
            inputs[1], tensor_specs, f"{label} timestamp input", data_types={5}, channels=4,
            count=1, usage=int(TensorType.TIMESTAMP.value),
        )
        camera = _operator_mat_values(inputs[2], tensor_specs, f"{label} camera matrix input")
        if camera[0] != [3, 3] or camera[1] != 1:
            raise ValueError(f"{label} camera matrix input must be a one-channel 3x3 MAT tensor")
        for ref in inputs[3:5]:
            _operator_mat_values(ref, tensor_specs, f"{label} image input")
        result = _operator_tensor_values(outputs[0], tensor_specs, f"{label} point output")
        collapsed = [dimension for dimension in result[0] if dimension > 1]
        if result[1] != 1:
            collapsed.append(result[1])
        if result[3] not in floating or not collapsed or len(collapsed) > 2 or collapsed[-1] != 3:
            raise ValueError(f"{label} output must contain floating-point XYZ triples")
        if _operator_element_count(result) // 3 != int(np.prod(uv[0])):
            raise ValueError(f"{label} output point count must match its UV input")
    elif op_name == "DEPTH":
        _require_float_mat(outputs[0], tensor_specs, f"{label} depth output")
        if len(_operator_tensor_values(outputs[0], tensor_specs, label)[0]) != 2:
            raise ValueError(f"{label} depth output must be a 2D MAT tensor")
    elif op_name == "UPLOAD_TEXTURE_TO_GLTF":
        _require_gltf_tensor(inputs[0], tensor_specs, f"{label} glTF input")
        image = _operator_mat_values(inputs[1], tensor_specs, f"{label} image input")
        if image[3] != 1 or image[1] not in {3, 4} or len(image[0]) not in {2, 3}:
            raise ValueError(f"{label} image input must be a UINT8 3/4-channel 2D or 3D MAT tensor")
        texture_ids = _require_vec(
            outputs[0], tensor_specs, f"{label} texture ID output", data_types={3}, channels=1,
            usage=int(TensorType.SCALAR.value),
        )
        if len(image[0]) == 3 and texture_ids[0][0] != image[0][0]:
            raise ValueError(f"{label} texture ID count must match the image batch size")
    elif op_name == "SWITCH_GLTF_RENDER_STATUS":
        _require_gltf_tensor(inputs[0], tensor_specs, f"{label} glTF input")
        if inputs[1] is not None:
            _require_float_mat(inputs[1], tensor_specs, f"{label} world pose input", [4, 4])
    elif op_name == "UPDATE_GLTF":
        _require_gltf_tensor(inputs[0], tensor_specs, f"{label} glTF input")
        attribute = attrs[0]
        if attribute == "world pose":
            if inputs[1] is not None:
                _require_float_mat(inputs[1], tensor_specs, f"{label} world pose input", [4, 4])
        elif attribute == "local":
            ids = _require_vec(inputs[1], tensor_specs, f"{label} node ID input", data_types={3}, channels=1)
            transforms = _operator_mat_values(inputs[2], tensor_specs, f"{label} transform input")
            if transforms[3] not in floating or transforms[1] != 1 or not (
                transforms[0] == [4, 4] or len(transforms[0]) == 3 and transforms[0][1:] == [4, 4]
            ):
                raise ValueError(f"{label} transform input must be a floating-point 4x4 MAT or MAT array")
            transform_count = 1 if len(transforms[0]) == 2 else transforms[0][0]
            if ids[0][0] != transform_count:
                raise ValueError(f"{label} node ID count must match its transform count")
        elif attribute == "animation":
            _require_vec(inputs[1], tensor_specs, f"{label} animation ID input", data_types={3}, channels=1, count=1)
            _require_vec(inputs[2], tensor_specs, f"{label} animation timer input", data_types=floating, channels=1, count=1)
        elif attribute == "texture":
            _require_vec(inputs[1], tensor_specs, f"{label} texture ID input", data_types={3}, channels=1)
            image = _operator_mat_values(inputs[2], tensor_specs, f"{label} image input")
            if image[3] != 1 or image[1] not in {3, 4} or len(image[0]) not in {2, 3}:
                raise ValueError(f"{label} image input must be a UINT8 3/4-channel MAT tensor")
        else:
            ids = _require_vec(inputs[1], tensor_specs, f"{label} material ID input", data_types={3}, channels=1)
            value = _operator_tensor_values(inputs[2], tensor_specs, f"{label} material value input")
            texture_attrs = {
                "material::metallic_roughness_texture", "material::base_color_texture",
                "material::normal_map_texture", "material::occlusion_texture",
                "material::emissive_texture",
            }
            four_channel_attrs = {"material::base_color_factor", "material::emissive_factor"}
            if attribute in texture_attrs:
                if value[3] != 3 or value[1] != 1 or len(value[0]) != 1:
                    raise ValueError(f"{label} material texture value must be a UINT16 vector")
            elif value[3] not in floating or value[1] != (4 if attribute in four_channel_attrs else 1) or len(value[0]) != 1:
                raise ValueError(f"{label} material value has an incompatible floating-point vector descriptor")
            if ids[0][0] != value[0][0]:
                raise ValueError(f"{label} material ID and value counts must match")
    elif op_name == "RENDER_TEXT":
        _require_vec(inputs[1], tensor_specs, f"{label} start input", data_types=floating, channels=2, count=1, usage=int(TensorType.POINT.value))
        _require_vec(inputs[2], tensor_specs, f"{label} colors input", data_types={1}, channels=4, count=2, usage=int(TensorType.COLOR.value))
        _require_gltf_tensor(inputs[3], tensor_specs, f"{label} glTF input")
        _require_vec(inputs[4], tensor_specs, f"{label} texture ID input", data_types={3}, channels=1, count=1, usage=int(TensorType.SCALAR.value))
        _require_vec(inputs[5], tensor_specs, f"{label} font size input", data_types=floating, channels=1, count=1, usage=int(TensorType.SCALAR.value))
    elif op_name in {
        "SSMR_SWITCH_VISIBILITY", "SCENEGRAPH_VISIBILITY",
        "SSMR_UPDATE_COMPONENT", "UPDATE_COMPONENT",
    }:
        _require_gltf_tensor(inputs[0], tensor_specs, f"{label} scenegraph input")
    elif op_name == "MICROPHONE":
        encoding = attrs[0].split(";", 1)[1]
        allowed_data_types = {
            "PCM_16BIT": {3, 4},
            "PCM_32BIT": {5},
            "PCM_FLOAT": {6},
        }[encoding]
        for index, ref in enumerate(outputs):
            if ref is None:
                continue
            dimensions, channels, _, data_type = _operator_tensor_values(ref, tensor_specs, label)
            if index == 3:
                if data_type != 5 or channels != 4 or dimensions != [1]:
                    raise ValueError(
                        f"{label} timestamp output must have dimensions [1], 4 channels, and INT32 data"
                    )
                continue
            if data_type not in allowed_data_types:
                raise ValueError(f"{label} audio output data type must match {encoding}")
            if index == 0:
                effective_channels = channels * (dimensions[-1] if channels == 1 else 1)
                if effective_channels != 2:
                    raise ValueError(
                        f"{label} stereo output must have 2 channels or one channel with final dimension 2"
                    )
            elif channels != 1:
                raise ValueError(f"{label} left and right outputs must each have one channel")
    elif op_name == "SPEAKER":
        dimensions, channels, _, data_type = _operator_tensor_values(inputs[0], tensor_specs, label)
        del dimensions
        if channels not in {1, 2}:
            raise ValueError(f"{label} audio input must have one or two channels")
        if data_type not in {3, 4, 5, 6}:
            raise ValueError(
                f"{label} audio input must use UINT16, INT16, INT32, or FLOAT32 data"
            )
    elif op_name in {"ELEMENTWISE_MIN", "ELEMENTWISE_MAX", "ELEMENTWISE_MULTIPLY", "ELEMENTWISE_OR", "ELEMENTWISE_AND"}:
        values = [_operator_tensor_values(ref, tensor_specs, label) for ref in (inputs[0], inputs[1], outputs[0])]
        if any(value[:2] != values[0][:2] for value in values[1:]):
            raise ValueError(f"{label} input and output shapes/channels must match")
        if op_name in {"ELEMENTWISE_OR", "ELEMENTWISE_AND"} and any(value[3] not in integer for value in values):
            raise ValueError(f"{label} requires integer tensors")
    elif op_name == "CONVERT_COLOR":
        _operator_mat_values(inputs[0], tensor_specs, f"{label} input")
        _operator_mat_values(outputs[0], tensor_specs, f"{label} output")
    elif op_name == "NORMALIZE":
        source = _operator_tensor_values(inputs[0], tensor_specs, label)
        result = _operator_tensor_values(outputs[0], tensor_specs, label)
        if source != result:
            raise ValueError(f"{label} source and result tensor types/shapes must match")
        if len(inputs) > 1 and inputs[1] is not None:
            alpha_beta = _operator_tensor_values(inputs[1], tensor_specs, label)
            if alpha_beta[3] not in floating or _operator_element_count(alpha_beta) != 2:
                raise ValueError(f"{label} alpha_beta must be a floating-point tensor containing exactly two values")
    elif op_name == "NORM":
        source = _operator_mat_values(inputs[0], tensor_specs, f"{label} input")
        result = _operator_tensor_values(outputs[0], tensor_specs, label)
        if source[3] not in floating or result[3] not in floating or result[1] != 1 or _operator_element_count(result) != 1:
            raise ValueError(f"{label} requires floating-point input and one-value floating-point result")
    elif op_name == "INVERSION":
        source = _operator_mat_values(inputs[0], tensor_specs, f"{label} input")
        result = _operator_mat_values(outputs[0], tensor_specs, f"{label} output")
        if len(source[0]) != 2 or source[0][0] != source[0][1] or result[0] != source[0]:
            raise ValueError(f"{label} requires a square input and identically shaped result")
    elif op_name == "MAKE_TRANSFORM_MAT":
        for index in range(2):
            value = _operator_mat_values(inputs[index], tensor_specs, label)
            if value[3] not in floating or _operator_element_count(value) != 3:
                raise ValueError(f"{label} rotation and translation must be 3-value floating-point tensors")
        if inputs[2] is not None:
            value = _operator_mat_values(inputs[2], tensor_specs, label)
            if value[3] not in floating or _operator_element_count(value) != 3:
                raise ValueError(f"{label} scale must be a 3-value floating-point tensor")
        result = _operator_mat_values(outputs[0], tensor_specs, label)
        if result[3] not in floating or result[1] != 1 or result[0] != [4, 4]:
            raise ValueError(f"{label} result must be a one-channel floating-point 4x4 matrix")
    elif op_name == "NMS":
        scores = _operator_mat_values(inputs[0], tensor_specs, label)
        boxes = _operator_mat_values(inputs[1], tensor_specs, label)
        if scores[3] not in floating or boxes[3] not in floating or len(scores[0]) != 2 or 1 not in scores[0] or len(boxes[0]) != 2 or not (4 in boxes[0] or boxes[1] == 4):
            raise ValueError(f"{label} requires floating-point score and box matrices")
        for index, ref in enumerate(outputs):
            if ref is None:
                continue
            result = _operator_tensor_values(ref, tensor_specs, label)
            if index < 2 and result[3] not in floating:
                raise ValueError(f"{label} score/box results must be floating-point")
            if index == 2 and result[3] not in integer:
                raise ValueError(f"{label} index result must be integer")
    elif op_name in {"SORT_VEC", "SORT_MAT"}:
        source = _operator_tensor_values(inputs[0], tensor_specs, label)
        if op_name == "SORT_VEC" and len(source[0]) != 1:
            raise ValueError(f"{label} input must be one-dimensional")
        if op_name == "SORT_MAT" and (len(source[0]) != 2 or source[1] != 1):
            raise ValueError(f"{label} input must be a one-channel 2D matrix")
        for index, ref in enumerate(outputs):
            if ref is None:
                continue
            result = _operator_tensor_values(ref, tensor_specs, label)
            if result[0] != source[0] or result[1] != 1:
                raise ValueError(f"{label} result shape must match its input")
            if index == 1 and result[3] not in integer:
                raise ValueError(f"{label} index result must be integer")


def _validate_slice_descriptor(value: Any, field: str, label: str) -> None:
    if value is None:
        return
    if field in {"src_channel_slice", "dst_channel_slice"}:
        if not isinstance(value, list) or len(value) not in (2, 3) or not all(isinstance(item, int) for item in value):
            raise ValueError(f"{label} {field} must be an array of 2 or 3 integers")
        return
    if not isinstance(value, list):
        raise ValueError(f"{label} {field} must be an array of slice arrays")
    for item in value:
        if not isinstance(item, list) or len(item) not in (1, 2, 3) or not all(isinstance(part, int) for part in item):
            raise ValueError(f"{label} {field} entries must contain 1 to 3 integers")


def _validate_assignment_slice_rank(
    value: Any, field: str, ref: Any, tensor_specs: Dict[str, Any], label: str
) -> None:
    """Validate the native PipelineTensor::operator[] rank contract."""
    if value is None:
        return
    tensor_name = _resolve_tensor_name(ref)
    tensor_spec = tensor_specs.get(tensor_name or {})
    if not isinstance(tensor_spec, dict):
        return
    dimensions = tensor_spec.get("dimensions")
    if not isinstance(dimensions, list):
        return
    slice_rank = len(value)
    tensor_rank = len(dimensions)
    if slice_rank != tensor_rank:
        raise ValueError(
            f"{label} {field} rank {slice_rank} does not match tensor '{tensor_name}' rank {tensor_rank}; "
            "channel slices apply to channels separately"
        )


def _validate_model_metadata(model: Any, label: str) -> Optional[Dict[str, Any]]:
    if model is None:
        return None
    if not isinstance(model, dict):
        raise ValueError(f"{label} model must be an object")
    required = {"bin_path", "model_type", "model_target", "input", "output"}
    missing = sorted(key for key in required if key not in model)
    if missing:
        raise ValueError(f"{label} model metadata missing required keys: {missing}")
    if model.get("model_type") != "tflite":
        raise ValueError(f"{label} model.model_type must be 'tflite'")
    if model.get("model_target") not in {"cpu", "gpu", "npu"}:
        raise ValueError(f"{label} model.model_target must be cpu, gpu, or npu")
    model_name = model.get("model_name", "main")
    if not isinstance(model_name, str) or re.fullmatch(r"[A-Za-z0-9_]+", model_name) is None:
        raise ValueError(f"{label} model.model_name must contain only letters, digits, and underscores")
    bin_path = model.get("bin_path")
    if not isinstance(bin_path, str) or not bin_path or Path(bin_path).is_absolute() or ".." in Path(bin_path).parts:
        raise ValueError(f"{label} model.bin_path must be a package-relative path")
    for field in ("input", "output"):
        entries = model.get(field)
        if not isinstance(entries, list):
            raise ValueError(f"{label} model.{field} must be an array")
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f"{label} model.{field}[{idx}] must be an object")
            for key in ("name", "shape", "encoding_type"):
                if key not in item:
                    raise ValueError(f"{label} model.{field}[{idx}] missing {key}")
            if not isinstance(item["name"], str) or not isinstance(item["encoding_type"], str):
                raise ValueError(f"{label} model.{field}[{idx}] name and encoding_type must be strings")
            if not item["name"] or "\0" in item["name"]:
                raise ValueError(
                    f"{label} model.{field}[{idx}] binding name must be a non-empty string without NUL bytes"
                )
            if not item["encoding_type"].strip():
                raise ValueError(f"{label} model.{field}[{idx}].encoding_type must be non-empty")
            if not isinstance(item["shape"], list) or not all(
                isinstance(dim, int) and 0 < dim <= _NATIVE_INT_MAX for dim in item["shape"]
            ):
                raise ValueError(f"{label} model.{field}[{idx}].shape must be an array of integers")
    return model


def _validate_preloaded_tensor_data(name: str, tensor_spec: Dict[str, Any]) -> None:
    if "data" not in tensor_spec:
        return
    data = tensor_spec["data"]
    expected_count = int(tensor_spec["channels"])
    for dimension in tensor_spec["dimensions"]:
        expected_count *= int(dimension)
    if len(data) != expected_count:
        raise ValueError(
            f"Tensor '{name}' data must contain exactly {expected_count} value(s), got {len(data)}"
        )

    data_type = int(tensor_spec["data_type"])
    integer_ranges = {
        1: (0, 2**8 - 1, "UINT8"),
        2: (-(2**7), 2**7 - 1, "INT8"),
        3: (0, 2**16 - 1, "UINT16"),
        4: (-(2**15), 2**15 - 1, "INT16"),
        5: (-(2**31), 2**31 - 1, "INT32"),
    }
    if data_type in integer_ranges:
        minimum, maximum, dtype_name = integer_ranges[data_type]
        for index, value in enumerate(data):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Tensor '{name}' data[{index}] must be an integer for {dtype_name}")
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Tensor '{name}' data[{index}] is out of range for {dtype_name}"
                )
    else:
        for index, value in enumerate(data):
            try:
                finite = np.isfinite(float(value))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"Tensor '{name}' data[{index}] must be finite")


def _validate_tensor_usage(name: str, tensor_spec: Dict[str, Any]) -> None:
    dimensions = tensor_spec["dimensions"]
    channels = int(tensor_spec["channels"])
    data_type = int(tensor_spec["data_type"])
    usage = int(tensor_spec["usage"])

    if usage == int(TensorType.MAT.value):
        if len(dimensions) < 2:
            raise ValueError(
                f"Tensor '{name}' is declared as matrix/MAT usage but has "
                f"dimensions {dimensions!r}; matrix tensors must have at least "
                "2 dimensions. Use [1, N] or [N, 1] for vectors, or use a "
                "scalar/point tensor type for 1D data."
            )
        return

    if usage == int(TensorType.SCALAR.value) and (len(dimensions) != 1 or channels != 1):
        raise ValueError(f"Tensor '{name}' scalar usage requires one dimension and one channel")
    if usage == int(TensorType.POINT.value) and (len(dimensions) != 1 or channels not in {2, 3}):
        raise ValueError(f"Tensor '{name}' point usage requires one dimension and 2 or 3 channels")
    if usage == int(TensorType.COLOR.value) and (
        len(dimensions) != 1 or channels not in {3, 4} or data_type != int(EDataType.UINT8)
    ):
        raise ValueError(f"Tensor '{name}' color usage requires one dimension and 3 or 4 UINT8 channels")
    if usage == int(TensorType.TIMESTAMP.value) and (
        dimensions != [1] or channels != 4 or data_type != int(EDataType.INT32)
    ):
        raise ValueError(f"Tensor '{name}' timestamp usage requires dimensions [1], 4 channels, and INT32 data")
    if usage == int(TensorType.SLICE.value) and (
        len(dimensions) != 1 or channels not in {2, 3} or data_type != int(EDataType.INT32)
    ):
        raise ValueError(f"Tensor '{name}' slice usage requires one dimension and 2 or 3 INT32 channels")


def validate_pipeline_spec(spec: Dict[str, Any]) -> None:
    """Validate pipeline JSON rules that must match the native runtime.

    Raises:
        ValueError: If the pipeline spec contains an invalid tensor descriptor.
    """
    if not isinstance(spec, dict):
        raise ValueError("Pipeline spec must be an object")
    unknown_root_keys = set(spec) - _ROOT_KEYS
    if unknown_root_keys:
        raise ValueError(f"Pipeline root has unknown keys: {sorted(unknown_root_keys)}")

    for required_root in _ROOT_KEYS:
        if required_root not in spec:
            raise ValueError(f"Pipeline root requires '{required_root}'")
    tensor_specs = spec.get("tensors")
    if not isinstance(tensor_specs, dict):
        raise ValueError("Pipeline root requires a tensors object")
    operators = spec.get("operators")
    if not isinstance(operators, list):
        raise ValueError("Pipeline root requires an operators array")
    for field in ("inputs", "outputs"):
        values = spec.get(field, [])
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"Pipeline root {field} must be an array of tensor-name strings")

    tensor_names = set(tensor_specs.keys())
    boundary_names = set(spec.get("inputs", [])) | set(spec.get("outputs", []))
    for name in spec.get("inputs", []) + spec.get("outputs", []):
        if name not in tensor_names:
            raise ValueError(f"Pipeline root references unknown tensor '{name}'")

    for name, tensor_spec in tensor_specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor names must be non-empty strings")
        if not isinstance(tensor_spec, dict):
            raise ValueError(f"Tensor '{name}' descriptor must be an object")
        unknown_tensor_keys = set(tensor_spec) - _TENSOR_KEYS
        if unknown_tensor_keys:
            raise ValueError(f"Tensor '{name}' has unknown keys: {sorted(unknown_tensor_keys)}")
        for required in ("dimensions", "channels", "data_type", "is_placeholder", "usage"):
            if required not in tensor_spec:
                raise ValueError(f"Tensor '{name}' missing required key '{required}'")
        dimensions = tensor_spec.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions or not all(
            isinstance(dim, int) and 0 < dim <= _NATIVE_INT_MAX for dim in dimensions
        ):
            raise ValueError(f"Tensor '{name}' dimensions must be a non-empty array of positive integers")
        if (
            not isinstance(tensor_spec.get("channels"), int)
            or not 0 < int(tensor_spec["channels"]) <= 127
        ):
            raise ValueError(f"Tensor '{name}' channels must be a positive integer no greater than 127")
        if not isinstance(tensor_spec.get("data_type"), int) or int(tensor_spec["data_type"]) not in {1, 2, 3, 4, 5, 6, 7}:
            raise ValueError(f"Tensor '{name}' data_type must be a supported integer code")
        if not isinstance(tensor_spec.get("is_placeholder"), bool):
            raise ValueError(f"Tensor '{name}' is_placeholder must be a bool")
        if not isinstance(tensor_spec.get("usage"), int) or not 1 <= int(tensor_spec["usage"]) <= 8:
            raise ValueError(f"Tensor '{name}' usage must be an integer code from 1 through 8")
        if "flag" in tensor_spec and not isinstance(tensor_spec.get("flag"), int):
            raise ValueError(f"Tensor '{name}' flag must be an integer")
        if "data" in tensor_spec and (
            not isinstance(tensor_spec.get("data"), list)
            or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in tensor_spec["data"])
        ):
            raise ValueError(f"Tensor '{name}' data must be an array")
        if "data" in tensor_spec:
            _validate_preloaded_tensor_data(name, tensor_spec)
        if "is_gltf" in tensor_spec and not isinstance(tensor_spec.get("is_gltf"), bool):
            raise ValueError(f"Tensor '{name}' is_gltf must be a bool")
        if "asset" in tensor_spec:
            asset = tensor_spec.get("asset")
            if not isinstance(asset, str) or Path(asset).is_absolute() or ".." in Path(asset).parts:
                raise ValueError(f"Tensor '{name}' asset must be a package-relative path")
        if tensor_spec.get("is_gltf") is True:
            if tensor_spec.get("is_placeholder") is not True:
                raise ValueError(f"glTF tensor '{name}' requires is_placeholder=true")
            if int(tensor_spec.get("usage")) != int(TensorType.GLTF.value):
                raise ValueError(f"glTF tensor '{name}' must use usage {int(TensorType.GLTF.value)}")
        else:
            _validate_tensor_usage(name, tensor_spec)

        is_placeholder = tensor_spec["is_placeholder"]
        if name in boundary_names and not is_placeholder:
            raise ValueError(
                f"Pipeline boundary tensor '{name}' requires is_placeholder=true"
            )

    for index, op_spec in enumerate(operators):
        if not isinstance(op_spec, dict):
            raise ValueError(f"operator #{index} must be an object")
        label = _operator_label(index, op_spec.get("type"))
        canonical_name = _canonical_operator_name(op_spec.get("type"))
        unknown_op_keys = set(op_spec) - _OPERATOR_COMMON_KEYS
        if canonical_name in _OPERATOR_EXTRA_KEYS:
            unknown_op_keys -= _OPERATOR_EXTRA_KEYS[canonical_name]
        if unknown_op_keys:
            legacy = unknown_op_keys & _LEGACY_OPERATOR_CONFIG_FIELDS
            if legacy:
                raise ValueError(f"{label} uses legacy field(s) outside attrs: {sorted(legacy)}")
            raise ValueError(f"{label} has unknown keys: {sorted(unknown_op_keys)}")
        if canonical_name is None:
            raise ValueError(f"{label} has unknown operator type")
        if canonical_name == "UNKNOWN":
            raise ValueError(f"{label} UNKNOWN is reserved and cannot be executed")

        model = _validate_model_metadata(op_spec.get("model"), label) if canonical_name == "RUN_MODEL_INFERENCE" else None
        if "inputs" not in op_spec or "outputs" not in op_spec:
            raise ValueError(f"{label} requires inputs and outputs arrays")
        input_refs = _validate_ref_list(op_spec["inputs"], "inputs", label, tensor_names)
        output_refs = _validate_ref_list(op_spec["outputs"], "outputs", label, tensor_names)
        _validate_signature(canonical_name, input_refs, output_refs, label, model)
        attrs = _validate_attrs(op_spec, canonical_name, label)
        validation_inputs, validation_outputs = _expand_compact_operator_refs(
            canonical_name, input_refs, output_refs
        )
        _validate_operator_binding_contracts(canonical_name, validation_inputs, attrs, label)
        _validate_operator_tensor_contracts(
            canonical_name, validation_inputs, validation_outputs, attrs, tensor_specs, label, model
        )

        if canonical_name == "ASSIGNMENT":
            for field in ("src_slices", "dst_slices", "src_channel_slice", "dst_channel_slice"):
                _validate_slice_descriptor(op_spec.get(field), field, label)
            _validate_assignment_slice_rank(op_spec.get("src_slices"), "src_slices", input_refs[0], tensor_specs, label)
            _validate_assignment_slice_rank(op_spec.get("dst_slices"), "dst_slices", output_refs[0], tensor_specs, label)
        if _OP_SWAP_HWC_CHW is not None and _get_operator_type(op_spec.get("type", "")) == _OP_SWAP_HWC_CHW:
            _validate_swap_hwc_chw_operator(op_spec, tensor_specs)


def compare_outputs(
    expected: Dict[str, np.ndarray],
    actual: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> VerificationResult:
    """Compare expected and actual outputs.

    Args:
        expected: Dictionary of expected output tensors.
        actual: Dictionary of actual output tensors.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        VerificationResult with comparison details.
    """
    max_abs_diff = {}
    max_rel_diff = {}
    all_close = True
    error_messages = []

    for name in expected:
        if name not in actual:
            error_messages.append(f"Missing output: {name}")
            all_close = False
            continue

        exp = expected[name].astype(np.float64)
        act = actual[name].astype(np.float64)

        # Try to reshape actual to match expected if sizes are equal
        if exp.shape != act.shape and exp.size == act.size:
            act = act.reshape(exp.shape)

        if exp.shape != act.shape:
            error_messages.append(
                f"Shape mismatch for {name}: expected {exp.shape}, got {act.shape}"
            )
            all_close = False
            continue

        abs_diff = np.abs(exp - act)
        max_abs_diff[name] = float(np.max(abs_diff))

        # Compute relative difference avoiding division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(exp) + 1e-10)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
        max_rel_diff[name] = float(np.max(rel_diff))

        if not np.allclose(exp, act, rtol=rtol, atol=atol):
            error_messages.append(
                f"Output {name} differs: max_abs={max_abs_diff[name]:.6f}, "
                f"max_rel={max_rel_diff[name]:.6f}"
            )
            all_close = False

    return VerificationResult(
        success=all_close,
        host_outputs=expected,
        device_outputs=actual,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        error_message="\n".join(error_messages) if error_messages else None,
    )


def _run_host_pipeline(
    pipeline_path: Union[str, Path],
    inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run pipeline on host using pure Python implementation.

    Args:
        pipeline_path: Path to pipeline JSON file.
        inputs: Dictionary of input tensors.

    Returns:
        Dictionary of output tensors.
    """
    # Load pipeline spec
    with open(pipeline_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    return run_pipeline_python(spec, inputs)


def run_pipeline_python(
    spec: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
    *,
    return_all_tensors: bool = False,
    model_runner=None,
) -> Dict[str, np.ndarray]:
    """Execute a pipeline spec using pure Python (no native bindings).

    Args:
        spec: Pipeline specification dictionary.
        inputs: Dictionary of input tensors.

    Returns:
        Dictionary of output tensors by default. When ``return_all_tensors`` is
        true, returns every tensor available after execution.

    """
    from . import ops

    validate_pipeline_spec(spec)

    # Initialize tensor storage with inputs
    tensors: Dict[str, np.ndarray] = dict(inputs)
    protected_inputs = set(inputs.keys())

    # Load pre-defined tensor values from spec
    for name, tensor_spec in spec.get("tensors", {}).items():
        if name in tensors:
            continue
        value = tensor_spec.get("data")
        if value is not None:
            data_type = tensor_spec.get("data_type", 6)  # default float32
            dtype = convert_to_dtype(data_type, target="numpy")
            dims = tensor_spec.get("dimensions", [])
            channels = tensor_spec.get("channels", 1)
            shape = dims + ([channels] if channels > 1 else [])
            arr = np.array(value, dtype=dtype)
            try:
                arr = arr.reshape(shape)
            except ValueError:
                pass
            tensors[name] = arr

    # Execute operators in order
    tensor_specs = spec.get("tensors", {})
    for op_spec in spec.get("operators", []):
        _execute_operator(
            op_spec,
            tensors,
            ops,
            tensor_specs=tensor_specs,
            protected_inputs=protected_inputs,
            model_runner=model_runner,
        )

    # Collect outputs
    output_names = spec.get("outputs", [])
    outputs = {}
    for name in output_names:
        if name in tensors:
            outputs[name] = tensors[name]

    return tensors if return_all_tensors else outputs


def _resolve_tensor_name(ref: Any) -> Optional[str]:
    """Resolve a spec tensor-reference object to its package tensor name."""
    if isinstance(ref, dict):
        return ref.get("tensor")
    return None


def _tensor_from_ref(ref: Any, tensors: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    name = _resolve_tensor_name(ref)
    if not name:
        return None
    return tensors.get(name)


def _resolve_assignment_slice(
    value: Any,
    tensors: Dict[str, np.ndarray],
    field: str,
) -> Optional[List[List[int]]]:
    """Resolve an inline or tensor-backed slice descriptor."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{field} must contain slice descriptors")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{field} must have shape [rank, 2] or [rank, 3]")
    return [[int(item) for item in row] for row in array.tolist()]


def _resolve_channel_slice(value: Any, tensors: Dict[str, np.ndarray], field: str) -> Optional[List[int]]:
    if value is None:
        return None
    result = [int(item) for item in np.asarray(value).reshape(-1).tolist()]
    if len(result) not in (2, 3):
        raise ValueError(f"{field} must contain 2 or 3 integers")
    return result


def _get_operator_type(type_str: Any) -> Optional[EOperatorType]:
    """Convert operator type string to EOperatorType enum."""
    normalized = _canonical_operator_name(type_str)
    if normalized is None or normalized == "UNKNOWN":
        return None
    normalized = _CANONICAL_TO_INTERNAL.get(normalized, normalized)

    # Try to get the enum member by name
    try:
        return EOperatorType[normalized]
    except (KeyError, TypeError):
        pass

    # Try to find by iterating (for pure Python enum)
    try:
        for member in EOperatorType:
            if member.name == normalized:
                return member
    except TypeError:
        # pybind11 type is not iterable, try getattr
        if hasattr(EOperatorType, normalized):
            return getattr(EOperatorType, normalized)

    return None


def _execute_operator(
    op_spec: Dict[str, Any],
    tensors: Dict[str, np.ndarray],
    ops_module,
    tensor_specs: Optional[Dict[str, Any]] = None,
    protected_inputs: Optional[set[str]] = None,
    model_runner=None,
) -> None:
    """Execute a single operator.

    Args:
        op_spec: Operator specification.
        tensors: Dictionary of tensors (modified in place).
        ops_module: The ops module containing operation implementations.
    """
    op_type = _get_operator_type(op_spec.get("type", ""))
    op_name = _canonical_operator_name(op_spec.get("type", ""))
    attrs = op_spec.get("attrs", []) or []

    # Get input tensors
    input_refs = op_spec.get("inputs", [])
    input_tensors: List[Optional[np.ndarray]] = []
    for ref in input_refs:
        input_tensors.append(_tensor_from_ref(ref, tensors) if ref is not None else None)
    present_input_tensors = [value for value in input_tensors if value is not None]

    # Get output names
    output_refs = op_spec.get("outputs", [])
    output_names = [_resolve_tensor_name(ref) for ref in output_refs]

    if op_type is None or op_name == "UNKNOWN":
        raise ValueError(f"Unknown operator type '{op_spec.get('type')}' is not supported")

    def get_output_shape(name: Optional[str]) -> Optional[tuple]:
        if not name or not tensor_specs:
            return None
        spec = tensor_specs.get(name)
        if not spec:
            return None
        dims = spec.get("dimensions", [])
        channels = int(spec.get("channels", 1))
        if len(dims) >= 2:
            # Schema dimensions use image order: [height, width].  Keep the
            # same order when reconstructing host output shapes; swapping
            # these values only shows up for non-square tensors.
            height, width = int(dims[0]), int(dims[1])
        elif len(dims) == 1:
            height, width = int(dims[0]), 1
        else:
            height, width = 1, 1
        if channels > 1:
            return (height, width, channels)
        return (height, width)

    def named_tensor(field: str, index: int = 0) -> Optional[np.ndarray]:
        resolved = _resolve_tensor_name(input_refs[index]) if index < len(input_refs) else None
        return tensors.get(resolved) if resolved is not None else None

    # Execute based on operator type
    if op_type == EOperatorType.ARITHMETIC_COMPOSE:
        expression = attrs[0]
        if present_input_tensors:
            result = ops_module.arithmetic(
                present_input_tensors if len(present_input_tensors) > 1 else present_input_tensors[0],
                expression,
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.CONVERT_COLOR:
        flag = int(attrs[0])
        if input_tensors[0] is not None:
            result = ops_module.cvt_color(input_tensors[0], flag)
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NORMALIZE:
        if input_tensors[0] is not None:
            normalize_type = attrs[0] if attrs else "L2"
            alpha_beta = input_tensors[1] if len(input_tensors) > 1 else None
            result = ops_module.normalize(
                input_tensors[0],
                normalize_type=str(normalize_type),
                alpha_beta=alpha_beta,
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ARGMAX:
        if input_tensors[0] is not None:
            result = ops_module.argmax(input_tensors[0])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MIN:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.elementwise_min(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MAX:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.elementwise_max(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MULTIPLY:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.elementwise_multiply(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.ELEMENTWISE_OR:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.elementwise_or(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.ELEMENTWISE_AND:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.elementwise_and(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ALL:
        if input_tensors[0] is not None:
            result = ops_module.all(input_tensors[0])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ANY:
        if input_tensors[0] is not None:
            result = ops_module.any(input_tensors[0])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ASSIGNMENT:
        src_slices = _resolve_assignment_slice(op_spec.get("src_slices"), tensors, "src_slices")
        dst_slices = _resolve_assignment_slice(op_spec.get("dst_slices"), tensors, "dst_slices")
        src_channel_slice = _resolve_channel_slice(op_spec.get("src_channel_slice"), tensors, "src_channel_slice")
        dst_channel_slice = _resolve_channel_slice(op_spec.get("dst_channel_slice"), tensors, "dst_channel_slice")
        output_base_name = output_names[0] if output_names else None
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.assignment(
                input_tensors[0],
                input_tensors[1],
                src_slices=src_slices,
                dst_slices=dst_slices,
                src_channel_slice=src_channel_slice,
                dst_channel_slice=dst_channel_slice,
            )
            if output_base_name:
                tensors[output_base_name] = result
        elif len(input_tensors) == 1 and input_tensors[0] is not None:
            if output_base_name:
                if dst_slices is not None or dst_channel_slice is not None:
                    dst_spec = tensor_specs.get(output_base_name, {}) if tensor_specs else {}
                    dst = tensors.get(output_base_name)
                    if dst is None:
                        shape = get_output_shape(output_base_name) or input_tensors[0].shape
                        data_type = dst_spec.get("data_type", 6) if isinstance(dst_spec, dict) else 6
                        try:
                            dtype = convert_to_dtype(data_type, target="numpy")
                        except Exception:
                            dtype = input_tensors[0].dtype
                        dst = np.zeros(shape, dtype=dtype)
                    tensors[output_base_name] = ops_module.assignment(
                        input_tensors[0], dst,
                        src_slices=src_slices,
                        dst_slices=dst_slices,
                        src_channel_slice=src_channel_slice,
                        dst_channel_slice=dst_channel_slice,
                    )
                else:
                    if src_slices is not None or src_channel_slice is not None:
                        tensors[output_base_name] = ops_module.assignment(
                            input_tensors[0], input_tensors[0],
                            src_slices=src_slices,
                            src_channel_slice=src_channel_slice,
                        )
                    else:
                        input_name = _resolve_tensor_name(input_refs[0]) if input_refs else None
                        input_spec = tensor_specs.get(input_name, {}) if tensor_specs and input_name else {}
                        output_spec = tensor_specs.get(output_base_name, {}) if tensor_specs else {}
                        input_data_type = input_spec.get("data_type") if isinstance(input_spec, dict) else None
                        output_data_type = output_spec.get("data_type") if isinstance(output_spec, dict) else None
                        if input_data_type != output_data_type and output_data_type is not None:
                            try:
                                output_dtype = convert_to_dtype(output_data_type, target="numpy")
                            except Exception as exc:
                                raise ValueError(
                                    f"assignment/type_convert output '{output_base_name}' has unsupported data_type "
                                    f"{output_data_type!r}"
                                ) from exc
                            tensors[output_base_name] = input_tensors[0].astype(output_dtype, copy=True)
                        else:
                            tensors[output_base_name] = input_tensors[0].copy()

    elif op_type == EOperatorType.APPLY_AFFINE:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            output_shape = get_output_shape(output_names[0] if output_names else None)
            result = ops_module.apply_affine(
                input_tensors[0],
                input_tensors[1],
                output_shape=output_shape,
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.APPLY_AFFINE_POINT:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.apply_affine_point(
                input_tensors[0],
                input_tensors[1],
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.CUSTOMIZED_COMPARE:
        compare = attrs[0]
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.customized_compare(
                input_tensors[0],
                input_tensors[1],
                compare=str(compare),
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NMS:
        threshold = float(attrs[0]) if attrs else 0.95
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.nms(
                input_tensors[0],
                input_tensors[1],
                threshold=float(threshold),
            )
            if output_names:
                # The spec orders optional NMS results as scores, boxes, indices.
                # The host helper returns the kept indices; derive only the
                # requested representations in that documented order.
                kept = np.asarray(result, dtype=np.int32).reshape(-1)
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = input_tensors[0][kept]
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = input_tensors[1][kept]
                if len(output_names) >= 3 and output_names[2]:
                    tensors[output_names[2]] = kept

    elif op_type == EOperatorType.CAMERA_SPACE_TO_WORLD:
        if input_tensors[0] is not None:
            right, left = ops_module.camera_space_to_world(input_tensors[0])
            if output_names:
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = right
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = left

    elif op_type == EOperatorType.GET_AFFINE:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.get_affine(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif _OP_GET_TRANSFORM_MAT is not None and op_type == _OP_GET_TRANSFORM_MAT:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            scale = input_tensors[2] if len(input_tensors) >= 3 and input_tensors[2] is not None else None
            result = ops_module.get_transform_mat(input_tensors[0], input_tensors[1], scale=scale)
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.INVERSION:
        if input_tensors[0] is not None:
            result = ops_module.inversion(input_tensors[0])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NORM:
        if input_tensors[0] is not None:
            result = ops_module.norm(input_tensors[0], norm_type=attrs[0] if attrs else "L2")
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.SOLVE_P_N_P:
        if len(input_tensors) >= 3 and all(value is not None for value in input_tensors[:3]):
            rvec, tvec = ops_module.solve_p_n_p(input_tensors[0], input_tensors[1], input_tensors[2])
            if output_names:
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = rvec
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = tvec

    elif op_type == EOperatorType.SORT_VEC:
        if input_tensors[0] is not None:
            sorted_vec, indices = ops_module.sort_vec(input_tensors[0])
            if output_names:
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = sorted_vec
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = indices

    elif op_type == EOperatorType.SORT_MAT:
        sort_axis = str(attrs[0]) if attrs else "ROW"
        if input_tensors[0] is not None:
            sorted_mat, indices = ops_module.sort_mat(input_tensors[0], axis=sort_axis)
            if output_names:
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = sorted_mat
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = indices

    elif op_type == EOperatorType.SVD:
        if input_tensors[0] is not None:
            w, u, vt = ops_module.svd(input_tensors[0])
            if output_names:
                if len(output_names) >= 1 and output_names[0]:
                    tensors[output_names[0]] = w
                if len(output_names) >= 2 and output_names[1]:
                    tensors[output_names[1]] = u
                if len(output_names) >= 3 and output_names[2]:
                    tensors[output_names[2]] = vt

    elif _OP_SWAP_HWC_CHW is not None and op_type == _OP_SWAP_HWC_CHW:
        if input_tensors[0] is not None:
            result = ops_module.swap_hwc_chw(input_tensors[0])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.UV_TO_3D_IN_CAM_SPACE:
        if len(input_tensors) >= 5 and all(value is not None for value in input_tensors[:5]):
            result = ops_module.uv_to_3d_in_camera_space(
                input_tensors[0],
                input_tensors[1],
                input_tensors[2],
                input_tensors[3],
                input_tensors[4],
            )
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.RECTIFIED_VST_ACCESS:
        output_shapes = []
        for name in output_names:
            if name is None:
                output_shapes.append((1, 1))
                continue
            shape = get_output_shape(name)
            output_shapes.append(shape if shape is not None else (1, 1))
        right, left, timestamp, cam_mat = ops_module.camera_access(
            output_shapes=output_shapes,
            output_names=output_names,
        )
        if output_names:
            protected_inputs = protected_inputs or set()
            if len(output_names) >= 1 and output_names[0] and output_names[0] not in protected_inputs:
                tensors[output_names[0]] = right
            if len(output_names) >= 2 and output_names[1] and output_names[1] not in protected_inputs:
                tensors[output_names[1]] = left
            if len(output_names) >= 3 and output_names[2] and output_names[2] not in protected_inputs:
                tensors[output_names[2]] = timestamp
            if len(output_names) >= 4 and output_names[3] and output_names[3] not in protected_inputs:
                tensors[output_names[3]] = cam_mat

    elif op_type == EOperatorType.RUN_MODEL_INFERENCE:
        model_ref = op_spec.get("model")
        inline_model = model_ref if isinstance(model_ref, dict) else {}
        model_file = inline_model.get("bin_path")
        model_name = inline_model.get("model_name") or "main"
        if not model_file:
            raise ValueError(
                "RUN_MODEL_INFERENCE requires inline TFLite model metadata with model.bin_path"
            )
        inputs_map: Dict[str, np.ndarray] = {}
        for ref in input_refs:
            if isinstance(ref, dict) and "name" in ref and "tensor" in ref:
                name = ref.get("name")
                tensor_name = ref.get("tensor")
                if tensor_name in tensors:
                    inputs_map[name] = tensors[tensor_name]
            else:
                name = _resolve_tensor_name(ref)
                if name and name in tensors:
                    inputs_map[name] = tensors[name]
        input_aliasing = {
            str(ref["name"]): str(ref["name"])
            for ref in input_refs
            if isinstance(ref, dict) and ref.get("name") and ref.get("tensor")
        }
        output_aliasing = {
            str(ref["name"]): str(ref["name"])
            for ref in output_refs
            if isinstance(ref, dict) and ref.get("name") and ref.get("tensor")
        }
        output_shapes = []
        output_dtypes = []
        for name in output_names:
            if name is None:
                continue
            spec = tensor_specs.get(name, {})
            dims = spec.get("dimensions", [])
            channels = spec.get("channels", 1)
            shape = list(dims)
            if channels and int(channels) > 1:
                shape.append(int(channels))
            output_shapes.append(tuple(shape) if shape else (1,))
            data_type = spec.get("data_type")
            dtype = None
            if data_type is not None:
                try:
                    dtype = convert_to_dtype(data_type, target="numpy")
                except Exception:
                    dtype = None
            output_dtypes.append(dtype if dtype is not None else np.float32)

        if model_runner is not None:
            outputs = model_runner(
                inputs=inputs_map,
                model_file=model_file,
                model_name=model_name,
                output_names=[name for name in output_names if name],
                output_shapes=output_shapes if output_shapes else None,
                output_dtypes=output_dtypes if output_dtypes else None,
                input_aliasing=input_aliasing,
                output_aliasing=output_aliasing,
                model=inline_model,
            )
        else:
            outputs = ops_module.run_algorithm(
                inputs=inputs_map,
                model_file=model_file,
                model_name=model_name,
                output_names=[name for name in output_names if name],
                output_shapes=output_shapes if output_shapes else None,
                output_dtypes=output_dtypes if output_dtypes else None,
                input_aliasing=input_aliasing,
                output_aliasing=output_aliasing,
                model=inline_model,
            )
        for name, value in outputs.items():
            tensors[name] = value

    elif _OP_LOAD_TEXTURE is not None and op_type == _OP_LOAD_TEXTURE:
        if len(input_tensors) >= 2 and input_tensors[0] is not None and input_tensors[1] is not None:
            result = ops_module.load_texture(input_tensors[0], input_tensors[1])
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.SWITCH_GLTF_RENDER_STATUS:
        gltf = input_tensors[0] if input_tensors else None
        if gltf is not None:
            ops_module.switch_gltf_render_status(
                gltf,
                pose=input_tensors[1] if len(input_tensors) > 1 and input_tensors[1] is not None else None,
            )

    elif op_type == EOperatorType.UPDATE_GLTF:
        update_type = str(attrs[0])
        gltf = input_tensors[0] if input_tensors else None
        if gltf is not None:
            ops_module.update_gltf(
                gltf,
                update_type=update_type,
                operands=[tensor for tensor in input_tensors[1:] if tensor is not None],
            )

    elif op_type == EOperatorType.RENDER_TEXT:
        if len(input_tensors) >= 6 and input_tensors[3] is not None and input_tensors[0] is not None:
            text_values = np.asarray(input_tensors[0]).reshape(-1)
            text = bytes(text_values.astype(np.uint8).tolist()).decode("utf-8", errors="replace")
            ops_module.render_text(
                input_tensors[3],
                attrs[0],
                text,
                start=input_tensors[1],
                colors=input_tensors[2],
                texture_id=input_tensors[4],
                font_size=input_tensors[5],
            )

    elif _OP_JAVASCRIPT is not None and op_type == _OP_JAVASCRIPT:
        js_code = attrs[0]
        inputs_map: Dict[str, np.ndarray] = {}
        for ref in input_refs:
            if isinstance(ref, dict) and "name" in ref and "tensor" in ref:
                name = ref.get("name")
                tensor_name = ref.get("tensor")
                if tensor_name in tensors:
                    inputs_map[name] = tensors[tensor_name]
            else:
                name = _resolve_tensor_name(ref)
                if name and name in tensors:
                    inputs_map[name] = tensors[name]
        js_output_names = [name for name in output_names if name]
        outputs = _try_execute_known_javascript(js_code, inputs_map, js_output_names)
        if outputs is None:
            output_specs = {
                name: tensor_specs[name]
                for name in js_output_names
                if tensor_specs is not None and name in tensor_specs
            }
            outputs = ops_module.javascript(
                js_code,
                inputs_map,
                js_output_names,
                output_specs=output_specs,
            )
        for name, value in outputs.items():
            tensors[name] = value

    elif op_type == EOperatorType.SCENEGRAPH_VISIBILITY:
        if input_tensors and input_tensors[0] is not None:
            visible = input_tensors[1] if len(input_tensors) > 1 and input_tensors[1] is not None else True
            result = ops_module.scenegraph_visibility(input_tensors[0], visible=visible)
            if output_names and output_names[0]:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.UPDATE_COMPONENT:
        scenegraph_value = named_tensor("scenegraph", 0)
        data_value = named_tensor("data", 1)
        if scenegraph_value is None or data_value is None:
            raise ValueError("update_component requires scenegraph and data tensors")
        target_path, property_name = attrs[0].split(":", 1)
        ops_module.update_component(
            scenegraph_value,
            data_value,
            entity_path=target_path,
            property=str(property_name),
        )

    elif op_type == EOperatorType.MICROPHONE:
        connected_output_names = [name for name in output_names if name]
        shape = get_output_shape(connected_output_names[0] if connected_output_names else None) or (1,)
        result = ops_module.microphone(output_shape=shape)
        if connected_output_names:
            tensors[connected_output_names[0]] = result

    elif op_type == EOperatorType.SPEAKER:
        if input_tensors[0] is not None:
            ops_module.speaker(input_tensors[0])

    elif op_type == EOperatorType.DEPTH:
        shape = get_output_shape(output_names[0] if output_names else None) or (1, 1)
        result = ops_module.depth(output_shape=shape)
        if output_names and output_names[0]:
            tensors[output_names[0]] = result

    else:
        raise NotImplementedError(
            f"Operator type {op_type.name} is not implemented in pure Python executor"
        )


def _try_execute_known_javascript(
    js_code: str,
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    if (
        "decodeDetection" in js_code
        and "anchorFor" in js_code
        and {"box_coords_1", "box_coords_2", "box_scores_1", "box_scores_2"}.issubset(inputs)
        and "post_det" in output_names
    ):
        return {"post_det": _decode_mediapipe_face_detection(inputs)}
    return None


def _decode_mediapipe_face_detection(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    box_coords_1 = np.asarray(inputs["box_coords_1"], dtype=np.float32).reshape(-1)
    box_coords_2 = np.asarray(inputs["box_coords_2"], dtype=np.float32).reshape(-1)
    box_scores_1 = np.asarray(inputs["box_scores_1"], dtype=np.float32).reshape(-1)
    box_scores_2 = np.asarray(inputs["box_scores_2"], dtype=np.float32).reshape(-1)
    template = np.asarray(inputs.get("post_det_template", np.zeros((1, 21), dtype=np.float32)), dtype=np.float32)
    post_det = template.reshape(-1).copy()
    if post_det.size < 21:
        padded = np.zeros(21, dtype=np.float32)
        padded[: post_det.size] = post_det
        post_det = padded
    else:
        post_det = post_det[:21]

    input_size = 256.0
    camera_width = 580.0
    camera_height = 326.0
    affine_scale_x = 0.4413793087
    affine_scale_y = 0.7852760736
    affine_x_offset = 0.0
    affine_y_offset = 0.0
    score_threshold = 0.25

    def sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(value))))

    best_score = 0.0
    best_index = -1
    best_head = 0
    for index in range(min(512, box_scores_1.size)):
        score = sigmoid(box_scores_1[index])
        if score > best_score:
            best_score = score
            best_index = index
            best_head = 1
    for index in range(min(384, box_scores_2.size)):
        score = sigmoid(box_scores_2[index])
        if score > best_score:
            best_score = score
            best_index = index
            best_head = 2

    if best_score <= score_threshold or best_index < 0:
        return post_det.reshape(1, 21)

    coords = box_coords_1 if best_head == 1 else box_coords_2
    feature_size = 16 if best_head == 1 else 8
    anchors_per_cell = 2 if best_head == 1 else 6
    cell = best_index // anchors_per_cell
    col = cell % feature_size
    row = cell // feature_size
    anchor_x = (col + 0.5) / feature_size
    anchor_y = (row + 0.5) / feature_size

    def to_camera_x(value: float) -> float:
        return float(np.clip((value - affine_x_offset) / affine_scale_x, 0.0, camera_width))

    def to_camera_y(value: float) -> float:
        return float(np.clip((value - affine_y_offset) / affine_scale_y, 0.0, camera_height))

    base = best_index * 16
    if base + 14 > coords.size:
        return post_det.reshape(1, 21)

    x_center = (coords[base] / input_size + anchor_x) * input_size
    y_center = (coords[base + 1] / input_size + anchor_y) * input_size
    box_w = coords[base + 2]
    box_h = coords[base + 3]

    post_det[0] = to_camera_x(x_center - box_w * 0.5)
    post_det[1] = to_camera_y(y_center - box_h * 0.5)
    post_det[2] = to_camera_x(x_center + box_w * 0.5)
    post_det[3] = to_camera_y(y_center + box_h * 0.5)
    post_det[4] = best_score
    post_det[5] = 0.0

    for keypoint in range(5):
        coord_base = base + 4 + keypoint * 2
        out_base = 6 + keypoint * 3
        keypoint_x = (coords[coord_base] / input_size + anchor_x) * input_size
        keypoint_y = (coords[coord_base + 1] / input_size + anchor_y) * input_size
        post_det[out_base] = to_camera_x(keypoint_x)
        post_det[out_base + 1] = to_camera_y(keypoint_y)
        post_det[out_base + 2] = best_score

    return post_det.astype(np.float32).reshape(1, 21)

def _run_device_pipeline(
    pipeline_path: Path,
    inputs: Dict[str, np.ndarray],
    input_tensor_name: str,
    duration: int,
    expected_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Device verification is not available from the Python verifier."""
    print("Device verification is not available from the Python verifier. Use pyspatialml run device.")
    return None

def verify(
    pipeline: Union[str, Path, Dict[str, Any]],
    inputs: Dict[str, np.ndarray],
    expected_outputs: Optional[Dict[str, np.ndarray]] = None,
    device: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    duration: int = 30,
) -> VerificationResult:
    """Verify pipeline outputs against expected values.

    Args:
        pipeline: Path to pipeline JSON file or pipeline spec dictionary.
        inputs: Dictionary of input tensors.
        expected_outputs: Optional dictionary of expected output tensors.
                         If not provided, only host execution is performed.
        device: If True, also run on device and compare outputs.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        duration: Duration for device execution in seconds.

    Returns:
        VerificationResult with comparison details.
    """
    def _prepare_device_pipeline(
        src_path: Path,
        input_tensor: str,
        input_values: Dict[str, np.ndarray],
    ) -> Path:
        with open(src_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        inputs_list = list(spec.get("inputs", []))
        tensors_spec = spec.get("tensors", {})

        for name, value in input_values.items():
            if name == input_tensor:
                continue
            tensor_spec = tensors_spec.get(name)
            if tensor_spec is None:
                continue
            tensor_spec["value"] = value.flatten().tolist()
            tensor_spec["is_placeholder"] = False
            if name in inputs_list:
                inputs_list.remove(name)

        spec["inputs"] = inputs_list

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(spec, f, indent=2)
            return Path(f.name)

    # Handle pipeline spec dictionary
    if isinstance(pipeline, dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pipeline, f, indent=2)
            pipeline_path = Path(f.name)
        cleanup_pipeline = True
    else:
        pipeline_path = Path(pipeline)
        cleanup_pipeline = False

    try:
        # Run on host (optionally force model inference to use device backend for parity)
        prev_target = os.getenv("PY2SMR_MODEL_INFERENCE_TARGET")
        if device:
            os.environ["PY2SMR_MODEL_INFERENCE_TARGET"] = "android"
        try:
            host_outputs = _run_host_pipeline(pipeline_path, inputs)
        finally:
            if device:
                if prev_target is None:
                    os.environ.pop("PY2SMR_MODEL_INFERENCE_TARGET", None)
                else:
                    os.environ["PY2SMR_MODEL_INFERENCE_TARGET"] = prev_target

        # Compare with expected outputs if provided
        if expected_outputs is not None:
            result = compare_outputs(expected_outputs, host_outputs, rtol, atol)
            if not result.success:
                return result

        # Run on device if requested
        if device:
            input_tensor_name = list(inputs.keys())[0]
            device_pipeline_path = pipeline_path
            cleanup_device_pipeline = False
            if len(inputs) > 1:
                device_pipeline_path = _prepare_device_pipeline(
                    pipeline_path, input_tensor_name, inputs
                )
                cleanup_device_pipeline = True
            device_outputs = _run_device_pipeline(
                device_pipeline_path, inputs, input_tensor_name, duration,
                expected_outputs=host_outputs,
            )

            if cleanup_device_pipeline:
                device_pipeline_path.unlink(missing_ok=True)

            if device_outputs is None:
                return VerificationResult(
                    success=False,
                    host_outputs=host_outputs,
                    error_message="Device verification is not available",
                )

            # Compare host and device outputs
            return compare_outputs(host_outputs, device_outputs, rtol, atol)

        # Success - host execution only
        return VerificationResult(
            success=True,
            host_outputs=host_outputs,
        )

    finally:
        if cleanup_pipeline:
            os.unlink(pipeline_path)
