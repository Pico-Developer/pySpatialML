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
"""Convert trace context to SecureMR pipeline JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from securemr.core.types import BaseType, EDataType, EOperatorType
from securemr.core.utils import TensorType, convert_from_dtype, convert_to_dtype, mat_flag

from .tracer import TraceContext, TensorInfo, TracedOp
from .verifier import validate_pipeline_spec

__all__ = ["convert", "trace_to_pipeline_spec"]

_OP_JAVASCRIPT = EOperatorType.JAVASCRIPT
_OP_SORT_MAT = getattr(EOperatorType, "SORT_MAT", None)
_OP_NORM = getattr(EOperatorType, "NORM", None)
_OP_RENDER_TEXT = getattr(EOperatorType, "RENDER_TEXT", None)
_OP_UPDATE_GLTF = getattr(EOperatorType, "UPDATE_GLTF", None)
_OP_RUN_MODEL_INFERENCE = getattr(EOperatorType, "RUN_MODEL_INFERENCE", None)
_OP_GET_AFFINE = getattr(EOperatorType, "GET_AFFINE", None)
_OP_SOLVE_PNP = getattr(EOperatorType, "SOLVE_P_N_P", None)
_OP_SORT_VEC = getattr(EOperatorType, "SORT_VEC", None)
_OP_SWITCH_GLTF_RENDER_STATUS = getattr(EOperatorType, "SWITCH_GLTF_RENDER_STATUS", None)
_OP_LOAD_TEXTURE = getattr(EOperatorType, "LOAD_TEXTURE", None)
_OP_SWAP_HWC_CHW = getattr(EOperatorType, "SWAP_HWC_CHW", None)


# Mapping from numpy dtype to EDataType
NUMPY_TO_EDATATYPE = {
    np.uint8: EDataType.UINT8,
    np.int8: EDataType.INT8,
    np.uint16: EDataType.UINT16,
    np.int16: EDataType.INT16,
    np.int32: EDataType.INT32,
    np.float32: EDataType.FLOAT32,
    np.float64: EDataType.FLOAT64,
}


def _get_edatatype(dtype: np.dtype) -> EDataType:
    """Convert numpy dtype to EDataType."""
    dtype_type = np.dtype(dtype).type
    if dtype_type in NUMPY_TO_EDATATYPE:
        return NUMPY_TO_EDATATYPE[dtype_type]
    raise ValueError(f"Unsupported traced tensor dtype: {np.dtype(dtype)}")


def _canonical_operator_type(operator_type: EOperatorType) -> str:
    """Return the one package spelling for an internal operator enum."""
    canonical_name = operator_type.name
    canonical_name = {
        "SWAP_HWC_CHW": "CHW_HWC",
        "GET_TRANSFORM_MAT": "MAKE_TRANSFORM_MAT",
        "LOAD_TEXTURE": "UPLOAD_TEXTURE_TO_GLTF",
        "SCENEGRAPH_VISIBILITY": "SSMR_SWITCH_VISIBILITY",
        "UPDATE_COMPONENT": "SSMR_UPDATE_COMPONENT",
        "JAVASCRIPT": "JS_SCRIPTING",
    }.get(canonical_name, canonical_name)
    return f"XR_SECURE_MR_OPERATOR_TYPE_{canonical_name}_PICO"


def _tensor_info_to_spec(info: TensorInfo) -> Dict[str, Any]:
    """Convert TensorInfo to pipeline tensor spec."""
    shape = info.shape
    dtype = info.dtype
    name_lower = info.name.lower()

    # Determine dimensions and channels
    if len(shape) == 0:
        dimensions = [1, 1]
        channels = 1
    elif len(shape) == 1:
        dimensions = [int(shape[0]), 1]
        channels = 1
    elif len(shape) == 2:
        dimensions = [int(shape[0]), int(shape[1])]  # H, W
        channels = 1
    elif len(shape) == 3:
        dimensions = [int(shape[0]), int(shape[1])]  # H, W
        channels = int(shape[2])
    else:
        # For higher dimensions, flatten to 2D
        dimensions = [int(np.prod(shape[1:])), int(shape[0])]
        channels = 1

    usage = 6  # MAT by default
    if "timestamp" in name_lower and np.dtype(dtype).type == np.int32:
        total = int(np.prod(shape)) if len(shape) > 0 else 1
        if total == 4:
            dimensions = [1]
            channels = 4
            usage = 5  # TIMESTAMP

    edatatype = _get_edatatype(dtype)
    data_type_val = convert_from_dtype(edatatype, source="smr")
    flag = mat_flag(edatatype, channels)

    spec = {
        "dimensions": dimensions,
        "channels": channels,
        "data_type": data_type_val,
        "is_placeholder": info.is_input or info.is_output,
        "usage": usage,
        "flag": flag,
    }

    # The package schema calls preloaded tensor contents `data`.
    if info.value is not None and not info.is_input:
        spec["data"] = [x.item() if isinstance(x, np.generic) else x for x in info.value.flatten()]

    return spec


def _op_to_spec(op: TracedOp) -> Dict[str, Any]:
    """Convert TracedOp to pipeline operator spec."""
    type_name = _canonical_operator_type(op.op_type)

    def refs(values):
        return [
            None if value is None else (value if isinstance(value, dict) else {"tensor": value})
            for value in values
        ]

    spec = {
        "type": type_name,
        "inputs": refs(op.input_names),
        "outputs": refs(op.output_names),
    }
    if op.op_type == EOperatorType.ARITHMETIC_COMPOSE:
        spec["inputs"] = refs(list(op.input_names) + [None] * (10 - len(op.input_names)))
    elif op.op_type in {EOperatorType.NMS, EOperatorType.SORT_VEC, EOperatorType.SORT_MAT, EOperatorType.SVD, EOperatorType.MICROPHONE}:
        max_outputs = {EOperatorType.NMS: 3, EOperatorType.SORT_VEC: 2, EOperatorType.SORT_MAT: 2, EOperatorType.SVD: 3, EOperatorType.MICROPHONE: 4}[op.op_type]
        if op.op_type == EOperatorType.NMS:
            if len(op.output_names) != 1:
                raise ValueError("NMS tracing must produce exactly one kept-index output")
            # The Python helper returns kept indices.  Native NMS reserves
            # slots 0/1 for scores/boxes and exposes indices at slot 2.
            spec["outputs"] = refs([None, None, op.output_names[0]])
        else:
            spec["outputs"] = refs(list(op.output_names) + [None] * (max_outputs - len(op.output_names)))
    elif op.op_type == EOperatorType.CAMERA_SPACE_TO_WORLD:
        spec["outputs"] = refs(list(op.output_names) + [None] * (2 - len(op.output_names)))
    elif op.op_type == EOperatorType.GET_TRANSFORM_MAT:
        spec["inputs"] = refs(list(op.input_names) + [None] * (3 - len(op.input_names)))
    elif op.op_type == EOperatorType.SWITCH_GLTF_RENDER_STATUS:
        spec["inputs"] = refs(list(op.input_names) + [None] * (4 - len(op.input_names)))
    elif op.op_type == EOperatorType.UPDATE_GLTF:
        spec["inputs"] = refs(list(op.input_names) + [None] * (3 - len(op.input_names)))
    elif op.op_type == EOperatorType.SCENEGRAPH_VISIBILITY:
        spec["inputs"] = refs(list(op.input_names) + [None] * (2 - len(op.input_names)))

    # Add operator-specific fields
    if op.op_type == EOperatorType.ARITHMETIC_COMPOSE and op.attrs:
        spec["attrs"] = [op.attrs[0]]
    elif op.op_type == EOperatorType.CONVERT_COLOR and op.attrs:
        spec["attrs"] = [str(op.attrs[0])]
    elif op.op_type == EOperatorType.NMS and op.attrs:
        spec["attrs"] = [str(op.attrs[0])]
    elif op.op_type in {EOperatorType.MICROPHONE, EOperatorType.SPEAKER} and op.attrs:
        spec["attrs"] = [str(op.attrs[0])]
    elif op.op_type == EOperatorType.CUSTOMIZED_COMPARE and op.attrs:
        spec["attrs"] = [op.attrs[0]]
    elif op.op_type == EOperatorType.NORMALIZE:
        if op.attrs:
            spec["attrs"] = [op.attrs[0]]
    # NORM is the legacy magnitude operator and has no serialized mode in
    # the pipeline spec.  Its traced attrs are an implementation detail and
    # must not be emitted as the unrelated normalize_type field.
    elif _OP_SORT_MAT is not None and op.op_type == _OP_SORT_MAT and op.attrs:
        spec["attrs"] = [str(op.attrs[0])]
    elif _OP_JAVASCRIPT is not None and op.op_type == _OP_JAVASCRIPT and op.attrs:
        spec["attrs"] = [op.attrs[0]]
        if op.extra_info.get("input_refs"):
            spec["inputs"] = refs(op.extra_info["input_refs"])
        if op.extra_info.get("output_refs"):
            spec["outputs"] = refs(op.extra_info["output_refs"])
    elif _OP_LOAD_TEXTURE is not None and op.op_type == _OP_LOAD_TEXTURE:
        spec["inputs"] = refs(op.input_names[:2])
    elif _OP_SWITCH_GLTF_RENDER_STATUS is not None and op.op_type == _OP_SWITCH_GLTF_RENDER_STATUS:
        spec["inputs"] = refs(list(op.input_names[:4]) + [None] * (4 - len(op.input_names)))
    elif _OP_UPDATE_GLTF is not None and op.op_type == _OP_UPDATE_GLTF:
        attribute = op.extra_info.get("attribute") or (op.attrs[0] if op.attrs else "")
        spec["attrs"] = [attribute]
        spec["inputs"] = refs(list(op.input_names[:3]) + [None] * (3 - len(op.input_names)))
    elif _OP_RENDER_TEXT is not None and op.op_type == _OP_RENDER_TEXT:
        if op.attrs:
            spec["attrs"] = [op.attrs[0]]
        spec["inputs"] = refs(list(op.input_names[:6]) + [None] * (6 - len(op.input_names)))
    elif op.op_type == EOperatorType.SCENEGRAPH_VISIBILITY:
        spec["inputs"] = refs(list(op.input_names[:2]) + [None] * (2 - len(op.input_names)))
        spec["outputs"] = []
    elif op.op_type == EOperatorType.UPDATE_COMPONENT:
        if op.input_names:
            spec["inputs"] = refs(op.input_names[:2])
        entity_path = op.extra_info.get("entity_path")
        property_name = op.extra_info.get("property") or op.extra_info.get("target_property")
        if entity_path and property_name:
            spec["attrs"] = [f"{entity_path}:{property_name}"]
        spec["outputs"] = []
    elif _OP_RUN_MODEL_INFERENCE is not None and op.op_type == _OP_RUN_MODEL_INFERENCE:
        if "model" in op.extra_info:
            spec["model"] = dict(op.extra_info["model"])
            if spec["model"].get("model_target") != "cpu":
                spec["model"].pop("cpu_target_num_threads", None)
        if op.extra_info.get("input_refs"):
            spec["inputs"] = refs(op.extra_info["input_refs"])
        if op.extra_info.get("output_refs"):
            spec["outputs"] = refs(op.extra_info["output_refs"])
    elif op.op_type == EOperatorType.ASSIGNMENT:
        # Strict package JSON represents assignment as src -> dst.  The trace
        # records the destination array as an execution input as well, but it
        # is represented by the recorded result/output in the package graph.
        spec["inputs"] = refs(op.input_names[:1])
        for field in ("src_slices", "dst_slices", "src_channel_slice", "dst_channel_slice"):
            if field in op.extra_info:
                spec[field] = op.extra_info[field]

    return spec


def _parse_bool_or_tensor(value: str) -> Union[bool, str]:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def trace_to_pipeline_spec(ctx: TraceContext) -> Dict[str, Any]:
    """Convert a TraceContext to a pipeline specification dictionary.

    Args:
        ctx: The trace context containing recorded operations.

    Returns:
        Pipeline specification dictionary ready for JSON serialization.
    """
    # Build tensor specs
    tensors = {}
    for name, info in ctx.tensors.items():
        tensors[name] = _tensor_info_to_spec(info)

    # The native CHW/HWC loader distinguishes the layouts by rank: HWC is a
    # 2D MAT with channels, while CHW is a 3D one-channel MAT.  Trace arrays
    # carry the channel axis in both NumPy shapes, so normalize the serialized
    # descriptors around each swap operation.
    if _OP_SWAP_HWC_CHW is not None:
        for op in ctx.operations:
            if op.op_type != _OP_SWAP_HWC_CHW or not op.input_names or not op.output_names:
                continue
            source_info = ctx.tensors.get(op.input_names[0])
            result_info = ctx.tensors.get(op.output_names[0])
            if source_info is None or result_info is None or len(source_info.shape) != 3 or len(result_info.shape) != 3:
                continue
            source_spec = tensors[op.input_names[0]]
            result_spec = tensors[op.output_names[0]]
            source_spec["dimensions"] = [int(source_info.shape[0]), int(source_info.shape[1])]
            source_spec["channels"] = int(source_info.shape[2])
            source_spec["usage"] = int(TensorType.MAT.value)
            result_spec["dimensions"] = [int(result_info.shape[0]), int(result_info.shape[1]), int(result_info.shape[2])]
            result_spec["channels"] = 1
            result_spec["usage"] = int(TensorType.MAT.value)

    # Build operator specs
    operators = []
    for op in ctx.operations:
        operators.append(_op_to_spec(op))

    def set_tensor_layout(
        name: str, *, dimensions: List[int], channels: int, usage: int, is_gltf: bool = False
    ) -> None:
        tensor = tensors.get(name)
        if tensor is None:
            return
        tensor["dimensions"] = dimensions
        tensor["channels"] = channels
        tensor["usage"] = usage
        if is_gltf:
            tensor["is_gltf"] = True
            tensor["is_placeholder"] = True
        tensor.pop("flag", None)

    # Normalize trace arrays to the non-MAT tensor forms required by OpenMR.
    for op in ctx.operations:
        if op.op_type == EOperatorType.UV_TO_3D_IN_CAM_SPACE and op.input_names:
            info = ctx.tensors.get(op.input_names[0])
            if info is not None and int(np.prod(info.shape)) % 2 == 0:
                set_tensor_layout(
                    op.input_names[0], dimensions=[int(np.prod(info.shape)) // 2],
                    channels=2, usage=int(TensorType.POINT.value),
                )
        if op.op_type in {
            EOperatorType.LOAD_TEXTURE, EOperatorType.SWITCH_GLTF_RENDER_STATUS,
            EOperatorType.UPDATE_GLTF, EOperatorType.SCENEGRAPH_VISIBILITY,
            EOperatorType.UPDATE_COMPONENT,
        } and op.input_names:
            set_tensor_layout(
                op.input_names[0], dimensions=[1, 1], channels=1,
                usage=int(TensorType.GLTF.value), is_gltf=True,
            )
        if op.op_type == EOperatorType.LOAD_TEXTURE and op.output_names:
            set_tensor_layout(
                op.output_names[0], dimensions=[1], channels=1,
                usage=int(TensorType.SCALAR.value),
            )
        if op.op_type == EOperatorType.UPDATE_GLTF and len(op.input_names) >= 2:
            attribute = op.attrs[0] if op.attrs else ""
            if attribute != "world pose":
                set_tensor_layout(
                    op.input_names[1], dimensions=[int(np.prod(ctx.tensors[op.input_names[1]].shape))],
                    channels=1, usage=int(TensorType.SCALAR.value),
                )
            if len(op.input_names) >= 3 and attribute in {
                "animation", "material::metallic_factor", "material::roughness_factor",
                "material::metallic_roughness_texture", "material::base_color_factor",
                "material::base_color_texture", "material::normal_map_texture",
                "material::occlusion_texture", "material::emissive_factor",
                "material::emissive_strength", "material::emissive_texture",
            }:
                value_info = ctx.tensors[op.input_names[2]]
                value_count = int(np.prod(value_info.shape))
                channels = 4 if attribute in {
                    "material::base_color_factor", "material::emissive_factor"
                } else 1
                set_tensor_layout(
                    op.input_names[2], dimensions=[max(value_count // channels, 1)],
                    channels=channels, usage=int(TensorType.SCALAR.value),
                )
            if len(op.input_names) >= 3 and attribute == "local":
                transform_info = ctx.tensors[op.input_names[2]]
                transform_shape = list(transform_info.shape)
                if transform_shape == [4, 4] or (
                    len(transform_shape) == 3 and transform_shape[1:] == [4, 4]
                ):
                    set_tensor_layout(
                        op.input_names[2], dimensions=transform_shape, channels=1,
                        usage=int(TensorType.MAT.value),
                    )
        if op.op_type == EOperatorType.RENDER_TEXT and len(op.input_names) >= 6:
            set_tensor_layout(
                op.input_names[1], dimensions=[1], channels=2, usage=int(TensorType.POINT.value)
            )
            set_tensor_layout(
                op.input_names[2], dimensions=[2], channels=4, usage=int(TensorType.COLOR.value)
            )
            set_tensor_layout(
                op.input_names[3], dimensions=[1, 1], channels=1,
                usage=int(TensorType.GLTF.value), is_gltf=True,
            )
            for name in (op.input_names[4], op.input_names[5]):
                set_tensor_layout(
                    name, dimensions=[1], channels=1, usage=int(TensorType.SCALAR.value)
                )

    # Determine inputs and outputs
    inputs = [name for name, info in ctx.tensors.items() if info.is_input]
    command_only_outputs = {
        output_name
        for op in ctx.operations
        if op.op_type in {EOperatorType.SCENEGRAPH_VISIBILITY, EOperatorType.UPDATE_COMPONENT}
        for output_name in op.output_names
    }
    outputs = [
        name
        for name, info in ctx.tensors.items()
        if info.is_output and name not in command_only_outputs
    ]

    # A tensor cannot be both a pipeline input placeholder and an output
    # placeholder in the package contract.  Give an in-place/identity output
    # its own local tensor name while preserving the declared input.
    overlapping = set(inputs) & set(outputs)
    for name in sorted(overlapping):
        output_name = f"{name}_output"
        suffix = 1
        while output_name in tensors:
            output_name = f"{name}_output_{suffix}"
            suffix += 1
        output_spec = dict(tensors[name])
        output_spec["is_placeholder"] = True
        tensors[output_name] = output_spec
        outputs[outputs.index(name)] = output_name
        for op in ctx.operations:
            op.output_names = [output_name if item == name else item for item in op.output_names]

    # Fix up tensor specs for scalar-result operators (ALL/ANY).
    scalar_result_ops = {EOperatorType.ALL, EOperatorType.ANY}
    scalar_outputs = set()
    for op in ctx.operations:
        if op.op_type in scalar_result_ops and op.output_names:
            scalar_outputs.add(op.output_names[0])
    for name in scalar_outputs:
        spec = tensors.get(name)
        if spec is None:
            continue
        spec["dimensions"] = [1]
        spec["channels"] = 1
        spec["usage"] = 2  # TensorType.SCALAR
        spec.pop("flag", None)

    # Fix up tensor specs for argmax outputs (channel-wise indices).
    for op in ctx.operations:
        if op.op_type != EOperatorType.ARGMAX or not op.input_names or not op.output_names:
            continue
        input_spec = tensors.get(op.input_names[0])
        output_spec = tensors.get(op.output_names[0])
        if input_spec is None or output_spec is None:
            continue
        input_channels = int(input_spec.get("channels", 1))
        input_dims = len(input_spec.get("dimensions", [])) or 1
        output_spec["dimensions"] = [1, input_channels] if input_channels > 0 else [1, 1]
        output_spec["channels"] = input_dims
        output_spec["usage"] = 6  # TensorType.MAT
        output_spec.pop("flag", None)

    # Fix up tensor specs for get_affine inputs (expects 2-channel point arrays).
    if _OP_GET_AFFINE is not None:
        for op in ctx.operations:
            if op.op_type != _OP_GET_AFFINE:
                continue
            for name in op.input_names[:2]:
                spec = tensors.get(name)
                if spec is None:
                    continue
                if int(spec.get("channels", 1)) == 2:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if total != 6:
                    continue
                spec["dimensions"] = [3, 1]
                spec["channels"] = 2
                data_type_val = spec.get("data_type")
                if data_type_val is not None:
                    try:
                        dtype_enum = convert_to_dtype(data_type_val, target="smr")
                        spec["flag"] = mat_flag(dtype_enum, 2)
                    except Exception:
                        pass

    # Fix up tensor specs for solve_p_n_p inputs (expects channelized point arrays).
    if _OP_SOLVE_PNP is not None:
        for op in ctx.operations:
            if op.op_type != _OP_SOLVE_PNP or len(op.input_names) < 2:
                continue
            for name, channels in zip(op.input_names[:2], [3, 2]):
                spec = tensors.get(name)
                if spec is None:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if channels <= 0 or total % channels != 0:
                    continue
                count = total // channels
                spec["dimensions"] = [int(count), 1]
                spec["channels"] = channels
                data_type_val = spec.get("data_type")
                if data_type_val is not None:
                    try:
                        dtype_enum = convert_to_dtype(data_type_val, target="smr")
                        spec["flag"] = mat_flag(dtype_enum, channels)
                    except Exception:
                        pass

    # Fix up tensor specs for sort_vec inputs/outputs (vector stored as scalar usage).
    if _OP_SORT_VEC is not None:
        for op in ctx.operations:
            if op.op_type != _OP_SORT_VEC or not op.input_names:
                continue
            for name in list(op.input_names) + list(op.output_names or []):
                spec = tensors.get(name)
                if spec is None:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if total <= 0:
                    continue
                spec["dimensions"] = [int(total)]
                spec["channels"] = 1
                spec["usage"] = 2  # TensorType.SCALAR

    # Build final spec
    spec = {
        "tensors": tensors,
        "operators": operators,
        "inputs": inputs,
        "outputs": outputs,
    }

    return spec


def convert(
    ctx: TraceContext,
    output: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert a trace context to pipeline JSON.

    Args:
        ctx: The trace context containing recorded operations.
        output: Optional path to save the pipeline JSON file.

    Returns:
        Pipeline specification dictionary.
    """
    spec = trace_to_pipeline_spec(ctx)
    validate_pipeline_spec(spec)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    return spec
