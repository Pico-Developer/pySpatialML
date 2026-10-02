# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File-based SpatialML pipeline builder commands."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from securemr.core.types import EDataType
from securemr.core.utils import mat_flag
from securemr.py2smr import convert
from securemr.py2smr.verifier import validate_microphone_attrs, validate_pipeline_spec
from securemr.operator_contracts import (
    OPERATOR_ARITIES,
    SPATIAL_ONLY_OPERATORS,
    XR_ONLY_OPERATORS,
    operator_contract,
    package_operator_name,
)


class PipelineCliError(RuntimeError):
    """Raised when a pipeline CLI operation cannot be completed."""


_DTYPE_ALIASES: Mapping[str, Any] = {
    "uint8": np.uint8,
    "u8": np.uint8,
    "int8": np.int8,
    "i8": np.int8,
    "uint16": np.uint16,
    "u16": np.uint16,
    "int16": np.int16,
    "i16": np.int16,
    "int32": np.int32,
    "i32": np.int32,
    "float32": np.float32,
    "fp32": np.float32,
    "float64": np.float64,
    "fp64": np.float64,
}

_DTYPE_TO_SCHEMA = {
    np.dtype(np.uint8).type: int(EDataType.UINT8),
    np.dtype(np.int8).type: int(EDataType.INT8),
    np.dtype(np.uint16).type: int(EDataType.UINT16),
    np.dtype(np.int16).type: int(EDataType.INT16),
    np.dtype(np.int32).type: int(EDataType.INT32),
    np.dtype(np.float32).type: int(EDataType.FLOAT32),
    np.dtype(np.float64).type: int(EDataType.FLOAT64),
}

_USAGE_ALIASES = {
    "mat": 6,
    "matrix": 6,
    "scalar": 2,
    "point": 1,
    "slice": 3,
    "color": 4,
    "timestamp": 5,
    "gltf": 7,
}

_LEGACY_OPERATOR_FIELDS = {
    "comparison",
    "config",
    "flag",
    "mode",
    "model_asset",
    "model_file",
    "model_id",
    "model_name",
    "model_target",
    "model_type",
    "cpu_target_num_threads",
    "normalize_type",
    "script",
    "target_property",
    "text",
    "threshold",
    "update_type",
    "visible",
}
_XR_ONLY_OPERATORS = XR_ONLY_OPERATORS
_SPATIAL_ONLY_OPERATORS = SPATIAL_ONLY_OPERATORS


def init_pipeline(path: Path, *, force: bool = False) -> int:
    """Create an empty pipeline JSON file."""
    if path.exists() and not force:
        raise PipelineCliError(f"Pipeline already exists: {path}")
    spec = {"tensors": {}, "operators": [], "inputs": [], "outputs": []}
    _write_json(path, spec)
    print(f"Initialized pipeline: {path}")
    return 0


def add_tensor(
    path: Path,
    name: str,
    *,
    shape: str,
    dtype: str,
    usage: str = "matrix",
    channels: Optional[int] = None,
    is_input: bool = False,
    is_output: bool = False,
    value: Optional[str] = None,
    asset: Optional[str] = None,
) -> int:
    """Add or replace a tensor descriptor."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    if name in tensors:
        raise PipelineCliError(f"Tensor already exists: {name}")

    parsed_shape = _parse_int_list(shape)
    usage_value = _usage_value(usage)
    if channels is None and usage_value != _USAGE_ALIASES["matrix"]:
        dimensions = [int(dim) for dim in parsed_shape]
        inferred_channels = {
            _USAGE_ALIASES["point"]: 2,
            _USAGE_ALIASES["color"]: 4,
            _USAGE_ALIASES["timestamp"]: 4,
            _USAGE_ALIASES["slice"]: 2,
        }.get(usage_value, 1)
    else:
        dimensions, inferred_channels = _shape_to_dimensions_and_channels(parsed_shape)
    if channels is None:
        channels = inferred_channels
    else:
        if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 127:
            raise PipelineCliError("Tensor channels must be an integer from 1 through 127")
        dimensions = [int(dim) for dim in parsed_shape]
    data_type = _schema_dtype(dtype)
    tensor_spec: dict[str, Any] = {
        "dimensions": dimensions,
        "channels": channels,
        "data_type": data_type,
        "is_placeholder": bool(is_input or is_output),
        "usage": usage_value,
    }
    if usage_value == _USAGE_ALIASES["matrix"]:
        tensor_spec["flag"] = mat_flag(EDataType(data_type), channels)
    if usage_value == _USAGE_ALIASES["gltf"]:
        tensor_spec["is_gltf"] = True
        tensor_spec["is_placeholder"] = True
    if value is not None:
        tensor_spec["data"] = _parse_value_list(value)
    if asset is not None:
        asset_value = asset.replace("\\", "/").lstrip("/")
        if not asset_value or any(part in {"", ".", ".."} for part in asset_value.split("/")):
            raise PipelineCliError(f"Invalid package-relative tensor asset: {asset}")
        tensor_spec["asset"] = asset_value

    tensors[name] = tensor_spec
    if is_input:
        _append_unique(spec.setdefault("inputs", []), name)
    if is_output:
        _append_unique(spec.setdefault("outputs", []), name)
    _validate_and_write(path, spec)
    print(f"Added tensor: {name}")
    return 0


def add_op(
    path: Path,
    op_type: str,
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    attrs: Sequence[str] = (),
    expression: Optional[str] = None,
    dtype: Optional[str] = None,
    flag: Optional[str] = None,
    threshold: Optional[float] = None,
    comparison: Optional[str] = None,
    normalize_type: Optional[str] = None,
    script: Optional[str] = None,
    config: Optional[str] = None,
    text: Optional[str] = None,
    update_type: Optional[str] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
    scenegraph: Optional[str] = None,
    entity_path: Optional[str] = None,
    property: Optional[str] = None,
    target_property: Optional[str] = None,
    data: Optional[str] = None,
    src_slices: Any = None,
    dst_slices: Any = None,
    src_channel_slice: Any = None,
    dst_channel_slice: Any = None,
) -> int:
    """Append an operator to a pipeline."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    normalized_op_type = _operator_type(op_type)
    if _operator_enum_name(normalized_op_type) not in OPERATOR_ARITIES:
        raise PipelineCliError(f"Unsupported operator type: {op_type}")
    effective_inputs = [_normalize_tensor_ref(ref) for ref in inputs]
    effective_outputs = [_normalize_tensor_ref(ref) for ref in outputs]
    op_name = _operator_enum_name(normalized_op_type)
    if op_name in {"SSMR_SWITCH_VISIBILITY", "SSMR_UPDATE_COMPONENT"} and scenegraph:
        if not effective_inputs:
            effective_inputs.append(_tensor_ref(scenegraph))
        elif _resolve_ref_name(effective_inputs[0]) != scenegraph:
            raise PipelineCliError(
                f"{op_name.lower()} --scenegraph must match the first --input tensor"
            )
    if op_name == "SSMR_UPDATE_COMPONENT" and data and data not in {_resolve_ref_name(ref) for ref in effective_inputs}:
        effective_inputs.append(_tensor_ref(data))
    effective_property = property or target_property
    referenced_names = [_resolve_ref_name(ref) for ref in [*effective_inputs, *effective_outputs]]
    for tensor_name in referenced_names:
        if tensor_name and tensor_name not in tensors:
            raise PipelineCliError(f"Unknown tensor referenced by operator: {tensor_name}")

    effective_inputs, effective_outputs = _normalize_operator_refs_for_arity(
        normalized_op_type,
        inputs=effective_inputs,
        outputs=effective_outputs,
    )
    _validate_operator_arity(
        normalized_op_type,
        inputs=effective_inputs,
        outputs=effective_outputs,
    )
    if op_name == "ARITHMETIC_COMPOSE" and not expression and not attrs:
        raise PipelineCliError("Arithmetic operators require --expression")
    canonical_attrs = _canonical_attrs(
        op_name,
        attrs=attrs,
        expression=expression,
        flag=flag,
        threshold=threshold,
        comparison=comparison,
        normalize_type=normalize_type,
        script=script,
        config=config,
        text=text,
        update_type=update_type,
        mode=mode,
        entity_path=entity_path,
        property=effective_property,
    )
    _validate_required_operator_metadata(
        normalized_op_type,
        attrs=canonical_attrs,
        model=model,
    )
    _validate_structured_operator_fields(
        normalized_op_type,
        inputs=effective_inputs,
        outputs=effective_outputs,
        scenegraph=scenegraph,
        entity_path=entity_path,
        property=effective_property,
        target_property=target_property,
        data=data,
        src_slices=src_slices,
        dst_slices=dst_slices,
    )

    op = {
        "type": normalized_op_type,
        "inputs": effective_inputs,
        "outputs": effective_outputs,
    }
    if canonical_attrs:
        op["attrs"] = canonical_attrs
    if model is not None:
        if not model_name:
            model_name = Path(model).stem or "model"
        op["model"] = {
            "bin_path": _normalize_model_path(model),
            "model_name": model_name,
            "model_type": "tflite",
            "model_target": model_target,
            "input": _model_io_metadata(effective_inputs, tensors),
            "output": _model_io_metadata(effective_outputs, tensors),
        }
        if model_target.lower() == "cpu":
            op["model"]["cpu_target_num_threads"] = int(cpu_target_num_threads)
    if src_slices is not None:
        op["src_slices"] = _parse_structured_list(src_slices, "src_slices")
    if dst_slices is not None:
        op["dst_slices"] = _parse_structured_list(dst_slices, "dst_slices")
    if src_channel_slice is not None:
        op["src_channel_slice"] = _parse_structured_list(src_channel_slice, "src_channel_slice")
    if dst_channel_slice is not None:
        op["dst_channel_slice"] = _parse_structured_list(dst_channel_slice, "dst_channel_slice")

    spec.setdefault("operators", []).append(op)
    _validate_and_write_operator_update(path, spec)
    print(f"Added operator: {op_type}")
    return 0


def remove_op(path: Path, index: int) -> int:
    """Remove an operator from a pipeline by index."""
    spec = _load_pipeline(path)
    operators = spec.get("operators", [])
    if not isinstance(operators, list):
        raise PipelineCliError("Pipeline 'operators' must be a list")
    if index < 0 or index >= len(operators):
        raise PipelineCliError(f"Operator index out of range: {index}")
    removed = operators.pop(index)
    if not isinstance(removed, Mapping):
        removed_type = "<invalid>"
    else:
        removed_type = str(removed.get("type") or "<unknown>")
    _validate_and_write_operator_update(path, spec)
    print(f"Removed operator #{index}: {removed_type}")
    return 0


def remove_tensor(path: Path, name: str, *, force: bool = False) -> int:
    """Remove a tensor descriptor from a pipeline."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    if name not in tensors:
        raise PipelineCliError(f"Tensor not found: {name}")

    references = _operator_references_to_tensor(spec, name)
    if references and not force:
        details = ", ".join(f"#{index} {op_type}" for index, op_type in references)
        raise PipelineCliError(
            f"Tensor '{name}' is referenced by operator(s): {details}. "
            "Remove those operators first or rerun with --force."
        )

    del tensors[name]
    for key in ("inputs", "outputs"):
        values = spec.get(key, [])
        if isinstance(values, list):
            spec[key] = [value for value in values if value != name]
    if force and references:
        _write_json(path, spec)
    else:
        _validate_and_write(path, spec)
    print(f"Removed tensor: {name}")
    if references and force:
        details = ", ".join(f"#{index} {op_type}" for index, op_type in references)
        print(f"Warning: removed tensor was referenced by operator(s): {details}")
    return 0


def _validate_operator_arity(
    op_type: str,
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    has_inline_affine_points: bool = False,
) -> None:
    op_name = _operator_enum_name(op_type)
    contract = operator_contract(op_name)
    if contract is None:
        return
    arity = contract.builder_arity
    min_inputs, max_inputs, min_outputs, max_outputs, nullable_inputs, nullable_outputs = arity
    if op_name == "GET_AFFINE" and has_inline_affine_points:
        min_inputs = max_inputs = 0
    _validate_count(
        op_name,
        "input",
        len(inputs),
        minimum=min_inputs,
        maximum=max_inputs,
    )
    _validate_count(
        op_name,
        "output",
        len(outputs),
        minimum=min_outputs,
        maximum=max_outputs,
    )
    _validate_non_nullable_slots(op_name, "input", inputs, nullable_inputs)
    _validate_non_nullable_slots(op_name, "output", outputs, nullable_outputs)


def _normalize_operator_refs_for_arity(
    op_type: str,
    *,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
    op_name = _operator_enum_name(op_type)
    contract = operator_contract(op_name)
    if contract is None:
        return list(inputs), list(outputs)
    if op_name in XR_ONLY_OPERATORS:
        arity = contract.builder_arity
        _, max_inputs, _, max_outputs, nullable_inputs, nullable_outputs = arity
        return (
            _pad_nullable_refs(list(inputs), max_inputs, nullable_inputs),
            _pad_nullable_refs(list(outputs), max_outputs, nullable_outputs),
        )
    return _trim_trailing_null_refs(list(inputs)), _trim_trailing_null_refs(list(outputs))


def _pad_nullable_refs(
    refs: list[Any], maximum: Optional[int], nullable_slots: set[int]
) -> list[Any]:
    if maximum is None:
        return refs
    for index in range(len(refs), maximum):
        if index in nullable_slots:
            refs.append(None)
    return refs


def _trim_trailing_null_refs(refs: list[Any]) -> list[Any]:
    """Use the dense schema-v2 form accepted by the current SpatialSDK loader."""
    while refs and _resolve_ref_name(refs[-1]) is None:
        refs.pop()
    if any(_resolve_ref_name(ref) is None for ref in refs):
        raise PipelineCliError(
            "Schema v2 cannot represent an omitted operator slot before a later "
            "connected slot; connect the earlier slot or omit all following slots"
        )
    return refs


def _validate_non_nullable_slots(
    op_name: str,
    label: str,
    refs: Sequence[Any],
    nullable_slots: set[int],
) -> None:
    for index, ref in enumerate(refs):
        if _resolve_ref_name(ref) is None and index not in nullable_slots:
            raise PipelineCliError(
                f"{op_name.lower()} operators do not allow null {label} slot {index}"
            )


def _validate_count(
    op_name: str,
    label: str,
    count: int,
    *,
    minimum: int,
    maximum: Optional[int],
) -> None:
    if count < minimum or (maximum is not None and count > maximum):
        expected = _format_count_range(minimum, maximum)
        raise PipelineCliError(
            f"{op_name.lower()} operators require {expected} {label} tensor(s); got {count}"
        )


def _format_count_range(minimum: int, maximum: Optional[int]) -> str:
    if maximum is None:
        return f"at least {minimum}"
    if minimum == maximum:
        return f"exactly {minimum}"
    return f"{minimum} to {maximum}"


def _validate_required_operator_metadata(
    op_type: str,
    *,
    attrs: Sequence[str],
    model: Optional[str],
) -> None:
    op_name = _operator_enum_name(op_type)
    if op_name == "CONVERT_COLOR" and not attrs:
        raise PipelineCliError("CONVERT_COLOR operators require flag or attrs[0]")
    if op_name == "CUSTOMIZED_COMPARE" and not attrs:
        raise PipelineCliError("customized_compare operators require comparison or attrs[0]")
    if op_name == "JS_SCRIPTING" and not attrs:
        raise PipelineCliError("JS_SCRIPTING operators require script or attrs[0]")
    if op_name == "RENDER_TEXT" and not attrs:
        raise PipelineCliError("render_text operators require config or attrs[0]")
    if op_name == "UPDATE_GLTF" and not attrs:
        raise PipelineCliError("update_gltf operators require update_type or attrs[0]")
    if op_name == "SSMR_UPDATE_COMPONENT" and not attrs:
        raise PipelineCliError("update_component operators require component path in attrs[0]")
    if op_name == "MICROPHONE" and not attrs:
        raise PipelineCliError("microphone operators require attrs[0]")
    if op_name == "MICROPHONE":
        try:
            validate_microphone_attrs(list(attrs), "microphone operators")
        except ValueError as exc:
            raise PipelineCliError(str(exc)) from exc
    if op_name == "SPEAKER" and not attrs:
        raise PipelineCliError("speaker operators require attrs[0]")
    if _operator_enum_name(op_type) == "RUN_MODEL_INFERENCE" and model is None:
        raise PipelineCliError("RUN_MODEL_INFERENCE operators require --model")


def _validate_structured_operator_fields(
    op_type: str,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    scenegraph: Optional[str],
    entity_path: Optional[str],
    property: Optional[str],
    target_property: Optional[str],
    data: Optional[str],
    src_slices: Any,
    dst_slices: Any,
) -> None:
    op_name = _operator_enum_name(op_type)
    if op_name == "SSMR_UPDATE_COMPONENT":
        if not scenegraph and not inputs:
            raise PipelineCliError("update_component requires --scenegraph")
        if entity_path and not entity_path.startswith("/"):
            raise PipelineCliError("update_component entity path must start with '/'")
        if not property and not target_property:
            raise PipelineCliError("update_component requires --property")
        if not data and len(inputs) < 2:
            raise PipelineCliError("update_component requires --data")
        if outputs:
            raise PipelineCliError("update_component does not produce output tensors")
    if op_name == "SSMR_SWITCH_VISIBILITY":
        if not scenegraph and not inputs:
            raise PipelineCliError("scenegraph_visibility requires --scenegraph")


def _canonical_attrs(
    op_name: str,
    *,
    attrs: Sequence[str],
    expression: Optional[str],
    flag: Optional[str],
    threshold: Optional[float],
    comparison: Optional[str],
    normalize_type: Optional[str],
    script: Optional[str],
    config: Optional[str],
    text: Optional[str],
    update_type: Optional[str],
    mode: Optional[str],
    entity_path: Optional[str],
    property: Optional[str],
) -> list[str]:
    values = list(attrs)
    if op_name == "ARITHMETIC_COMPOSE" and expression is not None:
        values = [expression]
    elif op_name == "CONVERT_COLOR" and flag is not None:
        values = [str(_parse_int(flag))]
    elif op_name == "CUSTOMIZED_COMPARE" and comparison is not None:
        values = [comparison]
    elif op_name == "NMS" and threshold is not None:
        values = [str(float(threshold))]
    elif op_name in {"NORMALIZE", "NORM"} and normalize_type is not None:
        values = [normalize_type]
    elif op_name == "JS_SCRIPTING" and script is not None:
        values = [script]
    elif op_name == "RENDER_TEXT" and config is not None:
        values = [config]
    elif op_name == "UPDATE_GLTF" and update_type is not None:
        values = [update_type]
    elif op_name == "SORT_MAT" and mode is not None:
        values = [mode]
    elif op_name == "SSMR_UPDATE_COMPONENT" and entity_path and property:
        values = [f"{entity_path}:{property}"]
    if op_name == "RENDER_TEXT" and text is not None:
        raise PipelineCliError("render_text text must be provided as input tensor, not --text")
    return values


def _parse_bool_or_tensor(value: str) -> Union[bool, str]:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def _parse_structured_list(value: Any, label: str) -> Any:
    """Parse JSON or comma-separated structured CLI values."""
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        raise PipelineCliError(f"{label} cannot be empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for row in text.split(";"):
            rows.append([_parse_number(item.strip(), label) for item in row.split(",") if item.strip()])
        parsed = rows[0] if len(rows) == 1 else rows
    if not isinstance(parsed, list):
        raise PipelineCliError(f"{label} must be a JSON array or comma-separated list")
    return parsed


def _parse_number(value: str, label: str) -> Union[int, float]:
    try:
        return float(value) if any(char in value.lower() for char in (".", "e")) else int(value, 0)
    except ValueError as exc:
        raise PipelineCliError(f"Invalid number in {label}: {value}") from exc


def set_input(path: Path, names: Sequence[str]) -> int:
    """Set top-level pipeline inputs."""
    return _set_boundary(path, "inputs", names)


def set_output(path: Path, names: Sequence[str]) -> int:
    """Set top-level pipeline outputs."""
    return _set_boundary(path, "outputs", names)


def validate_pipeline(path: Path) -> int:
    """Validate a pipeline JSON file."""
    spec = _load_pipeline(path)
    try:
        _validate_pipeline_update(spec)
    except ValueError as exc:
        raise PipelineCliError(str(exc)) from exc
    print(f"Pipeline is valid: {path}")
    return 0


def inspect_pipeline(path: Path) -> int:
    """Print a compact pipeline summary."""
    spec = _load_pipeline(path)
    tensors = spec.get("tensors", {})
    operators = spec.get("operators", [])
    print(f"Pipeline: {path}")
    print(f"Tensors: {len(tensors)}")
    print(f"Operators: {len(operators)}")
    print(f"Inputs: {', '.join(spec.get('inputs', [])) or '-'}")
    print(f"Outputs: {', '.join(spec.get('outputs', [])) or '-'}")
    for index, op in enumerate(operators):
        print(
            f"  [{index}] {op.get('type', '<unknown>')} "
            f"{op.get('inputs', [])} -> {op.get('outputs', [])}"
        )
    return 0


def trace_pipeline(
    source: Path,
    *,
    function_name: str,
    output: Path,
    inputs: Sequence[str],
) -> int:
    """Import a Python file, run a traceable function, and write pipeline JSON."""
    module = _load_module(source)
    target = getattr(module, function_name, None)
    if target is None:
        raise PipelineCliError(f"Function not found: {function_name}")
    trace_fn = getattr(target, "trace", None)
    if trace_fn is None:
        raise PipelineCliError(f"Function is not traceable with @trace: {function_name}")

    input_values = dict(_parse_trace_input(item) for item in inputs)
    _result, ctx = trace_fn(**input_values)
    convert(ctx, output=str(output))
    print(f"Wrote traced pipeline: {output}")
    return 0


def print_pipeline_error(exc: Exception) -> None:
    """Print a concise pipeline command error."""
    print(f"Error [PSM_PIPELINE]: {exc}", file=sys.stderr)


def _set_boundary(path: Path, key: str, names: Sequence[str]) -> int:
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    missing = [name for name in names if name not in tensors]
    if missing:
        raise PipelineCliError(f"Unknown tensor(s): {', '.join(missing)}")
    previous = set(spec.get(key, [])) if isinstance(spec.get(key), list) else set()
    spec[key] = list(dict.fromkeys(names))
    for name in previous - set(spec[key]):
        if name in tensors and name not in set(spec.get("inputs", [])) | set(spec.get("outputs", [])):
            tensors[name]["is_placeholder"] = False
    for name in names:
        tensors[name]["is_placeholder"] = True
    _validate_and_write(path, spec)
    print(f"Set {key}: {', '.join(names) or '-'}")
    return 0


def _load_pipeline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineCliError(f"Pipeline not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise PipelineCliError("Pipeline JSON must be an object")
    data.setdefault("tensors", {})
    data.setdefault("operators", [])
    data.setdefault("inputs", [])
    data.setdefault("outputs", [])
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_updates([(path, payload)])


def _write_json_updates(updates: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    temp_paths: list[Path] = []
    try:
        for path, payload in updates:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
            temp_paths.append(tmp)
        for tmp, (path, _payload) in zip(temp_paths, updates):
            tmp.replace(path)
    finally:
        for tmp in temp_paths:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _validate_and_write_operator_update(path: Path, spec: Mapping[str, Any]) -> None:
    _validate_pipeline_update(spec)
    manifest_update = _prepare_manifest_modes_for_pipeline(path, spec)
    updates: list[tuple[Path, Mapping[str, Any]]] = [(path, spec)]
    if manifest_update is not None:
        updates.append(manifest_update)
    _write_json_updates(updates)


def _validate_pipeline_update(spec: Mapping[str, Any]) -> None:
    _validate_tensor_references(spec)
    _reject_legacy_operator_fields(spec)
    _validate_canonical_operator_specs(spec)
    validate_pipeline_spec(spec)


def _validate_canonical_operator_specs(spec: Mapping[str, Any]) -> None:
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            continue
        op_name = _operator_enum_name(str(op.get("type") or ""))
        contract = operator_contract(op_name)
        if contract is None:
            continue
        arity = contract.serialized_arity
        attrs = op.get("attrs", [])
        if attrs is None:
            attrs = []
        if not isinstance(attrs, list) or not all(isinstance(item, str) for item in attrs):
            raise PipelineCliError(f"Operator #{index} attrs must be an array of strings")
        _validate_attr_count(index, op_name, attrs)
        min_inputs, max_inputs, min_outputs, max_outputs, nullable_inputs, nullable_outputs = arity
        inputs = op.get("inputs", [])
        outputs = op.get("outputs", [])
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            raise PipelineCliError(f"Operator #{index} inputs and outputs must be arrays")
        _validate_count(op_name, "input", len(inputs), minimum=min_inputs, maximum=max_inputs)
        _validate_count(op_name, "output", len(outputs), minimum=min_outputs, maximum=max_outputs)
        _validate_non_nullable_slots(op_name, "input", inputs, nullable_inputs)
        _validate_non_nullable_slots(op_name, "output", outputs, nullable_outputs)


def _validate_attr_count(index: int, op_name: str, attrs: Sequence[str]) -> None:
    required = 0
    maximum: Optional[int] = 0
    if op_name in {"ARITHMETIC_COMPOSE", "CONVERT_COLOR", "CUSTOMIZED_COMPARE"}:
        required = maximum = 1
    elif op_name in {"NORMALIZE", "NMS", "SORT_MAT", "NORM"}:
        maximum = 1
    elif op_name in {
        "UPDATE_GLTF",
        "RENDER_TEXT",
        "SSMR_UPDATE_COMPONENT",
        "MICROPHONE",
        "SPEAKER",
        "JS_SCRIPTING",
    }:
        required = maximum = 1
    if len(attrs) < required:
        raise PipelineCliError(f"Operator #{index} {op_name} requires at least {required} attrs value(s)")
    if maximum is not None and len(attrs) > maximum:
        raise PipelineCliError(f"Operator #{index} {op_name} accepts at most {maximum} attrs value(s)")


def _reject_legacy_operator_fields(spec: Mapping[str, Any]) -> None:
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            continue
        legacy_fields = sorted(key for key in op if key in _LEGACY_OPERATOR_FIELDS)
        if legacy_fields:
            raise PipelineCliError(
                f"Operator #{index} uses legacy package field(s): "
                f"{', '.join(legacy_fields)}. Use attrs and inline model metadata instead."
            )


def _validate_and_write(path: Path, spec: Mapping[str, Any]) -> None:
    _validate_pipeline_update(spec)
    _write_json(path, spec)


def _prepare_manifest_modes_for_pipeline(
    path: Path,
    spec: Mapping[str, Any],
) -> Optional[tuple[Path, Mapping[str, Any]]]:
    manifest_context = _find_manifest_context(path)
    if manifest_context is None:
        classification = _classify_mode_specific_operators(spec.get("operators", []))
        _raise_if_modes_conflict(classification)
        return None

    manifest_path, manifest = manifest_context
    runtime = manifest.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise PipelineCliError("Package manifest 'runtime' must be an object")
    supported_modes = runtime.get("supported_modes", [])
    if supported_modes and not isinstance(supported_modes, list):
        raise PipelineCliError("Package manifest runtime.supported_modes must be a list")

    pipeline_specs = _load_manifest_pipeline_specs_with_candidate(
        manifest,
        manifest_path.parent,
        path,
        spec,
    )
    classification = _classify_mode_specific_pipeline_specs(pipeline_specs)
    _raise_if_modes_conflict(classification)

    current_modes = {str(mode).strip().lower() for mode in supported_modes}
    if classification["xr"]:
        if current_modes and "xr" not in current_modes:
            raise PipelineCliError(
                "Cannot add XR-only operator to pipeline referenced by a manifest "
                "whose runtime.supported_modes only includes spatial."
            )
        desired_modes = ["xr"]
    elif classification["spatial"]:
        if current_modes and "spatial" not in current_modes:
            raise PipelineCliError(
                "Cannot add Spatial-only operator to pipeline referenced by a manifest "
                "whose runtime.supported_modes only includes xr."
            )
        desired_modes = ["spatial"]
    elif current_modes:
        desired_modes = ["xr", "spatial"]
    else:
        return None
    if list(supported_modes) == desired_modes:
        return None
    runtime["supported_modes"] = desired_modes
    return manifest_path, manifest


def _load_manifest_pipeline_specs_with_candidate(
    manifest: Mapping[str, Any],
    root: Path,
    candidate_path: Path,
    candidate_spec: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    pipelines = manifest.get("pipelines", [])
    if not isinstance(pipelines, list):
        raise PipelineCliError("Package manifest 'pipelines' must be a list")

    resolved_candidate_path = candidate_path.resolve()
    pipeline_specs: list[tuple[str, Mapping[str, Any]]] = []
    matched_candidate = False
    for index, item in enumerate(pipelines):
        if not isinstance(item, Mapping):
            raise PipelineCliError("Package manifest pipeline entries must be objects")
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise PipelineCliError("Package manifest pipeline entries require a non-empty path")
        pipeline_id = str(item.get("id") or rel_path or index)
        pipeline_path = _resolve_manifest_pipeline_path(root, rel_path)
        if pipeline_path == resolved_candidate_path:
            pipeline_specs.append((pipeline_id, candidate_spec))
            matched_candidate = True
            continue
        pipeline_specs.append((pipeline_id, _load_pipeline(pipeline_path)))
    if not matched_candidate:
        raise PipelineCliError("Package manifest no longer references the edited pipeline")
    return pipeline_specs


def _resolve_manifest_pipeline_path(root: Path, path: str) -> Path:
    raw_path = Path(path)
    normalized = path.replace("\\", "/").strip("/")
    if raw_path.is_absolute() or path.replace("\\", "/").startswith("/") or not normalized:
        raise PipelineCliError(f"Package manifest pipeline path must be package-relative: {path}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PipelineCliError(f"Invalid package manifest pipeline path: {path}")
    resolved_root = root.resolve()
    resolved_path = (root / normalized).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PipelineCliError(f"Package manifest pipeline path escapes package root: {path}") from exc
    return resolved_path


def _classify_mode_specific_pipeline_specs(
    pipeline_specs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, set[str]]:
    xr_only = []
    spatial_only = []
    for pipeline_id, spec in pipeline_specs:
        classification = _classify_mode_specific_operators(spec.get("operators", []))
        xr_only.extend(f"{pipeline_id}:{op_type}" for op_type in sorted(classification["xr"]))
        spatial_only.extend(f"{pipeline_id}:{op_type}" for op_type in sorted(classification["spatial"]))
    return {"xr": set(xr_only), "spatial": set(spatial_only)}


def _raise_if_modes_conflict(classification: Mapping[str, set[str]]) -> None:
    if classification["xr"] and classification["spatial"]:
        raise PipelineCliError(
            "Cannot update pipeline because it would mix XR-only and Spatial-only operators. "
            f"XR-only: {', '.join(sorted(classification['xr']))}. "
            f"Spatial-only: {', '.join(sorted(classification['spatial']))}."
        )


def _validate_tensor_references(spec: Mapping[str, Any]) -> None:
    tensors = spec.get("tensors", {})
    if not isinstance(tensors, Mapping):
        raise PipelineCliError("Pipeline 'tensors' must be an object")
    for key in ("inputs", "outputs"):
        refs = spec.get(key, [])
        if not isinstance(refs, list):
            raise PipelineCliError(f"Pipeline '{key}' must be a list")
        missing = [name for name in refs if name not in tensors]
        if missing:
            raise PipelineCliError(f"Unknown {key} tensor(s): {', '.join(missing)}")
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            raise PipelineCliError(f"Operator #{index} must be an object")
        for key in ("inputs", "outputs"):
            for ref in op.get(key, []):
                tensor_name = _resolve_ref_name(ref)
                if tensor_name and tensor_name not in tensors:
                    raise PipelineCliError(
                        f"Operator #{index} references unknown tensor '{tensor_name}'"
                    )
                if ref is not None and not isinstance(ref, Mapping):
                    raise PipelineCliError(
                        f"Operator #{index} {key} entries must be tensor-reference objects or null"
                    )


def _operator_references_to_tensor(spec: Mapping[str, Any], tensor_name: str) -> list[tuple[int, str]]:
    references = []
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            continue
        for key in ("inputs", "outputs"):
            for ref in op.get(key, []):
                if _resolve_ref_name(ref) == tensor_name:
                    references.append((index, str(op.get("type") or "<unknown>")))
                    break
            else:
                continue
            break
    return references


def _classify_mode_specific_operators(operators: Sequence[Any]) -> dict[str, set[str]]:
    xr_only = []
    spatial_only = []
    for op in operators:
        if not isinstance(op, Mapping):
            continue
        op_type = _operator_enum_name(str(op.get("type") or ""))
        if op_type in _XR_ONLY_OPERATORS:
            xr_only.append(op_type)
        if op_type in _SPATIAL_ONLY_OPERATORS:
            spatial_only.append(op_type)
    return {"xr": set(xr_only), "spatial": set(spatial_only)}


def _find_manifest_context(path: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    pipeline_path = path.resolve()
    for manifest_path in _candidate_manifest_paths(path):
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json_file(manifest_path)
        except PipelineCliError:
            continue
        if not _manifest_references_pipeline(manifest, manifest_path.parent, pipeline_path):
            continue
        return manifest_path, manifest
    return None


def _candidate_manifest_paths(path: Path) -> list[Path]:
    candidates = []
    for directory in [path.parent, *path.parents]:
        candidate = directory / "manifest.json"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _manifest_references_pipeline(manifest: Mapping[str, Any], root: Path, pipeline_path: Path) -> bool:
    pipelines = manifest.get("pipelines", [])
    if not isinstance(pipelines, list):
        return False
    for item in pipelines:
        if not isinstance(item, Mapping):
            continue
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        try:
            manifest_pipeline = _resolve_manifest_pipeline_path(root, rel_path)
        except (OSError, PipelineCliError):
            continue
        if manifest_pipeline == pipeline_path:
            return True
    return False


def _read_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise PipelineCliError(f"JSON file must contain an object: {path}")
    return payload


def _tensors(spec: dict[str, Any]) -> dict[str, Any]:
    tensors = spec.setdefault("tensors", {})
    if not isinstance(tensors, dict):
        raise PipelineCliError("Pipeline 'tensors' must be an object")
    return tensors


def _resolve_ref_name(ref: Any) -> Optional[str]:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, Mapping):
        return ref.get("tensor") or ref.get("name")
    return None


def _normalize_tensor_ref(ref: Any) -> Optional[dict[str, str]]:
    if isinstance(ref, str) and ref.strip().lower() in {"null", "none"}:
        return None
    tensor_name = _resolve_ref_name(ref)
    if tensor_name is None:
        return None
    result = _tensor_ref(tensor_name)
    if isinstance(ref, Mapping) and isinstance(ref.get("name"), str):
        result["name"] = ref["name"]
    return result


def _tensor_ref(name: str) -> dict[str, str]:
    return {"tensor": name}


def _model_io_metadata(refs: Sequence[Any], tensors: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = []
    for ref in refs:
        name = _resolve_ref_name(ref)
        if not name:
            continue
        tensor = tensors.get(name, {})
        dimensions = list(tensor.get("dimensions", [])) if isinstance(tensor, Mapping) else []
        channels = int(tensor.get("channels", 1) or 1) if isinstance(tensor, Mapping) else 1
        shape = [*dimensions, channels] if channels > 1 else dimensions
        logical_name = ref.get("name") if isinstance(ref, Mapping) and ref.get("name") else name
        metadata.append(
            {
                "name": logical_name,
                "shape": shape,
                "encoding_type": _encoding_type(tensor.get("data_type") if isinstance(tensor, Mapping) else None),
            }
        )
    return metadata


def _encoding_type(data_type: Any) -> str:
    try:
        code = int(data_type)
    except (TypeError, ValueError):
        code = int(EDataType.FLOAT32)
    return {
        int(EDataType.UINT8): "UINT8",
        int(EDataType.INT8): "INT8",
        int(EDataType.UINT16): "UINT16",
        int(EDataType.INT16): "INT16",
        int(EDataType.INT32): "INT32",
        int(EDataType.FLOAT32): "FP32",
        int(EDataType.FLOAT64): "FP64",
    }.get(code, "FP32")


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _parse_int_list(value: str) -> list[int]:
    if not value:
        raise PipelineCliError("Shape cannot be empty")
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise PipelineCliError(f"Invalid integer list: {value}") from exc


def _shape_to_dimensions_and_channels(shape: Sequence[int]) -> tuple[list[int], int]:
    if not shape:
        raise PipelineCliError("Shape cannot be empty")
    if any(int(dim) <= 0 for dim in shape):
        raise PipelineCliError(f"Shape dimensions must be positive: {shape}")
    dims = [int(dim) for dim in shape]
    if len(dims) == 1:
        return [dims[0], 1], 1
    if len(dims) == 2:
        return dims, 1
    if len(dims) == 3:
        return dims[:2], dims[2]
    return [int(np.prod(dims[1:])), dims[0]], 1


def _schema_dtype(dtype: str) -> int:
    key = dtype.strip().lower()
    np_type = _DTYPE_ALIASES.get(key)
    if np_type is None:
        raise PipelineCliError(f"Unsupported dtype: {dtype}")
    return _DTYPE_TO_SCHEMA[np.dtype(np_type).type]


def _usage_value(usage: str) -> int:
    key = usage.strip().lower().replace("-", "_")
    if key in _USAGE_ALIASES:
        return _USAGE_ALIASES[key]
    return _parse_int(usage)


def _parse_int(value: str) -> int:
    try:
        return int(str(value), 0)
    except ValueError as exc:
        raise PipelineCliError(f"Invalid integer value: {value}") from exc


def _parse_value_list(value: str) -> list[Any]:
    if not value:
        return []
    result: list[Any] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            if "." in text or "e" in text.lower():
                result.append(float(text))
            else:
                result.append(int(text, 0))
        except ValueError:
            raise PipelineCliError(f"Invalid numeric tensor data: {text}")
    return result


def _operator_type(op_type: str) -> str:
    key = str(op_type)
    if not package_operator_name(key):
        raise PipelineCliError(
            f"Unsupported operator type {op_type!r}; use a canonical XR_SECURE_MR_OPERATOR_TYPE_*_PICO name"
        )
    return key


def _operator_enum_name(op_type: str) -> str:
    return package_operator_name(str(op_type))


def _normalize_model_path(model: str) -> str:
    if not model.endswith(".tflite"):
        raise PipelineCliError("Model path must point to a .tflite file")
    normalized = model.replace("\\", "/").lstrip("/")
    if normalized.startswith("../") or "/../" in normalized:
        raise PipelineCliError(f"Invalid package-relative model path: {model}")
    return normalized


def _load_module(source: Path):
    if not source.is_file():
        raise PipelineCliError(f"Python source file not found: {source}")
    module_name = f"_pyspatialml_trace_{source.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise PipelineCliError(f"Unable to import source file: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source.parent.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(source.parent.resolve()))
        except ValueError:
            pass
    return module


def _parse_trace_input(value: str) -> tuple[str, np.ndarray]:
    if "=" not in value:
        raise PipelineCliError("Trace input must use name=path format")
    name, raw_path = value.split("=", 1)
    if not name:
        raise PipelineCliError("Trace input name cannot be empty")
    path = Path(raw_path)
    if not path.is_file():
        raise PipelineCliError(f"Trace input file not found: {path}")
    return name, _load_input_array(path)


def _load_input_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise PipelineCliError(f"NPZ input has no arrays: {path}")
        return data[data.files[0]]
    if suffix in {".png", ".jpg", ".jpeg"}:
        try:
            import cv2
        except ImportError as exc:
            raise PipelineCliError("Image trace inputs require opencv-python") from exc
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PipelineCliError(f"Failed to read image input: {path}")
        return image
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return np.asarray(payload)
    if suffix == ".bin":
        return np.fromfile(path, dtype=np.float32)
    raise PipelineCliError(f"Unsupported trace input file type: {path.suffix}")


def copy_pipeline(source: Path, destination: Path) -> None:
    """Copy a pipeline JSON file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
