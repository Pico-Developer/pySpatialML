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

"""Operator discovery commands."""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from securemr.core.types import EOperatorType
from securemr.py2smr import ops
from securemr.operator_contracts import (
    DYNAMIC_INPUT_OPERATORS,
    DYNAMIC_OUTPUT_OPERATORS,
    INTERNAL_OPERATOR_NAMES,
    input_switches,
    output_switches,
)


class OperatorCliError(RuntimeError):
    """Raised when operator discovery fails."""


@dataclass(frozen=True)
class OperatorInfo:
    """One discoverable SecureMR operator."""

    enum_name: str
    type_name: str
    creator: Optional[str]
    signature: Optional[str]
    summary: str
    supported: bool
    input_switches: tuple[str, ...]
    dynamic_inputs: bool
    output_switches: tuple[str, ...]
    dynamic_outputs: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "enum_name": self.enum_name,
            "type": self.type_name,
            "creator": self.creator,
            "signature": self.signature,
            "summary": self.summary,
            "supported": self.supported,
            "input_switches": list(self.input_switches),
            "dynamic_inputs": self.dynamic_inputs,
            "output_switches": list(self.output_switches),
            "dynamic_outputs": self.dynamic_outputs,
        }


_OPERATOR_CREATORS: Mapping[str, str] = {
    "UNKNOWN": "unknown",
    "ARITHMETIC_COMPOSE": "arithmetic",
    "ELEMENTWISE_MIN": "elementwise_min",
    "ELEMENTWISE_MAX": "elementwise_max",
    "ELEMENTWISE_MULTIPLY": "elementwise_multiply",
    "CUSTOMIZED_COMPARE": "customized_compare",
    "ELEMENTWISE_OR": "elementwise_or",
    "ELEMENTWISE_AND": "elementwise_and",
    "ALL": "all",
    "ANY": "any",
    "NMS": "nms",
    "SOLVE_P_N_P": "solve_p_n_p",
    "GET_AFFINE": "get_affine",
    "APPLY_AFFINE": "apply_affine",
    "APPLY_AFFINE_POINT": "apply_affine_point",
    "UV_TO_3D_IN_CAM_SPACE": "uv_to_3d_in_camera_space",
    "ASSIGNMENT": "assignment",
    "RUN_MODEL_INFERENCE": "run_algorithm",
    "NORMALIZE": "normalize",
    "CAMERA_SPACE_TO_WORLD": "camera_space_to_world",
    "RECTIFIED_VST_ACCESS": "camera_access",
    "ARGMAX": "argmax",
    "CONVERT_COLOR": "cvt_color",
    "SORT_VEC": "sort_vec",
    "INVERSION": "inversion",
    "GET_TRANSFORM_MAT": "get_transform_mat",
    "SORT_MAT": "sort_mat",
    "SWITCH_GLTF_RENDER_STATUS": "switch_gltf_render_status",
    "UPDATE_GLTF": "update_gltf",
    "RENDER_TEXT": "render_text",
    "LOAD_TEXTURE": "load_texture",
    "SVD": "svd",
    "NORM": "norm",
    "SWAP_HWC_CHW": "swap_hwc_chw",
    "SCENEGRAPH_VISIBILITY": "scenegraph_visibility",
    "UPDATE_COMPONENT": "update_component",
    "JAVASCRIPT": "javascript",
    "MICROPHONE": "microphone",
    "SPEAKER": "speaker",
    "DEPTH": "depth",
}
_CANONICAL_OPERATOR_NAMES: Mapping[str, str] = {
    internal: canonical for canonical, internal in INTERNAL_OPERATOR_NAMES.items()
}
_CANONICAL_CREATOR_ALIASES: Mapping[str, str] = {
    "CHW_HWC": "swap_hwc_chw",
    "MAKE_TRANSFORM_MAT": "get_transform_mat",
    "UPLOAD_TEXTURE_TO_GLTF": "load_texture",
    "SSMR_SWITCH_VISIBILITY": "scenegraph_visibility",
    "SSMR_UPDATE_COMPONENT": "update_component",
    "JS_SCRIPTING": "javascript",
}
_LEGACY_OPERATOR_NAMES = frozenset(_CANONICAL_OPERATOR_NAMES)

def list_operators(*, as_json: bool = False) -> int:
    """Print discoverable operators."""
    operators = discover_operators()
    if as_json:
        print(json.dumps([item.to_dict() for item in operators], indent=2))
        return 0
    print(f"Operators: {len(operators)}")
    for item in operators:
        marker = "yes" if item.supported else "no"
        creator = item.creator or "-"
        print(f"  {item.enum_name:<32} creator={creator:<28} supported={marker}")
    return 0


def describe_operator(name: str, *, as_json: bool = False) -> int:
    """Print details for one operator."""
    info = find_operator(name)
    if info is None:
        raise OperatorCliError(f"Unknown operator: {name}")
    if as_json:
        print(json.dumps(info.to_dict(), indent=2))
        return 0
    print(f"Operator: {info.enum_name}")
    print(f"Type: {info.type_name}")
    print(f"Creator: {info.creator or '-'}")
    print(f"Supported: {'yes' if info.supported else 'no'}")
    if info.dynamic_inputs:
        print("Input switches: dynamic (--NAME TENSOR or --named-input NAME=TENSOR)")
    else:
        print(f"Input switches: {', '.join(info.input_switches) or '-'}")
    if info.dynamic_outputs:
        print("Output switches: dynamic (--named-output NAME=TENSOR)")
    else:
        print(f"Output switches: {', '.join(info.output_switches) or '-'}")
    if info.signature:
        print(f"Signature: {info.signature}")
    if info.summary:
        print(f"Summary: {info.summary}")
    return 0


def discover_operators() -> list[OperatorInfo]:
    """Return all enum-backed operators with py2smr creator metadata."""
    result = []
    emitted_names = set()
    for enum_name in _enum_names():
        canonical_name = _CANONICAL_OPERATOR_NAMES.get(enum_name, enum_name)
        # EOperatorType is an internal/native enum and still contains the
        # historical spellings.  They are translated once at this boundary;
        # never expose those spellings as package-facing operator names.
        if canonical_name in emitted_names:
            continue
        emitted_names.add(canonical_name)
        creator = _CANONICAL_CREATOR_ALIASES.get(canonical_name, _OPERATOR_CREATORS.get(enum_name))
        fn = getattr(ops, creator, None) if creator else None
        signature = None
        summary = ""
        if fn is not None:
            try:
                signature = f"{creator}{inspect.signature(fn)}"
            except (TypeError, ValueError):
                signature = f"{creator}(...)"
            summary = _doc_summary(fn)
        result.append(
            OperatorInfo(
                enum_name=canonical_name,
                type_name=f"XR_SECURE_MR_OPERATOR_TYPE_{canonical_name}_PICO",
                creator=creator,
                signature=signature,
                summary=summary,
                supported=fn is not None,
                input_switches=input_switches(canonical_name),
                dynamic_inputs=canonical_name in DYNAMIC_INPUT_OPERATORS,
                output_switches=output_switches(canonical_name),
                dynamic_outputs=canonical_name in DYNAMIC_OUTPUT_OPERATORS,
            )
        )
    return result


def find_operator(name: str) -> Optional[OperatorInfo]:
    """Resolve operator by enum name, JSON type name, or creator name."""
    normalized = _normalize_name(name)
    # Do not canonicalize legacy input here.  That would make an old alias
    # appear supported even though strict schema packages must not contain it.
    if normalized in _LEGACY_OPERATOR_NAMES:
        return None
    for item in discover_operators():
        if normalized in {
            _normalize_name(item.enum_name),
            _normalize_name(item.type_name),
            _normalize_name(item.creator or ""),
        }:
            return item
    return None


def print_operator_error(exc: Exception) -> None:
    """Print a concise operator command error."""
    print(f"Error [PSM_OPERATOR]: {exc}", file=sys.stderr)


def _enum_names() -> list[str]:
    names = []
    try:
        iterator = iter(EOperatorType)
        for member in iterator:
            names.append(member.name)
    except TypeError:
        for attr in dir(EOperatorType):
            if attr.startswith("_") or not attr.isupper():
                continue
            try:
                int(getattr(EOperatorType, attr))
            except Exception:
                continue
            names.append(attr)
    return sorted(set(names), key=lambda item: int(getattr(EOperatorType, item)))


def _doc_summary(fn) -> str:
    doc = inspect.getdoc(fn) or ""
    return doc.splitlines()[0] if doc else ""


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().upper()
    if normalized.startswith("XR_SECURE_MR_OPERATOR_TYPE_"):
        normalized = normalized[len("XR_SECURE_MR_OPERATOR_TYPE_"):]
    if normalized.endswith("_PICO"):
        normalized = normalized[: -len("_PICO")]
    return normalized
