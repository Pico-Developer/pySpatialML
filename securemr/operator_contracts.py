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

"""Canonical names, slots, and arity contracts for SpatialML operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


class OperatorInputError(ValueError):
    """Raised when named CLI operands do not match an operator signature."""


@dataclass(frozen=True)
class InputOperand:
    """One fixed operator input or output slot."""

    name: str
    slot: int
    cli_name: str | None = None

    @property
    def switch(self) -> str:
        return "--" + (self.cli_name or self.name).lower().replace(" ", "-")


@dataclass(frozen=True)
class OperatorArity:
    """Canonical serialized slot counts and required slot indices."""

    min_inputs: int
    max_inputs: Optional[int]
    min_outputs: int
    max_outputs: Optional[int]
    required_inputs: frozenset[int]
    required_outputs: frozenset[int]
    builder_min_inputs: Optional[int] = None
    builder_required_inputs: Optional[frozenset[int]] = None

    @property
    def nullable_inputs(self) -> frozenset[int]:
        return _nullable_slots(self.max_inputs, self.required_inputs)

    @property
    def nullable_outputs(self) -> frozenset[int]:
        return _nullable_slots(self.max_outputs, self.required_outputs)

    @property
    def builder_arity(self) -> tuple[int, Optional[int], int, Optional[int], set[int], set[int]]:
        required_inputs = self.required_inputs if self.builder_required_inputs is None else self.builder_required_inputs
        return (
            self.min_inputs if self.builder_min_inputs is None else self.builder_min_inputs,
            self.max_inputs,
            self.min_outputs,
            self.max_outputs,
            set(_nullable_slots(self.max_inputs, required_inputs)),
            set(self.nullable_outputs),
        )

    @property
    def serialized_arity(self) -> tuple[int, Optional[int], int, Optional[int], set[int], set[int]]:
        return (
            self.min_inputs, self.max_inputs, self.min_outputs, self.max_outputs,
            set(self.nullable_inputs), set(self.nullable_outputs),
        )

    @property
    def verifier_signature(self) -> tuple[int, int, int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            self.min_inputs, 64 if self.max_inputs is None else self.max_inputs,
            self.min_outputs, 64 if self.max_outputs is None else self.max_outputs,
            tuple(sorted(self.required_inputs)), tuple(sorted(self.required_outputs)),
        )


def _arity(
    input_slots: int,
    output_slots: int,
    *,
    min_inputs: Optional[int] = None,
    min_outputs: Optional[int] = None,
    required_inputs: Sequence[int] | None = None,
    required_outputs: Sequence[int] | None = None,
    builder_min_inputs: Optional[int] = None,
    builder_required_inputs: Sequence[int] | None = None,
) -> OperatorArity:
    return OperatorArity(
        input_slots if min_inputs is None else min_inputs, input_slots,
        output_slots if min_outputs is None else min_outputs, output_slots,
        frozenset(range(input_slots) if required_inputs is None else required_inputs),
        frozenset(range(output_slots) if required_outputs is None else required_outputs),
        builder_min_inputs,
        None if builder_required_inputs is None else frozenset(builder_required_inputs),
    )


def _dynamic_arity(min_inputs: int, min_outputs: int) -> OperatorArity:
    return OperatorArity(
        min_inputs, None, min_outputs, None,
        frozenset(), frozenset(),
    )


def _nullable_slots(maximum: Optional[int], required: frozenset[int]) -> frozenset[int]:
    if maximum is None:
        return frozenset()
    return frozenset(set(range(maximum)) - set(required))


OPERATOR_ALIASES: Mapping[str, str] = {
    "SCENEGRAPH_VISIBILITY": "SSMR_SWITCH_VISIBILITY",
    "UPDATE_COMPONENT": "SSMR_UPDATE_COMPONENT",
    "JAVASCRIPT": "JS_SCRIPTING",
}

XR_ONLY_OPERATORS = frozenset({
    "UPLOAD_TEXTURE_TO_GLTF",
    "RENDER_TEXT",
    "SWITCH_GLTF_RENDER_STATUS",
    "UPDATE_GLTF",
})
SPATIAL_ONLY_OPERATORS = frozenset({
    "SSMR_SWITCH_VISIBILITY",
    "SSMR_UPDATE_COMPONENT",
})

INTERNAL_OPERATOR_NAMES: Mapping[str, str] = {
    "CHW_HWC": "SWAP_HWC_CHW",
    "MAKE_TRANSFORM_MAT": "GET_TRANSFORM_MAT",
    "UPLOAD_TEXTURE_TO_GLTF": "LOAD_TEXTURE",
    "SSMR_SWITCH_VISIBILITY": "SCENEGRAPH_VISIBILITY",
    "SSMR_UPDATE_COMPONENT": "UPDATE_COMPONENT",
    "JS_SCRIPTING": "JAVASCRIPT",
}
INTERNAL_TO_CANONICAL: Mapping[str, str] = {
    internal: canonical for canonical, internal in INTERNAL_OPERATOR_NAMES.items()
}

OPERATOR_ARITIES: Mapping[str, OperatorArity] = {
    "ARITHMETIC_COMPOSE": _arity(
        10, 1, min_inputs=1, required_inputs=(0,), builder_min_inputs=1,
        builder_required_inputs=(0,)
    ),
    "ELEMENTWISE_MIN": _arity(2, 1),
    "ELEMENTWISE_MAX": _arity(2, 1),
    "ELEMENTWISE_MULTIPLY": _arity(2, 1),
    "CUSTOMIZED_COMPARE": _arity(2, 1),
    "ELEMENTWISE_OR": _arity(2, 1),
    "ELEMENTWISE_AND": _arity(2, 1),
    "ALL": _arity(1, 1),
    "ANY": _arity(1, 1),
    "NMS": _arity(2, 3, min_outputs=1, required_outputs=()),
    "SOLVE_P_N_P": _arity(3, 2, min_outputs=1, required_outputs=(0,)),
    "GET_AFFINE": _arity(2, 1),
    "APPLY_AFFINE": _arity(2, 1),
    "APPLY_AFFINE_POINT": _arity(2, 1),
    "UV_TO_3D_IN_CAM_SPACE": _arity(5, 1),
    "ASSIGNMENT": _arity(1, 1),
    "RUN_MODEL_INFERENCE": _dynamic_arity(1, 1),
    "NORMALIZE": _arity(2, 1, min_inputs=1, required_inputs=(0,), builder_min_inputs=1),
    "CAMERA_SPACE_TO_WORLD": _arity(1, 2, min_outputs=1, required_outputs=(0,)),
    "RECTIFIED_VST_ACCESS": _arity(0, 4),
    "ARGMAX": _arity(1, 1),
    "CONVERT_COLOR": _arity(1, 1),
    "SORT_VEC": _arity(1, 2, min_outputs=1, required_outputs=()),
    "INVERSION": _arity(1, 1),
    "MAKE_TRANSFORM_MAT": _arity(3, 1, min_inputs=2, required_inputs=(0, 1)),
    "SORT_MAT": _arity(1, 2, min_outputs=1, required_outputs=()),
    "SWITCH_GLTF_RENDER_STATUS": _arity(4, 0, required_inputs=(0,)),
    "UPDATE_GLTF": _arity(3, 0, required_inputs=(0,)),
    "RENDER_TEXT": _arity(6, 0),
    "UPLOAD_TEXTURE_TO_GLTF": _arity(2, 1),
    "SVD": _arity(1, 3, min_outputs=1, required_outputs=()),
    "NORM": _arity(1, 1),
    "CHW_HWC": _arity(1, 1),
    "SSMR_SWITCH_VISIBILITY": _arity(2, 0, required_inputs=(0,)),
    "SSMR_UPDATE_COMPONENT": _arity(2, 0),
    "JS_SCRIPTING": _dynamic_arity(0, 1),
    "MICROPHONE": _arity(0, 4, min_outputs=1, required_outputs=()),
    "SPEAKER": _arity(1, 0),
    "DEPTH": _arity(0, 1),
}


# Names come from the package specification when it defines them, otherwise
# from the names passed to xrSetSecureMrOperatorOperandByNamePICO in OpenMR.
# Tuple order is the canonical order written to the JSON inputs array.
OPERATOR_INPUTS: Mapping[str, tuple[InputOperand, ...]] = {
    "ARITHMETIC_COMPOSE": tuple(InputOperand(f"operand{index}", index) for index in range(10)),
    "ELEMENTWISE_MIN": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "ELEMENTWISE_MAX": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "ELEMENTWISE_MULTIPLY": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "CUSTOMIZED_COMPARE": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "ELEMENTWISE_OR": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "ELEMENTWISE_AND": (InputOperand("operand0", 0), InputOperand("operand1", 1)),
    "ALL": (InputOperand("operand", 0),),
    "ANY": (InputOperand("operand", 0),),
    "NMS": (InputOperand("scores", 0), InputOperand("boxes", 1)),
    "SOLVE_P_N_P": (
        InputOperand("object points", 0),
        InputOperand("image points", 1),
        InputOperand("camera matrix", 2),
    ),
    "GET_AFFINE": (InputOperand("src", 0), InputOperand("dst", 1)),
    "APPLY_AFFINE": (InputOperand("affine", 0), InputOperand("src image", 1)),
    "APPLY_AFFINE_POINT": (InputOperand("affine", 0), InputOperand("src points", 1)),
    "UV_TO_3D_IN_CAM_SPACE": (
        InputOperand("uv", 0),
        InputOperand("timestamp", 1),
        InputOperand("camera intrinsic", 2),
        InputOperand("left image", 3),
        InputOperand("right image", 4),
    ),
    "ASSIGNMENT": (InputOperand("src", 0),),
    "NORMALIZE": (InputOperand("source", 0), InputOperand("alpha_beta", 1)),
    "CAMERA_SPACE_TO_WORLD": (InputOperand("timestamp", 0),),
    "RECTIFIED_VST_ACCESS": (),
    "ARGMAX": (InputOperand("operand", 0),),
    "CONVERT_COLOR": (InputOperand("src", 0),),
    "SORT_VEC": (InputOperand("input", 0),),
    "INVERSION": (InputOperand("operand", 0),),
    "MAKE_TRANSFORM_MAT": (
        InputOperand("rotation", 0),
        InputOperand("translation", 1),
        InputOperand("scale", 2),
    ),
    "SORT_MAT": (InputOperand("input", 0),),
    "SWITCH_GLTF_RENDER_STATUS": (
        InputOperand("gltf", 0),
        InputOperand("world pose", 1),
        InputOperand("visible", 2),
        InputOperand("view locked", 3),
    ),
    # UPDATE_GLTF selects the semantic names for slots 1 and 2 through
    # attrs[0]. Multiple names intentionally target those same slots.
    "UPDATE_GLTF": (
        InputOperand("gltf", 0),
        InputOperand("node ID", 1),
        InputOperand("animation ID", 1),
        InputOperand("world pose", 1),
        InputOperand("material ID", 1),
        InputOperand("texture ID", 1),
        InputOperand("transform", 2),
        InputOperand("animation timer", 2),
        InputOperand("value", 2),
        InputOperand("rgb image", 2),
    ),
    "RENDER_TEXT": (
        InputOperand("text", 0),
        InputOperand("start", 1),
        InputOperand("colors", 2),
        InputOperand("gltf", 3),
        InputOperand("texture ID", 4),
        InputOperand("font size", 5),
    ),
    "UPLOAD_TEXTURE_TO_GLTF": (InputOperand("gltf", 0), InputOperand("rgb image", 1)),
    "SVD": (InputOperand("src", 0),),
    "NORM": (InputOperand("operand0", 0),),
    "CHW_HWC": (InputOperand("operand0", 0),),
    "SSMR_SWITCH_VISIBILITY": (InputOperand("scenegraph", 0), InputOperand("visible", 1)),
    "SSMR_UPDATE_COMPONENT": (InputOperand("scenegraph", 0), InputOperand("data", 1)),
    "MICROPHONE": (),
    "SPEAKER": (InputOperand("audio", 0),),
    "DEPTH": (),
}

DYNAMIC_INPUT_OPERATORS = frozenset({"RUN_MODEL_INFERENCE", "JS_SCRIPTING"})

# OpenMR result names in canonical JSON slot order. NMS result names collide
# with its operand names, so their CLI spelling is explicitly disambiguated.
OPERATOR_OUTPUTS: Mapping[str, tuple[InputOperand, ...]] = {
    "ARITHMETIC_COMPOSE": (InputOperand("result", 0),),
    "ELEMENTWISE_MIN": (InputOperand("result", 0),),
    "ELEMENTWISE_MAX": (InputOperand("result", 0),),
    "ELEMENTWISE_MULTIPLY": (InputOperand("result", 0),),
    "CUSTOMIZED_COMPARE": (InputOperand("result", 0),),
    "ELEMENTWISE_OR": (InputOperand("result", 0),),
    "ELEMENTWISE_AND": (InputOperand("result", 0),),
    "ALL": (InputOperand("result", 0),),
    "ANY": (InputOperand("result", 0),),
    "NMS": (
        InputOperand("scores", 0, "result-scores"),
        InputOperand("boxes", 1, "result-boxes"),
        InputOperand("indices", 2),
    ),
    "SOLVE_P_N_P": (InputOperand("rotation", 0), InputOperand("translation", 1)),
    "GET_AFFINE": (InputOperand("result", 0),),
    "APPLY_AFFINE": (InputOperand("dst image", 0),),
    "APPLY_AFFINE_POINT": (InputOperand("dst points", 0),),
    "UV_TO_3D_IN_CAM_SPACE": (InputOperand("point_xyz", 0),),
    "ASSIGNMENT": (InputOperand("dst", 0),),
    "NORMALIZE": (InputOperand("result", 0),),
    "CAMERA_SPACE_TO_WORLD": (InputOperand("right", 0), InputOperand("left", 1)),
    "RECTIFIED_VST_ACCESS": (
        InputOperand("right image", 0),
        InputOperand("left image", 1),
        InputOperand("timestamp", 2),
        InputOperand("camera matrix", 3),
    ),
    "ARGMAX": (InputOperand("result", 0),),
    "CONVERT_COLOR": (InputOperand("dst", 0),),
    "SORT_VEC": (InputOperand("sorted", 0), InputOperand("indices", 1)),
    "INVERSION": (InputOperand("result", 0),),
    "MAKE_TRANSFORM_MAT": (InputOperand("result", 0),),
    "SORT_MAT": (InputOperand("sorted", 0), InputOperand("indices", 1)),
    "SWITCH_GLTF_RENDER_STATUS": (),
    "UPDATE_GLTF": (),
    "RENDER_TEXT": (),
    "UPLOAD_TEXTURE_TO_GLTF": (InputOperand("texture ID", 0),),
    "SVD": (InputOperand("w", 0), InputOperand("u", 1), InputOperand("vt", 2)),
    "NORM": (InputOperand("result0", 0),),
    "CHW_HWC": (InputOperand("result0", 0),),
    "SSMR_SWITCH_VISIBILITY": (),
    "SSMR_UPDATE_COMPONENT": (),
    "MICROPHONE": (
        InputOperand("stereo audio", 0),
        InputOperand("timestamp", 1),
    ),
    "SPEAKER": (),
    "DEPTH": (InputOperand("depth map", 0),),
}

DYNAMIC_OUTPUT_OPERATORS = frozenset({"RUN_MODEL_INFERENCE", "JS_SCRIPTING"})

def canonical_operator_name(op_type: str) -> str:
    """Return the canonical enum token from a package operator type."""
    value = str(op_type).strip().upper()
    prefix = "XR_SECURE_MR_OPERATOR_TYPE_"
    suffix = "_PICO"
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.endswith(suffix):
        value = value[:-len(suffix)]
    return OPERATOR_ALIASES.get(value, value)


def package_operator_name(op_type: str) -> str:
    """Validate and return the canonical package-facing operator name."""
    value = str(op_type)
    prefix = "XR_SECURE_MR_OPERATOR_TYPE_"
    suffix = "_PICO"
    if not (value.startswith(prefix) and value.endswith(suffix)):
        return ""
    raw_name = value[len(prefix):-len(suffix)]
    if not raw_name or raw_name != raw_name.upper():
        return ""
    canonical = OPERATOR_ALIASES.get(raw_name, raw_name)
    return canonical if canonical in OPERATOR_ARITIES else ""


def internal_operator_name(op_type: str) -> str:
    """Return the legacy enum spelling used by the Python runtime."""
    canonical = canonical_operator_name(op_type)
    return INTERNAL_OPERATOR_NAMES.get(canonical, canonical)


def operator_contract(op_type: str) -> Optional[OperatorArity]:
    """Resolve an operator contract from canonical, alias, or internal spelling."""
    name = canonical_operator_name(op_type)
    name = INTERNAL_TO_CANONICAL.get(name, name)
    return OPERATOR_ARITIES.get(name)


def all_input_switches() -> tuple[str, ...]:
    """Return every fixed operand switch accepted by ``pipeline add-op``."""
    return tuple(sorted({operand.switch for operands in OPERATOR_INPUTS.values() for operand in operands}))


def all_output_switches() -> tuple[str, ...]:
    """Return every fixed result switch accepted by ``pipeline add-op``."""
    return tuple(sorted({result.switch for results in OPERATOR_OUTPUTS.values() for result in results}))


def input_switches(op_type: str) -> tuple[str, ...]:
    """Return fixed input switches in canonical slot order for one operator."""
    return tuple(operand.switch for operand in OPERATOR_INPUTS.get(canonical_operator_name(op_type), ()))


def output_switches(op_type: str) -> tuple[str, ...]:
    """Return fixed result switches in canonical slot order for one operator."""
    return tuple(result.switch for result in OPERATOR_OUTPUTS.get(canonical_operator_name(op_type), ()))


def order_named_inputs(
    op_type: str,
    values: Sequence[tuple[str, str]],
    dynamic_values: Sequence[str] = (),
) -> list[Any]:
    """Convert command-line named operands to the canonical JSON slot order."""
    op_name = canonical_operator_name(op_type)
    if op_name in DYNAMIC_INPUT_OPERATORS:
        bindings = list(values)
        for value in dynamic_values:
            if "=" not in value:
                raise OperatorInputError("--named-input must use NAME=TENSOR format")
            name, tensor = value.split("=", 1)
            if not name or not tensor:
                raise OperatorInputError("--named-input must use NAME=TENSOR format")
            bindings.append((name, tensor))
        _reject_duplicate_names(bindings)
        return [{"name": name, "tensor": tensor} for name, tensor in bindings]

    if dynamic_values:
        raise OperatorInputError(
            f"{op_name.lower()} has fixed operands; use: {_format_switches(input_switches(op_name))}"
        )

    operands = OPERATOR_INPUTS.get(op_name)
    if operands is None:
        if values:
            raise OperatorInputError(f"{op_name.lower()} does not accept named input operands")
        return []

    by_switch: dict[str, InputOperand] = {operand.switch[2:]: operand for operand in operands}
    slots: dict[int, str] = {}
    seen_switches: set[str] = set()
    for switch, tensor in values:
        operand = by_switch.get(switch)
        if operand is None:
            allowed = _format_switches(input_switches(op_name))
            raise OperatorInputError(
                f"--{switch} is not an input operand for {op_name}; expected {allowed}"
            )
        if switch in seen_switches:
            raise OperatorInputError(f"--{switch} may only be specified once")
        if operand.slot in slots:
            raise OperatorInputError(
                f"--{switch} selects input slot {operand.slot}, which is already set"
            )
        seen_switches.add(switch)
        slots[operand.slot] = tensor

    if not slots:
        return []
    return [({"tensor": slots[index]} if index in slots else None) for index in range(max(slots) + 1)]


def order_named_outputs(
    op_type: str,
    values: Sequence[tuple[str, str]],
    dynamic_values: Sequence[str] = (),
) -> list[Any]:
    """Convert command-line named results to the canonical JSON slot order."""
    op_name = canonical_operator_name(op_type)
    if op_name in DYNAMIC_OUTPUT_OPERATORS:
        bindings = list(values)
        for value in dynamic_values:
            if "=" not in value:
                raise OperatorInputError("--named-output must use NAME=TENSOR format")
            name, tensor = value.split("=", 1)
            if not name or not tensor:
                raise OperatorInputError("--named-output must use NAME=TENSOR format")
            bindings.append((name, tensor))
        _reject_duplicate_names(bindings, label="Output")
        return [{"name": name, "tensor": tensor} for name, tensor in bindings]

    if dynamic_values:
        raise OperatorInputError(
            f"{op_name.lower()} has fixed results; use: {_format_switches(output_switches(op_name))}"
        )
    if op_name == "MICROPHONE":
        _reject_duplicate_names(values, label="Output")
        by_switch = dict(values)
        unsupported = set(by_switch) - {"stereo-audio", "timestamp"}
        if unsupported:
            raise OperatorInputError(
                f"--{sorted(unsupported)[0]} is not a result for MICROPHONE; "
                "schema v2 supports --stereo-audio and --timestamp"
            )
        if "timestamp" in by_switch and "stereo-audio" not in by_switch:
            raise OperatorInputError(
                "Schema v2 cannot represent microphone --timestamp without --stereo-audio"
            )
        outputs = []
        if "stereo-audio" in by_switch:
            outputs.append({"tensor": by_switch["stereo-audio"]})
        if "timestamp" in by_switch:
            outputs.append({"tensor": by_switch["timestamp"]})
        return outputs
    return _order_fixed_refs(op_name, values, OPERATOR_OUTPUTS, "result", output_switches(op_name))


def split_named_bindings(
    op_type: str, values: Sequence[tuple[str, str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Partition parsed switches into input and output bindings for one operator."""
    op_name = canonical_operator_name(op_type)
    input_names = {switch[2:] for switch in input_switches(op_name)}
    output_names = {switch[2:] for switch in output_switches(op_name)}
    inputs: list[tuple[str, str]] = []
    outputs: list[tuple[str, str]] = []
    for name, tensor in values:
        if name in input_names:
            inputs.append((name, tensor))
        elif name in output_names:
            outputs.append((name, tensor))
        elif op_name in DYNAMIC_INPUT_OPERATORS:
            inputs.append((name, tensor))
        else:
            allowed = _format_switches([*input_switches(op_name), *output_switches(op_name)])
            raise OperatorInputError(f"--{name} is not an input or result for {op_name}; expected {allowed}")
    return inputs, outputs


def _order_fixed_refs(
    op_name: str,
    values: Sequence[tuple[str, str]],
    definitions: Mapping[str, tuple[InputOperand, ...]],
    label: str,
    expected_switches: Sequence[str],
) -> list[Any]:
    refs = definitions.get(op_name)
    if refs is None:
        if values:
            raise OperatorInputError(f"{op_name.lower()} does not accept named {label}s")
        return []
    by_switch = {ref.switch[2:]: ref for ref in refs}
    slots: dict[int, str] = {}
    seen: set[str] = set()
    for switch, tensor in values:
        ref = by_switch.get(switch)
        if ref is None:
            raise OperatorInputError(
                f"--{switch} is not a {label} for {op_name}; expected {_format_switches(expected_switches)}"
            )
        if switch in seen:
            raise OperatorInputError(f"--{switch} may only be specified once")
        if ref.slot in slots:
            raise OperatorInputError(f"--{switch} selects {label} slot {ref.slot}, which is already set")
        seen.add(switch)
        slots[ref.slot] = tensor
    if not slots:
        return []
    return [({"tensor": slots[index]} if index in slots else None) for index in range(max(slots) + 1)]


def _reject_duplicate_names(values: Sequence[tuple[str, str]], *, label: str = "Input") -> None:
    seen: set[str] = set()
    for name, _ in values:
        if name in seen:
            raise OperatorInputError(f"{label} --{name} may only be specified once")
        seen.add(name)


def _format_switches(switches: Sequence[str]) -> str:
    return ", ".join(switches) if switches else "no input switches"
