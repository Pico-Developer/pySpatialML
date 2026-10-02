import pytest

from pyspatialml import package_cli, pipeline_cli
from securemr.operator_contracts import (
    DYNAMIC_INPUT_OPERATORS,
    DYNAMIC_OUTPUT_OPERATORS,
    OPERATOR_ARITIES,
    OPERATOR_INPUTS,
    OPERATOR_OUTPUTS,
    OperatorInputError,
    order_named_inputs,
    order_named_outputs,
)
from securemr.py2smr import verifier


def test_every_add_op_operator_has_fixed_or_dynamic_input_metadata():
    expected = set(OPERATOR_ARITIES)
    assert expected == set(OPERATOR_INPUTS) | set(DYNAMIC_INPUT_OPERATORS)
    assert expected == set(OPERATOR_OUTPUTS) | set(DYNAMIC_OUTPUT_OPERATORS)
    assert pipeline_cli.OPERATOR_ARITIES is OPERATOR_ARITIES
    assert package_cli.operator_contract("SOLVE_P_N_P") is OPERATOR_ARITIES["SOLVE_P_N_P"]
    assert verifier.operator_contract("GET_TRANSFORM_MAT") is OPERATOR_ARITIES["MAKE_TRANSFORM_MAT"]


@pytest.mark.parametrize("op_name,operands", OPERATOR_INPUTS.items())
def test_fixed_named_inputs_are_always_serialized_in_slot_order(op_name, operands):
    selected = []
    used_slots = set()
    for operand in reversed(operands):
        if operand.slot not in used_slots:
            selected.append((operand.switch[2:], f"tensor_{operand.slot}"))
            used_slots.add(operand.slot)

    inputs = order_named_inputs(op_name, selected)

    assert inputs == [
        {"tensor": f"tensor_{slot}"}
        for slot in range(len(used_slots))
    ]


def test_input_switches_replace_spaces_with_dashes():
    inputs = order_named_inputs(
        "XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO",
        [
            ("right-image", "right"),
            ("camera-intrinsic", "camera"),
            ("uv", "uvs"),
            ("left-image", "left"),
            ("timestamp", "time"),
        ],
    )

    assert inputs == [
        {"tensor": "uvs"},
        {"tensor": "time"},
        {"tensor": "camera"},
        {"tensor": "left"},
        {"tensor": "right"},
    ]
    assert order_named_inputs("NORMALIZE", [("alpha_beta", "range"), ("source", "image")]) == [
        {"tensor": "image"},
        {"tensor": "range"},
    ]


def test_conditional_update_gltf_operands_share_their_canonical_slots():
    assert order_named_inputs(
        "UPDATE_GLTF",
        [("rgb-image", "pixels"), ("texture-id", "texture"), ("gltf", "scene")],
    ) == [
        {"tensor": "scene"},
        {"tensor": "texture"},
        {"tensor": "pixels"},
    ]

    with pytest.raises(OperatorInputError, match="already set"):
        order_named_inputs("UPDATE_GLTF", [("node-id", "node"), ("texture-id", "texture")])


def test_dynamic_inputs_keep_names_in_json_refs():
    assert order_named_inputs(
        "RUN_MODEL_INFERENCE",
        [("serving_default_image", "image")],
        ["anchors=anchors_tensor"],
    ) == [
        {"name": "serving_default_image", "tensor": "image"},
        {"name": "anchors", "tensor": "anchors_tensor"},
    ]


def test_fixed_operator_rejects_another_operators_input_switch():
    with pytest.raises(OperatorInputError, match="not an input operand"):
        order_named_inputs("GET_AFFINE", [("scores", "scores_tensor")])


def test_solve_pnp_outputs_are_serialized_in_openmr_order():
    assert order_named_outputs(
        "SOLVE_P_N_P", [("translation", "tvec"), ("rotation", "rvec")]
    ) == [{"tensor": "rvec"}, {"tensor": "tvec"}]


def test_nms_uses_disambiguated_result_switches():
    assert order_named_outputs(
        "NMS", [("indices", "keep"), ("result-scores", "out_scores")]
    ) == [{"tensor": "out_scores"}, None, {"tensor": "keep"}]


def test_microphone_uses_spatial_sdk_v2_output_order():
    assert order_named_outputs(
        "MICROPHONE", [("timestamp", "time"), ("stereo-audio", "audio")]
    ) == [{"tensor": "audio"}, {"tensor": "time"}]


def test_microphone_rejects_outputs_not_representable_in_schema_v2():
    with pytest.raises(OperatorInputError, match="not a result"):
        order_named_outputs("MICROPHONE", [("left-audio", "left")])
