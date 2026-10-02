import json
from pathlib import Path

import pytest

from pyspatialml import pipeline_cli
from securemr.operator_contracts import (
    OPERATOR_ARITIES,
    SPATIAL_ONLY_OPERATORS,
    XR_ONLY_OPERATORS,
    canonical_operator_name,
)
from securemr.py2smr.verifier import validate_pipeline_spec


EXAMPLES = Path(__file__).parents[1] / "examples"
XR_FIXTURE = EXAMPLES / "all_xr_operators_pipeline.json"
SPATIAL_FIXTURE = EXAMPLES / "all_spatial_operators_pipeline.json"
def _operator_names(path: Path) -> list[str]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    return [
        canonical_operator_name(operator["type"])
        for operator in spec["operators"]
    ]


def test_all_xr_operators_pipeline_parses_and_validates():
    spec = json.loads(XR_FIXTURE.read_text(encoding="utf-8"))

    validate_pipeline_spec(spec)
    assert pipeline_cli.validate_pipeline(XR_FIXTURE) == 0


def test_all_xr_operators_pipeline_covers_every_supported_xr_operator_once():
    actual = _operator_names(XR_FIXTURE)
    expected = sorted(set(OPERATOR_ARITIES) - set(SPATIAL_ONLY_OPERATORS))

    assert len(actual) == len(set(actual))
    assert sorted(actual) == expected


def test_all_spatial_operators_pipeline_parses_and_validates():
    spec = json.loads(SPATIAL_FIXTURE.read_text(encoding="utf-8"))

    validate_pipeline_spec(spec)
    assert pipeline_cli.validate_pipeline(SPATIAL_FIXTURE) == 0


def test_all_spatial_operators_pipeline_covers_every_supported_spatial_operator_once():
    actual = _operator_names(SPATIAL_FIXTURE)
    expected = sorted(set(OPERATOR_ARITIES) - set(XR_ONLY_OPERATORS))

    assert len(actual) == len(set(actual))
    assert sorted(actual) == expected


@pytest.mark.parametrize("fixture", [XR_FIXTURE, SPATIAL_FIXTURE], ids=["xr", "spatial"])
@pytest.mark.parametrize("operator_name", ["ALL", "ANY"])
def test_all_operator_pipeline_logical_reduction_outputs(fixture, operator_name):
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    operator = next(
        operator for operator in spec["operators"]
        if canonical_operator_name(operator["type"]) == operator_name
    )
    assert len(operator["outputs"]) == 1
    result = spec["tensors"][operator["outputs"][0]["tensor"]]

    assert result["dimensions"] == [1]
    assert result["channels"] == 1
    assert result["usage"] == 2
    assert result["data_type"] in {1, 2, 3, 4, 5}


@pytest.mark.parametrize("fixture", [XR_FIXTURE, SPATIAL_FIXTURE], ids=["xr", "spatial"])
def test_all_operator_pipeline_camera_image_order(fixture):
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    operators = {
        canonical_operator_name(operator["type"]): operator
        for operator in spec["operators"]
    }
    right_image, left_image = operators["RECTIFIED_VST_ACCESS"]["outputs"][:2]
    uv_inputs = operators["UV_TO_3D_IN_CAM_SPACE"]["inputs"]

    assert left_image["tensor"] != right_image["tensor"]
    assert uv_inputs[3]["tensor"] == left_image["tensor"]
    assert uv_inputs[4]["tensor"] == right_image["tensor"]


@pytest.mark.parametrize(
    "operator_name",
    sorted(set(OPERATOR_ARITIES) - set(XR_ONLY_OPERATORS) - set(SPATIAL_ONLY_OPERATORS)),
)
def test_all_operator_pipeline_common_operators_match(operator_name):
    xr_spec = json.loads(XR_FIXTURE.read_text(encoding="utf-8"))
    spatial_spec = json.loads(SPATIAL_FIXTURE.read_text(encoding="utf-8"))
    operators = [
        next(
            operator for operator in spec["operators"]
            if canonical_operator_name(operator["type"]) == operator_name
        )
        for spec in (xr_spec, spatial_spec)
    ]
    if operator_name == "JS_SCRIPTING":
        operators = [dict(operator, type="JS_SCRIPTING") for operator in operators]

    assert operators[0] == operators[1]
    for field in ("inputs", "outputs"):
        for ref in operators[0][field]:
            if ref is not None:
                tensor = ref["tensor"]
                assert xr_spec["tensors"][tensor] == spatial_spec["tensors"][tensor], tensor


@pytest.mark.parametrize("fixture", [XR_FIXTURE, SPATIAL_FIXTURE], ids=["xr", "spatial"])
def test_all_operator_pipeline_shared_output_contracts(fixture):
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    operators = {
        canonical_operator_name(operator["type"]): operator
        for operator in spec["operators"]
    }
    argmax = spec["tensors"][operators["ARGMAX"]["outputs"][0]["tensor"]]
    assert argmax["dimensions"] == [1]
    assert argmax["channels"] == 2
    assert argmax["data_type"] in {1, 2, 3, 4, 5}
    svd_w = spec["tensors"][operators["SVD"]["outputs"][0]["tensor"]]
    assert svd_w["dimensions"] == [2, 1]
    assert svd_w["channels"] == 1
    assert svd_w["data_type"] in {6, 7}
    assert svd_w["usage"] == 6
    assert operators["MICROPHONE"]["outputs"] == [
        {"tensor": "audio_stereo"}, {"tensor": "timestamp"},
    ]


def test_all_operator_pipeline_fixtures_do_not_contain_null_slots():
    for fixture in (XR_FIXTURE, SPATIAL_FIXTURE):
        spec = json.loads(fixture.read_text(encoding="utf-8"))
        assert all(
            ref is not None
            for operator in spec["operators"]
            for field in ("inputs", "outputs")
            for ref in operator[field]
        )
