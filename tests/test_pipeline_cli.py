import json

import numpy as np
import pytest

from pyspatialml import pipeline_cli


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _add_render_text_tensors(path):
    existing = _read_json(path)["tensors"]
    specs = {
        "text_data": ("1", "uint8", "scalar", 1),
        "text_origin": ("1", "float32", "point", 2),
        "text_color": ("2", "uint8", "color", 4),
        "gltf": ("1,1", "uint8", "gltf", None),
        "texture_ids": ("1", "uint16", "scalar", 1),
        "text_bounds": ("1", "float32", "scalar", 1),
    }
    for name, (shape, dtype, usage, channels) in specs.items():
        if name not in existing:
            pipeline_cli.add_tensor(
                path, name, shape=shape, dtype=dtype, usage=usage, channels=channels
            )


def _render_text_inputs():
    return ["text_data", "text_origin", "text_color", "gltf", "texture_ids", "text_bounds"]


def test_init_pipeline_refuses_existing_file_without_force(tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert pipeline_cli.init_pipeline(pipeline) == 0
    with pytest.raises(pipeline_cli.PipelineCliError, match="already exists"):
        pipeline_cli.init_pipeline(pipeline)

    assert pipeline_cli.init_pipeline(pipeline, force=True) == 0
    assert _read_json(pipeline) == {"tensors": {}, "operators": [], "inputs": [], "outputs": []}


def test_add_tensor_writes_matrix_descriptor_and_boundaries(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline,
        "image",
        shape="128,64,3",
        dtype="uint8",
        usage="matrix",
        is_input=True,
        is_output=True,
    )

    spec = _read_json(pipeline)
    tensor = spec["tensors"]["image"]
    assert tensor["dimensions"] == [128, 64]
    assert tensor["channels"] == 3
    assert tensor["data_type"] == 1
    assert tensor["usage"] == 6
    assert tensor["is_placeholder"] is True
    assert "flag" in tensor
    assert spec["inputs"] == ["image"]
    assert spec["outputs"] == ["image"]


def test_add_tensor_marks_gltf_usage_for_sdk_loader(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline,
        "scene",
        shape="1,1",
        dtype="uint8",
        usage="gltf",
        asset="gltf/frame.gltf",
    )

    tensor = _read_json(pipeline)["tensors"]["scene"]
    assert tensor["usage"] == 7
    assert tensor["is_gltf"] is True
    assert tensor["is_placeholder"] is True
    assert tensor["asset"] == "gltf/frame.gltf"


def test_add_tensor_supports_scalar_values(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline,
        "threshold",
        shape="1",
        dtype="float32",
        usage="scalar",
        value="0.5",
    )

    tensor = _read_json(pipeline)["tensors"]["threshold"]
    assert tensor["dimensions"] == [1]
    assert tensor["channels"] == 1
    assert tensor["usage"] == 2
    assert tensor["data"] == [0.5]
    assert "flag" not in tensor


def test_add_tensor_supports_explicit_timestamp_channels(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline, "timestamp", shape="1", dtype="int32",
        usage="timestamp", channels=4,
    )

    tensor = _read_json(pipeline)["tensors"]["timestamp"]
    assert tensor["dimensions"] == [1]
    assert tensor["channels"] == 4
    assert tensor["data_type"] == 5
    assert tensor["usage"] == 5


def test_add_op_supports_affine_refs_and_assignment_slices(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    for name in ("src", "dst", "affine"):
        pipeline_cli.add_tensor(pipeline, name, shape="3,2", dtype="float32")
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO",
        inputs=["src", "dst"],
        outputs=["affine"],
    )
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
        inputs=["src"],
        outputs=["dst"],
        src_slices="[[0, 2, 1], [0, 2, 1]]",
        dst_slices="[[1, 3, 1], [0, 2, 1]]",
    )
    operators = _read_json(pipeline)["operators"]
    assert operators[0]["inputs"] == [{"tensor": "src"}, {"tensor": "dst"}]
    assert operators[1]["src_slices"] == [[0, 2, 1], [0, 2, 1]]


def test_add_op_type_convert_uses_assignment_operation_identity(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "input", shape="2,2", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "output", shape="2,2", dtype="int32")

    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", inputs=["input"], outputs=["output"])

    operator = _read_json(pipeline)["operators"][0]
    assert operator == {
        "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
        "inputs": [{"tensor": "input"}],
        "outputs": [{"tensor": "output"}],
    }


def test_add_tensor_rejects_duplicate_and_invalid_dtype(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="already exists"):
        pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unsupported dtype"):
        pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="bad_dtype")


def test_add_op_writes_common_operator_fields(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
        inputs=["x"],
        outputs=["y"],
        expression="{0} + 1.0",
        attrs=["unused"],
    )

    op = _read_json(pipeline)["operators"][0]
    assert op["type"] == "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO"
    assert op["inputs"] == [{"tensor": "x"}]
    assert op["outputs"] == [{"tensor": "y"}]
    assert op["attrs"] == ["{0} + 1.0"]
    assert "expression" not in op


def test_add_op_normalize_accepts_optional_alpha_beta_slot(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)
    pipeline_cli.add_tensor(pipeline, "alpha_beta", shape="2", dtype="float32")

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO",
        inputs=["x"],
        outputs=["y"],
    )
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO",
        inputs=["x", "alpha_beta"],
        outputs=["y"],
    )

    operators = _read_json(pipeline)["operators"]
    assert operators[0]["inputs"] == [{"tensor": "x"}]
    assert operators[1]["inputs"] == [{"tensor": "x"}, {"tensor": "alpha_beta"}]


def test_add_op_rejects_middle_gap_not_representable_in_schema_v2(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "rotation", shape="1,3", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "scale", shape="1,3", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "result", shape="4,4", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="cannot represent an omitted operator slot"):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_MAKE_TRANSFORM_MAT_PICO",
            inputs=["rotation", None, "scale"],
            outputs=["result"],
        )


def test_add_op_microphone_writes_spatialsdk_v2_outputs(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "stereo", shape="128,2", dtype="float32")
    pipeline_cli.add_tensor(
        pipeline, "timestamp", shape="1", dtype="int32", usage="timestamp", channels=4
    )

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO",
        inputs=[],
        outputs=["stereo", "timestamp"],
        attrs=["48000;PCM_FLOAT"],
    )

    assert _read_json(pipeline)["operators"][0]["outputs"] == [
        {"tensor": "stereo"},
        {"tensor": "timestamp"},
    ]


def test_add_op_arithmetic_requires_expression(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Arithmetic operators require --expression"):
        pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO", inputs=["x"], outputs=["y"])


@pytest.mark.parametrize(
    ("op_type", "inputs", "outputs", "message"),
    [
        ("XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", ["x"], ["y"], r"CONVERT_COLOR operators require flag or attrs\[0\]"),
        ("XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO", ["x", "y"], ["y"], r"customized_compare operators require comparison or attrs\[0\]"),
        ("XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO", ["x"], ["y"], r"JS_SCRIPTING operators require script or attrs\[0\]"),
        ("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO", ["gltf", "x", "x", "gltf", "x", "x"], [], r"render_text operators require config or attrs\[0\]"),
        ("XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO", ["gltf"], [], r"update_gltf operators require update_type or attrs\[0\]"),
        ("XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO", ["x"], ["y"], "RUN_MODEL_INFERENCE operators require --model"),
    ],
)
def test_add_op_requires_operator_metadata(tmp_path, op_type, inputs, outputs, message):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")

    with pytest.raises(pipeline_cli.PipelineCliError, match=message):
        pipeline_cli.add_op(pipeline, op_type, inputs=inputs, outputs=outputs)


@pytest.mark.parametrize(
    "attrs",
    [["8000;PCM_16BIT"], ["96000;PCM_32BIT"], ["48000;PCM_FLOAT"]],
)
def test_add_op_accepts_valid_microphone_attrs(tmp_path, attrs):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    encoding = attrs[0].split(";", 1)[1]
    dtype = {"PCM_16BIT": "int16", "PCM_32BIT": "int32", "PCM_FLOAT": "float32"}[encoding]
    pipeline_cli.add_tensor(pipeline, "stereo", shape="128,2", dtype=dtype)
    pipeline_cli.add_tensor(pipeline, "left", shape="128,1", dtype=dtype)
    pipeline_cli.add_tensor(pipeline, "right", shape="128,1", dtype=dtype)
    pipeline_cli.add_tensor(
        pipeline, "timestamp", shape="1", dtype="int32", usage="timestamp", channels=4
    )

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO",
        inputs=[],
        outputs=["stereo", "left", "right", "timestamp"],
        attrs=attrs,
    )

    assert _read_json(pipeline)["operators"][0]["attrs"] == attrs


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
def test_add_op_rejects_invalid_microphone_attrs(tmp_path, attrs):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "stereo", shape="128,2", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "left", shape="128,1", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "right", shape="128,1", dtype="float32")
    pipeline_cli.add_tensor(
        pipeline, "timestamp", shape="1", dtype="int32", usage="timestamp", channels=4
    )

    with pytest.raises(pipeline_cli.PipelineCliError):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO",
            inputs=[],
            outputs=["stereo", "left", "right", "timestamp"],
            attrs=attrs,
        )


@pytest.mark.parametrize(
    ("op_type", "inputs", "outputs", "message"),
    [
        ("XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", [], ["y"], "convert_color operators require exactly 1 input"),
        ("XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO", ["x"], ["y"], "elementwise_min operators require exactly 2 input"),
        ("XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO", ["x"], ["y"], "nms operators require exactly 2 input"),
        ("XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO", ["x", "y"], ["y", "z"], "solve_p_n_p operators require exactly 3 input"),
        ("XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO", ["x"], ["y"], "rectified_vst_access operators require exactly 0 input"),
    ],
)
def test_add_op_rejects_bad_operator_arity(tmp_path, op_type, inputs, outputs, message):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    for name in {"x", "y", "z", *inputs, *outputs}:
        pipeline_cli.add_tensor(pipeline, name, shape="2,2", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match=message):
        pipeline_cli.add_op(pipeline, op_type, inputs=inputs, outputs=outputs)


@pytest.mark.parametrize("outputs", [["w"], ["w", "u"], ["w", "u", "vt"]])
def test_add_op_accepts_one_to_three_svd_outputs(tmp_path, outputs):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32")
    for name in {"w", "u", "vt"}:
        pipeline_cli.add_tensor(pipeline, name, shape="2,2", dtype="float32")

    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO", inputs=["x"], outputs=outputs)

    spec = json.loads(pipeline.read_text(encoding="utf-8"))
    assert spec["operators"][-1]["type"].endswith("_SVD_PICO")
    expected_outputs = [{"tensor": name} for name in outputs]
    assert spec["operators"][-1]["outputs"] == expected_outputs


@pytest.mark.parametrize("outputs", [["scores"], ["scores", "boxes"], ["scores", "boxes", "indices"]])
def test_add_op_accepts_one_to_three_nms_outputs(tmp_path, outputs):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "scores", shape="3", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "boxes", shape="3,4", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "filtered_scores", shape="3", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "filtered_boxes", shape="3,4", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "indices", shape="3", dtype="int32")

    output_names = {"scores": "filtered_scores", "boxes": "filtered_boxes", "indices": "indices"}
    pipeline_cli.add_op(
        pipeline, "XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO",
        inputs=["scores", "boxes"],
        outputs=[output_names[name] for name in outputs], threshold=0.5,
    )

    spec = json.loads(pipeline.read_text(encoding="utf-8"))
    assert spec["operators"][-1]["type"].endswith("_NMS_PICO")
    expected_outputs = [{"tensor": output_names[name]} for name in outputs]
    assert spec["operators"][-1]["outputs"] == expected_outputs


def test_add_op_accepts_required_operator_metadata(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model")
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)
    _add_render_text_tensors(pipeline)
    pipeline_cli.add_tensor(pipeline, "texture", shape="2,2,3", dtype="uint8")

    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", inputs=["x"], outputs=["y"], flag="4")
    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO", inputs=["x", "y"], outputs=["y"], attrs=[">="])
    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO", inputs=["x"], outputs=["y"], attrs=["out = in;"])
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )

    operators = _read_json(pipeline)["operators"]
    assert operators[0]["attrs"] == ["4"]
    assert operators[1]["attrs"] == [">="]
    assert operators[2]["attrs"] == ["out = in;"]
    assert operators[2]["type"] == "XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO"
    assert operators[3]["attrs"] == ["bold#en-us#512#64"]
    assert "flag" not in operators[0]
    assert "script" not in operators[2]
    assert "config" not in operators[3]
    assert "text" not in operators[3]
    assert operators[3]["inputs"] == [{"tensor": name} for name in _render_text_inputs()]


@pytest.mark.parametrize(
    ("op_type", "inputs", "outputs", "attrs"),
    [
        ("XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO", ["scene", "data"], [], []),
        ("XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO", ["scene", "data"], [], []),
        ("XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO", ["input"], ["output"], ["output = input;"]),
    ],
)
def test_add_op_accepts_schema_operator_aliases(tmp_path, op_type, inputs, outputs, attrs):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    for name in {"scene", "data", "input", "output"}:
        pipeline_cli.add_tensor(
            pipeline,
            name,
            shape="1,1",
            dtype="uint8" if name == "scene" else "float32",
            usage="gltf" if name == "scene" else "matrix",
        )

    kwargs = {"attrs": attrs}
    if op_type.endswith("UPDATE_COMPONENT_PICO"):
        kwargs.update(entity_path="/target", property="Transform.Scale")
    pipeline_cli.add_op(pipeline, op_type, inputs=inputs, outputs=outputs, **kwargs)

    operator = _read_json(pipeline)["operators"][0]
    assert operator["type"] == op_type
    expected_attrs = ["/target:Transform.Scale"] if op_type.endswith("UPDATE_COMPONENT_PICO") else attrs
    if expected_attrs:
        assert operator["attrs"] == expected_attrs
    else:
        assert "attrs" not in operator


def test_add_op_rejects_unknown_tensor_reference(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown tensor"):
        pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", inputs=["x"], outputs=["missing"])


def test_remove_tensor_removes_tensor_and_boundaries(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True, is_output=True)

    assert pipeline_cli.remove_tensor(pipeline, "x") == 0

    spec = _read_json(pipeline)
    assert "x" not in spec["tensors"]
    assert spec["inputs"] == []
    assert spec["outputs"] == []


def test_remove_tensor_rejects_operator_references_without_force(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32", is_output=True)
    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", inputs=["x"], outputs=["y"])

    with pytest.raises(pipeline_cli.PipelineCliError) as exc_info:
        pipeline_cli.remove_tensor(pipeline, "x")

    message = str(exc_info.value)
    assert "Tensor 'x' is referenced by operator(s): #0 XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO" in message
    assert "--force" in message
    assert "x" in _read_json(pipeline)["tensors"]


def test_remove_tensor_force_allows_dangling_operator_reference(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32", is_output=True)
    pipeline_cli.add_op(pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", inputs=["x"], outputs=["y"])

    assert pipeline_cli.remove_tensor(pipeline, "x", force=True) == 0

    spec = _read_json(pipeline)
    assert "x" not in spec["tensors"]
    assert spec["inputs"] == []
    assert spec["operators"][0]["inputs"] == [{"tensor": "x"}]


def test_remove_tensor_rejects_missing_tensor(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Tensor not found"):
        pipeline_cli.remove_tensor(pipeline, "missing")


def test_add_op_rejects_xr_only_operator_when_manifest_supports_spatial(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    _add_render_text_tensors(pipeline)

    with pytest.raises(pipeline_cli.PipelineCliError, match="XR-only operator.*only includes spatial"):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
            inputs=_render_text_inputs(),
            outputs=[],
            attrs=["bold#en-us#512#64"],
        )


def test_add_op_rejects_spatial_only_operator_when_manifest_supports_xr(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "scene.json"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "data", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Spatial-only operator.*only includes xr"):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
            inputs=["component", "data"],
            outputs=[],
            entity_path="/target",
            property="Transform.Scale",
        )


def test_add_op_with_xr_only_operator_narrows_both_mode_manifest(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    _add_render_text_tensors(pipeline)

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]


def test_add_op_with_spatial_only_operator_narrows_both_mode_manifest(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "scene.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "data", shape="1,1", dtype="float32")

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
        inputs=["component", "data"],
        outputs=[],
        entity_path="/target",
        property="Transform.Scale",
    )

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["spatial"]


def test_add_op_spatial_only_aliases_write_sdk_fields(tmp_path):
    pipeline = tmp_path / "scene.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "scene", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "scale", shape="1,3", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "visible", shape="1", dtype="int32", usage="scalar")

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
        inputs=["scene", "visible"],
        outputs=[],
    )
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
        inputs=["scene", "scale"],
        outputs=[],
        entity_path="/target",
        property="Transform.Scale",
    )

    operators = _read_json(pipeline)["operators"]
    assert operators[0] == {
        "type": "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
        "inputs": [{"tensor": "scene"}, {"tensor": "visible"}],
        "outputs": [],
    }
    assert operators[1] == {
        "type": "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
        "inputs": [{"tensor": "scene"}, {"tensor": "scale"}],
        "outputs": [],
        "attrs": ["/target:Transform.Scale"],
    }


def test_remove_op_widens_manifest_when_no_exclusive_operators_remain(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    _add_render_text_tensors(pipeline)
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )
    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]

    pipeline_cli.remove_op(pipeline, 0)

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert _read_json(pipeline)["operators"] == []


def test_add_op_keeps_manifest_narrowed_by_mode_specific_sibling(tmp_path):
    package = tmp_path / "pkg"
    neutral_pipeline = package / "pipeline" / "neutral.json"
    display_pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [
                {"id": "neutral", "path": "pipeline/neutral.json"},
                {"id": "display", "path": "pipeline/display.json"},
            ],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(neutral_pipeline)
    pipeline_cli.add_tensor(neutral_pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_tensor(neutral_pipeline, "y", shape="1,1", dtype="float32")
    pipeline_cli.init_pipeline(display_pipeline)
    _add_render_text_tensors(display_pipeline)
    pipeline_cli.add_op(
        display_pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )

    pipeline_cli.add_op(neutral_pipeline, "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO", inputs=["x"], outputs=["y"])

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]


def test_remove_op_validation_failure_leaves_pipeline_and_manifest_unchanged(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    _add_render_text_tensors(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )
    original_manifest = manifest.read_text(encoding="utf-8")
    spec = _read_json(pipeline)
    spec["tensors"]["x"]["dimensions"] = [1]
    _write_json(pipeline, spec)
    original_pipeline = pipeline.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="matrix tensors must have at least 2 dimensions"):
        pipeline_cli.remove_op(pipeline, 0)

    assert manifest.read_text(encoding="utf-8") == original_manifest
    assert pipeline.read_text(encoding="utf-8") == original_pipeline


def test_remove_op_rejects_out_of_range_index(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Operator index out of range"):
        pipeline_cli.remove_op(pipeline, 0)


def test_add_op_rejects_mixing_xr_and_spatial_only_operators_without_manifest(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    _add_render_text_tensors(pipeline)
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "data", shape="1,1", dtype="float32")
    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
        inputs=_render_text_inputs(),
        outputs=[],
        attrs=["bold#en-us#512#64"],
    )

    with pytest.raises(pipeline_cli.PipelineCliError, match="mix XR-only and Spatial-only"):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
            inputs=["component", "data"],
            outputs=[],
            entity_path="/target",
            property="Transform.Scale",
        )


def test_add_op_model_requires_tflite_and_writes_inline_metadata(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "input", shape="1,4", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "output", shape="1,2", dtype="float32", is_output=True)

    with pytest.raises(pipeline_cli.PipelineCliError, match=".tflite"):
        pipeline_cli.add_op(
            pipeline,
            "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
            inputs=["input"],
            outputs=["output"],
            model="model/demo.bin",
        )

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
        inputs=["input"],
        outputs=["output"],
        model="model/demo.tflite",
        model_name="demo",
        model_target="cpu",
        cpu_target_num_threads=4,
    )

    op = _read_json(pipeline)["operators"][0]
    assert op["model"]["bin_path"] == "model/demo.tflite"
    assert op["model"]["model_name"] == "demo"
    assert op["model"]["model_type"] == "tflite"
    assert op["model"]["model_target"] == "cpu"
    assert op["model"]["cpu_target_num_threads"] == 4
    assert op["model"]["input"] == [{"name": "input", "shape": [1, 4], "encoding_type": "FP32"}]
    assert op["model"]["output"] == [{"name": "output", "shape": [1, 2], "encoding_type": "FP32"}]
    assert "model_type" not in op
    assert "model_target" not in op
    assert "cpu_target_num_threads" not in op
    assert "model_file" not in op
    assert "model_asset" not in op
    assert "model_id" not in op


def test_add_op_model_preserves_model_io_aliases(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "image_tensor", shape="1,224,224,3", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "scores_tensor", shape="1,10", dtype="float32", is_output=True)

    pipeline_cli.add_op(
        pipeline,
        "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
        inputs=[{"tensor": "image_tensor", "name": "input_0"}],
        outputs=[{"tensor": "scores_tensor", "name": "output_0"}],
        model="model/demo.tflite",
    )

    model = _read_json(pipeline)["operators"][0]["model"]
    assert model["input"][0]["name"] == "input_0"
    assert model["output"][0]["name"] == "output_0"
    assert model["input"][0]["encoding_type"] == "FP32"
    assert model["output"][0]["encoding_type"] == "FP32"


def test_set_input_and_set_output_mark_placeholders(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32")

    pipeline_cli.set_input(pipeline, ["x"])
    pipeline_cli.set_output(pipeline, ["y"])

    spec = _read_json(pipeline)
    assert spec["inputs"] == ["x"]
    assert spec["outputs"] == ["y"]
    assert spec["tensors"]["x"]["is_placeholder"] is True
    assert spec["tensors"]["y"]["is_placeholder"] is True

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown tensor"):
        pipeline_cli.set_input(pipeline, ["missing"])


def test_validate_pipeline_reports_bad_references(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps({"tensors": {}, "operators": [], "inputs": ["missing"], "outputs": []}),
        encoding="utf-8",
    )

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown inputs tensor"):
        pipeline_cli.validate_pipeline(pipeline)


def test_inspect_pipeline_prints_summary(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)

    assert pipeline_cli.inspect_pipeline(pipeline) == 0

    captured = capsys.readouterr()
    assert "Tensors: 1" in captured.out
    assert "Operators: 0" in captured.out
    assert "Inputs: x" in captured.out


def test_trace_pipeline_writes_converted_spec(tmp_path):
    source = tmp_path / "source.py"
    sample = tmp_path / "sample.npy"
    output = tmp_path / "pipeline.json"
    source.write_text(
        "\n".join(
            [
                "from securemr.py2smr import trace, ops",
                "@trace(inputs=['x'], outputs=['y'])",
                "def build(x):",
                "    return ops.arithmetic(x, '{0} * 2.0', output_name='y')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert pipeline_cli.trace_pipeline(
        source,
        function_name="build",
        output=output,
        inputs=[f"x={sample}"],
    ) == 0

    spec = _read_json(output)
    assert spec["inputs"] == ["x"]
    assert spec["outputs"] == ["y"]
    assert spec["operators"][0]["attrs"] == ["{0} * 2.0"]


def test_trace_pipeline_requires_trace_decorator_and_input_files(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def build(x):\n    return x\n", encoding="utf-8")

    with pytest.raises(pipeline_cli.PipelineCliError, match="not traceable"):
        pipeline_cli.trace_pipeline(
            source,
            function_name="build",
            output=tmp_path / "pipeline.json",
            inputs=[],
        )

    with pytest.raises(pipeline_cli.PipelineCliError, match="file not found"):
        pipeline_cli.trace_pipeline(
            tmp_path / "missing.py",
            function_name="build",
            output=tmp_path / "pipeline.json",
            inputs=[],
        )
