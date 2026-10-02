import json

import pytest

from pyspatialml.package_cli import validate_package
from securemr.pipeline_zoo import (
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    configure_litert_inference_operator,
    create_litert_model_spec,
    load_pipeline_zoo_manifest,
    validate_pipeline_zoo_manifest,
    write_pipeline_zoo_package,
)
from securemr.py2smr.verifier import validate_pipeline_spec


def test_create_litert_model_spec_defaults_to_pipeline_zoo_schema():
    model_spec = create_litert_model_spec(
        "model/face_detector.tflite",
        "face_detector",
        input_tensors=[{"name": "input", "shape": [1, 128, 128, 3], "encoding_type": "FP32"}],
        output_tensors=[{"name": "scores", "shape": [1, 896, 1], "encoding_type": "FP32"}],
    )

    assert model_spec["bin_path"] == "model/face_detector.tflite"
    assert model_spec["model_name"] == "face_detector"
    assert model_spec["model_type"] == "tflite"
    assert model_spec["model_target"] == "npu"
    assert "cpu_target_num_threads" not in model_spec
    assert "path_to_zoo" not in model_spec
    assert "specific_config" not in model_spec
    assert model_spec["input"][0]["name"] == "input"
    assert model_spec["output"][0]["name"] == "scores"


@pytest.mark.parametrize(
    "model_path",
    [
        "/tmp/model.tflite",
        "C:/tmp/model.tflite",
        r"C:\tmp\model.tflite",
        r"\\server\share\model.tflite",
    ],
)
def test_create_litert_model_spec_rejects_absolute_paths(model_path):
    with pytest.raises(ValueError, match="Package paths must be relative"):
        create_litert_model_spec(
            model_path, "model",
            input_tensors=[{"name": "input", "shape": [1], "encoding_type": "FP32"}],
            output_tensors=[],
        )


@pytest.mark.parametrize("model_name", ["", "face-model", "face model", "模型"])
def test_create_litert_model_spec_rejects_invalid_model_name(model_name):
    with pytest.raises(ValueError, match="model_name"):
        create_litert_model_spec(
            "model/model.tflite", model_name,
            input_tensors=[{"name": "input", "shape": [1], "encoding_type": "FP32"}],
            output_tensors=[{"name": "output", "shape": [1], "encoding_type": "FP32"}],
        )


def test_write_pipeline_zoo_package_writes_manifest_and_assets(tmp_path):
    model_file = tmp_path / "source.tflite"
    model_file.write_bytes(b"model")
    package = PipelineZooPackageSpec(
        package_id="face",
        supported_modes=["xr", "spatial", "xr"],
        pipelines=[PipelinePackageEntry("detection", "pipeline/face_detection_pipeline.json")],
        runtime={"detection_tensor": "detections"},
    )
    pipeline = {
        "tensors": {
            "input": {"dimensions": [1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 2},
            "output": {"dimensions": [1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 2},
        },
        "operators": [
            configure_litert_inference_operator(
                {"type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO", "inputs": [{"name": "input", "tensor": "input"}], "outputs": [{"name": "output", "tensor": "output"}]},
                model_path="model/face_detector.tflite",
                model_name="face_detector",
                input_tensors=[{"name": "input", "shape": [1], "encoding_type": "FP32"}],
                output_tensors=[{"name": "output", "shape": [1], "encoding_type": "FP32"}],
            )
        ],
        "inputs": ["input"],
        "outputs": ["output"],
    }
    validate_pipeline_spec(pipeline)

    manifest = write_pipeline_zoo_package(
        tmp_path / "pkg",
        package,
        pipelines={"detection": pipeline},
        assets={"model/face_detector.tflite": model_file},
    )

    assert manifest["schema_version"] == "2"
    assert manifest["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert "display_name" not in manifest
    assert "task" not in manifest
    assert "labels" not in manifest
    assert "model" not in manifest
    assert "models" not in manifest
    assert (tmp_path / "pkg" / "manifest.json").exists()
    assert (tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json").exists()
    assert not (tmp_path / "pkg" / "model" / "model.json").exists()
    assert (tmp_path / "pkg" / "model" / "face_detector.tflite").read_bytes() == b"model"

    loaded_manifest = load_pipeline_zoo_manifest(tmp_path / "pkg")
    assert loaded_manifest["runtime"]["detection_tensor"] == "detections"
    with open(tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json", encoding="utf-8") as file:
        loaded_pipeline = json.load(file)
    validate_pipeline_spec(loaded_pipeline)
    validate_package(tmp_path / "pkg")
    assert loaded_pipeline["operators"][0]["model"]["model_type"] == "tflite"
    assert loaded_pipeline["operators"][0]["model"]["bin_path"] == "model/face_detector.tflite"
    assert "model_file" not in loaded_pipeline["operators"][0]


def test_pipeline_zoo_package_spec_default_manifest_is_valid():
    package = PipelineZooPackageSpec(
        package_id="portable",
        pipelines=[PipelinePackageEntry("main", "pipeline/main.json")],
    )

    manifest = package.to_manifest_dict()

    assert manifest["runtime"]["supported_modes"] == ["xr", "spatial"]
    validate_pipeline_zoo_manifest(manifest)


def test_pipeline_zoo_package_spec_normalizes_runtime_supported_modes():
    package = PipelineZooPackageSpec(
        package_id="spatial",
        pipelines=[PipelinePackageEntry("main", "pipeline/main.json")],
        runtime={"supported_modes": ["SPATIAL", "spatial"]},
    )

    manifest = package.to_manifest_dict()

    assert manifest["runtime"]["supported_modes"] == ["spatial"]
    validate_pipeline_zoo_manifest(manifest)


def test_configure_litert_inference_operator_requires_inline_model():
    with pytest.raises(ValueError, match="requires input tensor metadata"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model_path="model/detector.tflite",
            model_name="detector",
        )

    by_model_path = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model_path="model/detector.tflite",
        model_name="detector",
        input_tensors=[{"name": "input", "shape": [1], "encoding_type": "FP32"}],
        output_tensors=[],
    )
    by_inline_model = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model={"bin_path": "model/inline.tflite", "model_name": "inline", "model_type": "tflite", "model_target": "npu",
               "input": [{"name": "input", "shape": [1], "encoding_type": "FP32"}],
               "output": [{"name": "output", "shape": [1], "encoding_type": "FP32"}]},
    )

    assert by_model_path["model"]["bin_path"] == "model/detector.tflite"
    assert by_inline_model["model"]["bin_path"] == "model/inline.tflite"
    assert by_inline_model["model"]["model_type"] == "tflite"

    by_gpu_model = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model={"bin_path": "model/gpu.tflite", "model_type": "tflite", "model_target": "gpu",
               "input": [{"name": "input", "shape": [1], "encoding_type": "FP32"}],
               "output": [{"name": "output", "shape": [1], "encoding_type": "FP32"}],
               "cpu_target_num_threads": 4},
    )
    assert by_gpu_model["model"]["model_target"] == "gpu"
    assert "cpu_target_num_threads" not in by_gpu_model["model"]

    with pytest.raises(ValueError, match="requires inline model metadata"):
        configure_litert_inference_operator({"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []})

    with pytest.raises(ValueError, match="either model or model_path"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model={"bin_path": "model/inline.tflite"},
            model_path="model/detector.tflite",
        )

    with pytest.raises(ValueError, match="missing required keys: encoding_type"):
        create_litert_model_spec("model/bad.tflite", "bad",
                                 input_tensors=[{"name": "input", "shape": [1]}],
                                 output_tensors=[])

    with pytest.raises(ValueError, match="missing required keys: model_target, input, output"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model={"bin_path": "model/incomplete.tflite", "model_type": "tflite"},
        )


def test_validate_pipeline_zoo_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid package-relative path"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad", "pipelines": [{"id": "p", "path": "../pipeline.json"}]}
        )


def test_validate_pipeline_zoo_rejects_absolute_paths():
    with pytest.raises(ValueError, match="Package paths must be relative"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad",
             "pipelines": [{"id": "p", "path": "/tmp/pipeline.json"}]}
        )


def test_pipeline_zoo_rejects_duplicate_pipeline_ids():
    package = PipelineZooPackageSpec(
        package_id="bad",
        pipelines=[
            PipelinePackageEntry("same", "pipeline/one.json"),
            PipelinePackageEntry("same", "pipeline/two.json"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate pipeline id: same"):
        package.to_manifest_dict()

    with pytest.raises(ValueError, match="Duplicate pipeline id: same"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [
                    {"id": "same", "path": "pipeline/one.json"},
                    {"id": "same", "path": "pipeline/two.json"},
                ],
            }
        )


def test_validate_pipeline_zoo_requires_schema_v2():
    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_pipeline_zoo_manifest({"schema_version": "1.0", "id": "bad", "pipelines": []})
    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_pipeline_zoo_manifest({"schema_version": 2, "id": "bad", "pipelines": []})


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"id": 1}, "manifest id must be a non-empty string"),
        ({"pipelines": [{"id": 1, "path": "pipeline.json"}]}, "pipeline #0.id"),
        ({"pipelines": [{"id": "main", "path": 1}]}, "pipeline #0.path"),
    ],
)
def test_validate_pipeline_zoo_requires_string_identifiers_and_paths(manifest_update, message):
    manifest = {
        "schema_version": "2",
        "id": "demo",
        "pipelines": [{"id": "main", "path": "pipeline.json"}],
        "runtime": {"supported_modes": ["xr"]},
    }
    manifest.update(manifest_update)

    with pytest.raises(ValueError, match=message):
        validate_pipeline_zoo_manifest(manifest)


def test_validate_pipeline_zoo_requires_declared_supported_modes():
    with pytest.raises(ValueError, match="requires runtime.supported_modes"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad",
             "pipelines": [{"id": "p", "path": "pipeline.json"}]}
        )


def test_validate_pipeline_zoo_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "runtime": {"supported_modes": ["desktop"]},
            }
        )

    with pytest.raises(ValueError, match="supported_modes must be a list"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "runtime": {"supported_modes": "xr"},
            }
        )
