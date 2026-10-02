import json
import zipfile
from pathlib import Path

import pytest

from pyspatialml import package_cli


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline_with_model(model_path="face.tflite"):
    return {
        "tensors": {
            "input": {"dimensions": [1, 4], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            "output": {"dimensions": [1, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
        },
        "operators": [
            {
                "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                "inputs": [{"tensor": "input"}],
                "outputs": [{"tensor": "output"}],
                "model": {
                    "bin_path": model_path,
                    "model_name": "face",
                    "model_type": "tflite",
                    "model_target": "npu",
                    "input": [
                        {"name": "input", "shape": [1, 4], "encoding_type": "FP32"},
                    ],
                    "output": [
                        {"name": "output", "shape": [1, 2], "encoding_type": "FP32"},
                    ],
                },
            }
        ],
        "inputs": ["input"],
        "outputs": ["output"],
    }


def _pipeline_with_operator(op_type):
    prefix = "XR_SECURE_MR_OPERATOR_TYPE_"
    suffix = "_PICO"
    assert op_type.startswith(prefix) and op_type.endswith(suffix)
    normalized = op_type[len(prefix):-len(suffix)]
    if normalized == "MICROPHONE":
        return {
            "tensors": {
                "stereo": {
                    "dimensions": [128, 2],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "left": {
                    "dimensions": [128, 1],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "right": {
                    "dimensions": [128, 1],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "timestamp": {
                    "dimensions": [1],
                    "channels": 4,
                    "data_type": 5,
                    "is_placeholder": True,
                    "usage": 5,
                },
            },
            "operators": [{
                "type": op_type,
                "inputs": [],
                "outputs": [
                    {"tensor": "stereo"},
                    {"tensor": "left"},
                    {"tensor": "right"},
                    {"tensor": "timestamp"},
                ],
                "attrs": ["48000;PCM_FLOAT"],
            }],
            "inputs": [],
            "outputs": [],
        }
    if normalized in {"RENDER_TEXT", "SWITCH_GLTF_RENDER_STATUS", "UPDATE_GLTF", "UPLOAD_TEXTURE_TO_GLTF"}:
        tensors = {
            "gltf": {"dimensions": [1, 1], "channels": 1, "data_type": 1, "is_placeholder": True, "usage": 7, "is_gltf": True},
            "text": {"dimensions": [1], "channels": 1, "data_type": 1, "is_placeholder": False, "usage": 2},
            "start": {"dimensions": [1], "channels": 2, "data_type": 6, "is_placeholder": False, "usage": 1},
            "colors": {"dimensions": [2], "channels": 4, "data_type": 1, "is_placeholder": False, "usage": 4},
            "texture": {"dimensions": [1], "channels": 1, "data_type": 3, "is_placeholder": False, "usage": 2},
            "font_size": {"dimensions": [1], "channels": 1, "data_type": 6, "is_placeholder": False, "usage": 2},
        }
        operator = {
            "type": op_type,
            "inputs": [{"tensor": "text"}, {"tensor": "start"}, {"tensor": "colors"}, {"tensor": "gltf"}, {"tensor": "texture"}, {"tensor": "font_size"}],
            "outputs": [],
            "attrs": ["bold#en-us#512#64"],
        }
        if normalized == "SWITCH_GLTF_RENDER_STATUS":
            operator = {"type": op_type, "inputs": [{"tensor": "gltf"}, None, None, None], "outputs": []}
        elif normalized == "UPDATE_GLTF":
            operator = {"type": op_type, "inputs": [{"tensor": "gltf"}, None, None], "outputs": [], "attrs": ["world pose"]}
        elif normalized == "UPLOAD_TEXTURE_TO_GLTF":
            tensors["image"] = {"dimensions": [2, 2], "channels": 3, "data_type": 1, "is_placeholder": False, "usage": 6}
            operator = {"type": op_type, "inputs": [{"tensor": "gltf"}, {"tensor": "image"}], "outputs": [{"tensor": "texture"}]}
        return {
            "tensors": tensors,
            "operators": [operator],
            "inputs": [],
            "outputs": [],
        }
    if normalized in {
        "SSMR_UPDATE_COMPONENT", "SSMR_SWITCH_VISIBILITY",
        "UPDATE_COMPONENT", "SCENEGRAPH_VISIBILITY",
    }:
        canonical = {
            "UPDATE_COMPONENT": "SSMR_UPDATE_COMPONENT",
            "SCENEGRAPH_VISIBILITY": "SSMR_SWITCH_VISIBILITY",
        }.get(normalized, normalized)
        operator = {
            "type": op_type,
            "inputs": [{"tensor": "scene"}, {"tensor": "data"}],
            "outputs": [],
        }
        if canonical == "SSMR_UPDATE_COMPONENT":
            operator["attrs"] = ["/target:Transform.Scale"]
        return {
            "tensors": {
                "scene": {"dimensions": [1, 1], "channels": 1, "data_type": 1, "is_placeholder": True, "usage": 7, "is_gltf": True},
                "data": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": False, "usage": 6},
            },
            "operators": [operator],
            "inputs": [],
            "outputs": [],
        }
    if normalized == "JAVASCRIPT":
        return {
            "tensors": {
                "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [{
                "type": op_type,
                "inputs": [{"name": "input", "tensor": "input"}],
                "outputs": [{"name": "output", "tensor": "output"}],
                "attrs": ["output = input;"],
            }],
            "inputs": ["input"],
            "outputs": ["output"],
        }
    return {
        "tensors": {
            "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": False, "usage": 6},
            "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": False, "usage": 6},
        },
        "operators": [{"type": op_type, "inputs": [{"tensor": "input"}], "outputs": [{"tensor": "output"}]}],
        "inputs": [],
        "outputs": [],
    }


def test_create_package_normalizes_pipeline_and_model_paths(tmp_path):
    source_dir = tmp_path / "src"
    pipeline = source_dir / "detection.json"
    model = source_dir / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"detection={pipeline}"],
        output=output,
        supported_modes=["spatial"],
    )

    manifest = _read_json(output / "manifest.json")
    packaged_pipeline = _read_json(output / "pipeline" / "detection.json")
    assert manifest["schema_version"] == "2"
    assert manifest["id"] == "face-demo"
    assert manifest["pipelines"] == [{"id": "detection", "path": "pipeline/detection.json"}]
    assert manifest["runtime"]["supported_modes"] == ["spatial"]
    assert (output / "model" / "face.tflite").read_bytes() == b"model"
    assert packaged_pipeline["operators"][0]["model"]["bin_path"] == "model/face.tflite"
    assert packaged_pipeline["operators"][0]["model"]["model_target"] == "npu"
    assert "model_target" not in packaged_pipeline["operators"][0]
    assert "model_file" not in packaged_pipeline["operators"][0]
    assert "model_asset" not in packaged_pipeline["operators"][0]
    assert "model_id" not in packaged_pipeline["operators"][0]


def test_create_package_resolves_model_from_asset_root(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    asset_root = tmp_path / "assets"
    model = asset_root / "model" / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("model/face.tflite"))
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        asset_roots=[asset_root],
    )

    assert (output / "model" / "face.tflite").read_bytes() == b"model"
    assert _read_json(output / "pipeline" / "main.json")["operators"][0]["model"]["bin_path"] == "model/face.tflite"


def test_create_package_normalizes_pipeline_id_before_building_path(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, {"tensors": {}, "operators": [], "inputs": [], "outputs": []})

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"group\\main={pipeline}"],
        output=output,
    )

    manifest = _read_json(output / "manifest.json")
    assert manifest["pipelines"] == [{"id": "group/main", "path": "pipeline/group/main.json"}]
    assert (output / "pipeline" / "group" / "main.json").is_file()
    assert package_cli.validate_package(output) == 0


def test_create_package_normalizes_gltf_tensor_assets(tmp_path):
    source_dir = tmp_path / "src"
    pipeline = source_dir / "display.json"
    gltf = source_dir / "frame.gltf"
    _write_json(
        pipeline,
        {
            "tensors": {
                "scene": {
                    "asset": "frame.gltf",
                    "is_gltf": True,
                    "is_placeholder": True,
                    "dimensions": [1, 1],
                    "channels": 1,
                    "data_type": 1,
                    "usage": 7,
                }
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        },
    )
    gltf.write_text("{}", encoding="utf-8")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="display-demo",
        pipelines=[f"display={pipeline}"],
        output=output,
    )

    packaged_pipeline = _read_json(output / "pipeline" / "display.json")
    assert (output / "gltf" / "frame.gltf").read_text(encoding="utf-8") == "{}"
    assert packaged_pipeline["tensors"]["scene"]["asset"] == "gltf/frame.gltf"


def test_validate_package_reports_gltf_materialization_requirement(tmp_path, capsys):
    source_dir = tmp_path / "src"
    pipeline = source_dir / "display.json"
    gltf = source_dir / "frame.gltf"
    _write_json(
        pipeline,
        {
            "tensors": {"scene": {"dimensions": [1, 1], "channels": 1, "data_type": 1, "is_placeholder": True, "usage": 7, "is_gltf": True, "asset": "frame.gltf"}},
            "operators": [], "inputs": [], "outputs": [],
        },
    )
    gltf.write_text("{}", encoding="utf-8")
    output = tmp_path / "pkg"
    package_cli.create_package(package_id="display-demo", pipelines=[f"display={pipeline}"], output=output)

    assert package_cli.validate_package(output) == 0
    assert "GLTF tensor asset materialization" in capsys.readouterr().out


def test_create_package_infers_common_operator_modes_when_supported_modes_omitted(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO"))

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"main={pipeline}"],
        output=output,
    )

    assert _read_json(output / "manifest.json")["runtime"]["supported_modes"] == ["xr", "spatial"]


def test_create_package_accepts_valid_microphone_attrs(tmp_path):
    pipeline = tmp_path / "microphone.json"
    _write_json(pipeline, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO"))

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="microphone-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
    )

    assert package_cli.validate_package(output) == 0


@pytest.mark.parametrize(
    "attrs",
    [
        [],
        ["48000;PCM_FLOAT", "16000;PCM_16BIT"],
        ["7999;PCM_16BIT"],
        ["96001;PCM_16BIT"],
        ["48000;PCM_8BIT"],
        ["48000PCM_16BIT"],
    ],
)
def test_create_package_rejects_invalid_microphone_attrs(tmp_path, attrs):
    pipeline = _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO")
    pipeline["operators"][0]["attrs"] = attrs
    source = tmp_path / "microphone.json"
    _write_json(source, pipeline)

    with pytest.raises(package_cli.PackageCliError):
        package_cli.create_package(
            package_id="microphone-demo",
            pipelines=[f"main={source}"],
            output=tmp_path / "pkg",
        )


def test_create_package_infers_xr_mode_from_xr_only_operator(tmp_path):
    pipeline = tmp_path / "display.json"
    _write_json(pipeline, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"display={pipeline}"],
        output=output,
    )

    assert _read_json(output / "manifest.json")["runtime"]["supported_modes"] == ["xr"]


@pytest.mark.parametrize(
    "op_type",
    [
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
        "XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO",
        "XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO",
    ],
)
def test_create_package_infers_spatial_mode_from_spatial_only_operator(tmp_path, op_type):
    pipeline = tmp_path / "scene.json"
    _write_json(pipeline, _pipeline_with_operator(op_type))

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"scene={pipeline}"],
        output=output,
    )

    assert _read_json(output / "manifest.json")["runtime"]["supported_modes"] == ["spatial"]
    assert _read_json(output / "pipeline" / "scene.json")["operators"][0]["type"] == op_type


def test_create_package_accepts_javascript_alias_and_preserves_type(tmp_path):
    op_type = "XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO"
    pipeline = tmp_path / "script.json"
    _write_json(pipeline, _pipeline_with_operator(op_type))

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"script={pipeline}"],
        output=output,
    )

    assert _read_json(output / "manifest.json")["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert _read_json(output / "pipeline" / "script.json")["operators"][0]["type"] == op_type


@pytest.mark.parametrize(
    "fixture",
    [
        "spatial_scenegraph_visibility_package",
        "spatial_update_component_package",
    ],
)
def test_spatial_only_fixture_packages_validate(fixture):
    assert package_cli.validate_package(Path("tests/data") / fixture) == 0


def test_create_package_errors_for_missing_asset(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _pipeline_with_model("missing.tflite"))

    with pytest.raises(package_cli.PackageCliError, match="Referenced asset not found"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"main={pipeline}"],
            output=tmp_path / "pkg",
        )


def test_create_package_errors_on_model_name_collision(tmp_path):
    first = tmp_path / "a" / "pipeline.json"
    second = tmp_path / "b" / "pipeline.json"
    first_model = tmp_path / "a" / "shared.tflite"
    second_model = tmp_path / "b" / "shared.tflite"
    _write_json(first, _pipeline_with_model("shared.tflite"))
    _write_json(second, _pipeline_with_model("shared.tflite"))
    first_model.write_bytes(b"one")
    second_model.write_bytes(b"two")

    with pytest.raises(package_cli.PackageCliError, match="Asset name collision"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"first={first}", f"second={second}"],
            output=tmp_path / "pkg",
        )


def test_create_package_zip_output_and_validate(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")

    archive = tmp_path / "pkg.zip"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=archive,
        zip_output=True,
    )

    assert archive.is_file()
    with zipfile.ZipFile(archive) as package_zip:
        names = set(package_zip.namelist())
    assert "manifest.json" in names
    assert "pipeline/main.json" in names
    assert "model/face.tflite" in names
    assert package_cli.validate_package(archive) == 0


def test_validate_package_zip_does_not_delete_archive_sibling_directory(tmp_path):
    archive = tmp_path / "foo.zip"
    sibling = tmp_path / "foo"
    sentinel = sibling / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("must survive validation", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": "2",
                    "id": "demo",
                    "runtime": {"supported_modes": ["xr", "spatial"]},
                    "pipelines": [
                        {"id": "main", "path": "pipeline/main.json"}
                    ],
                }
            ),
        )
        package_zip.writestr("pipeline/main.json", "{}")

    with pytest.raises(package_cli.PackageCliError, match="Pipeline root requires"):
        package_cli.validate_package(archive)
    assert sentinel.read_text(encoding="utf-8") == "must survive validation"


def test_create_package_repacks_existing_package_root_without_pipeline_arg(tmp_path):
    source_pipeline = tmp_path / "src" / "pipeline.json"
    model = tmp_path / "src" / "face.tflite"
    _write_json(source_pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    source = tmp_path / "source-package"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={source_pipeline}"],
        output=source,
    )

    archive = tmp_path / "repacked.zip"
    package_cli.create_package(
        package_id="",
        pipelines=[],
        source=source,
        output=archive,
        zip_output=True,
    )

    assert archive.is_file()
    assert package_cli.validate_package(archive) == 0
    with zipfile.ZipFile(archive) as package_zip:
        names = set(package_zip.namelist())
    assert "manifest.json" in names
    assert "pipeline/main.json" in names
    assert "model/face.tflite" in names


def test_create_package_uses_source_without_manifest_as_asset_root(tmp_path):
    source = tmp_path / "source"
    pipeline = source / "pipeline.json"
    model = source / "model" / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("model/face.tflite"))
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        source=source,
        output=tmp_path / "out.zip",
        zip_output=True,
    )

    assert package_cli.validate_package(tmp_path / "out.zip") == 0


def test_create_package_reconciles_existing_manifest_modes_with_force(tmp_path):
    source = tmp_path / "source-package"
    _write_json(
        source / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["spatial"]},
        },
    )
    _write_json(source / "pipeline" / "display.json", _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))
    original_manifest = _read_json(source / "manifest.json")

    with pytest.raises(package_cli.PackageCliError, match="includes spatial.*XR-only"):
        package_cli.create_package(package_id="", pipelines=[], source=source, output=tmp_path / "bad.zip")

    archive = tmp_path / "fixed.zip"
    package_cli.create_package(package_id="", pipelines=[], source=source, output=archive, force=True)

    with zipfile.ZipFile(archive) as package_zip:
        manifest = json.loads(package_zip.read("manifest.json"))
    assert manifest["runtime"]["supported_modes"] == ["xr"]
    assert _read_json(source / "manifest.json") == original_manifest
    assert package_cli.validate_package(archive) == 0


def test_create_package_force_reconcile_failure_leaves_existing_output_unchanged(tmp_path):
    source = tmp_path / "source-package"
    _write_json(
        source / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [
                {"id": "display", "path": "pipeline/display.json"},
                {"id": "scene", "path": "pipeline/scene.json"},
            ],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    _write_json(source / "pipeline" / "display.json", _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))
    _write_json(source / "pipeline" / "scene.json", _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO"))
    output = tmp_path / "existing-output"
    sentinel = output / "sentinel.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(package_cli.PackageCliError, match="mix XR-only and Spatial-only"):
        package_cli.create_package(package_id="", pipelines=[], source=source, output=output, force=True)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_validate_package_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr("../evil.txt", "bad")
        package_zip.writestr("manifest.json", '{"schema_version":"2","id":"bad","pipelines":[]}')

    with pytest.raises(package_cli.PackageCliError, match="Unsafe zip entry path"):
        package_cli.validate_package(archive)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_validate_package_accepts_windows_relative_zip_paths(tmp_path):
    archive = tmp_path / "windows.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr(
            "pkg\\manifest.json",
            json.dumps(
                {
                    "schema_version": "2",
                    "id": "demo",
                    "runtime": {"supported_modes": ["xr", "spatial"]},
                    "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
                }
            ),
        )
        package_zip.writestr(
            "pkg\\pipeline\\main.json",
            json.dumps({"tensors": {}, "operators": [], "inputs": [], "outputs": []}),
        )

    assert package_cli.validate_package(archive) == 0


def test_validate_package_rejects_absolute_manifest_pipeline_path(tmp_path):
    external_pipeline = tmp_path / "external.json"
    _write_json(external_pipeline, {"tensors": {}, "operators": [], "inputs": [], "outputs": []})
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "bad",
            "runtime": {"supported_modes": ["xr", "spatial"]},
            "pipelines": [{"id": "main", "path": str(external_pipeline)}],
        },
    )

    with pytest.raises(package_cli.PackageCliError, match="pipeline path must be package-relative"):
        package_cli.validate_package(package)


def test_validate_package_rejects_manifest_pipeline_path_traversal(tmp_path):
    external_pipeline = tmp_path / "external.json"
    _write_json(external_pipeline, {"tensors": {}, "operators": [], "inputs": [], "outputs": []})
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "bad",
            "runtime": {"supported_modes": ["xr", "spatial"]},
            "pipelines": [{"id": "main", "path": "../external.json"}],
        },
    )

    with pytest.raises(package_cli.PackageCliError, match="Invalid package-relative path"):
        package_cli.validate_package(package)


def test_validate_package_rejects_absolute_model_asset_path(tmp_path):
    package = tmp_path / "pkg"
    external_model = tmp_path / "external.tflite"
    external_model.write_bytes(b"model")
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "bad",
            "runtime": {"supported_modes": ["xr", "spatial"]},
            "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
        },
    )
    _write_json(package / "pipeline" / "main.json", _pipeline_with_model(str(external_model)))

    with pytest.raises(package_cli.PackageCliError, match="Invalid package-relative path"):
        package_cli.validate_package(package)


def test_validate_package_rejects_asset_symlink_escape(tmp_path):
    package = tmp_path / "pkg"
    external_model = tmp_path / "external.tflite"
    external_model.write_bytes(b"model")
    package_model = package / "model" / "external.tflite"
    package_model.parent.mkdir(parents=True)
    package_model.symlink_to(external_model)
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "bad",
            "runtime": {"supported_modes": ["xr", "spatial"]},
            "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
        },
    )
    _write_json(package / "pipeline" / "main.json", _pipeline_with_model("model/external.tflite"))

    with pytest.raises(package_cli.PackageCliError, match="asset path escapes package root"):
        package_cli.validate_package(package)


def test_validate_package_rejects_xr_only_operator_for_spatial_manifest(tmp_path):
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["spatial"]},
        },
    )
    _write_json(
        package / "pipeline" / "display.json",
        _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"),
    )

    with pytest.raises(package_cli.PackageCliError, match="includes spatial.*XR-only"):
        package_cli.validate_package(package)


def test_validate_package_rejects_spatial_only_operator_for_xr_manifest(tmp_path):
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    _write_json(
        package / "pipeline" / "scene.json",
        _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO"),
    )

    with pytest.raises(package_cli.PackageCliError, match="includes xr.*Spatial-only"):
        package_cli.validate_package(package)


def test_validate_package_rejects_mixed_xr_and_spatial_only_operators(tmp_path):
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "mixed", "path": "pipeline/mixed.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    _write_json(
        package / "pipeline" / "mixed.json",
            {
                "tensors": {
                    "text": {"dimensions": [1], "channels": 1, "data_type": 1, "is_placeholder": False, "usage": 2},
                    "start": {"dimensions": [1], "channels": 2, "data_type": 6, "is_placeholder": False, "usage": 1},
                    "colors": {"dimensions": [2], "channels": 4, "data_type": 1, "is_placeholder": False, "usage": 4},
                    "gltf": {"dimensions": [1, 1], "channels": 1, "data_type": 1, "is_placeholder": True, "usage": 7, "is_gltf": True},
                    "texture": {"dimensions": [1], "channels": 1, "data_type": 3, "is_placeholder": False, "usage": 2},
                    "font": {"dimensions": [1], "channels": 1, "data_type": 6, "is_placeholder": False, "usage": 2},
                    "visible": {"dimensions": [1], "channels": 1, "data_type": 1, "is_placeholder": False, "usage": 2},
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO",
                        "inputs": [
                            {"tensor": "text"},
                            {"tensor": "start"},
                            {"tensor": "colors"},
                            {"tensor": "gltf"},
                            {"tensor": "texture"},
                            {"tensor": "font"},
                        ],
                        "outputs": [],
                        "attrs": ["bold#en-us#512#64"],
                    },
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
                        "inputs": [{"tensor": "gltf"}, {"tensor": "visible"}],
                        "outputs": [],
                    },
                ],
                "inputs": [],
                "outputs": [],
            },
    )

    with pytest.raises(package_cli.PackageCliError, match="mix XR-only and Spatial-only"):
        package_cli.validate_package(package)


def test_validate_package_rejects_xr_only_operator_when_manifest_claims_both_modes(tmp_path):
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    _write_json(
        package / "pipeline" / "display.json",
        _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"),
    )

    with pytest.raises(package_cli.PackageCliError, match="includes spatial.*XR-only"):
        package_cli.validate_package(package)


def test_validate_package_rejects_spatial_only_operator_when_manifest_claims_both_modes(tmp_path):
    package = tmp_path / "pkg"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    _write_json(
        package / "pipeline" / "scene.json",
        _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO"),
    )

    with pytest.raises(package_cli.PackageCliError, match="includes xr.*Spatial-only"):
        package_cli.validate_package(package)


def test_create_package_rejects_manifest_modes_for_mode_specific_operators(tmp_path):
    pipeline = tmp_path / "display.json"
    _write_json(pipeline, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))

    with pytest.raises(package_cli.PackageCliError, match="includes spatial.*XR-only"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"display={pipeline}"],
            output=tmp_path / "pkg",
            supported_modes=["spatial"],
        )


def test_create_package_rejects_mixed_exclusive_operators_across_pipelines(tmp_path):
    display = tmp_path / "display.json"
    scene = tmp_path / "scene.json"
    _write_json(display, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))
    _write_json(scene, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO"))

    with pytest.raises(package_cli.PackageCliError, match="mix XR-only and Spatial-only"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"display={display}", f"scene={scene}"],
            output=tmp_path / "pkg",
        )


def test_create_package_rejects_explicit_spatial_mode_with_xr_only_operator(tmp_path):
    pipeline = tmp_path / "display.json"
    _write_json(pipeline, _pipeline_with_operator("XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO"))

    with pytest.raises(package_cli.PackageCliError, match="includes spatial.*XR-only"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"display={pipeline}"],
            output=tmp_path / "pkg",
            supported_modes=["spatial"],
        )


def test_create_package_existing_output_requires_confirmation(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(package_cli.PackageCliError, match="already exists"):
        package_cli.create_package(
            package_id="face-demo",
            pipelines=[f"main={pipeline}"],
            output=output,
        )
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"

    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
    ) == 0
    assert not (output / "old.txt").exists()
    assert (output / "manifest.json").is_file()


def test_create_package_force_bypasses_confirmation(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output.mkdir()

    def _fail_input(_prompt):
        raise AssertionError("input should not be called when force=True")

    monkeypatch.setattr("builtins.input", _fail_input)
    assert package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        force=True,
    ) == 0


def test_inspect_package_prints_summary(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        supported_modes=["xr"],
    )

    package_cli.inspect_package(output)

    captured = capsys.readouterr()
    assert "Package: face-demo" in captured.out
    assert "Pipelines: 1" in captured.out
    assert "main -> pipeline/main.json" in captured.out
    assert "Supported modes: xr" in captured.out
    assert "model/face.tflite" in captured.out
