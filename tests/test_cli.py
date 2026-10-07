import subprocess
import json
from pathlib import Path
from types import SimpleNamespace

from pyspatialml import cli as cli_module
from pyspatialml.litert_tools import LiteRTCli, LiteRTToolError

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = REPO_ROOT / "tests" / "data" / "face_mediapipe_package" / "model" / "face_detector.tflite"


def _create_cli_run_package(tmp_path, pipeline, *, pipeline_id="main"):
    package = tmp_path / f"{pipeline.stem}-package"
    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "run-demo",
            "--pipeline",
            f"{pipeline_id}={pipeline}",
            "--output",
            str(package),
        ]
    ) == 0
    return package


def test_tools_litert_status_prints_resolved_cli(monkeypatch, capsys, tmp_path):
    class _FakeLiteRT:
        path = tmp_path / "bin" / "litert"
        managed = False

        def version(self):
            return "litert 2.0.3"

    litert = _FakeLiteRT()
    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda cache_dir=None: litert)

    exit_code = cli_module.main(["tools", "litert", "status", "--tool-cache", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"LiteRT CLI: {litert.path}" in captured.out
    assert "Source: system" in captured.out
    assert "Version: litert 2.0.3" in captured.out
    assert f"Managed cache: {tmp_path}" in captured.out


def test_tools_litert_status_json(monkeypatch, capsys, tmp_path):
    class _FakeLiteRT:
        path = tmp_path / "bin" / "litert"
        managed = False

        def version(self):
            return "litert 2.0.3"

    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda cache_dir=None: _FakeLiteRT())

    exit_code = cli_module.main(["tools", "litert", "status", "--tool-cache", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "tools litert status"
    assert payload["source"] == "system"
    assert payload["version"] == "litert 2.0.3"


def test_tools_litert_status_format_json_alias(monkeypatch, capsys, tmp_path):
    class _FakeLiteRT:
        path = tmp_path / "bin" / "litert"
        managed = False

        def version(self):
            return "litert 2.0.3"

    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda cache_dir=None: _FakeLiteRT())

    exit_code = cli_module.main(["tools", "litert", "status", "--tool-cache", str(tmp_path), "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "tools litert status"


def test_tools_litert_status_reports_missing_tool(monkeypatch, capsys, tmp_path):
    def _raise_missing(cache_dir=None):
        raise LiteRTToolError("missing litert")

    monkeypatch.setattr(cli_module, "resolve_litert_cli", _raise_missing)

    exit_code = cli_module.main(["tools", "litert", "status", "--tool-cache", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PSM_LITERT_UNAVAILABLE" in captured.err
    assert "missing litert" in captured.err


def test_tools_litert_status_json_reports_missing_tool(monkeypatch, capsys, tmp_path):
    def _raise_missing(cache_dir=None):
        raise LiteRTToolError("missing litert")

    monkeypatch.setattr(cli_module, "resolve_litert_cli", _raise_missing)

    exit_code = cli_module.main(["tools", "litert", "status", "--tool-cache", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["error"]["code"] == "PSM_LITERT_UNAVAILABLE"
    assert payload["error"]["category"] == "litert_tool"
    assert payload["error"]["message"] == "missing litert"


def test_tools_litert_install_uses_requested_package_and_version(monkeypatch, capsys, tmp_path):
    calls = []
    litert = LiteRTCli(path=tmp_path / "litert-cli" / "bin" / "litert", managed=True)

    def _install_litert_cli(*, cache_dir=None, package="", version="", force=False):
        calls.append((cache_dir, package, version, force))
        return litert

    monkeypatch.setattr(cli_module, "install_litert_cli", _install_litert_cli)

    exit_code = cli_module.main(
        [
            "tools",
            "litert",
            "install",
            "--tool-cache",
            str(tmp_path),
            "--package",
            "ai-edge-litert-nightly",
            "--version",
            "9.9.9",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [(tmp_path, "ai-edge-litert-nightly", "9.9.9", False)]
    assert f"LiteRT CLI installed: {litert.path}" in captured.out


def test_tools_litert_install_force_recreates_managed_env(monkeypatch, capsys, tmp_path):
    calls = []
    litert = LiteRTCli(path=tmp_path / "litert-cli" / "bin" / "litert", managed=True)

    def _install_litert_cli(*, cache_dir=None, package="", version="", force=False):
        calls.append((cache_dir, package, version, force))
        return litert

    monkeypatch.setattr(cli_module, "install_litert_cli", _install_litert_cli)

    exit_code = cli_module.main(["tools", "litert", "install", "--tool-cache", str(tmp_path), "--force"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [(tmp_path, "litert-cli", "0.1.0", True)]
    assert f"LiteRT CLI installed: {litert.path}" in captured.out


def test_tools_litert_repair_recreates_managed_env(monkeypatch, capsys, tmp_path):
    calls = []
    litert = LiteRTCli(path=tmp_path / "litert-cli" / "bin" / "litert", managed=True)

    def _repair_litert_cli(*, cache_dir=None, package="", version=""):
        calls.append((cache_dir, package, version))
        return litert

    monkeypatch.setattr(cli_module, "repair_litert_cli", _repair_litert_cli)

    exit_code = cli_module.main(["tools", "litert", "repair", "--tool-cache", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert calls == [(tmp_path, "litert-cli", "0.1.0")]
    assert payload["command"] == "tools litert repair"
    assert payload["recreated"] is True


def test_compare_command_passes(capsys, tmp_path):
    expected = tmp_path / "expected.npy"
    actual = tmp_path / "actual.npy"
    import numpy as np

    np.save(expected, np.array([1.0, 2.0], dtype=np.float32))
    np.save(actual, np.array([1.0, 2.0], dtype=np.float32))

    assert cli_module.main(["compare", str(expected), str(actual)]) == 0

    captured = capsys.readouterr()
    assert "passed: yes" in captured.out


def test_compare_command_mismatch_returns_compare_code(capsys, tmp_path):
    expected = tmp_path / "expected.npy"
    actual = tmp_path / "actual.npy"
    import numpy as np

    np.save(expected, np.array([1.0], dtype=np.float32))
    np.save(actual, np.array([2.0], dtype=np.float32))

    assert cli_module.main(["compare", str(expected), str(actual), "--rtol", "1e-6", "--atol", "1e-6"]) == 4

    captured = capsys.readouterr()
    assert "passed: no" in captured.out


def test_compare_command_reports_errors(capsys, tmp_path):
    expected = tmp_path / "expected.bin"
    actual = tmp_path / "actual.bin"
    expected.write_bytes(b"1")
    actual.write_bytes(b"1")

    assert cli_module.main(["compare", str(expected), str(actual)]) == 1

    captured = capsys.readouterr()
    assert "PSM_COMPARE" in captured.err
    assert "Only .npy files" in captured.err


def test_model_command_delegates_to_litert_with_remainder_separator(monkeypatch, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    resolve_calls = []
    subprocess_calls = []

    def _resolve_litert_cli(*, ensure=False, cache_dir=None):
        resolve_calls.append((ensure, cache_dir))
        return litert

    def _run(argv, env=None):
        subprocess_calls.append(argv)
        return subprocess.CompletedProcess(argv, 7)

    monkeypatch.setattr(cli_module, "resolve_litert_cli", _resolve_litert_cli)
    monkeypatch.setattr(cli_module.subprocess, "run", _run)

    exit_code = cli_module.main(
        [
            "model",
            "benchmark",
            "--tool-cache",
            str(tmp_path),
            "--",
            "model.tflite",
            "--target",
            "host",
        ]
    )

    assert exit_code == 7
    assert resolve_calls == [(True, tmp_path)]
    assert subprocess_calls == [[str(litert.path), "benchmark", "model.tflite", "--target", "host"]]


def test_model_command_json_wraps_delegated_litert(monkeypatch, capsys, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda ensure=False, cache_dir=None: litert)

    def _run(argv, env=None, text=False, stdout=None, stderr=None):
        return subprocess.CompletedProcess(argv, 0, stdout="litert stdout", stderr="litert stderr")

    monkeypatch.setattr(cli_module.subprocess, "run", _run)

    exit_code = cli_module.main(["model", "benchmark", "--json", "--", "model.tflite"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "model benchmark"
    assert payload["argv"] == [str(litert.path), "benchmark", "model.tflite"]
    assert payload["stdout"] == "litert stdout"
    assert payload["stderr"] == "litert stderr"


def test_model_convert_help_includes_onnx_and_litert_help(monkeypatch, capsys, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda ensure=False, cache_dir=None: litert)

    def _run(argv, env=None, text=False, stdout=None, stderr=None):
        return subprocess.CompletedProcess(argv, 0, stdout="Usage: litert convert [OPTIONS] MODEL_OR_SCRIPT\n", stderr="")

    monkeypatch.setattr(cli_module.subprocess, "run", _run)

    exit_code = cli_module.main(["model", "convert", "--", "--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ONNX input:" in captured.out
    assert "pyspatialml model convert -- model.onnx --output ./converted_tflite" in captured.out
    assert "--input-shape NAME:DIMS" in captured.out
    assert "--onnx2tf-arg VALUE" in captured.out
    assert "LiteRT convert help:" in captured.out
    assert "Usage: litert convert" in captured.out


def test_model_convert_help_json_includes_onnx_and_litert_help(monkeypatch, capsys, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda ensure=False, cache_dir=None: litert)

    def _run(argv, env=None, text=False, stdout=None, stderr=None):
        return subprocess.CompletedProcess(argv, 0, stdout="Usage: litert convert [OPTIONS] MODEL_OR_SCRIPT\n", stderr="")

    monkeypatch.setattr(cli_module.subprocess, "run", _run)

    exit_code = cli_module.main(["model", "convert", "--json", "--", "--help"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["command"] == "model convert"
    assert "ONNX input:" in payload["stdout"]
    assert "--no-large-tensor" in payload["stdout"]
    assert "Usage: litert convert" in payload["stdout"]


def test_model_convert_routes_onnx_without_litert(monkeypatch, capsys, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")
    calls = []

    def _resolve_litert_cli(*_args, **_kwargs):
        raise AssertionError("ONNX convert should not resolve LiteRT")

    def _convert_onnx_to_tflite(**kwargs):
        calls.append(kwargs)
        tflite = output / "model.tflite"
        tflite.parent.mkdir()
        tflite.write_bytes(b"tflite")
        return SimpleNamespace(
            model=kwargs["model"],
            output=kwargs["output"],
            tflite_models=[tflite],
            argv=["onnx2tf", "-i", str(model), "-o", str(output)],
            stdout="stdout",
            stderr="stderr",
            tool=SimpleNamespace(path=tmp_path / "onnx2tf", managed=True),
        )

    monkeypatch.setattr(cli_module, "resolve_litert_cli", _resolve_litert_cli)
    monkeypatch.setattr(cli_module.onnx_tools, "convert_onnx_to_tflite", _convert_onnx_to_tflite)

    exit_code = cli_module.main(["model", "convert", "--", str(model), "--output", str(output), "--verbosity", "debug"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "model": model,
            "output": output,
            "extra_args": ["--verbosity", "debug"],
            "cache_dir": None,
            "verbose": True,
        }
    ]
    assert f"Converted ONNX model: {model}" in captured.out
    assert f"TFLite model: {output / 'model.tflite'}" in captured.out
    assert "stderr" in captured.err


def test_model_convert_routes_onnx_json(monkeypatch, capsys, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")

    def _convert_onnx_to_tflite(**kwargs):
        tflite = output / "model.tflite"
        return SimpleNamespace(
            model=kwargs["model"],
            output=kwargs["output"],
            tflite_models=[tflite],
            argv=["onnx2tf", "-i", str(model), "-o", str(output)],
            stdout="stdout",
            stderr="stderr",
            tool=SimpleNamespace(path=tmp_path / "onnx2tf", managed=True),
        )

    monkeypatch.setattr(cli_module.onnx_tools, "convert_onnx_to_tflite", _convert_onnx_to_tflite)

    exit_code = cli_module.main(["model", "convert", "--json", "--", str(model), "--output", str(output)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["converter"] == "onnx2tf"
    assert payload["model"] == str(model)
    assert payload["output"] == str(output)
    assert payload["tflite_models"] == [str(output / "model.tflite")]


def test_model_convert_translates_onnx_alias_flags(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")
    calls = []

    def _convert_onnx_to_tflite(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            output=kwargs["output"],
            tflite_models=[output / "model.tflite"],
            argv=[],
            stdout="",
            stderr="",
            tool=SimpleNamespace(path=tmp_path / "onnx2tf", managed=True),
        )

    monkeypatch.setattr(cli_module.onnx_tools, "convert_onnx_to_tflite", _convert_onnx_to_tflite)

    assert cli_module.main(
        [
            "model",
            "convert",
            "--",
            str(model),
            "--output",
            str(output),
            "--input-shape",
            "images:1,3,640,640",
            "--shape-hint=tokens:1,128",
            "--no-large-tensor",
            "--keep-nchw",
            "images",
            "--keep-nhwc=features",
            "--non-verbose",
            "--copy-input-output-names",
            "--dynamic-range-quantize",
            "--integer-quantize",
            "--onnx2tf-arg",
            "--disable_strict_mode",
            "--onnx2tf-arg=--output_nms_with_dynamic_tensor",
        ]
    ) == 0

    assert calls[0]["extra_args"] == [
        "--overwrite_input_shape",
        "images:1,3,640,640",
        "--shape_hints",
        "tokens:1,128",
        "--no_large_tensor",
        "--keep_ncw_or_nchw_or_ncdhw_input_names",
        "images",
        "--keep_nwc_or_nhwc_or_ndhwc_input_names",
        "features",
        "--non_verbose",
        "--copy_onnx_input_output_names_to_tflite",
        "--output_dynamic_range_quantized_tflite",
        "--output_integer_quantized_tflite",
        "--disable_strict_mode",
        "--output_nms_with_dynamic_tensor",
    ]


def test_model_convert_onnx_alias_requires_value(capsys, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")

    exit_code = cli_module.main(["model", "convert", "--", str(model), "--output", str(output), "--input-shape"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_ONNX_CONVERT" in captured.err
    assert "--input-shape requires a value" in captured.err


def test_model_convert_reports_missing_local_model_before_litert(capsys, tmp_path):
    missing = tmp_path / "model.onn"
    output = tmp_path / "model.tflite"

    exit_code = cli_module.main(["model", "convert", str(missing), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_ONNX_CONVERT" in captured.err
    assert f"Model input file not found: {missing}" in captured.err
    assert "Pass an existing model file" in captured.err


def test_model_convert_allows_non_path_model_names(monkeypatch, tmp_path):
    litert = SimpleNamespace(path=tmp_path / "litert", managed=False)
    calls = []

    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda ensure=False, cache_dir=None: litert)
    monkeypatch.setattr(cli_module.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(returncode=0))

    assert cli_module.main(["model", "convert", "repo/model-name", "--output", str(tmp_path / "out")]) == 0
    assert calls[0][0] == [str(litert.path), "convert", "repo/model-name", "--output", str(tmp_path / "out")]


def test_model_convert_rejects_shape_hint_for_overwritten_input(capsys, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")

    exit_code = cli_module.main(
        [
            "model",
            "convert",
            "--",
            str(model),
            "--output",
            str(output),
            "--input-shape",
            "images:1,3,640,640",
            "--shape-hint=images:1,3,640,640",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_ONNX_CONVERT" in captured.err
    assert "Do not pass both --input-shape and --shape-hint" in captured.err
    assert "images" in captured.err


def test_model_convert_allows_shape_hint_for_different_input(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    output = tmp_path / "converted"
    model.write_bytes(b"onnx")
    calls = []

    def _convert_onnx_to_tflite(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            output=kwargs["output"],
            tflite_models=[output / "model.tflite"],
            argv=[],
            stdout="",
            stderr="",
            tool=SimpleNamespace(path=tmp_path / "onnx2tf", managed=True),
        )

    monkeypatch.setattr(cli_module.onnx_tools, "convert_onnx_to_tflite", _convert_onnx_to_tflite)

    assert cli_module.main(
        [
            "model",
            "convert",
            "--",
            str(model),
            "--output",
            str(output),
            "--input-shape=images:1,3,640,640",
            "--shape-hint",
            "tokens:1,128",
        ]
    ) == 0

    assert calls[0]["extra_args"] == [
        "--overwrite_input_shape",
        "images:1,3,640,640",
        "--shape_hints",
        "tokens:1,128",
    ]


def test_model_info_command_prints_model_metadata(capsys):
    assert cli_module.main(["model", "info", str(FACE_MODEL)]) == 0

    captured = capsys.readouterr()
    assert "Inputs:" in captured.out
    assert "image: shape=(1, 256, 256, 3) dtype=float32 index=0" in captured.out
    assert "box_coords_1: shape=(1, 512, 16) dtype=float32" in captured.out


def test_model_info_command_reports_missing_model(capsys, tmp_path):
    exit_code = cli_module.main(["model", "info", str(tmp_path / "missing.tflite")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_MODEL" in captured.err
    assert "Model file not found" in captured.err


def test_operator_list_command(capsys):
    assert cli_module.main(["operator", "list"]) == 0

    captured = capsys.readouterr()
    assert "Operators:" in captured.out
    assert "ARITHMETIC_COMPOSE" in captured.out
    assert "creator=arithmetic" in captured.out


def test_operator_describe_command(capsys):
    assert cli_module.main(["operator", "describe-op", "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO"]) == 0

    captured = capsys.readouterr()
    assert "Operator: ARITHMETIC_COMPOSE" in captured.out
    assert "Creator: arithmetic" in captured.out
    assert "Input switches: --operand0" in captured.out
    assert "Output switches: --result" in captured.out


def test_operator_describe_command_reports_unknown(capsys):
    exit_code = cli_module.main(["operator", "describe-op", "missing"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_OPERATOR" in captured.err
    assert "Unknown operator" in captured.err


def test_visualize_model_delegates_to_litert(monkeypatch, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    subprocess_calls = []

    resolve_calls = []

    def _resolve_litert_cli(*, ensure=False, cache_dir=None):
        resolve_calls.append((ensure, cache_dir))
        return litert

    monkeypatch.setattr(cli_module, "resolve_litert_cli", _resolve_litert_cli)

    subprocess_envs = []

    def _run(argv, env=None):
        subprocess_calls.append(argv)
        subprocess_envs.append(env)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        _run,
    )

    exit_code = cli_module.main(["visualize", "model", "model.tflite", "--output", "model.html"])

    assert exit_code == 0
    assert resolve_calls == [(True, None)]
    assert subprocess_calls == [[str(litert.path), "visualize", "model.tflite", "--output", "model.html"]]
    assert subprocess_envs[0]["UV_SYSTEM_CERTS"] == "true"


def test_called_process_error_is_reported(monkeypatch, capsys, tmp_path):
    litert = LiteRTCli(path=tmp_path / "bin" / "litert", managed=False)
    monkeypatch.setattr(cli_module, "resolve_litert_cli", lambda ensure=False, cache_dir=None: litert)

    def _run(_argv, env=None):
        raise subprocess.CalledProcessError(
            4,
            "litert run",
            output="stdout message\n",
            stderr="stderr message\n",
        )

    monkeypatch.setattr(cli_module.subprocess, "run", _run)

    exit_code = cli_module.main(["model", "run", "model.tflite"])

    captured = capsys.readouterr()
    assert exit_code == 4
    assert "stdout message" in captured.out
    assert "stderr message" in captured.err


def test_pipeline_builder_commands_create_validate_and_inspect_pipeline(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "image",
            "--shape",
            "2,3",
            "--dtype",
            "uint8",
            "--input",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "image_f32",
            "--shape",
            "2,3",
            "--dtype",
            "float32",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "normalized",
            "--shape",
            "2,3",
            "--dtype",
            "float32",
            "--output",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
            "--src",
            "image",
            "--dst",
            "image_f32",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
            "--operand0",
            "image_f32",
            "--result",
            "normalized",
            "--expression",
            "{0} / 255.0",
        ]
    ) == 0
    assert cli_module.main(["pipeline", "set-input", str(pipeline), "image"]) == 0
    assert cli_module.main(["pipeline", "set-output", str(pipeline), "normalized"]) == 0
    assert cli_module.main(["pipeline", "validate", str(pipeline)]) == 0
    assert cli_module.main(["pipeline", "inspect", str(pipeline)]) == 0

    captured = capsys.readouterr()
    spec = json.loads(pipeline.read_text(encoding="utf-8"))
    assert spec["inputs"] == ["image"]
    assert spec["outputs"] == ["normalized"]
    assert len(spec["operators"]) == 2
    assert spec["operators"][1]["attrs"] == ["{0} / 255.0"]
    assert "Pipeline is valid" in captured.out
    assert "Operators: 2" in captured.out


def test_pipeline_add_op_arithmetic_requires_expression(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "x", "--shape", "2,2", "--dtype", "float32", "--input"]
    ) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "y", "--shape", "2,2", "--dtype", "float32", "--output"]
    ) == 0

    exit_code = cli_module.main(["pipeline", "add-op", str(pipeline), "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO", "--operand0", "x", "--result", "y"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_PIPELINE" in captured.err
    assert "Arithmetic operators require --expression" in captured.err


def test_pipeline_add_op_js_scripting_requires_code(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "x", "--shape", "2,2", "--dtype", "float32", "--input"]
    ) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "y", "--shape", "2,2", "--dtype", "float32", "--output"]
    ) == 0

    exit_code = cli_module.main(["pipeline", "add-op", str(pipeline), "XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO", "--x", "x", "--named-output", "y=y"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_PIPELINE" in captured.err
    assert "JS_SCRIPTING operators require script or attrs[0]" in captured.err


def test_pipeline_add_op_cvt_color_requires_input(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "y", "--shape", "2,2,3", "--dtype", "uint8", "--output"]
    ) == 0

    exit_code = cli_module.main(["pipeline", "add-op", str(pipeline), "XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO", "--dst", "y", "--flag", "4"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_PIPELINE" in captured.err
    assert "convert_color operators require exactly 1 input" in captured.err


def test_pipeline_add_op_named_inputs_are_written_in_operator_order(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    for name in ("left", "right", "result"):
        assert cli_module.main(
            [
                "pipeline", "add-tensor", str(pipeline), name,
                "--shape", "2,2", "--dtype", "float32",
            ]
        ) == 0

    assert cli_module.main(
        [
            "pipeline", "add-op", str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO",
            "--operand1", "right",
            "--result", "result",
            "--operand0", "left",
        ]
    ) == 0

    inputs = json.loads(pipeline.read_text(encoding="utf-8"))["operators"][0]["inputs"]
    assert inputs == [{"tensor": "left"}, {"tensor": "right"}]


def test_pipeline_add_op_named_outputs_are_written_in_operator_order(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    tensors = (
        ("objects", "4", "point", "3"),
        ("points", "4", "point", "2"),
        ("camera", "3,3", "matrix", None),
        ("rvec", "3,1", "matrix", None),
        ("tvec", "3,1", "matrix", None),
    )
    for name, shape, usage, channels in tensors:
        command = [
            "pipeline", "add-tensor", str(pipeline), name,
            "--shape", shape, "--dtype", "float64", "--usage", usage,
        ]
        if channels is not None:
            command.extend(["--channels", channels])
        assert cli_module.main(
            command
        ) == 0

    assert cli_module.main(
        [
            "pipeline", "add-op", str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO",
            "--translation", "tvec",
            "--camera-matrix", "camera",
            "--rotation", "rvec",
            "--image-points", "points",
            "--object-points", "objects",
        ]
    ) == 0

    outputs = json.loads(pipeline.read_text(encoding="utf-8"))["operators"][0]["outputs"]
    assert outputs == [
        {"tensor": "rvec"}, {"tensor": "tvec"}
    ]


def test_pipeline_add_op_rejects_generic_input_for_fixed_operator(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0

    exit_code = cli_module.main(
        [
            "pipeline", "add-op", str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
            "--input", "x", "--dst", "y",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--input is not an input or result for ASSIGNMENT" in captured.err
    assert "expected --src, --dst" in captured.err


def test_pipeline_commands_json_output(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert cli_module.main(["pipeline", "init", str(pipeline), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "pipeline.init"
    assert payload["pipeline"] == str(pipeline)

    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "x",
            "--shape",
            "2,2",
            "--dtype",
            "float32",
            "--input",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "pipeline.add_tensor"
    assert payload["tensor"] == "x"

    assert cli_module.main(["pipeline", "remove-tensor", str(pipeline), "x", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "pipeline.remove_tensor"
    assert payload["tensor"] == "x"
    assert payload["force"] is False

    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "x",
            "--shape",
            "2,2",
            "--dtype",
            "float32",
            "--input",
        ]
    ) == 0

    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "y",
            "--shape",
            "2,2",
            "--dtype",
            "float32",
            "--output",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
            "--src",
            "x",
            "--dst",
            "y",
        ]
    ) == 0
    capsys.readouterr()
    assert cli_module.main(["pipeline", "remove-op", str(pipeline), "--index", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "pipeline.remove_op"
    assert payload["index"] == 0

    assert cli_module.main(["pipeline", "inspect", str(pipeline), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "pipeline.inspect"
    assert payload["tensors"] == 2
    assert payload["operators"] == 0
    assert payload["inputs"] == ["x"]


def test_pipeline_add_op_model_writes_inline_litert_metadata(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "input",
            "--shape",
            "1,4",
            "--dtype",
            "float32",
            "--input",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "scores",
            "--shape",
            "1,2",
            "--dtype",
            "float32",
            "--output",
        ]
    ) == 0

    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
            "--input",
            "input",
            "--named-output",
            "scores=scores",
            "--model",
            "model/demo.tflite",
            "--model-name",
            "demo",
        ]
    ) == 0

    op = json.loads(pipeline.read_text(encoding="utf-8"))["operators"][0]
    assert op["model"]["bin_path"] == "model/demo.tflite"
    assert op["model"]["model_type"] == "tflite"
    assert "model_type" not in op
    assert "model_file" not in op
    assert "model_asset" not in op
    assert "model_id" not in op


def test_pipeline_add_op_accepts_spatial_only_canonical_types(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    assert cli_module.main(["pipeline", "init", str(pipeline)]) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "scene", "--shape", "1,1", "--dtype", "uint8", "--usage", "gltf"]
    ) == 0
    assert cli_module.main(
        ["pipeline", "add-tensor", str(pipeline), "scale", "--shape", "1,3", "--dtype", "float32"]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-tensor",
            str(pipeline),
            "visible",
            "--shape",
            "1",
            "--dtype",
            "int32",
            "--usage",
            "scalar",
        ]
    ) == 0

    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
            "--scenegraph",
            "scene",
            "--visible",
            "visible",
        ]
    ) == 0
    assert cli_module.main(
        [
            "pipeline",
            "add-op",
            str(pipeline),
            "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
            "--scenegraph",
            "scene",
            "--data",
            "scale",
            "--entity-path",
            "/target",
            "--property",
            "Transform.Scale",
        ]
    ) == 0

    operators = json.loads(pipeline.read_text(encoding="utf-8"))["operators"]
    assert operators[0]["type"] == "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO"
    assert "attrs" not in operators[0]
    assert operators[1]["type"] == "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO"
    assert operators[1]["attrs"] == ["/target:Transform.Scale"]


def test_pipeline_command_reports_builder_errors(capsys, tmp_path):
    missing = tmp_path / "missing.json"

    exit_code = cli_module.main(["pipeline", "validate", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_PIPELINE" in captured.err
    assert "Pipeline not found" in captured.err


def test_pipeline_trace_writes_pipeline_from_decorated_function(tmp_path):
    source = tmp_path / "trace_source.py"
    sample = tmp_path / "sample.npy"
    output = tmp_path / "traced.json"
    source.write_text(
        "\n".join(
            [
                "from securemr.py2smr import trace, ops",
                "@trace(inputs=['image'], outputs=['normalized'])",
                "def preprocess(image):",
                "    return ops.arithmetic(image, '{0} / 255.0', output_name='normalized')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))

    exit_code = cli_module.main(
        [
            "pipeline",
            "trace",
            str(source),
            "--function",
            "preprocess",
            "--input",
            f"image={sample}",
            "--output",
            str(output),
        ]
    )

    spec = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert spec["inputs"] == ["image"]
    assert spec["outputs"] == ["normalized"]
    assert len(spec["operators"]) == 1
    assert spec["operators"][0]["attrs"] == ["{0} / 255.0"]


def test_package_create_validate_and_inspect_commands(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                    "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                        "inputs": [{"tensor": "input"}],
                        "outputs": [{"tensor": "output"}],
                        "model": {
                            "bin_path": "face.tflite",
                            "model_name": "face",
                            "model_type": "tflite",
                            "model_target": "cpu",
                            "input": [{"name": "input", "shape": [1, 1], "encoding_type": "float32"}],
                            "output": [{"name": "output", "shape": [1, 1], "encoding_type": "float32"}],
                        },
                    }
                ],
                "inputs": ["input"],
                "outputs": ["output"],
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"model")

    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "face-demo",
            "--pipeline",
            f"main={pipeline}",
            "--supported-mode",
            "spatial",
            "--output",
            str(output),
        ]
    ) == 0
    assert cli_module.main(["package", "validate", str(output)]) == 0
    assert cli_module.main(["package", "inspect", str(output)]) == 0

    captured = capsys.readouterr()
    packaged_pipeline = json.loads((output / "pipeline" / "main.json").read_text(encoding="utf-8"))
    assert (output / "model" / "face.tflite").read_bytes() == b"model"
    assert packaged_pipeline["operators"][0]["model"]["bin_path"] == "model/face.tflite"
    assert "Package: face-demo" in captured.out
    assert "main -> pipeline/main.json" in captured.out


def test_package_create_from_existing_source_package(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    source = tmp_path / "source-package"
    archive = tmp_path / "source-package.zip"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                    "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                        "inputs": [{"tensor": "input"}],
                        "outputs": [{"tensor": "output"}],
                        "model": {
                            "bin_path": "face.tflite",
                            "model_name": "face",
                            "model_type": "tflite",
                            "model_target": "cpu",
                            "input": [{"name": "input", "shape": [1, 1], "encoding_type": "float32"}],
                            "output": [{"name": "output", "shape": [1, 1], "encoding_type": "float32"}],
                        },
                    }
                ],
                "inputs": ["input"],
                "outputs": ["output"],
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"model")

    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "face-demo",
            "--pipeline",
            f"main={pipeline}",
            "--output",
            str(source),
        ]
    ) == 0
    assert cli_module.main(
        [
            "package",
            "create",
            str(source),
            "--output",
            str(archive),
            "--yes",
        ]
    ) == 0
    assert cli_module.main(["package", "validate", str(archive)]) == 0


def test_package_inspect_json(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    package = tmp_path / "pkg"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                    "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                        "inputs": [{"tensor": "input"}],
                        "outputs": [{"tensor": "output"}],
                        "model": {
                            "bin_path": "face.tflite",
                            "model_name": "face",
                            "model_type": "tflite",
                            "model_target": "cpu",
                            "input": [{"name": "input", "shape": [1, 1], "encoding_type": "float32"}],
                            "output": [{"name": "output", "shape": [1, 1], "encoding_type": "float32"}],
                        },
                    }
                ],
                "inputs": ["input"],
                "outputs": ["output"],
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"model")

    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "face-demo",
            "--pipeline",
            f"main={pipeline}",
            "--output",
            str(package),
            "--force",
        ]
    ) == 0
    capsys.readouterr()

    assert cli_module.main(["package", "inspect", str(package), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "package.inspect"
    assert payload["manifest"]["id"] == "face-demo"
    assert payload["assets"] == ["model/face.tflite"]


def test_package_command_reports_errors(capsys, tmp_path):
    exit_code = cli_module.main(
        [
            "package",
            "create",
            "--id",
            "bad",
            "--pipeline",
            "main=missing.json",
            "--output",
            str(tmp_path / "pkg"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_PACKAGE" in captured.err
    assert "Pipeline file not found" in captured.err


def test_package_validate_wraps_manifest_errors(capsys, tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "id": "bad", "pipelines": []}),
        encoding="utf-8",
    )

    assert cli_module.main(["package", "validate", str(package)]) == 1

    captured = capsys.readouterr()
    assert "PSM_PACKAGE" in captured.err
    assert "schema_version must be 2" in captured.err


def test_package_validate_format_json_wraps_manifest_errors(capsys, tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "id": "bad", "pipelines": []}),
        encoding="utf-8",
    )

    assert cli_module.main(["package", "validate", str(package), "--format", "json"]) == 1

    payload = json.loads(capsys.readouterr().err)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "PSM_PACKAGE"
    assert "schema_version must be 2" in payload["error"]["message"]


def test_package_create_yes_overwrites_existing_output(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "input": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                    "output": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                        "inputs": [{"tensor": "input"}],
                        "outputs": [{"tensor": "output"}],
                        "model": {
                            "bin_path": "face.tflite",
                            "model_name": "face",
                            "model_type": "tflite",
                            "model_target": "cpu",
                            "input": [{"name": "input", "shape": [1, 1], "encoding_type": "float32"}],
                            "output": [{"name": "output", "shape": [1, 1], "encoding_type": "float32"}],
                        },
                    }
                ],
                "inputs": ["input"],
                "outputs": ["output"],
            }
        ),
        encoding="utf-8",
    )
    model.write_bytes(b"model")
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "face-demo",
            "--pipeline",
            f"main={pipeline}",
            "--output",
            str(output),
            "--yes",
        ]
    ) == 0

    assert not (output / "old.txt").exists()
    assert (output / "manifest.json").is_file()


def test_run_host_command_runs_pipeline_and_saves_outputs(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "x"}] + [None] * 9,
                        "outputs": [{"tensor": "y"}],
            "attrs": ["{0} + 3.0"],
                    }
                ],
                "inputs": ["x"],
                "outputs": ["y"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = _create_cli_run_package(tmp_path, pipeline)

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"x={sample}",
            "--output-dir",
            str(output_dir),
            "--duration",
            "2.5",
        ]
    ) == 0

    captured = capsys.readouterr()
    np.testing.assert_allclose(np.load(output_dir / "main" / "y.npy"), np.ones((2, 2), dtype=np.float32) + 3.0)
    assert "Outputs: 1" in captured.out
    assert "y: shape=(2, 2)" in captured.out


def test_run_host_command_json_wraps_summary(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "x"}] + [None] * 9,
                        "outputs": [{"tensor": "y"}],
                        "attrs": ["{0} + 3.0"],
                    }
                ],
                "inputs": ["x"],
                "outputs": ["y"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = _create_cli_run_package(tmp_path, pipeline)
    capsys.readouterr()

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"x={sample}",
            "--output-dir",
            str(output_dir),
            "--duration",
            "2.5",
            "--json",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "run.host"
    assert payload["target"] == str(package)
    assert payload["duration"] == 2.5
    assert "Host Run Summary" in payload["stdout"]


def test_run_host_command_runs_model_operator_with_litert(monkeypatch, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "demo.tflite"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    model.write_bytes(b"model")
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "scores": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                        "inputs": [{"name": "input", "tensor": "x"}],
                        "outputs": [{"name": "scores", "tensor": "scores"}],
                        "model": {
                            "bin_path": "demo.tflite",
                            "model_name": "demo",
                            "model_type": "tflite",
                            "model_target": "cpu",
                            "input": [{"name": "input", "shape": [2, 2], "encoding_type": "float32"}],
                            "output": [{"name": "scores", "shape": [2, 2], "encoding_type": "float32"}],
                        },
                    }
                ],
                "inputs": ["x"],
                "outputs": ["scores"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = _create_cli_run_package(tmp_path, pipeline)

    monkeypatch.setattr(cli_module.run_cli, "resolve_litert_cli", lambda ensure=True: object())

    def _run_litert_runtime_model(**kwargs):
        return {"scores": kwargs["inputs"]["input"] * 4.0}

    monkeypatch.setattr(cli_module.run_cli, "_run_litert_runtime_model", _run_litert_runtime_model)

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"x={sample}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    np.testing.assert_allclose(np.load(output_dir / "main" / "scores.npy"), np.ones((2, 2), dtype=np.float32) * 4.0)


def test_run_host_command_dump_all(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "x"}] + [None] * 9,
                        "outputs": [{"tensor": "y"}],
                        "attrs": ["{0} * 2.0"],
                    }
                ],
                "inputs": ["x"],
                "outputs": ["y"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = _create_cli_run_package(tmp_path, pipeline)

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"x={sample}",
            "--dump",
            "all",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert (output_dir / "main" / "all_tensors" / "x.npy").is_file()
    assert (output_dir / "main" / "all_tensors" / "y.npy").is_file()


def test_run_host_command_writes_display_summary(tmp_path):
    pipeline = tmp_path / "display.json"
    sample = tmp_path / "pose.npy"
    output_dir = tmp_path / "outputs"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "pose_in": {
                        "dimensions": [4, 4],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "frame_pose": {
                        "dimensions": [4, 4],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "frame_gltf": {
                        "dimensions": [1, 1],
                        "channels": 1,
                        "data_type": 1,
                        "is_gltf": True,
                        "asset": "gltf/frame.gltf",
                        "is_placeholder": True,
                        "usage": 7,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                        "inputs": [{"tensor": "pose_in"}],
                        "outputs": [{"tensor": "frame_pose"}],
                    }
                ],
                "inputs": ["pose_in"],
                "outputs": ["frame_pose", "frame_gltf"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    asset = tmp_path / "gltf" / "frame.gltf"
    asset.parent.mkdir()
    asset.write_text("{}", encoding="utf-8")
    pose = np.eye(4, dtype=np.float32)
    np.save(sample, pose)
    package = _create_cli_run_package(tmp_path, pipeline)

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"pose_in={sample}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert (output_dir / "main" / "display_summary.json").is_file()


def test_run_host_command_writes_post_det_json(tmp_path):
    pipeline = tmp_path / "detection.json"
    sample = tmp_path / "post_det_input.npy"
    output_dir = tmp_path / "outputs"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "post_det_input": {
                        "dimensions": [1, 21],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "post_det": {
                        "dimensions": [1, 21],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                        "inputs": [{"tensor": "post_det_input"}],
                        "outputs": [{"tensor": "post_det"}],
                    }
                ],
                "inputs": ["post_det_input"],
                "outputs": ["post_det"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    values = np.arange(21, dtype=np.float32).reshape(1, 21)
    np.save(sample, values)
    package = _create_cli_run_package(tmp_path, pipeline)

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--input",
            f"post_det_input={sample}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    decoded = json.loads((output_dir / "main" / "post_det.json").read_text(encoding="utf-8"))
    assert decoded["bbox"] == {"x1": 0.0, "y1": 1.0, "x2": 2.0, "y2": 3.0}
    assert decoded["keypoints"][0] == {"index": 0, "x": 6.0, "y": 7.0, "score": 8.0}


def test_run_host_command_runs_package_pipeline_chain(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sample = tmp_path / "x.npy"
    package = tmp_path / "pkg"
    output_dir = tmp_path / "outputs"
    first.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "x"}] + [None] * 9,
                        "outputs": [{"tensor": "y"}],
                        "attrs": ["{0} * 2.0"],
                    }
                ],
                "inputs": ["x"],
                "outputs": ["y"],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "tensors": {
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "z": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "y"}] + [None] * 9,
                        "outputs": [{"tensor": "z"}],
                        "attrs": ["{0} + 3.0"],
                    }
                ],
                "inputs": ["y"],
                "outputs": ["z"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "chain-demo",
            "--pipeline",
            f"first={first}",
            "--pipeline",
            f"second={second}",
            "--output",
            str(package),
        ]
    ) == 0

    assert cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--pipeline",
            "first",
            "--pipeline",
            "second",
            "--input",
            f"x={sample}",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    np.testing.assert_allclose(np.load(output_dir / "second" / "z.npy"), np.ones((2, 2), dtype=np.float32) * 5.0)


def test_run_host_command_rejects_duplicate_pipeline_ids(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    package = tmp_path / "pkg"
    pipeline.write_text(
        json.dumps(
            {
                "tensors": {
                    "x": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                    "y": {
                        "dimensions": [2, 2],
                        "channels": 1,
                        "data_type": 6,
                        "is_placeholder": True,
                        "usage": 6,
                    },
                },
                "operators": [
                    {
                        "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                        "inputs": [{"tensor": "x"}] + [None] * 9,
                        "outputs": [{"tensor": "y"}],
                        "attrs": ["{0}"],
                    }
                ],
                "inputs": ["x"],
                "outputs": ["y"],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    np.save(sample, np.ones((2, 2), dtype=np.float32))
    assert cli_module.main(
        [
            "package",
            "create",
            "--id",
            "dup-demo",
            "--pipeline",
            f"main={pipeline}",
            "--output",
            str(package),
        ]
    ) == 0

    exit_code = cli_module.main(
        [
            "run",
            "host",
            str(package),
            "--pipeline",
            "main",
            "--pipeline",
            "main",
            "--input",
            f"x={sample}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Duplicate pipeline id in run order: main" in captured.err


def test_run_host_command_reports_errors(capsys, tmp_path):
    exit_code = cli_module.main(["run", "host", str(tmp_path / "missing.json")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "PSM_RUN" in captured.err
    assert "Run targets must be a SpatialML pipeline package directory" in captured.err


def test_run_device_command_forwards_options(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    image = tmp_path / "face.jpg"
    image.write_bytes(b"jpg")
    output_dir = tmp_path / "outputs"
    apk = tmp_path / "runner.apk"
    apk.write_bytes(b"apk")
    calls = []

    def _run_device(*args, **kwargs):
        calls.append((args, kwargs))
        return 9

    monkeypatch.setattr(cli_module.run_cli, "run_device", _run_device)

    assert cli_module.main(
        [
            "run",
            "device",
            str(package),
            "--mode",
            "xr",
            "--input",
            str(image),
            "--input",
            f"vst_right_image={image}",
            "--pipeline",
            "detection",
            "--pipeline",
            "display",
            "--output-dir",
            str(output_dir),
            "--dump",
            "all",
            "--duration",
            "2.5",
            "--loop",
            "--keep-running",
            "--use-vst",
            "--backend",
            "gpu",
            "--interval-ms",
            "33",
            "--apk",
            str(apk),
            "--device",
            "serial-1",
        ]
    ) == 9

    args, kwargs = calls[0]
    assert args == (package,)
    assert kwargs["mode"] == "xr"
    assert kwargs["inputs"] == [str(image), f"vst_right_image={image}"]
    assert kwargs["pipeline_ids"] == ["detection", "display"]
    assert kwargs["output_dir"] == output_dir
    assert kwargs["dumps"] == ["all"]
    assert kwargs["duration"] == 2.5
    assert kwargs["loop"] is True
    assert kwargs["keep_running"] is True
    assert kwargs["use_vst"] is True
    assert kwargs["backend"] == "gpu"
    assert kwargs["interval_ms"] == 33
    assert kwargs["apk"] == apk
    assert kwargs["device"] == "serial-1"
