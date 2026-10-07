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

"""Host-side pipeline execution commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pyspatialml import litert_runtime
from pyspatialml.litert_tools import LiteRTToolError, resolve_litert_cli
from pyspatialml.pipeline_cli import PipelineCliError, _load_input_array
from pyspatialml.zip_utils import ZipSafetyError, safe_extract_zip
from securemr.py2smr.verifier import validate_pipeline_spec


class RunCliError(RuntimeError):
    """Raised when host run cannot be completed."""


@dataclass(frozen=True)
class PipelineRunTarget:
    """Resolved pipeline run target."""

    id: Optional[str]
    path: Path
    asset_root: Path


@dataclass(frozen=True)
class PackageContext:
    """Materialized package root plus optional cleanup path."""

    root: Path
    cleanup_root: Optional[Path] = None


@dataclass
class HostPipelineResult:
    """Host execution result for one pipeline."""

    run_target: PipelineRunTarget
    path: Path
    elapsed_ms: float
    declared_outputs: list[str]
    outputs: dict[str, np.ndarray]
    display_outputs: list[dict[str, Any]]
    detection_outputs: dict[str, dict[str, Any]]
    output_dir: Optional[Path]


_POST_DET_BBOX_VALUES = 4
_POST_DET_SCORE_VALUES = 1
_POST_DET_CLASS_VALUES = 1
_POST_DET_KEYPOINT_COUNT = 5
_POST_DET_KEYPOINT_VALUES = 3
_POST_DET_TENSOR_SIZE = (
    _POST_DET_BBOX_VALUES
    + _POST_DET_SCORE_VALUES
    + _POST_DET_CLASS_VALUES
    + _POST_DET_KEYPOINT_COUNT * _POST_DET_KEYPOINT_VALUES
)


_PACKAGE_TARGET_HINT = (
    "Run targets must be a SpatialML pipeline package directory containing "
    "manifest.json, or a .zip package containing manifest.json. The manifest "
    "must include a non-empty pipelines list with paths to pipeline JSON files. "
    "Use `pyspatialml package create` to create a package from pipeline JSON."
)


def run_host(
    target: Path,
    *,
    pipeline_ids: Sequence[str] = (),
    inputs: Sequence[str] = (),
    output_dir: Optional[Path] = None,
    dumps: Sequence[str] = (),
    duration: float = 15.0,
) -> int:
    """Run a pipeline package on the host Python executor."""
    _validate_host_run_args(duration=duration)
    run_targets, package_context = _resolve_pipeline_targets(target, pipeline_ids=pipeline_ids)
    input_values, default_input_path = _parse_host_inputs(inputs)
    try:
        total_start = time.perf_counter()
        _normalize_inputs_for_targets(input_values, run_targets)
        all_outputs: dict[str, np.ndarray] = {}
        results: list[HostPipelineResult] = []
        for index, run_target in enumerate(run_targets):
            pipeline_output_dir = _pipeline_output_dir(output_dir, run_target, len(run_targets))
            result = _run_one_pipeline(
                run_target,
                input_values=input_values,
                default_input_path=default_input_path,
                output_dir=pipeline_output_dir,
                dumps=dumps,
            )
            outputs = result.outputs
            input_values.update(outputs)
            all_outputs.update(outputs)
            results.append(result)
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        _print_host_summary(results, total_elapsed_ms)
        return 0
    finally:
        if package_context.cleanup_root is not None:
            shutil.rmtree(package_context.cleanup_root, ignore_errors=True)


def run_device(
    target: Path,
    *,
    mode: Optional[str] = None,
    input_path: Optional[str] = None,
    inputs: Sequence[str] = (),
    pipeline_ids: Sequence[str] = (),
    output_dir: Optional[Path] = None,
    dumps: Sequence[str] = (),
    duration: float = 15.0,
    loop: bool = False,
    keep_running: bool = False,
    use_vst: bool = False,
    backend: Optional[str] = None,
    interval_ms: int = 50,
    apk: Optional[Path] = None,
    device: Optional[str] = None,
    as_json: bool = False,
) -> int:
    """Run a pipeline package on a connected device through a runner APK."""
    pipeline_ids = _normalize_pipeline_ids(pipeline_ids)
    normalized_mode = _validate_run_package_target(target, mode=mode)
    _validate_device_run_args(
        inputs=inputs,
        dumps=dumps,
        duration=duration,
        interval_ms=interval_ms,
        apk=apk,
    )
    script = _device_runner_script(normalized_mode)
    if not script.is_file():
        raise RunCliError(f"{normalized_mode.upper()} runner script not found: {script}")

    cmd = [sys.executable, str(script), str(target)]
    all_inputs = list(inputs)
    if input_path:
        all_inputs.insert(0, input_path)
    for input_item in all_inputs:
        cmd.extend(["--input", input_item])
    for pipeline_id in pipeline_ids:
        cmd.extend(["--pipeline", pipeline_id])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    for dump in dumps:
        cmd.extend(["--dump", dump])
    cmd.extend(["--duration", str(duration)])
    if loop:
        cmd.append("--loop")
    if keep_running:
        cmd.append("--keep-running")
    if use_vst:
        cmd.append("--use-vst")
    if backend:
        cmd.extend(["--backend", backend])
    cmd.extend(["--interval-ms", str(interval_ms)])
    if apk is not None:
        cmd.extend(["--apk", str(apk)])
    if device:
        cmd.extend(["--device", device])

    env = _runner_subprocess_env()
    if as_json:
        result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        payload = {
            "ok": result.returncode == 0,
            "command": "run.device",
            "target": str(target),
            "mode": normalized_mode,
            "argv": cmd,
            "returncode": result.returncode,
            "pipelines": list(pipeline_ids),
            "output_dir": str(output_dir) if output_dir is not None else None,
            "dumps": list(dumps),
            "duration": duration,
            "loop": loop,
            "keep_running": keep_running,
            "use_vst": use_vst,
            "backend": backend,
            "device": device,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return result.returncode
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _runner_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_root if not existing else f"{package_root}{os.pathsep}{existing}"
    return env


def _normalize_device_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in {"xr", "spatial"}:
        raise RunCliError(f"--mode must be 'xr' or 'spatial', got: {mode}")
    return normalized


def _validate_run_package_target(target: Path, *, mode: Optional[str] = None) -> str:
    if target.is_file() and target.suffix.lower() == ".json":
        raise RunCliError(f"Raw pipeline JSON is not a valid run target: {target}. {_PACKAGE_TARGET_HINT}")
    if not target.exists():
        raise RunCliError(f"Target package not found: {target}. {_PACKAGE_TARGET_HINT}")
    package_context = _materialize_package(target)
    try:
        manifest = _load_run_manifest(package_context.root)
        normalized_mode = _normalize_device_mode(mode)
        resolved_mode = (
            _validate_manifest_supports_mode(manifest, normalized_mode)
            if normalized_mode is not None
            else _infer_device_mode_from_manifest(manifest)
        )
        _validate_manifest_pipeline_files(package_context.root, manifest)
        return resolved_mode
    finally:
        if package_context.cleanup_root is not None:
            shutil.rmtree(package_context.cleanup_root, ignore_errors=True)


def _infer_device_mode_from_manifest(manifest: Mapping[str, Any]) -> str:
    supported_modes = _manifest_supported_modes(manifest)
    if supported_modes is None:
        return "spatial"
    if "spatial" in supported_modes:
        return "spatial"
    if "xr" in supported_modes:
        return "xr"
    raise RunCliError(
        "Package manifest runtime.supported_modes must include 'spatial' or 'xr': "
        f"{', '.join(supported_modes) or '-'}"
    )


def _manifest_supported_modes(manifest: Mapping[str, Any]) -> Optional[list[str]]:
    runtime = manifest.get("runtime")
    supported_modes = runtime.get("supported_modes") if isinstance(runtime, Mapping) else None
    if supported_modes is None:
        return None
    if not isinstance(supported_modes, list):
        raise RunCliError("Package manifest runtime.supported_modes must be a list")
    return [str(item).strip().lower() for item in supported_modes]


def _validate_manifest_supports_mode(manifest: Mapping[str, Any], mode: str) -> str:
    supported_modes = _manifest_supported_modes(manifest)
    if supported_modes is None:
        return mode
    if mode not in supported_modes:
        raise RunCliError(
            f"Package manifest runtime.supported_modes does not include '{mode}': "
            f"{', '.join(str(item) for item in supported_modes) or '-'}"
        )
    return mode


def _validate_manifest_pipeline_files(root: Path, manifest: Mapping[str, Any]) -> None:
    for item in manifest.get("pipelines", []):
        if not isinstance(item, Mapping):
            raise RunCliError("Package manifest pipelines entries must be objects")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RunCliError("Package manifest pipeline entries require a non-empty path")
        path = _resolve_manifest_path(root, path_value, label="pipeline")
        if not path.is_file():
            raise RunCliError(f"Package manifest pipeline file not found: {path_value}. {_PACKAGE_TARGET_HINT}")
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            validate_pipeline_spec(spec)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RunCliError(f"Invalid pipeline '{path_value}': {exc}") from exc


def _resolve_manifest_path(root: Path, path_value: str, *, label: str) -> Path:
    """Resolve a manifest-relative path without allowing package escapes."""
    if not isinstance(path_value, str) or not path_value:
        raise RunCliError(f"Package manifest {label} entries require a non-empty path")

    normalized = path_value.replace("\\", "/")
    windows_path = PureWindowsPath(path_value)
    if (
        PurePosixPath(normalized).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise RunCliError(f"Package manifest {label} path must be package-relative: {path_value}")

    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RunCliError(f"Invalid package-relative {label} path: {path_value}")

    resolved_root = root.resolve()
    resolved_path = (root / PurePosixPath(*parts)).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RunCliError(
            f"Package {label} path escapes package root: {path_value}"
        ) from exc
    return resolved_path


def _validate_device_run_args(
    *,
    inputs: Sequence[str],
    dumps: Sequence[str],
    duration: float,
    interval_ms: int,
    apk: Optional[Path],
) -> None:
    if duration <= 0:
        raise RunCliError("--duration must be greater than 0")
    if interval_ms <= 0:
        raise RunCliError("--interval-ms must be greater than 0")
    unsupported_dumps = [item for item in dumps if str(item).lower() != "all"]
    if unsupported_dumps:
        raise RunCliError(f"Device run only supports --dump all, got: {', '.join(unsupported_dumps)}")
    if apk is not None and not apk.is_file():
        raise RunCliError(f"Runner APK not found: {apk}")
    for value in inputs:
        raw_path = value.split("=", 1)[1] if "=" in value else value
        if not raw_path:
            raise RunCliError("Input path cannot be empty")
        path = Path(raw_path)
        if not path.exists():
            raise RunCliError(f"Input file not found: {path}")


def _validate_host_run_args(*, duration: float) -> None:
    if duration <= 0:
        raise RunCliError("--duration must be greater than 0")


def print_run_error(exc: Exception) -> None:
    """Print a concise run command error."""
    print(f"Error [PSM_RUN]: {exc}", file=sys.stderr)


def _section_header(title: str) -> str:
    return f"========== {title} =========="


def _xr_runner_script() -> Path:
    return Path(__file__).resolve().parent / "xr_pipeline_runner" / "scripts" / "run_xr_pipeline.py"


def _spatial_runner_script() -> Path:
    return Path(__file__).resolve().parent / "spatial_pipeline_runner" / "scripts" / "run_spatial_pipeline.py"


def _device_runner_script(mode: str) -> Path:
    if mode == "spatial":
        return _spatial_runner_script()
    return _xr_runner_script()


def _run_one_pipeline(
    run_target: PipelineRunTarget,
    *,
    input_values: dict[str, np.ndarray],
    default_input_path: Optional[Path],
    output_dir: Optional[Path],
    dumps: Sequence[str],
) -> HostPipelineResult:
    from securemr.py2smr.verifier import run_pipeline_python

    spec = _read_json(run_target.path)
    spec = _normalize_run_pipeline_spec(spec)
    if default_input_path is not None:
        spec = _apply_default_image_input(spec, default_input_path, input_values)
    model_runner = _LiteRTModelRunner(asset_root=run_target.asset_root)
    pipeline_start = time.perf_counter()
    all_tensors = run_pipeline_python(
        spec,
        dict(input_values),
        return_all_tensors=True,
        model_runner=model_runner,
    )
    pipeline_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000.0
    outputs = {name: all_tensors[name] for name in spec.get("outputs", []) if name in all_tensors}
    dumped_tensors = _select_dump_tensors(all_tensors, outputs, dumps)

    declared_outputs = list(spec.get("outputs", []))
    display_outputs = _display_output_summary(spec, outputs)
    detection_outputs = _detection_output_summary(outputs)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            safe_name = _safe_filename(name)
            np.save(output_dir / f"{safe_name}.npy", np.asarray(value))
            detection = detection_outputs.get(name)
            if detection is not None:
                _write_json(output_dir / f"{safe_name}.json", detection)
        if dumped_tensors:
            dump_dir = output_dir / ("all_tensors" if _dump_all(dumps) else "dumped")
            dump_dir.mkdir(parents=True, exist_ok=True)
            for name, value in dumped_tensors.items():
                np.save(dump_dir / f"{_safe_filename(name)}.npy", np.asarray(value))
            print(f"Saved dumped tensors: {dump_dir}")
        if display_outputs:
            _write_display_summary(output_dir / "display_summary.json", run_target.path, display_outputs)
    return HostPipelineResult(
        run_target=run_target,
        path=run_target.path,
        elapsed_ms=pipeline_elapsed_ms,
        declared_outputs=declared_outputs,
        outputs=outputs,
        display_outputs=display_outputs,
        detection_outputs=detection_outputs,
        output_dir=output_dir,
    )


def _print_host_summary(results: Sequence[HostPipelineResult], total_elapsed_ms: float) -> None:
    print(_section_header("Host Run Summary"))
    pipelines = ", ".join(result.run_target.id or result.path.stem for result in results) or "-"
    print(f"  Pipelines: {pipelines}")
    print(f"  Total host run time: {total_elapsed_ms:.2f} ms")
    for result in results:
        pipeline_id = result.run_target.id or result.path.stem
        print(f"  {pipeline_id}: {result.elapsed_ms:.2f} ms")
    print(_section_header("Host Outputs"))
    total_outputs = sum(len(result.declared_outputs) for result in results)
    print(f"  Outputs: {total_outputs}")
    for result in results:
        pipeline_id = result.run_target.id or result.path.stem
        print(f"  Pipeline: {pipeline_id}")
        print(f"    Path: {result.path}")
        if len(result.outputs) != len(result.declared_outputs):
            print(f"    Tensor outputs: {len(result.outputs)}")
        for name, value in result.outputs.items():
            array = np.asarray(value)
            print(_tensor_summary(name, array, indent="    "))
            detection = result.detection_outputs.get(name)
            if detection is not None:
                print(_format_detection_summary(name, detection, indent="    "))
        if result.display_outputs:
            print("    Host note: spatial display outputs are not rendered on host.")
            for item in result.display_outputs:
                if item["kind"] == "gltf":
                    exists = "yes" if item.get("exists") else "no"
                    print(f"      {item['name']}: asset reference {item['asset']} exists={exists}")
                elif item["kind"] == "pose":
                    translation = item.get("translation")
                    if translation is not None:
                        print(f"      {item['name']}: translation={translation}")
        if result.output_dir is not None:
            print(f"    Saved outputs: {result.output_dir}")
    print(_section_header("End Host Outputs"))


def _normalize_pipeline_ids(pipeline_ids: Sequence[str]) -> list[str]:
    return [pipeline_id.strip() for pipeline_id in pipeline_ids if pipeline_id.strip()]


def _resolve_pipeline_targets(target: Path, *, pipeline_ids: Sequence[str]) -> tuple[list[PipelineRunTarget], PackageContext]:
    if target.is_file() and target.suffix.lower() == ".json":
        raise RunCliError(f"Raw pipeline JSON is not a valid run target: {target}. {_PACKAGE_TARGET_HINT}")
    package_context = _materialize_package(target)
    root = package_context.root
    manifest = _load_run_manifest(root)
    pipelines = manifest.get("pipelines", [])
    if not pipelines:
        raise RunCliError(f"Package has no pipelines: {target}")
    selected_items = []
    if pipeline_ids:
        _ensure_unique_pipeline_ids(pipeline_ids)
        for pipeline_id in pipeline_ids:
            selected = None
            for item in pipelines:
                if item.get("id") == pipeline_id:
                    selected = item
                    break
            if selected is None:
                available = ", ".join(str(item.get("id")) for item in pipelines)
                raise RunCliError(f"Pipeline '{pipeline_id}' not found. Available: {available}")
            selected_items.append(selected)
    else:
        selected_items = list(pipelines)

    run_targets = []
    for selected in selected_items:
        path = _resolve_manifest_path(root, selected.get("path"), label="pipeline")
        if not path.is_file():
            raise RunCliError(f"Package pipeline file not found: {selected['path']}")
        run_targets.append(
            PipelineRunTarget(
                id=selected.get("id"),
                path=path,
                asset_root=root,
            )
        )
    return run_targets, package_context


def _ensure_unique_pipeline_ids(pipeline_ids: Sequence[str]) -> None:
    seen = set()
    duplicates = []
    for pipeline_id in pipeline_ids:
        if pipeline_id in seen and pipeline_id not in duplicates:
            duplicates.append(pipeline_id)
        seen.add(pipeline_id)
    if duplicates:
        raise RunCliError(f"Duplicate pipeline id in run order: {', '.join(duplicates)}")


def _pipeline_output_dir(
    output_dir: Optional[Path],
    run_target: PipelineRunTarget,
    target_count: int,
) -> Optional[Path]:
    if output_dir is None:
        return None
    if run_target.id:
        return output_dir / _safe_filename(run_target.id)
    if target_count <= 1:
        return output_dir
    return output_dir / _safe_filename(run_target.id or run_target.path.stem)


def _normalize_inputs_for_targets(input_values: dict[str, np.ndarray], run_targets: Sequence[PipelineRunTarget]) -> None:
    for run_target in run_targets:
        spec = _read_json(run_target.path)
        tensor_specs = spec.get("tensors", {})
        if not isinstance(tensor_specs, Mapping):
            continue
        for name, value in list(input_values.items()):
            tensor_spec = tensor_specs.get(name)
            if not isinstance(tensor_spec, Mapping):
                continue
            target_shape = _tensor_spec_shape(tensor_spec)
            if target_shape is None:
                continue
            input_values[name] = _normalize_input_array(value, target_shape)


def _tensor_spec_shape(tensor_spec: Mapping[str, Any]) -> Optional[tuple[int, ...]]:
    dims = tensor_spec.get("dimensions")
    if not isinstance(dims, list):
        return None
    channels = int(tensor_spec.get("channels", 1) or 1)
    if len(dims) >= 2:
        height = int(dims[0])
        width = int(dims[1])
        return (height, width, channels) if channels > 1 else (height, width)
    if len(dims) == 1:
        return (int(dims[0]), channels) if channels > 1 else (int(dims[0]),)
    return None


def _normalize_input_array(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == target_shape:
        return array
    if array.ndim == 3 and len(target_shape) == 3 and array.shape[2] == target_shape[2]:
        try:
            import cv2
            resized = cv2.resize(array, (int(target_shape[1]), int(target_shape[0])), interpolation=cv2.INTER_LINEAR)
            return resized.astype(array.dtype, copy=False)
        except Exception:
            return array
    if array.ndim > 2 and array.size == int(np.prod(target_shape)):
        return array.reshape(target_shape)
    return array


def _materialize_package(path: Path) -> PackageContext:
    if path.is_dir():
        return PackageContext(root=_find_package_root(path))
    if path.is_file() and path.suffix.lower() == ".zip":
        cleanup_root = Path(tempfile.mkdtemp(prefix="pyspatialml-run-"))
        with zipfile.ZipFile(path) as archive:
            try:
                safe_extract_zip(archive, cleanup_root)
            except ZipSafetyError as exc:
                raise RunCliError(str(exc)) from exc
        return PackageContext(root=_find_package_root(cleanup_root), cleanup_root=cleanup_root)
    raise RunCliError(f"Target is not a SpatialML pipeline package directory or .zip package: {path}. {_PACKAGE_TARGET_HINT}")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise RunCliError(f"Pipeline JSON must contain an object: {path}")
    return payload


def _normalize_run_pipeline_spec(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(spec))
    _force_host_model_backend_cpu(normalized)
    return normalized


def _force_host_model_backend_cpu(spec: dict[str, Any]) -> None:
    operators = spec.get("operators", [])
    if not isinstance(operators, list):
        return
    for op in operators:
        if not isinstance(op, dict):
            continue
        op_type = op.get("type")
        if op_type != "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO":
            continue
        model = op.get("model")
        if isinstance(model, dict):
            model["model_target"] = "cpu"


def _apply_default_image_input(
    spec: Mapping[str, Any],
    image_path: Path,
    input_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(spec))
    applied = False
    for op in normalized.get("operators", []):
        if not isinstance(op, dict):
            continue
        op_type = str(op.get("type") or "").upper()
        if "RECTIFIED_VST_ACCESS" in op_type:
            try:
                image = _load_input_array(image_path)
            except PipelineCliError as exc:
                raise RunCliError(str(exc)) from exc
            output_names = [
                ref.get("tensor")
                for ref in op.get("outputs", [])
                if isinstance(ref, dict) and isinstance(ref.get("tensor"), str)
            ]
            for name in output_names[:2]:
                tensor_spec = normalized.get("tensors", {}).get(name)
                if isinstance(tensor_spec, Mapping):
                    dimensions = tensor_spec.get("dimensions")
                    channels = int(tensor_spec.get("channels", 1) or 1)
                    if isinstance(dimensions, list) and len(dimensions) >= 2:
                        target_shape = (int(dimensions[0]), int(dimensions[1]))
                        if channels > 1:
                            target_shape += (channels,)
                        image = _normalize_input_array(image, target_shape)
                input_values.setdefault(name, image)
            applied = True
    if not applied:
        targets = _default_image_tensor_names(normalized)
        if not targets:
            targets = _declared_input_tensor_names(normalized)
        if not targets:
            raise RunCliError("Bare --input path could not be applied: no declared input tensors found")
        try:
            image = _load_input_array(image_path)
        except PipelineCliError as exc:
            raise RunCliError(str(exc)) from exc
        for name in targets:
            input_values.setdefault(name, image)
    return normalized


def _load_run_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RunCliError(f"Package manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if "id" not in manifest or "pipelines" not in manifest:
        raise RunCliError("Package manifest requires id and pipelines")
    if str(manifest.get("schema_version", "")) != "2":
        raise RunCliError("Package manifest schema_version must be 2")
    if "model" in manifest or "models" in manifest:
        raise RunCliError("Package manifest must not contain model/models; v2 stores model metadata inline")
    if not isinstance(manifest["pipelines"], list) or not manifest["pipelines"]:
        raise RunCliError("Package manifest requires a non-empty pipelines list")
    return manifest


def _parse_host_inputs(values: Sequence[str]) -> tuple[dict[str, np.ndarray], Optional[Path]]:
    input_values: dict[str, np.ndarray] = {}
    default_input_path: Optional[Path] = None
    for value in values:
        if "=" in value:
            name, array = _parse_input_arg(value)
            input_values[name] = array
            continue
        if default_input_path is not None:
            raise RunCliError("Only one bare --input path is supported; use tensor=path for explicit inputs")
        path = Path(value)
        if not path.exists():
            raise RunCliError(f"Input file not found: {path}")
        default_input_path = path
    return input_values, default_input_path


def _default_image_tensor_names(spec: Mapping[str, Any]) -> list[str]:
    tensors = spec.get("tensors", {})
    if not isinstance(tensors, Mapping):
        return []
    result = []
    for name, tensor_spec in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor_spec, Mapping):
            continue
        lowered = name.lower()
        if "image" not in lowered:
            continue
        if lowered.startswith("vst_") or "vst" in lowered:
            result.append(name)
    return result


def _declared_input_tensor_names(spec: Mapping[str, Any]) -> list[str]:
    tensors = spec.get("tensors", {})
    if not isinstance(tensors, Mapping):
        return []
    inputs = spec.get("inputs", [])
    if not isinstance(inputs, list):
        return []
    result = []
    for name in inputs:
        if isinstance(name, str) and isinstance(tensors.get(name), Mapping):
            result.append(name)
    return result


def _parse_input_arg(value: str) -> tuple[str, np.ndarray]:
    if "=" not in value:
        raise RunCliError("--input must use name=path format")
    name, raw_path = value.split("=", 1)
    if not name:
        raise RunCliError("Input name cannot be empty")
    path = Path(raw_path)
    try:
        return name, _load_input_array(path)
    except PipelineCliError as exc:
        raise RunCliError(str(exc)) from exc


class _LiteRTModelRunner:
    """Adapter that runs pipeline model operators through LiteRT runtime."""

    def __init__(self, *, asset_root: Path):
        self.asset_root = asset_root
        self._litert = None

    def __call__(
        self,
        *,
        inputs: Mapping[str, np.ndarray],
        model_file: str,
        model_name: str,
        output_names: Sequence[str],
        output_shapes: Optional[Sequence[tuple]] = None,
        output_dtypes: Optional[Sequence[Any]] = None,
        input_aliasing: Optional[Mapping[str, str]] = None,
        output_aliasing: Optional[Mapping[str, str]] = None,
        model: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        model_path = _resolve_model_path(model_file, asset_root=self.asset_root)
        try:
            self._litert = self._litert or resolve_litert_cli(ensure=True)
        except LiteRTToolError as exc:
            raise RunCliError(str(exc)) from exc

        try:
            start = time.perf_counter()
            outputs = _run_litert_runtime_model(
                model_path=model_path,
                inputs=inputs,
                output_names=output_names,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"RUN_MODEL_INFERENCE {model_name}: {elapsed_ms:.2f} ms target=cpu model={model_path}")
            return outputs
        except RunCliError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RunCliError(f"LiteRT model run failed for {model_name}: {exc}") from exc


def _resolve_model_path(model_file: str, *, asset_root: Path) -> Path:
    path = Path(model_file)
    candidates = [path] if path.is_absolute() else [asset_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise RunCliError(f"Model file not found: {model_file}\nSearched:\n  {searched}")


def _run_litert_runtime_model(
    *,
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    output_shapes: Optional[Sequence[tuple]] = None,
    output_dtypes: Optional[Sequence[Any]] = None,
    litert_cli=None,
) -> dict[str, np.ndarray]:
    try:
        return litert_runtime.run_model(
            model_path=model_path,
            inputs=inputs,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            litert_cli=litert_cli,
        )
    except litert_runtime.LiteRTRuntimeError as exc:
        raise RunCliError(str(exc)) from exc


def _select_dump_tensors(
    all_tensors: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    dumps: Sequence[str],
) -> dict[str, np.ndarray]:
    if not dumps:
        return {}
    if _dump_all(dumps):
        return dict(all_tensors)
    selected = {}
    missing = []
    for name in dumps:
        if name in all_tensors:
            if name not in outputs:
                selected[name] = all_tensors[name]
        else:
            missing.append(name)
    if missing:
        raise RunCliError(f"Requested dump tensor not found: {', '.join(missing)}")
    return selected


def _dump_all(dumps: Sequence[str]) -> bool:
    return any(str(item).lower() == "all" for item in dumps)


def _tensor_summary(name: str, array: np.ndarray, *, indent: str = "  ") -> str:
    preview_indent = indent + "  "
    summary = f"{indent}{name}: shape={tuple(array.shape)} dtype={array.dtype}"
    if array.size:
        if np.issubdtype(array.dtype, np.number):
            summary += (
                f" min={float(np.min(array)):.6g}"
                f" max={float(np.max(array)):.6g}"
                f" mean={float(np.mean(array)):.6g}"
            )
            if np.all(array == 0):
                summary += " all_zero=true"
            summary += f"\n{preview_indent}preview={_tensor_preview(array)}"
        else:
            summary += f" values={array.size}"
            summary += f"\n{preview_indent}preview={_tensor_preview(array)}"
    return summary


def _tensor_preview(array: np.ndarray, limit: int = 8) -> str:
    flat = np.asarray(array).reshape(-1)
    values = flat[:limit].tolist()
    result = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            result.append(f"{value:.6g}")
        else:
            result.append(str(value))
    if flat.size > limit:
        result.append("...")
    return "[" + ", ".join(result) + "]"


def _detection_output_summary(outputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for name, value in outputs.items():
        array = np.asarray(value)
        if name == "post_det" and array.size == _POST_DET_TENSOR_SIZE:
            result[name] = _decode_post_det(array)
    return result


def _decode_post_det(array: np.ndarray) -> dict[str, Any]:
    """Decode [bbox(4), score, class_id, 5 keypoints(x, y, score)]."""
    values = [float(value) for value in np.asarray(array).reshape(-1)]
    keypoints = []
    keypoint_start = _POST_DET_BBOX_VALUES + _POST_DET_SCORE_VALUES + _POST_DET_CLASS_VALUES
    for index in range(_POST_DET_KEYPOINT_COUNT):
        base = keypoint_start + index * _POST_DET_KEYPOINT_VALUES
        keypoints.append(
            {
                "index": index,
                "x": values[base],
                "y": values[base + 1],
                "score": values[base + 2],
            }
        )
    return {
        "bbox": {
            "x1": values[0],
            "y1": values[1],
            "x2": values[2],
            "y2": values[3],
        },
        "score": values[4],
        "class_id": values[5],
        "keypoints": keypoints,
    }


def _format_detection_summary(name: str, detection: Mapping[str, Any], *, indent: str = "  ") -> str:
    bbox = detection["bbox"]
    lines = [
        f"{indent}{name} decoded:",
        f"{indent}  bbox:",
        f"{indent}    x1: {_format_float(bbox['x1'])}",
        f"{indent}    y1: {_format_float(bbox['y1'])}",
        f"{indent}    x2: {_format_float(bbox['x2'])}",
        f"{indent}    y2: {_format_float(bbox['y2'])}",
        f"{indent}  score: {_format_float(detection['score'])}",
        f"{indent}  class_id: {_format_float(detection['class_id'])}",
        f"{indent}  keypoints:",
    ]
    for keypoint in detection["keypoints"]:
        lines.extend(
            [
                f"{indent}    {keypoint['index']}:",
                f"{indent}      x: {_format_float(keypoint['x'])}",
                f"{indent}      y: {_format_float(keypoint['y'])}",
                f"{indent}      score: {_format_float(keypoint['score'])}",
            ]
        )
    return "\n".join(lines)


def _format_float(value: Any) -> str:
    return f"{float(value):.6g}"


def _display_output_summary(spec: Mapping[str, Any], outputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    tensors = spec.get("tensors", {})
    if not isinstance(tensors, Mapping):
        return []
    result: list[dict[str, Any]] = []
    for name in spec.get("outputs", []):
        tensor_spec = tensors.get(name, {})
        if not isinstance(tensor_spec, Mapping):
            tensor_spec = {}
        tensor_type = str(tensor_spec.get("tensor_type") or tensor_spec.get("type") or "").lower()
        asset = tensor_spec.get("asset")
        if tensor_type == "gltf" or (isinstance(asset, str) and asset):
            result.append(
                {
                    "name": name,
                    "kind": "gltf",
                    "asset": asset,
                    "exists": isinstance(asset, str) and bool(asset),
                }
            )
            continue
        if name in outputs and _looks_like_pose(name, np.asarray(outputs[name])):
            array = np.asarray(outputs[name])
            item: dict[str, Any] = {
                "name": name,
                "kind": "pose",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            translation = _pose_translation(array)
            if translation is not None:
                item["translation"] = translation
            result.append(item)
    return result


def _looks_like_pose(name: str, array: np.ndarray) -> bool:
    lowered = name.lower()
    return ("pose" in lowered or "transform" in lowered) and array.shape in {(4, 4), (3, 4)}


def _pose_translation(array: np.ndarray) -> Optional[list[float]]:
    if array.shape == (4, 4):
        return [float(value) for value in array[:3, 3]]
    if array.shape == (3, 4):
        return [float(value) for value in array[:3, 3]]
    return None


def _write_display_summary(path: Path, pipeline_path: Path, display_outputs: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "pipeline": str(pipeline_path),
        "outputs": list(display_outputs),
        "host_note": "Host mode does not render spatial glTF output.",
    }
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _safe_filename(name: str) -> str:
    safe = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "output"


def _find_package_root(root: Path) -> Path:
    if (root / "manifest.json").is_file():
        return root
    candidates = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith("__MACOSX")]
    manifest_dirs = [path for path in candidates if (path / "manifest.json").is_file()]
    if len(manifest_dirs) == 1:
        return manifest_dirs[0]
    recursive_manifest_dirs = [
        path.parent
        for path in root.rglob("manifest.json")
        if "__MACOSX" not in path.parts and not any(part.startswith("._") for part in path.parts)
    ]
    if len(recursive_manifest_dirs) == 1:
        return recursive_manifest_dirs[0]
    return root
