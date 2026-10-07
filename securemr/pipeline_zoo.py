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
"""Helpers for authoring SpatialML packages."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
SUPPORTED_EXECUTION_MODES = {"xr", "spatial"}


@dataclass(frozen=True)
class PipelinePackageEntry:
    """Manifest entry for one pipeline JSON file in a SpatialML package."""

    id: str
    path: str

    def to_dict(self) -> JsonDict:
        """Return the pipeline zoo representation."""
        return {"id": self.id, "path": _normalize_package_path(self.path)}


@dataclass(frozen=True)
class PipelineZooPackageSpec:
    """Top-level SpatialML manifest schema."""

    package_id: str
    pipelines: Sequence[PipelinePackageEntry]
    supported_modes: Sequence[str] = field(default_factory=tuple)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> JsonDict:
        """Return a manifest dictionary matching the SpatialML package schema."""
        manifest: JsonDict = {"schema_version": self.schema_version, "id": self.package_id}
        pipeline_ids = []
        for entry in self.pipelines:
            if any(entry.id == pipeline_id for pipeline_id in pipeline_ids):
                raise ValueError(f"Duplicate pipeline id: {entry.id}")
            pipeline_ids.append(entry.id)
        manifest["pipelines"] = [entry.to_dict() for entry in self.pipelines]
        runtime = dict(self.runtime)
        if self.supported_modes:
            runtime["supported_modes"] = _normalize_supported_modes(self.supported_modes)
        elif "supported_modes" in runtime:
            runtime["supported_modes"] = _normalize_supported_modes(runtime["supported_modes"])
        else:
            runtime["supported_modes"] = ["xr", "spatial"]
        manifest["runtime"] = runtime
        if self.metadata:
            manifest["metadata"] = dict(self.metadata)
        return manifest


def create_litert_model_spec(
    model_path: str,
    model_name: str,
    *,
    model_target: str = "npu",
    input_tensors: Sequence[Mapping[str, Any]],
    output_tensors: Sequence[Mapping[str, Any]],
    cpu_target_num_threads: int = 1,
) -> JsonDict:
    """Create an inline LiteRT/TFLite model spec for a model inference operator.

    Args:
        model_path: Package-relative path to the ``.tflite`` model file.
        model_name: Logical model name referenced by pipeline inference operators.
        model_target: Runtime target requested by the package (``npu`` by default).
        input_tensors: Required model input tensor metadata. Each entry must
            contain ``name``, ``shape``, and ``encoding_type``.
        output_tensors: Required model output tensor metadata. Each entry must
            contain ``name``, ``shape``, and ``encoding_type``.
        cpu_target_num_threads: Number of CPU threads when ``model_target`` selects CPU.

    Returns:
        A dictionary ready to place under a ``run_algorithm`` operator's ``model`` key.
    """
    _validate_model_io_metadata(input_tensors, field="input")
    _validate_model_io_metadata(output_tensors, field="output")
    _validate_model_name(model_name)
    if not isinstance(model_target, str):
        raise ValueError("model_target must be a string")
    normalized_target = model_target.lower()
    if normalized_target not in {"cpu", "gpu", "npu"}:
        raise ValueError("model_target must be cpu, gpu, or npu")
    model_spec: JsonDict = {
        "bin_path": _normalize_package_path(model_path),
        "model_name": model_name,
        "model_type": "tflite",
        "model_target": normalized_target,
        "input": [dict(tensor) for tensor in input_tensors],
        "output": [dict(tensor) for tensor in output_tensors],
    }
    if normalized_target == "cpu":
        model_spec["cpu_target_num_threads"] = int(cpu_target_num_threads)
    return model_spec


def configure_litert_inference_operator(
    operator_spec: JsonDict,
    *,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model: Optional[Mapping[str, Any]] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
    input_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    output_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
) -> JsonDict:
    """Return an inference operator spec with inline LiteRT/TFLite model metadata."""
    result = dict(operator_spec)
    if model is not None and model_path is not None:
        raise ValueError("Specify either model or model_path for an inference operator, not both")
    if model is not None:
        result["model"] = dict(model)
        _validate_inline_model_metadata(result["model"])
        result["model"]["bin_path"] = _normalize_package_path(result["model"]["bin_path"])
        result["model"]["model_type"] = result["model"]["model_type"].lower()
        result["model"]["model_target"] = result["model"]["model_target"].lower()
        if result["model"]["model_target"] == "cpu":
            result["model"].setdefault("cpu_target_num_threads", int(cpu_target_num_threads))
        else:
            result["model"].pop("cpu_target_num_threads", None)
    elif model_path is not None:
        result["model"] = create_litert_model_spec(
            model_path,
            model_name or "main",
            model_target=model_target,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            cpu_target_num_threads=cpu_target_num_threads,
        )
    else:
        raise ValueError("A run_algorithm operator requires inline model metadata")
    if model_name is not None:
        result["model"].setdefault("model_name", model_name)
    result.pop("model_asset", None)
    result.pop("model_file", None)
    result.pop("model_id", None)
    result.pop("bin_path", None)
    return result


def _validate_model_io_metadata(
    entries: Optional[Sequence[Mapping[str, Any]]], *, field: str
) -> None:
    if entries is None:
        raise ValueError(f"Model metadata requires {field} tensor metadata")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError(f"Model metadata {field} must be an array")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Model metadata {field}[{index}] must be an object")
        missing = [key for key in ("name", "shape", "encoding_type") if key not in entry]
        if missing:
            raise ValueError(
                f"Model metadata {field}[{index}] missing required keys: {', '.join(missing)}"
            )
        if not isinstance(entry["name"], str) or not entry["name"]:
            raise ValueError(f"Model metadata {field}[{index}].name must be a non-empty string")
        shape = entry["shape"]
        if (not isinstance(shape, list) or not shape or
                any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
                    for dimension in shape)):
            raise ValueError(f"Model metadata {field}[{index}].shape must be an array of positive integers")
        if not isinstance(entry["encoding_type"], str) or not entry["encoding_type"]:
            raise ValueError(f"Model metadata {field}[{index}].encoding_type must be a non-empty string")


def _validate_inline_model_metadata(model: Mapping[str, Any]) -> None:
    required = ("bin_path", "model_type", "model_target", "input", "output")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"Inline model metadata missing required keys: {', '.join(missing)}")
    if not isinstance(model["bin_path"], str) or not model["bin_path"]:
        raise ValueError("Inline model metadata bin_path must be a non-empty string")
    if not isinstance(model["model_type"], str) or model["model_type"].lower() != "tflite":
        raise ValueError("Inline model metadata model_type must be 'tflite'")
    if not isinstance(model["model_target"], str) or model["model_target"].lower() not in {"cpu", "gpu", "npu"}:
        raise ValueError("Inline model metadata model_target must be cpu, gpu, or npu")
    _validate_model_name(model.get("model_name", "main"))
    _validate_model_io_metadata(model["input"], field="input")
    _validate_model_io_metadata(model["output"], field="output")


def _validate_model_name(model_name: Any) -> None:
    if not isinstance(model_name, str) or re.fullmatch(r"[A-Za-z0-9_]+", model_name) is None:
        raise ValueError("model_name must contain only letters, digits, and underscores")


def write_pipeline_zoo_package(
    output_dir: PathLike,
    package: PipelineZooPackageSpec,
    *,
    pipelines: Mapping[str, Union[JsonDict, PathLike]],
    assets: Optional[Mapping[str, PathLike]] = None,
    indent: int = 2,
) -> JsonDict:
    """Write a SpatialML package directory.

    The generated layout follows the package schema: ``manifest.json`` at the
    package root, package-relative pipeline paths, and binary assets copied
    without rewriting their package paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = package.to_manifest_dict()
    _write_json(root / "manifest.json", manifest, indent=indent)

    for entry in package.pipelines:
        source = pipelines.get(entry.id)
        if source is None:
            source = pipelines.get(entry.path)
        if source is None:
            raise KeyError(f"Missing pipeline content for '{entry.id}' ({entry.path})")
        _write_json_or_copy(root / entry.path, source, indent=indent)

    for package_path, source_path in (assets or {}).items():
        destination = _package_destination(root, package_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    return manifest


def load_pipeline_zoo_manifest(path: PathLike) -> JsonDict:
    """Load and minimally validate a SpatialML ``manifest.json`` file."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    validate_pipeline_zoo_manifest(manifest)
    return manifest


def validate_pipeline_zoo_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the manifest fields required by the SpatialML package schema."""
    if not isinstance(manifest, Mapping):
        raise ValueError("SpatialML manifest must be an object")
    required = ["id", "pipelines"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"SpatialML manifest missing required fields: {', '.join(missing)}")
    if manifest.get("schema_version") != "2":
        raise ValueError("SpatialML manifest schema_version must be 2")
    if not isinstance(manifest["id"], str) or not manifest["id"]:
        raise ValueError("SpatialML manifest id must be a non-empty string")
    if "model" in manifest or "models" in manifest:
        raise ValueError("SpatialML manifest must not contain model/models; v2 stores model metadata inline")
    if not isinstance(manifest["pipelines"], list) or not manifest["pipelines"]:
        raise ValueError("SpatialML manifest requires a non-empty 'pipelines' list")
    pipeline_ids = []
    for index, pipeline in enumerate(manifest["pipelines"]):
        if not isinstance(pipeline, Mapping):
            raise ValueError(f"SpatialML manifest pipeline #{index} requires 'id' and 'path'")
        if not isinstance(pipeline.get("id"), str) or not pipeline["id"]:
            raise ValueError(f"SpatialML manifest pipeline #{index}.id must be a non-empty string")
        if not isinstance(pipeline.get("path"), str) or not pipeline["path"]:
            raise ValueError(f"SpatialML manifest pipeline #{index}.path must be a non-empty string")
        pipeline_id = pipeline["id"]
        if any(pipeline_id == existing_id for existing_id in pipeline_ids):
            raise ValueError(f"Duplicate pipeline id: {pipeline_id}")
        pipeline_ids.append(pipeline_id)
        _normalize_package_path(pipeline["path"])
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("SpatialML manifest requires runtime.supported_modes")
    if "supported_modes" not in runtime:
        raise ValueError("SpatialML manifest requires runtime.supported_modes")
    _normalize_supported_modes(runtime["supported_modes"])


def _write_json(path: Path, payload: Mapping[str, Any], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)
        file.write("\n")


def _write_json_or_copy(path: Path, source: Union[JsonDict, PathLike], *, indent: int) -> None:
    if isinstance(source, Mapping):
        _write_json(path, source, indent=indent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)


def _package_destination(root: Path, package_path: PathLike) -> Path:
    normalized = _normalize_package_path(str(package_path))
    destination = root / normalized
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Package path escapes output directory: {package_path}") from exc
    return destination


def _normalize_package_path(package_path: str) -> str:
    raw_path = str(package_path)
    normalized = raw_path.replace("\\", "/")
    # A package path is always relative to the package root.  Do this check
    # before any normalization so absolute paths cannot be made to look
    # relative by stripping their leading separator.  PureWindowsPath also
    # catches drive-qualified and UNC paths on this POSIX host.
    if (
        PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
        or bool(PureWindowsPath(raw_path).drive)
    ):
        raise ValueError(
            f"Package paths must be relative (pipeline path must be package-relative): {package_path}"
        )
    if not normalized:
        raise ValueError("Package paths must not be empty")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid package-relative path: {package_path}")
    return str(PurePosixPath(*parts))


def _normalize_supported_modes(supported_modes: Sequence[str]) -> List[str]:
    if isinstance(supported_modes, str):
        raise ValueError("supported_modes must be a list containing 'xr', 'spatial', or both")
    normalized_modes = []
    for mode in supported_modes:
        normalized = str(mode).strip().lower()
        if normalized not in SUPPORTED_EXECUTION_MODES:
            allowed = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
            raise ValueError(f"Unsupported execution mode '{mode}'. Expected one of: {allowed}")
        if normalized not in normalized_modes:
            normalized_modes.append(normalized)
    if not normalized_modes:
        raise ValueError("supported_modes must contain at least one execution mode")
    return normalized_modes


__all__ = [
    "PipelinePackageEntry",
    "PipelineZooPackageSpec",
    "SUPPORTED_EXECUTION_MODES",
    "configure_litert_inference_operator",
    "create_litert_model_spec",
    "load_pipeline_zoo_manifest",
    "validate_pipeline_zoo_manifest",
    "write_pipeline_zoo_package",
]
