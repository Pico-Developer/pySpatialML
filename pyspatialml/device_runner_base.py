#!/usr/bin/env python3
"""Shared host-side staging for pySpatialML device runner APKs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pyspatialml.zip_utils import ZipSafetyError, safe_extract_zip


EMPTY_PROP = "__pyspatialml_empty__"
APK_HASH_FILE = ".runner_apk_sha256"
LOGCAT_BUFFERS = "all"
LOGCAT_LINE_LIMIT = "12000"
LITERT_LOG_TAG_SAMPLE = "litert"
LITERT_IMPORTANT_SAMPLE = "compiler_plugin"
SECUREMR_LOG_TAG_SAMPLE = "Secure MR::Server"
BENIGN_READBACK_LOG_TOKENS = ("ackreadbacktensorcontent", "invalid parameter", "no shared memory")
SPATIAL_LOADER_OPERATOR_ALIASES = {
    "XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO":
        "XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO",
    "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO":
        "XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO",
    "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO":
        "XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO",
}


@dataclass(frozen=True)
class RunnerConfig:
    mode: str
    package_name: str
    component: str
    prop_prefix: str
    staging_root: str
    default_apk_name: str
    log_sample: str
    supports_vst: bool = True

    @property
    def app_root(self) -> str:
        return f"/data/user/0/{self.package_name}/files"

    @property
    def app_package(self) -> str:
        return f"{self.app_root}/package"

    @property
    def app_package_zip(self) -> str:
        return f"{self.app_root}/package.zip"

    @property
    def app_output(self) -> str:
        return f"{self.app_root}/outputs"

    @property
    def app_input(self) -> str:
        return f"{self.app_root}/input"

    @property
    def staging_package(self) -> str:
        return f"{self.staging_root}/package"

    @property
    def staging_package_zip(self) -> str:
        return f"{self.staging_root}/package.zip"

    @property
    def staging_input(self) -> str:
        return f"{self.staging_root}/input"


XR_CONFIG = RunnerConfig(
    mode="xr",
    package_name="com.bytedance.pico.pyspatialml.xr_runner",
    component="com.bytedance.pico.pyspatialml.xr_runner/android.app.NativeActivity",
    prop_prefix="debug.pyspatialml.xr_runner",
    staging_root="/data/local/tmp/pyspatialml_xr_runner",
    default_apk_name="pyspatialml_xr_runner-debug.apk",
    log_sample="pySpatialML XR runner",
    supports_vst=True,
)

SPATIAL_CONFIG = RunnerConfig(
    mode="spatial",
    package_name="com.bytedance.pico.pyspatialml.spatial_runner",
    component=(
        "com.bytedance.pico.pyspatialml.spatial_runner/"
        "com.bytedance.pico.pyspatialml.spatialrunner.MainActivity"
    ),
    prop_prefix="debug.pyspatialml.spatial_runner",
    staging_root="/data/local/tmp/pyspatialml_spatial_runner",
    default_apk_name="pyspatialml_spatial_runner-debug.apk",
    log_sample="pySpatialML Spatial runner",
    supports_vst=True,
)


def main(config: RunnerConfig, argv: list[str] | None = None) -> int:
    args = parse_args(config, argv)
    package_root = Path(__file__).resolve().parent
    packaged_apk = package_root / "apks" / config.default_apk_name
    apk = args.apk or packaged_apk

    if not apk.is_file():
        raise SystemExit(f"APK not found: {apk}. Build/copy the runner APK or pass --apk.")

    with tempfile.TemporaryDirectory(prefix=f"pyspatialml-{config.mode}-package-") as tmp:
        tmp_root = Path(tmp)
        package_dir = prepare_package(args.package, tmp_root)
        package_dir = filter_package_pipelines(package_dir, args.pipeline or [], tmp_root)
        asset_output_metadata = collect_asset_output_metadata(package_dir)
        package_dir = strip_unused_gltf_outputs(package_dir, tmp_root)
        package_dir = override_model_backend(package_dir, args.backend, tmp_root)
        if config.mode == "spatial":
            package_dir = normalize_spatial_loader_compatibility(package_dir, tmp_root)
            validate_spatial_runner_scene_ops(package_dir)
        model_backends = collect_model_backends(package_dir)
        package_zip = make_runner_package_zip(package_dir, tmp_root)
        adb = adb_prefix(args.device)

        ensure_apk_installed(config, adb, apk)

        stop_runner_apps(adb)
        run(adb + ["shell", "rm", "-rf", config.staging_root])
        run(adb + ["shell", "mkdir", "-p", config.staging_root])
        run(adb + ["push", str(package_dir), config.staging_package])
        run(adb + ["push", str(package_zip), config.staging_package_zip])
        run_as(
            config,
            adb,
            "rm -rf files/package files/package.zip files/outputs files/input files/i.* && "
            "mkdir -p files/package files/outputs && "
            f"cp -R {config.staging_package}/. files/package/ && "
            f"cp {config.staging_package_zip} files/package.zip",
        )

        input_remote = stage_inputs(config, adb, args.input or [])

        setprop(config, adb, "package", config.app_package)
        setprop(config, adb, "package_zip", config.app_package_zip)
        setprop(config, adb, "asset_root", "package")
        setprop(config, adb, "output", config.app_output)
        setprop(config, adb, "input", input_remote)
        setprop(config, adb, "use_vst", "true" if should_use_vst(args.input, args.use_vst) else "false")
        setprop(config, adb, "loop", "true" if args.loop else "false")
        setprop(config, adb, "dump_all", "true" if dump_all(args.dump or []) else "false")
        setprop(config, adb, "interval_ms", str(args.interval_ms))
        setprop(config, adb, "pipelines", ",".join(args.pipeline or []))

        run(adb + ["logcat", "-c"], check=False)
        run(adb + ["shell", "am", "start", "-n", config.component])
        if args.loop:
            print(f"Running for {args.duration}s...")
            time.sleep(args.duration)
        else:
            wait_for_outputs(config, adb, duration=args.duration)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            local_outputs = pull_app_outputs(config, adb, output_dir, asset_output_metadata=asset_output_metadata)
            print(f"Pulled outputs to {local_outputs}")
            print_device_summary(config, adb, local_outputs, model_backends=model_backends)
        finally:
            if args.loop and not args.keep_running:
                stop_runner_apps(adb)
    return 0


def parse_args(config: RunnerConfig, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Push a SpatialML package to the {config.mode} runner APK.")
    parser.add_argument("package", type=Path, help="Pipeline package directory or .zip")
    parser.add_argument("--input", action="append", help="Image/raw input path or tensor=path binding. Repeatable")
    parser.add_argument("--pipeline", action="append", help="Pipeline id to run; repeat to chain specific pipelines")
    parser.add_argument("--output-dir", default=f"{config.mode}_runner_outputs", help="Local directory for pulled outputs")
    parser.add_argument("--dump", action="append", default=[], help="Only 'all' is supported for device tensor dumps")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Max seconds to wait for one-shot output; fixed run time for loop",
    )
    parser.add_argument("--loop", action="store_true", help="Run the pipeline repeatedly")
    parser.add_argument("--keep-running", action="store_true", help="Leave the runner app running after wait")
    parser.add_argument("--use-vst", action="store_true", help="Use device VST; this is the default when no --input is supplied")
    parser.add_argument("--backend", choices=["npu", "gpu", "cpu"], help="Override run-model backend in staged package")
    parser.add_argument("--interval-ms", type=int, default=50, help="Loop interval in milliseconds")
    parser.add_argument("--apk", type=Path, help="Runner APK path")
    parser.add_argument("--device", help="ADB device serial")
    return parser.parse_args(argv)


def stop_runner_apps(adb: list[str]) -> None:
    for package_name in (XR_CONFIG.package_name, SPATIAL_CONFIG.package_name):
        run(adb + ["shell", "am", "force-stop", package_name], check=False)


def should_use_vst(input_args: list[str] | None, requested: bool) -> bool:
    """Use device VST unless raw/image input was supplied explicitly."""
    return requested or not input_args


def collect_asset_output_metadata(package_dir: Path) -> list[dict]:
    manifest_path = package_dir / "manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(manifest, dict):
        return []
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return []

    result: list[dict] = []
    for item in pipelines:
        if not isinstance(item, dict):
            continue
        pipeline_id = str(item.get("id") or "").strip()
        path_value = item.get("path")
        if not pipeline_id or not isinstance(path_value, str) or not path_value:
            continue
        pipeline_path = safe_package_path(package_dir, path_value)
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        try:
            with open(pipeline_path, "r", encoding="utf-8") as file:
                spec = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        tensors = spec.get("tensors")
        outputs = spec.get("outputs")
        if not isinstance(tensors, dict) or not isinstance(outputs, list):
            continue
        for tensor_name in outputs:
            if not isinstance(tensor_name, str):
                continue
            tensor_spec = tensors.get(tensor_name)
            if not isinstance(tensor_spec, dict) or not is_asset_tensor(tensor_spec):
                continue
            asset = tensor_spec.get("asset")
            metadata = {
                "pipeline": pipeline_id,
                "tensor": tensor_name,
                "kind": "asset",
                "is_output": True,
                "written": False,
                "reason": "asset_reference",
            }
            if isinstance(asset, str) and asset:
                metadata["asset"] = asset
                asset_path = safe_package_path(package_dir, asset)
                metadata["exists"] = bool(asset_path and asset_path.is_file())
            result.append(metadata)
    return result


def is_asset_tensor(tensor_spec: dict) -> bool:
    tensor_type = str(tensor_spec.get("tensor_type") or tensor_spec.get("type") or "").lower()
    return bool(tensor_spec.get("is_gltf")) or tensor_type == "gltf" or isinstance(tensor_spec.get("asset"), str)


def safe_package_path(package_dir: Path, path_value: str) -> Path | None:
    path = Path(path_value)
    if path.is_absolute():
        return None
    root = package_dir.resolve()
    candidate = (package_dir / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def dump_all(dumps: list[str]) -> bool:
    unsupported = [item for item in dumps if str(item).lower() != "all"]
    if unsupported:
        raise SystemExit(f"Device run only supports --dump all, got: {', '.join(unsupported)}")
    return any(str(item).lower() == "all" for item in dumps)


def prepare_package(package: Path, tmp_root: Path) -> Path:
    if not package.exists():
        raise SystemExit(f"Package not found: {package}")
    if package.is_dir():
        root = package
    elif package.suffix.lower() == ".zip":
        root = tmp_root / "package"
        with zipfile.ZipFile(package) as archive:
            try:
                safe_extract_zip(archive, root)
            except ZipSafetyError as exc:
                raise SystemExit(str(exc)) from exc
        root = normalize_extracted_root(root)
    else:
        raise SystemExit(f"Package must be a directory or .zip: {package}")
    if not (root / "manifest.json").is_file():
        raise SystemExit(f"Package manifest not found under {root}")
    return root


def normalize_extracted_root(root: Path) -> Path:
    if (root / "manifest.json").is_file():
        return root
    children = [child for child in root.iterdir() if child.is_dir() and child.name != "__MACOSX"]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0]
    manifest_dirs = [
        path.parent
        for path in root.rglob("manifest.json")
        if "__MACOSX" not in path.parts and not any(part.startswith("._") for part in path.parts)
    ]
    if len(manifest_dirs) == 1:
        return manifest_dirs[0]
    return root


def filter_package_pipelines(package_dir: Path, pipeline_ids: list[str], tmp_root: Path) -> Path:
    if not pipeline_ids:
        return package_dir

    filtered_root = tmp_root / "filtered-package"
    if filtered_root.exists():
        shutil.rmtree(filtered_root)
    shutil.copytree(package_dir, filtered_root, ignore=shutil.ignore_patterns("__MACOSX", "._*"))

    manifest_path = filtered_root / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        raise SystemExit(f"Package manifest has no pipelines list: {manifest_path}")

    by_id = {item.get("id"): item for item in pipelines if isinstance(item, dict)}
    selected = []
    missing = []
    for pipeline_id in pipeline_ids:
        item = by_id.get(pipeline_id)
        if item is None:
            missing.append(pipeline_id)
        else:
            selected.append(item)
    if missing:
        available = ", ".join(str(item.get("id")) for item in pipelines if isinstance(item, dict))
        raise SystemExit(f"Pipeline not found: {', '.join(missing)}. Available: {available}")

    manifest["pipelines"] = selected
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return filtered_root


def strip_unused_gltf_outputs(package_dir: Path, tmp_root: Path) -> Path:
    package_dir = ensure_mutable_package(package_dir, tmp_root)
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return package_dir

    for item in pipelines:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        pipeline_path = safe_package_path(package_dir, str(item["path"]))
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        with open(pipeline_path, "r", encoding="utf-8") as file:
            spec = json.load(file)
        tensors = spec.get("tensors")
        if not isinstance(tensors, dict):
            continue
        referenced = referenced_operator_strings(spec.get("operators", []))
        removed = []
        for tensor_name, tensor_spec in list(tensors.items()):
            if not isinstance(tensor_spec, dict):
                continue
            tensor_type = str(tensor_spec.get("tensor_type") or tensor_spec.get("type") or "").lower()
            is_gltf = bool(tensor_spec.get("is_gltf")) or tensor_type == "gltf"
            if is_gltf and tensor_name not in referenced:
                removed.append(tensor_name)
                tensors.pop(tensor_name, None)
        if removed:
            outputs = spec.get("outputs")
            if isinstance(outputs, list):
                spec["outputs"] = [name for name in outputs if name not in removed]
            with open(pipeline_path, "w", encoding="utf-8") as file:
                json.dump(spec, file, indent=2)
                file.write("\n")
    return package_dir


def override_model_backend(package_dir: Path, backend: str | None, tmp_root: Path) -> Path:
    if not backend:
        return package_dir
    package_dir = ensure_mutable_package(package_dir, tmp_root)
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return package_dir

    changed = 0
    for item in pipelines:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        pipeline_path = safe_package_path(package_dir, str(item["path"]))
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        with open(pipeline_path, "r", encoding="utf-8") as file:
            spec = json.load(file)
        changed += override_pipeline_model_backend(spec, backend)
        with open(pipeline_path, "w", encoding="utf-8") as file:
            json.dump(spec, file, indent=2)
            file.write("\n")
    print(f"Overrode model backend to {backend} for {changed} model operator(s).")
    return package_dir


def normalize_spatial_loader_compatibility(package_dir: Path, tmp_root: Path) -> Path:
    """Adapt spec-valid operators for the current Spatial SDK loader.

    The package format defines arithmetic as ten positional slots and uses null
    for unused slots, while the current loader rejects those null entries. The
    loader accepts compatibility aliases for JavaScript and Spatial scene
    operators instead of their canonical enum names. The Spatial runner owns
    this temporary staging copy, so adapt only the staged records. The source
    package and spec remain unchanged.
    """
    package_dir = ensure_mutable_package(package_dir, tmp_root)
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return package_dir

    changed = 0
    for item in pipelines:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        pipeline_path = safe_package_path(package_dir, str(item["path"]))
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        with open(pipeline_path, "r", encoding="utf-8") as file:
            spec = json.load(file)
        operators = spec.get("operators")
        if not isinstance(operators, list):
            continue
        pipeline_changed = False
        for index, op in enumerate(operators):
            if not isinstance(op, dict):
                continue
            compatibility_type = SPATIAL_LOADER_OPERATOR_ALIASES.get(op.get("type"))
            if compatibility_type is not None:
                op["type"] = compatibility_type
                pipeline_changed = True
                changed += 1
            op_type = str(op.get("type") or "").upper()
            if not op_type.endswith("ARITHMETIC_COMPOSE_PICO"):
                continue
            inputs = op.get("inputs")
            if not isinstance(inputs, list) or not any(value is None for value in inputs):
                continue
            connected = [slot for slot in inputs if slot is not None]
            if not connected:
                raise RuntimeError(
                    f"Spatial staging cannot compact arithmetic operator #{index}: no connected inputs"
                )
            old_to_new = {}
            new_index = 0
            for old_index, slot in enumerate(inputs):
                if slot is not None:
                    old_to_new[old_index] = new_index
                    new_index += 1

            attrs = op.get("attrs")
            if not isinstance(attrs, list) or not attrs or not isinstance(attrs[0], str):
                raise RuntimeError(
                    f"Spatial staging cannot compact arithmetic operator #{index}: missing expression"
                )
            expression = attrs[0]

            def remap_reference(match: re.Match[str]) -> str:
                old_index = int(match.group(1))
                if old_index not in old_to_new:
                    raise RuntimeError(
                        f"Spatial staging cannot compact arithmetic operator #{index}: "
                        f"expression references null input slot {old_index}"
                    )
                return "{" + str(old_to_new[old_index]) + "}"

            remapped_expression = re.sub(r"\{(\d+)\}", remap_reference, expression)
            op["inputs"] = connected
            attrs[0] = remapped_expression
            pipeline_changed = True
            changed += 1

        if pipeline_changed:
            with open(pipeline_path, "w", encoding="utf-8") as file:
                json.dump(spec, file, indent=2)
                file.write("\n")

    if changed:
        print(f"Normalized {changed} Spatial operator record(s) for runner staging.")
    return package_dir


def normalize_operator_type(op_type: str) -> str:
    if op_type in {
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO",
        "XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO",
    }:
        return "ssmr_switch_visibility"
    if op_type in {
        "XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO",
        "XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO",
    }:
        return "ssmr_update_component"
    return ""


def validate_spatial_runner_scene_ops(package_dir: Path) -> None:
    """Validate scene operations before staging them for the Spatial runner."""
    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return
    for item in pipelines:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        pipeline_id = str(item.get("id") or "<unknown>")
        pipeline_path = safe_package_path(package_dir, str(item["path"]))
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        with open(pipeline_path, "r", encoding="utf-8") as file:
            spec = json.load(file)
        for index, op in enumerate(spec.get("operators", [])):
            if not isinstance(op, dict):
                continue
            if normalize_operator_type(str(op.get("type") or "")) != "ssmr_update_component":
                continue
            inputs = op.get("inputs")
            attrs = op.get("attrs")
            if (not isinstance(inputs, list) or len(inputs) != 2 or
                    not isinstance(inputs[0], dict) or not isinstance(inputs[1], dict) or
                    not isinstance(attrs, list) or len(attrs) != 1 or
                    not isinstance(attrs[0], str) or not attrs[0].startswith("/") or ":" not in attrs[0]):
                raise SystemExit(
                    f"Invalid SSMR_UPDATE_COMPONENT at {pipeline_id}[{index}]: requires "
                    "two tensor inputs and component path in attrs[0]"
                )


def override_pipeline_model_backend(spec: dict, backend: str) -> int:
    changed = 0
    operators = spec.get("operators", [])
    if not isinstance(operators, list):
        return 0
    for op in operators:
        if not isinstance(op, dict) or not is_model_operator(op):
            continue
        model = op.get("model")
        if isinstance(model, dict):
            model["model_target"] = backend
            changed += 1
    return changed


def is_model_operator(op: dict) -> bool:
    return op.get("type") == "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO"


def collect_model_backends(package_dir: Path) -> list[str]:
    """Read effective model targets from the finalized staged package."""
    try:
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        return []

    backends = []
    for item in pipelines:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        pipeline_path = safe_package_path(package_dir, str(item["path"]))
        if pipeline_path is None or not pipeline_path.is_file():
            continue
        try:
            spec = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for op in spec.get("operators", []):
            if not isinstance(op, dict) or not is_model_operator(op):
                continue
            model = op.get("model")
            if not isinstance(model, dict):
                continue
            backend = str(model.get("model_target") or "npu").strip().lower()
            if backend and backend not in backends:
                backends.append(backend)
    return backends


def ensure_mutable_package(package_dir: Path, tmp_root: Path) -> Path:
    try:
        package_dir.resolve().relative_to(tmp_root.resolve())
        return package_dir
    except ValueError:
        mutable_root = tmp_root / "device-package"
        if mutable_root.exists():
            shutil.rmtree(mutable_root)
        shutil.copytree(package_dir, mutable_root, ignore=shutil.ignore_patterns("__MACOSX", "._*"))
        return mutable_root


def make_runner_package_zip(package_dir: Path, tmp_root: Path) -> Path:
    archive_path = tmp_root / "runner-package.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file() and "__MACOSX" not in path.parts and not path.name.startswith("._"):
                archive.write(path, f"assets/package/{path.relative_to(package_dir).as_posix()}")
    return archive_path


def referenced_operator_strings(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str):
        result.add(value)
    elif isinstance(value, list):
        for item in value:
            result.update(referenced_operator_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            result.update(referenced_operator_strings(item))
    return result


def adb_prefix(device: str | None) -> list[str]:
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    return cmd


def stage_inputs(config: RunnerConfig, adb: list[str], input_args: list[str]) -> str:
    default_inputs, named_inputs = parse_input_args(input_args)
    if not default_inputs and not named_inputs:
        return ""
    if len(default_inputs) > 1:
        raise SystemExit("Only one bare --input path is supported; use tensor=path for explicit inputs.")

    if named_inputs:
        remote_names = named_input_remote_names(named_inputs)
        run(adb + ["shell", "rm", "-rf", config.staging_input])
        run(adb + ["shell", "mkdir", "-p", config.staging_input])
        run_as(config, adb, "rm -rf files/input && mkdir -p files/input")
        for (tensor_name, input_path), remote_name in zip(named_inputs, remote_names):
            suffix = input_path.suffix or ".bin"
            staging_input_file = f"{config.staging_input}/{remote_name}"
            run(adb + ["push", str(input_path), staging_input_file])
            run_as(config, adb, f"cp {staging_input_file} files/input/{shlex.quote(remote_name)}")
        if default_inputs:
            input_path = default_inputs[0]
            suffix = input_path.suffix or ".bin"
            staging_input_file = f"{config.staging_input}/__default{suffix}"
            run(adb + ["push", str(input_path), staging_input_file])
            run_as(config, adb, f"cp {staging_input_file} files/input/__default{suffix}")
        return config.app_input

    input_path = default_inputs[0]
    if input_path.is_dir():
        run(adb + ["push", str(input_path), config.staging_input])
        run_as(config, adb, f"rm -rf files/input && mkdir -p files/input && cp -R {config.staging_input}/. files/input/")
        return config.app_input

    suffix = input_path.suffix or ".bin"
    input_remote = f"{config.app_root}/i{suffix}"
    staging_input_file = f"{config.staging_root}/i{suffix}"
    run(adb + ["push", str(input_path), staging_input_file])
    run_as(config, adb, f"cp {staging_input_file} files/i{suffix}")
    return input_remote


def parse_input_args(input_args: list[str]) -> tuple[list[Path], list[tuple[str, Path]]]:
    default_inputs = []
    named_inputs = []
    for item in input_args:
        if "=" in item:
            name, raw_path = item.split("=", 1)
            if not name:
                raise SystemExit("Input name cannot be empty.")
            input_path = Path(raw_path)
            if not input_path.exists():
                raise SystemExit(f"Input not found: {input_path}")
            named_inputs.append((name, input_path))
        else:
            input_path = Path(item)
            if not input_path.exists():
                raise SystemExit(f"Input not found: {input_path}")
            default_inputs.append(input_path)
    return default_inputs, named_inputs


def safe_input_filename(tensor_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tensor_name)


def named_input_remote_names(named_inputs: list[tuple[str, Path]]) -> list[str]:
    remote_names = [
        f"{safe_input_filename(tensor_name)}{input_path.suffix or '.bin'}"
        for tensor_name, input_path in named_inputs
    ]
    seen: dict[str, str] = {}
    for (tensor_name, _), remote_name in zip(named_inputs, remote_names):
        previous_name = seen.get(remote_name)
        if previous_name is not None:
            raise SystemExit(
                f"Named input filenames collide after sanitization: "
                f"{previous_name!r} and {tensor_name!r} both map to {remote_name!r}. "
                "Use distinct tensor names."
            )
        seen[remote_name] = tensor_name
    return remote_names


def ensure_apk_installed(config: RunnerConfig, adb: list[str], apk: Path) -> None:
    apk_hash = hashlib.sha256(apk.read_bytes()).hexdigest()
    installed_hash = run_as_capture(config, adb, f"cat files/{APK_HASH_FILE} 2>/dev/null || true", check=False).strip()
    if installed_hash == apk_hash:
        print(f"Runner APK already installed: {apk}")
        return

    run(adb + ["install", "-r", str(apk)])
    run(adb + ["shell", "pm", "grant", config.package_name, "android.permission.CAMERA"], check=False)
    run(adb + ["shell", "pm", "grant", config.package_name, "com.picovr.permission.SPATIAL_DATA"], check=False)
    run_as(config, adb, f"mkdir -p files && printf %s {shlex.quote(apk_hash)} > files/{APK_HASH_FILE}", check=False)


def setprop(config: RunnerConfig, adb: list[str], key: str, value: str) -> None:
    run(adb + ["shell", "setprop", f"{config.prop_prefix}.{key}", value or EMPTY_PROP])


def run_as(config: RunnerConfig, adb: list[str], command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(adb + ["shell", f"run-as {config.package_name} sh -c {shlex.quote(command)}"], check=check)


def run_as_capture(config: RunnerConfig, adb: list[str], command: str, *, check: bool = True) -> str:
    result = subprocess.run(
        [*adb, "shell", f"run-as {config.package_name} sh -c {shlex.quote(command)}"],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def wait_for_outputs(config: RunnerConfig, adb: list[str], *, duration: float) -> None:
    print(f"Waiting up to {duration}s for output files...")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        status_text = run_as_capture(config, adb, "cat files/outputs/status.json 2>/dev/null || true", check=False)
        status = parse_status(status_text)
        if status.get("state") == "complete":
            return
        if status.get("state") == "error":
            raise SystemExit(f"Runner reported error: {status_text.strip()}")
        time.sleep(0.25)
    print_device_logs(config, adb)
    raise SystemExit(f"Timed out after {duration}s waiting for output files.")


def parse_status(status_text: str) -> dict:
    if not status_text.strip():
        return {}
    try:
        status = json.loads(status_text)
    except json.JSONDecodeError:
        return {}
    return status if isinstance(status, dict) else {}


def section_header(title: str) -> str:
    return f"========== {title} =========="


def pull_app_outputs(
    config: RunnerConfig,
    adb: list[str],
    output_dir: Path,
    *,
    asset_output_metadata: list[dict] | None = None,
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        return _pull_app_outputs_into(
            config,
            adb,
            staging_dir,
            asset_output_metadata=asset_output_metadata,
            output_dir=output_dir,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _pull_app_outputs_into(
    config: RunnerConfig,
    adb: list[str],
    staging_dir: Path,
    *,
    asset_output_metadata: list[dict] | None,
    output_dir: Path,
) -> Path:

    listing = run_as_capture(config, adb, "find files/outputs -maxdepth 1 -type f -print", check=False)
    files = [line.strip() for line in listing.splitlines() if line.strip()]
    status = pull_remote_status(config, adb, files)
    metadata_by_file = output_metadata_by_file(status)
    for remote_path in files:
        filename = Path(remote_path).name
        if filename == "status.json":
            local_path = staging_dir / filename
        else:
            metadata = metadata_by_file.get(filename, {})
            pipeline_id = pipeline_id_for_output(Path(filename), metadata)
            local_root = staging_dir / safe_input_filename(pipeline_id)
            if metadata and metadata.get("is_output") is False:
                local_root = local_root / "all_tensors"
            local_root.mkdir(parents=True, exist_ok=True)
            local_path = local_root / filename
        print("+", " ".join([*adb, "exec-out", "run-as", config.package_name, "cat", remote_path]), ">", local_path)
        result = subprocess.run(
            [*adb, "exec-out", "run-as", config.package_name, "cat", remote_path],
            check=True,
            stdout=subprocess.PIPE,
        )
        local_path.write_bytes(result.stdout)
    if not files:
        raise SystemExit("No output files were produced by the runner APK.")
    if asset_output_metadata:
        annotate_local_status(staging_dir / "status.json", asset_output_metadata)
    _commit_pulled_outputs(staging_dir, output_dir)
    return output_dir


def _commit_pulled_outputs(staging_dir: Path, output_dir: Path) -> None:
    """Atomically publish pulled outputs after every remote read succeeded."""
    backup_dir: Path | None = None
    try:
        if output_dir.exists() or output_dir.is_symlink():
            backup_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_dir.name}.backup-",
                    dir=output_dir.parent,
                )
            )
            backup_dir.rmdir()
            output_dir.rename(backup_dir)
        staging_dir.rename(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


def annotate_local_status(status_path: Path, asset_output_metadata: list[dict]) -> None:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    if not isinstance(status, dict):
        status = {}
    metadata = status.get("outputs_metadata")
    if not isinstance(metadata, list):
        metadata = []
    existing = {
        (item.get("pipeline"), item.get("tensor"))
        for item in metadata
        if isinstance(item, dict) and item.get("is_output") is True
    }
    added = 0
    for item in asset_output_metadata:
        key = (item.get("pipeline"), item.get("tensor"))
        if key in existing:
            continue
        metadata.append(item)
        existing.add(key)
        added += 1
    status["outputs_metadata"] = metadata
    if added:
        status["asset_references"] = int(status.get("asset_references") or 0) + added
    with open(status_path, "w", encoding="utf-8") as file:
        json.dump(status, file, indent=2)
        file.write("\n")


def pull_remote_status(config: RunnerConfig, adb: list[str], files: list[str]) -> dict:
    for remote_path in files:
        if Path(remote_path).name != "status.json":
            continue
        result = subprocess.run(
            [*adb, "exec-out", "run-as", config.package_name, "cat", remote_path],
            check=False,
            stdout=subprocess.PIPE,
        )
        try:
            status = json.loads(result.stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {}
        return status if isinstance(status, dict) else {}
    return {}


def print_device_summary(
    config: RunnerConfig,
    adb: list[str],
    local_outputs: Path,
    *,
    model_backends: list[str] | None = None,
) -> None:
    status_path = local_outputs / "status.json"
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}

    metadata_by_file = output_metadata_by_file(status)
    metadata_by_pipeline = output_metadata_by_pipeline(status)
    output_files = sorted(
        path
        for path in local_outputs.glob("*/*")
        if path.is_file() and path.parent.name != "all_tensors" and path.name != "status.json"
    )
    print(section_header("Device Run Summary"))
    print(f"  Mode: {config.mode}")
    if status:
        runtime_modes = ", ".join(str(item) for item in status.get("runtime_modes", [])) or "-"
        print(f"  Runtime modes: {runtime_modes}")
        pipelines = ", ".join(str(item) for item in status.get("pipelines", [])) or "-"
        print(f"  Pipelines: {pipelines}")
        print(f"  Backend: {', '.join(model_backends or []) or '-'}")
        print(f"  Iteration: {status.get('iteration', '-')}")
        print(f"  Total time: {format_ms(status.get('total_elapsed_ms'))}")
        print(f"  Loop: {str(status.get('loop', '-')).lower()}")
        print(f"  Use VST: {str(status.get('use_vst', '-')).lower()}")
    print_device_logs(config, adb)
    print(section_header("Device Outputs"))
    declared_output_count = count_declared_outputs(status.get("outputs_metadata", []))
    print(f"  Outputs: {declared_output_count or len(output_files)}")
    for pipeline_id, paths, asset_outputs in group_device_outputs(
        output_files,
        metadata_by_file,
        metadata_by_pipeline,
    ):
        print(f"  Pipeline: {pipeline_id}")
        for path in paths:
            metadata = metadata_by_file.get(path.name, {})
            print(device_tensor_summary(path, metadata))
            if path.name.endswith("post_det_1.bin"):
                print_post_det_summary(path)
        for item in asset_outputs:
            print(device_asset_summary(item))
    print(section_header("End Device Outputs"))


def print_device_logs(config: RunnerConfig, adb: list[str]) -> None:
    logs = collect_relevant_logs(config, adb)
    if not logs:
        return
    print(section_header("Device Logs"))
    print("  Device logs:")
    for line in logs:
        print(f"    {line}")
    print(section_header("End Device Logs"))


def format_ms(value) -> str:
    if value is None or value == "":
        return "-"
    return f"{value} ms"


def group_outputs_by_pipeline(output_files: list[Path], metadata_by_file: dict[str, dict]) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    order = []
    for path in output_files:
        pipeline_id = pipeline_id_for_output(path, metadata_by_file.get(path.name, {}))
        if pipeline_id not in grouped:
            grouped[pipeline_id] = []
            order.append(pipeline_id)
        grouped[pipeline_id].append(path)
    return [(pipeline_id, grouped[pipeline_id]) for pipeline_id in order]


def group_device_outputs(
    output_files: list[Path],
    metadata_by_file: dict[str, dict],
    metadata_by_pipeline: dict[str, list[dict]],
) -> list[tuple[str, list[Path], list[dict]]]:
    order = []
    output_by_pipeline: dict[str, list[Path]] = {}
    for path in output_files:
        metadata = metadata_by_file.get(path.name, {})
        if metadata and metadata.get("is_output") is False:
            continue
        pipeline_id = pipeline_id_for_output(path, metadata)
        if pipeline_id not in order:
            order.append(pipeline_id)
        output_by_pipeline.setdefault(pipeline_id, []).append(path)
    for pipeline_id, items in metadata_by_pipeline.items():
        for item in items:
            if item.get("kind") == "asset" and item.get("is_output") is True:
                if pipeline_id not in order:
                    order.append(pipeline_id)
                continue
    return [
        (
            pipeline_id,
            output_by_pipeline.get(pipeline_id, []),
            [
                item
                for item in metadata_by_pipeline.get(pipeline_id, [])
                if item.get("kind") == "asset" and item.get("is_output") is True
            ],
        )
        for pipeline_id in order
    ]


def pipeline_id_for_output(path: Path, metadata: dict) -> str:
    pipeline = metadata.get("pipeline") if isinstance(metadata, dict) else None
    if isinstance(pipeline, str) and pipeline:
        return pipeline
    name = path.name
    if "_" in name:
        return name.split("_", 1)[0]
    return "unknown"


def output_metadata_by_file(status: dict) -> dict[str, dict]:
    metadata = status.get("outputs_metadata", [])
    if not isinstance(metadata, list):
        return {}
    result = {}
    for item in metadata:
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            result[item["file"]] = item
    return result


def output_metadata_by_pipeline(status: dict) -> dict[str, list[dict]]:
    metadata = status.get("outputs_metadata", [])
    if not isinstance(metadata, list):
        return {}
    result: dict[str, list[dict]] = {}
    for item in metadata:
        if not isinstance(item, dict):
            continue
        pipeline = item.get("pipeline")
        if isinstance(pipeline, str) and pipeline:
            result.setdefault(pipeline, []).append(item)
    return result


def count_declared_outputs(metadata: object) -> int:
    if not isinstance(metadata, list):
        return 0
    return sum(1 for item in metadata if isinstance(item, dict) and item.get("is_output") is True)


def device_asset_summary(metadata: dict, *, indent: str = "    ") -> str:
    tensor = str(metadata.get("tensor") or metadata.get("name") or "asset")
    asset = metadata.get("asset")
    if isinstance(asset, str) and asset:
        suffix = f" {asset}"
    else:
        suffix = ""
    exists = metadata.get("exists")
    exists_suffix = ""
    if isinstance(exists, bool):
        exists_suffix = f" exists={'yes' if exists else 'no'}"
    return f"{indent}{tensor}: asset reference{suffix}{exists_suffix}"


def device_tensor_summary(path: Path, metadata: dict, *, indent: str = "    ") -> str:
    size = path.stat().st_size
    if not metadata:
        return f"{indent}{path.name}: {size} bytes"

    shape = tensor_shape_from_metadata(metadata)
    dtype = str(metadata.get("dtype") or "unknown")
    summary = f"{indent}{path.name}: {size} bytes shape={shape} dtype={dtype}"
    array = read_tensor_array(path, dtype, shape)
    if array is None:
        return summary
    if array.size:
        summary += (
            f" min={float(array.min()):.6g}"
            f" max={float(array.max()):.6g}"
            f" mean={float(array.mean()):.6g}"
        )
        if bool((array == 0).all()):
            summary += " all_zero=true"
        summary += f"\n      preview={tensor_preview(array)}"
    return summary


def tensor_shape_from_metadata(metadata: dict) -> tuple[int, ...]:
    raw_shape = metadata.get("shape", [])
    shape = tuple(int(value) for value in raw_shape) if isinstance(raw_shape, list) else ()
    channels = int(metadata.get("channels") or 0)
    if channels > 1:
        return (*shape, channels)
    return shape


def read_tensor_array(path: Path, dtype: str, shape: tuple[int, ...]):
    try:
        import numpy as np
    except Exception:
        return None
    dtype_map = {
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    np_dtype = dtype_map.get(dtype)
    if np_dtype is None:
        return None
    raw = path.read_bytes()
    if not raw:
        return np.asarray([], dtype=np_dtype)
    array = np.frombuffer(raw, dtype=np_dtype)
    if shape and array.size == int(np.prod(shape)):
        array = array.reshape(shape)
    return array


def tensor_preview(array, limit: int = 8) -> str:
    flat = array.reshape(-1)
    values = flat[:limit].tolist()
    result = []
    for value in values:
        try:
            value = value.item()
        except AttributeError:
            pass
        if isinstance(value, float):
            result.append(f"{value:.6g}")
        else:
            result.append(str(value))
    if flat.size > limit:
        result.append("...")
    return "[" + ", ".join(result) + "]"


def print_post_det_summary(path: Path) -> None:
    try:
        import struct

        raw = path.read_bytes()
        if len(raw) != 21 * 4:
            return
        values = list(struct.unpack("<21f", raw))
    except Exception:
        return
    print(
        "      decoded: "
        f"bbox=[{values[0]:.4g}, {values[1]:.4g}, {values[2]:.4g}, {values[3]:.4g}] "
        f"score={values[4]:.4g}"
    )


def collect_relevant_logs(config: RunnerConfig, adb: list[str]) -> list[str]:
    result = subprocess.run(
        [*adb, "logcat", "-d", "-b", LOGCAT_BUFFERS, "-t", LOGCAT_LINE_LIMIT],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    litert_pattern = re.compile(
        r"(\s[VDIWEF]\s+(litert|tflite)\s*:|LiteRT|LiteRt|TFLite|tflite|"
        r"compiler_plugin|libLiteRtCompilerPlugin|LiteRtDispatch)"
    )
    securemr_pattern = re.compile(re.escape(SECUREMR_LOG_TAG_SAMPLE))
    runner_pattern = re.compile(rf"({re.escape(config.log_sample)}|OpenMR:|SpatialML)")
    litert_lines = []
    securemr_lines = []
    runner_lines = []
    output = result.stdout.decode("utf-8", errors="replace")
    for line in output.splitlines():
        # SecureMR owns the authoritative pipeline and inference errors. Keep
        # every line from that tag, including messages that would otherwise
        # match the benign readback filter, so early failures are not hidden
        # by later teardown or rendering logs.
        if securemr_pattern.search(line):
            securemr_lines.append(line)
            continue
        if is_benign_readback_log(line):
            continue
        if litert_pattern.search(line):
            litert_lines.append(line)
        elif runner_pattern.search(line):
            runner_lines.append(line)
    return (
        select_litert_logs(litert_lines, 24)
        + securemr_lines
        + tail_unique(runner_lines, 12)
    )


def is_benign_readback_log(line: str) -> bool:
    lowered = line.lower()
    return all(token in lowered for token in BENIGN_READBACK_LOG_TOKENS)


def select_litert_logs(lines: list[str], limit: int) -> list[str]:
    if len(lines) <= limit:
        return tail_unique(lines, limit)
    important = [line for line in lines if LITERT_IMPORTANT_SAMPLE in line]
    selected = tail_unique(important, limit)
    remaining = limit - len(selected)
    if remaining > 0:
        selected = tail_unique(lines, remaining) + selected
    return tail_unique(selected, limit)


def tail_unique(lines: list[str], limit: int) -> list[str]:
    seen = set()
    result = []
    for line in reversed(lines):
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
        if len(result) >= limit:
            break
    return list(reversed(result))


def run(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(shlex.quote(str(part)) for part in cmd))
    return subprocess.run(cmd, check=check, **kwargs)

