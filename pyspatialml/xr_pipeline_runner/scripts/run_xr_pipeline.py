#!/usr/bin/env python3
"""Push a SpatialML package to the XR runner APK, run it, and pull outputs."""

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
from pathlib import Path

from pyspatialml.zip_utils import ZipSafetyError, safe_extract_zip


PACKAGE_NAME = "com.bytedance.pico.pyspatialml.xr_runner"
SPATIAL_PACKAGE_NAME = "com.bytedance.pico.pyspatialml.spatial_runner"
COMPONENT = f"{PACKAGE_NAME}/android.app.NativeActivity"
REMOTE_ROOT = f"/sdcard/Android/data/{PACKAGE_NAME}/files"
REMOTE_PACKAGE = f"{REMOTE_ROOT}/package"
REMOTE_OUTPUT = f"{REMOTE_ROOT}/outputs"
REMOTE_INPUT = f"{REMOTE_ROOT}/input"
APP_ROOT = f"/data/user/0/{PACKAGE_NAME}/files"
APP_PACKAGE = f"{APP_ROOT}/package"
APP_OUTPUT = f"{APP_ROOT}/outputs"
APP_INPUT = f"{APP_ROOT}/input"
STAGING_ROOT = "/data/local/tmp/pyspatialml_xr_runner"
STAGING_PACKAGE = f"{STAGING_ROOT}/package"
STAGING_INPUT = f"{STAGING_ROOT}/input"
STAGING_OUTPUT = f"{STAGING_ROOT}/outputs"
PROP_PREFIX = "debug.pyspatialml.xr_runner"
EMPTY_PROP = "__pyspatialml_empty__"
APK_HASH_FILE = ".runner_apk_sha256"
LOGCAT_BUFFERS = "all"
LOGCAT_LINE_LIMIT = "12000"
LITERT_LOG_TAG_SAMPLE = "litert"
LITERT_IMPORTANT_SAMPLE = "compiler_plugin"
SECUREMR_LOG_TAG_SAMPLE = "Secure MR::Server"
RUNNER_LOG_SAMPLE = "pySpatialML XR runner"
BENIGN_READBACK_LOG_TOKENS = ("ackreadbacktensorcontent", "invalid parameter", "no shared memory")


def resolve_package_path(package_root: Path, value: str, *, label: str) -> Path:
    """Resolve a schema package path without allowing host-path escapes."""
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} must be a non-empty package-relative path")
    normalized = value.replace("\\", "/")
    if (normalized.startswith("/") or normalized.startswith("//") or
            (len(normalized) >= 2 and normalized[1] == ":" and normalized[0].isalpha())):
        raise SystemExit(f"{label} must be package-relative: {value!r}")
    parts = normalized.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise SystemExit(f"{label} contains an invalid path component: {value!r}")
    root = package_root.resolve()
    resolved = (root / Path(*parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} escapes the package root: {value!r}") from exc
    return resolved


def validate_package_paths(package_dir: Path) -> None:
    manifest_path = package_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read package manifest: {manifest_path}") from exc
    pipelines = manifest.get("pipelines")
    if not isinstance(pipelines, list):
        raise SystemExit("Package manifest pipelines must be an array")
    for index, entry in enumerate(pipelines):
        if not isinstance(entry, dict):
            raise SystemExit(f"Package manifest pipelines[{index}] must be an object")
        pipeline_path = resolve_package_path(package_dir, entry.get("path"), label=f"manifest pipelines[{index}].path")
        if not pipeline_path.is_file():
            raise SystemExit(f"Manifest pipeline does not exist: {entry.get('path')!r}")
        try:
            pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unable to read pipeline: {pipeline_path}") from exc
        for op_index, operator in enumerate(pipeline.get("operators", [])):
            if not isinstance(operator, dict):
                continue
            model = operator.get("model")
            if isinstance(model, dict) and "bin_path" in model:
                resolve_package_path(package_dir, model["bin_path"],
                                     label=f"pipeline operator {op_index} model.bin_path")
        tensors = pipeline.get("tensors", {})
        if isinstance(tensors, dict):
            for tensor_name, tensor in tensors.items():
                if isinstance(tensor, dict) and "asset" in tensor:
                    resolve_package_path(package_dir, tensor["asset"],
                                         label=f"pipeline tensor {tensor_name!r} asset")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.pipeline = normalize_pipeline_ids(args.pipeline)
    package_root = Path(__file__).resolve().parents[2]
    packaged_apk = package_root / "apks" / "pyspatialml_xr_runner-debug.apk"
    apk = args.apk or packaged_apk

    if not apk.is_file():
        raise SystemExit(f"APK not found: {apk}. Build/copy the runner APK or pass --apk.")

    with tempfile.TemporaryDirectory(prefix="pyspatialml-xr-package-") as tmp:
        tmp_root = Path(tmp)
        package_dir = prepare_package(args.package, tmp_root)
        validate_package_paths(package_dir)
        package_dir = filter_package_pipelines(package_dir, args.pipeline or [], tmp_root)
        package_dir = strip_unused_gltf_outputs(package_dir, tmp_root)
        package_dir = override_model_backend(package_dir, args.backend, tmp_root)
        model_backends = collect_model_backends(package_dir)
        adb = adb_prefix(args.device)

        ensure_apk_installed(adb, apk)

        stop_conflicting_runner(adb)
        run(adb + ["shell", "am", "force-stop", PACKAGE_NAME], check=False)
        run(adb + ["shell", "rm", "-rf", STAGING_ROOT])
        run(adb + ["shell", "mkdir", "-p", STAGING_ROOT])
        run(adb + ["push", str(package_dir), STAGING_PACKAGE])
        run_as(
            adb,
            "rm -rf files/package files/outputs files/input files/i.* && "
            "mkdir -p files/package files/outputs && "
            f"cp -R {STAGING_PACKAGE}/. files/package/",
        )

        input_remote = stage_inputs(adb, args.input or [])

        setprop(adb, "package", APP_PACKAGE)
        setprop(adb, "output", APP_OUTPUT)
        setprop(adb, "input", input_remote)
        setprop(adb, "use_vst", "true" if should_use_vst(args.input, args.use_vst) else "false")
        setprop(adb, "loop", "true" if args.loop else "false")
        setprop(adb, "dump_all", "true" if dump_all(args.dump or []) else "false")
        setprop(adb, "interval_ms", str(args.interval_ms))
        setprop(adb, "pipelines", ",".join(args.pipeline or []))

        run(adb + ["logcat", "-c"], check=False)
        run(adb + ["shell", "am", "start", "-n", COMPONENT])
        if args.loop:
            print(f"Running for {args.duration}s...")
            time.sleep(args.duration)
        else:
            wait_for_outputs(adb, duration=args.duration)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            local_outputs = pull_app_outputs(adb, output_dir)
            print(f"Pulled outputs to {local_outputs}")
            print_device_summary(adb, local_outputs, model_backends=model_backends)
        finally:
            if args.loop and not args.keep_running:
                run(adb + ["shell", "am", "force-stop", PACKAGE_NAME], check=False)
    return 0


def stop_conflicting_runner(adb: list[str]) -> None:
    """Stop the Spatial runner before starting the native XR runner."""
    run(adb + ["shell", "am", "force-stop", SPATIAL_PACKAGE_NAME], check=False)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Pipeline package directory or .zip")
    parser.add_argument("--input", action="append", help="Image/raw input path or tensor=path binding. Repeatable")
    parser.add_argument("--pipeline", action="append", help="Pipeline id to run; repeat to chain specific pipelines")
    parser.add_argument("--output-dir", default="xr_runner_outputs", help="Local directory for pulled outputs")
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


def normalize_pipeline_ids(values: list[str] | None) -> list[str]:
    """Drop empty optional selections before filtering or setting properties."""
    return [value.strip() for value in (values or []) if value.strip()]


def dump_all(dumps: list[str]) -> bool:
    unsupported = [item for item in dumps if str(item).lower() != "all"]
    if unsupported:
        raise SystemExit(f"Device run only supports --dump all, got: {', '.join(unsupported)}")
    return any(str(item).lower() == "all" for item in dumps)


def should_use_vst(input_args: list[str] | None, requested: bool) -> bool:
    """Use device VST unless raw/image input was supplied explicitly."""
    return requested or not input_args


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

    for item in selected:
        resolve_package_path(filtered_root, item.get("path"), label="manifest pipeline path")
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
        pipeline_path = resolve_package_path(package_dir, item["path"], label="manifest pipeline path")
        if not pipeline_path.is_file():
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
        pipeline_path = resolve_package_path(package_dir, item["path"], label="manifest pipeline path")
        if not pipeline_path.is_file():
            continue
        with open(pipeline_path, "r", encoding="utf-8") as file:
            spec = json.load(file)
        changed += override_pipeline_model_backend(spec, backend)
        with open(pipeline_path, "w", encoding="utf-8") as file:
            json.dump(spec, file, indent=2)
            file.write("\n")
    print(f"Overrode model backend to {backend} for {changed} model operator(s).")
    return package_dir


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
        pipeline_path = resolve_package_path(package_dir, str(item["path"]), label="manifest pipeline path")
        if not pipeline_path.is_file():
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


def stage_inputs(adb: list[str], input_args: list[str]) -> str:
    default_inputs, named_inputs = parse_input_args(input_args)
    if not default_inputs and not named_inputs:
        return ""
    if len(default_inputs) > 1:
        raise SystemExit("Only one bare --input path is supported; use tensor=path for explicit inputs.")

    if named_inputs:
        remote_names = named_input_remote_names(named_inputs)
        run(adb + ["shell", "rm", "-rf", STAGING_INPUT])
        run(adb + ["shell", "mkdir", "-p", STAGING_INPUT])
        run_as(adb, "rm -rf files/input && mkdir -p files/input")
        for (tensor_name, input_path), remote_name in zip(named_inputs, remote_names):
            suffix = input_path.suffix or ".bin"
            staging_input_file = f"{STAGING_INPUT}/{remote_name}"
            run(adb + ["push", str(input_path), staging_input_file])
            run_as(adb, f"cp {staging_input_file} files/input/{shlex.quote(remote_name)}")
        if default_inputs:
            input_path = default_inputs[0]
            suffix = input_path.suffix or ".bin"
            staging_input_file = f"{STAGING_INPUT}/__default{suffix}"
            run(adb + ["push", str(input_path), staging_input_file])
            run_as(adb, f"cp {staging_input_file} files/input/__default{suffix}")
        return APP_INPUT

    input_path = default_inputs[0]
    if input_path.is_dir():
        run(adb + ["push", str(input_path), STAGING_INPUT])
        run_as(adb, f"rm -rf files/input && mkdir -p files/input && cp -R {STAGING_INPUT}/. files/input/")
        return APP_INPUT

    suffix = input_path.suffix or ".bin"
    input_remote = f"{APP_ROOT}/i{suffix}"
    staging_input_file = f"{STAGING_ROOT}/i{suffix}"
    run(adb + ["push", str(input_path), staging_input_file])
    run_as(adb, f"cp {staging_input_file} files/i{suffix}")
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


def ensure_apk_installed(adb: list[str], apk: Path) -> None:
    apk_hash = hashlib.sha256(apk.read_bytes()).hexdigest()
    installed_hash = run_as_capture(adb, f"cat files/{APK_HASH_FILE} 2>/dev/null || true", check=False).strip()
    if installed_hash == apk_hash:
        print(f"Runner APK already installed: {apk}")
        return

    run(adb + ["install", "-r", str(apk)])
    run(adb + ["shell", "pm", "grant", PACKAGE_NAME, "android.permission.CAMERA"], check=False)
    run(adb + ["shell", "pm", "grant", PACKAGE_NAME, "com.picovr.permission.SPATIAL_DATA"], check=False)
    run_as(adb, f"mkdir -p files && printf %s {shlex.quote(apk_hash)} > files/{APK_HASH_FILE}", check=False)


def setprop(adb: list[str], key: str, value: str) -> None:
    run(adb + ["shell", "setprop", f"{PROP_PREFIX}.{key}", value or EMPTY_PROP])


def run_as(adb: list[str], command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(adb + ["shell", f"run-as {PACKAGE_NAME} sh -c {shlex.quote(command)}"], check=check)


def run_as_capture(adb: list[str], command: str, *, check: bool = True) -> str:
    result = subprocess.run(
        [*adb, "shell", f"run-as {PACKAGE_NAME} sh -c {shlex.quote(command)}"],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.stdout


def wait_for_outputs(adb: list[str], *, duration: float) -> None:
    print(f"Waiting up to {duration}s for output files...")
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        status_text = run_as_capture(adb, "cat files/outputs/status.json 2>/dev/null || true", check=False)
        status = parse_status(status_text)
        if status.get("state") == "complete":
            return
        if status.get("state") == "error":
            raise SystemExit(f"Runner reported error: {status_text.strip()}")
        time.sleep(0.25)
    print_device_logs(adb)
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


def pull_app_outputs(adb: list[str], output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )
    try:
        return _pull_app_outputs_into(adb, staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def _pull_app_outputs_into(adb: list[str], staging_dir: Path, output_dir: Path) -> Path:

    listing = run_as_capture(adb, "find files/outputs -maxdepth 1 -type f -print", check=False)
    files = [line.strip() for line in listing.splitlines() if line.strip()]
    status = pull_remote_status(adb, files)
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
        print("+", " ".join([*adb, "exec-out", "run-as", PACKAGE_NAME, "cat", remote_path]), ">", local_path)
        result = subprocess.run(
            [*adb, "exec-out", "run-as", PACKAGE_NAME, "cat", remote_path],
            check=True,
            stdout=subprocess.PIPE,
        )
        local_path.write_bytes(result.stdout)
    if not files:
        raise SystemExit("No output files were produced by the runner APK.")
    _commit_pulled_outputs(staging_dir, output_dir)
    return output_dir


def _commit_pulled_outputs(staging_dir: Path, output_dir: Path) -> None:
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


def pull_remote_status(adb: list[str], files: list[str]) -> dict:
    for remote_path in files:
        if Path(remote_path).name != "status.json":
            continue
        result = subprocess.run(
            [*adb, "exec-out", "run-as", PACKAGE_NAME, "cat", remote_path],
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
    output_files = sorted(
        path
        for path in local_outputs.glob("*/*")
        if path.is_file() and path.parent.name != "all_tensors" and path.name != "status.json"
    )
    print(section_header("Device Run Summary"))
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
    print_device_logs(adb)
    print(section_header("Device Outputs"))
    print(f"  Outputs: {len(output_files)}")
    for pipeline_id, paths in group_outputs_by_pipeline(output_files, metadata_by_file):
        print(f"  Pipeline: {pipeline_id}")
        for path in paths:
            metadata = metadata_by_file.get(path.name, {})
            print(device_tensor_summary(path, metadata))
            if path.name.endswith("post_det_1.bin"):
                print_post_det_summary(path)
    print(section_header("End Device Outputs"))


def print_device_logs(adb: list[str]) -> None:
    logs = collect_relevant_logs(adb)
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


def device_tensor_summary(path: Path, metadata: dict) -> str:
    size = path.stat().st_size
    if not metadata:
        return f"    {path.name}: {size} bytes"

    shape = tensor_shape_from_metadata(metadata)
    dtype = str(metadata.get("dtype") or "unknown")
    summary = f"    {path.name}: {size} bytes shape={shape} dtype={dtype}"
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


def collect_relevant_logs(adb: list[str]) -> list[str]:
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
    runner_pattern = re.compile(rf"({re.escape(RUNNER_LOG_SAMPLE)}|OpenMR:)")
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
    normalized = line.casefold()
    return all(token in normalized for token in BENIGN_READBACK_LOG_TOKENS)


def select_litert_logs(lines: list[str], limit: int) -> list[str]:
    important_pattern = re.compile(
        r"(compiler_plugin|libLiteRtCompilerPlugin|compiled_model|"
        r"LiteRtDispatch|dispatch_delegate|Initialized TensorFlow Lite runtime)"
    )
    important = [line for line in lines if important_pattern.search(line)]
    selected = []
    seen = set()
    for line in important + lines:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)
    if len(selected) <= limit:
        return selected
    head_count = min(len(important), limit // 2)
    head = head_unique(important, head_count)
    tail = tail_unique(selected, limit - len(head))
    merged = []
    seen.clear()
    for line in head + tail:
        if line not in seen:
            seen.add(line)
            merged.append(line)
    return merged


def head_unique(lines: list[str], limit: int) -> list[str]:
    unique = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
        if len(unique) >= limit:
            break
    return unique


def tail_unique(lines: list[str], limit: int) -> list[str]:
    unique = []
    seen = set()
    for line in reversed(lines):
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
        if len(unique) >= limit:
            break
    return list(reversed(unique))


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run([str(part) for part in cmd], cwd=cwd, check=check, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
