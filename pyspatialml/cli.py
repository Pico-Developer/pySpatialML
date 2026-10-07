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

"""pySpatialML command line interface."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence

from pyspatialml import __version__
from pyspatialml.litert_tools import (
    DEFAULT_LITERT_PACKAGE,
    DEFAULT_LITERT_VERSION,
    LiteRTToolError,
    default_tool_cache,
    install_litert_cli,
    managed_litert_bin,
    print_litert_error,
    repair_litert_cli,
    resolve_litert_cli,
)
from securemr.operator_contracts import (
    OperatorInputError,
    all_input_switches,
    all_output_switches,
    order_named_inputs,
    order_named_outputs,
    split_named_bindings,
)


class CliError(RuntimeError):
    """Raised for lightweight CLI validation failures before domain dispatch."""


class RunTargetCliError(RuntimeError):
    """Raised for lightweight run target validation failures before run dispatch."""


class _StoreOperatorBinding(argparse.Action):
    """Collect an operator tensor together with the switch that named it."""

    def __call__(self, parser, namespace, values, option_string=None):
        bindings = list(getattr(namespace, self.dest, None) or [])
        bindings.append((str(option_string)[2:], values))
        setattr(namespace, self.dest, bindings)


_DOMAIN_MODULES = {
    "compare_cli": "pyspatialml.compare_cli",
    "model_cli": "pyspatialml.model_cli",
    "onnx_tools": "pyspatialml.onnx_tools",
    "operator_cli": "pyspatialml.operator_cli",
    "package_cli": "pyspatialml.package_cli",
    "pipeline_cli": "pyspatialml.pipeline_cli",
    "run_cli": "pyspatialml.run_cli",
}


_LITERT_MODEL_COMMANDS = {
    "run": "run",
    "convert": "convert",
    "quantize": "quantize",
    "benchmark": "benchmark",
    "visualize": "visualize",
}

_DOMAIN_ERRORS = {
    "CompareCliError": ("PSM_COMPARE", "compare", "Error [PSM_COMPARE]"),
    "ModelCliError": ("PSM_MODEL", "model", "Error [PSM_MODEL]"),
    "OnnxToolError": ("PSM_ONNX_CONVERT", "model", "Error [PSM_ONNX_CONVERT]"),
    "OperatorCliError": ("PSM_OPERATOR", "operator", "Error [PSM_OPERATOR]"),
    "PackageCliError": ("PSM_PACKAGE", "package", "Error [PSM_PACKAGE]"),
    "PipelineCliError": ("PSM_PIPELINE", "pipeline", "Error [PSM_PIPELINE]"),
    "RunCliError": ("PSM_RUN", "run", "Error [PSM_RUN]"),
    "CliError": ("PSM_PIPELINE", "pipeline", "Error [PSM_PIPELINE]"),
    "RunTargetCliError": ("PSM_RUN", "run", "Error [PSM_RUN]"),
}

_PACKAGE_TARGET_HINT = (
    "Run targets must be a SpatialML pipeline package directory containing "
    "manifest.json, or a .zip package containing manifest.json. The manifest "
    "must include a non-empty pipelines list with paths to pipeline JSON files. "
    "Use `pyspatialml package create` to create a package from pipeline JSON."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="pyspatialml",
        description="SpatialML package and LiteRT/TFLite workflow tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_tools_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_model_parser(subparsers)
    _add_visualize_parser(subparsers)
    _add_operator_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_package_parser(subparsers)
    _add_run_parser(subparsers)

    return parser


def _add_tools_parser(subparsers: argparse._SubParsersAction) -> None:
    tools = subparsers.add_parser("tools", help="Manage external tool dependencies.")
    tools_subparsers = tools.add_subparsers(dest="tools_command", required=True)
    litert = tools_subparsers.add_parser("litert", help="Resolve or install LiteRT CLI.")
    litert.add_argument(
        "action",
        choices=["status", "install", "repair"],
        nargs="?",
        default="status",
        help="Tool action to run.",
    )
    litert.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
    litert.add_argument("--package", default=DEFAULT_LITERT_PACKAGE, help="Pip package to install.")
    litert.add_argument("--version", default=DEFAULT_LITERT_VERSION, help="LiteRT package version.")
    litert.add_argument(
        "--force",
        action="store_true",
        help="Recreate the managed LiteRT environment before installing.",
    )
    _add_json_output_argument(litert)
    litert.set_defaults(func=_run_litert_tool)


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    compare = subparsers.add_parser("compare", help="Compare tensor output .npy files or directories.")
    compare.add_argument("expected", type=Path, help="Expected .npy file or directory.")
    compare.add_argument("actual", type=Path, help="Actual .npy file or directory.")
    compare.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance.")
    compare.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance.")
    _add_json_output_argument(compare)
    compare.set_defaults(func=_run_compare)


def _add_model_parser(subparsers: argparse._SubParsersAction) -> None:
    model = subparsers.add_parser("model", help="Delegate model-level work to LiteRT CLI.")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_info = model_subparsers.add_parser("info", help="Inspect LiteRT/TFLite model inputs and outputs.")
    model_info.add_argument("model", type=Path, help="Path to .tflite model.")
    model_info.add_argument("--signature-index", type=int, default=0, help="Signature index to inspect.")
    _add_json_output_argument(model_info)
    model_info.set_defaults(func=_run_model_info)

    for command_name, litert_command in _LITERT_MODEL_COMMANDS.items():
        command = model_subparsers.add_parser(
            command_name,
            help=f"Run `litert {litert_command}` through pySpatialML tool resolution.",
            add_help=False,
        )
        command.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
        _add_json_output_argument(command, help="Print pySpatialML JSON wrapper output.")
        command.add_argument(
            "litert_args",
            nargs=argparse.REMAINDER,
            help=f"Arguments passed to `litert {litert_command}`.",
        )
        command.set_defaults(func=_run_litert_model_command, litert_command=litert_command)


def _add_visualize_parser(subparsers: argparse._SubParsersAction) -> None:
    visualize = subparsers.add_parser("visualize", help="Visualization commands.")
    visualize_subparsers = visualize.add_subparsers(dest="visualize_command", required=True)
    visualize_model = visualize_subparsers.add_parser(
        "model",
        help="Run `litert visualize` through pySpatialML tool resolution.",
        add_help=False,
    )
    visualize_model.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
    _add_json_output_argument(visualize_model, help="Print pySpatialML JSON wrapper output.")
    visualize_model.add_argument("litert_args", nargs=argparse.REMAINDER, help="Arguments passed to `litert visualize`.")
    visualize_model.set_defaults(func=_run_litert_model_command, litert_command="visualize")


def _add_operator_parser(subparsers: argparse._SubParsersAction) -> None:
    operator = subparsers.add_parser("operator", help="Discover supported SecureMR operators.")
    operator_subparsers = operator.add_subparsers(dest="operator_command", required=True)
    operator_list = operator_subparsers.add_parser("list", help="List supported operators.")
    _add_json_output_argument(operator_list)
    operator_list.set_defaults(func=_run_operator_list)

    operator_describe = operator_subparsers.add_parser("describe-op", help="Describe one operator.")
    operator_describe.add_argument("name", help="Operator enum name, JSON type name, or creator name.")
    _add_json_output_argument(operator_describe)
    operator_describe.set_defaults(func=_run_operator_describe)


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    pipeline = subparsers.add_parser("pipeline", help="Build and inspect pipeline JSON files.")
    pipeline_subparsers = pipeline.add_subparsers(dest="pipeline_command", required=True)

    pipeline_init = pipeline_subparsers.add_parser("init", help="Create an empty pipeline JSON file.")
    pipeline_init.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    pipeline_init.add_argument("--force", action="store_true", help="Overwrite an existing pipeline file.")
    _add_json_output_argument(pipeline_init)
    pipeline_init.set_defaults(func=_run_pipeline_init)

    add_tensor = pipeline_subparsers.add_parser("add-tensor", help="Add a tensor descriptor.")
    add_tensor.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    add_tensor.add_argument("name", help="Tensor name.")
    add_tensor.add_argument("--shape", required=True, help="Tensor shape, for example 128,128,3.")
    add_tensor.add_argument("--dtype", required=True, help="Tensor dtype, for example uint8 or float32.")
    add_tensor.add_argument("--usage", default="matrix", help="Tensor usage: matrix, scalar, point, slice, color, timestamp, gltf.")
    add_tensor.add_argument("--channels", type=int, help="Explicit channel count; keeps --shape as spatial dimensions.")
    add_tensor.add_argument("--input", action="store_true", help="Also mark this tensor as a pipeline input.")
    add_tensor.add_argument("--output", action="store_true", help="Also mark this tensor as a pipeline output.")
    add_tensor.add_argument("--value", help="Optional comma-separated tensor values.")
    add_tensor.add_argument("--asset", help="Package-relative asset path, primarily for glTF tensors.")
    _add_json_output_argument(add_tensor)
    add_tensor.set_defaults(func=_run_pipeline_add_tensor)

    remove_tensor = pipeline_subparsers.add_parser("remove-tensor", help="Remove a tensor descriptor.")
    remove_tensor.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    remove_tensor.add_argument("name", help="Tensor name.")
    remove_tensor.add_argument("--force", action="store_true", help="Remove even when operators reference the tensor.")
    _add_json_output_argument(remove_tensor)
    remove_tensor.set_defaults(func=_run_pipeline_remove_tensor)

    add_op = pipeline_subparsers.add_parser(
        "add-op", help="Append an operator.", allow_abbrev=False
    )
    add_op.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    add_op.add_argument("op_type", help="Canonical XR_SECURE_MR_OPERATOR_TYPE_*_PICO operator name.")
    for switch in sorted(set(all_input_switches()) | set(all_output_switches())):
        add_op.add_argument(
            switch,
            dest="operator_bindings",
            action=_StoreOperatorBinding,
            metavar="TENSOR",
            help="Named operator input or result tensor. Availability depends on OP_TYPE.",
        )
    add_op.add_argument(
        "--named-input",
        action="append",
        default=[],
        metavar="NAME=TENSOR",
        help="Dynamic model or JavaScript input binding. Repeatable.",
    )
    add_op.add_argument(
        "--named-output",
        action="append",
        default=[],
        metavar="NAME=TENSOR",
        help="Dynamic model or JavaScript result binding. Repeatable.",
    )
    add_op.add_argument("--attr", action="append", default=[], help="Raw operator attr. Repeatable.")
    add_op.add_argument("--expression", help="Arithmetic expression for arithmetic operators.")
    add_op.add_argument("--dtype", help="Operator dtype hint when supported.")
    add_op.add_argument("--flag", help="Integer flag for operators such as cvt_color.")
    add_op.add_argument("--threshold", type=float, help="Threshold for operators such as nms.")
    add_op.add_argument("--comparison", help="Comparison operator for customized_compare.")
    add_op.add_argument("--script", help="JavaScript source for javascript operators.")
    add_op.add_argument("--config", help="Text render config for render_text.")
    add_op.add_argument("--update-type", help="Update command for update_gltf.")
    add_op.add_argument("--mode", help="Sort orientation for sort_mat.")
    add_op.add_argument("--model", help="Package-relative .tflite model path for run_algorithm.")
    add_op.add_argument("--model-name", help="Logical model name.")
    add_op.add_argument("--model-target", default="npu", help="Model target, defaults to npu.")
    add_op.add_argument("--cpu-target-num-threads", type=int, default=1, help="CPU thread count for CPU target.")
    add_op.add_argument("--entity-path", help="Entity path for update_component, starting with '/'.")
    add_op.add_argument("--property", dest="property", help="SceneGraphProperty name for update_component.")
    add_op.add_argument("--target-property", dest="target_property", help=argparse.SUPPRESS)
    add_op.add_argument("--src-slices", help="JSON or semicolon/comma slice array for assignment.")
    add_op.add_argument("--dst-slices", help="JSON or semicolon/comma slice array for assignment.")
    add_op.add_argument("--src-channel-slice", help="JSON or comma-separated source channel slice.")
    add_op.add_argument("--dst-channel-slice", help="JSON or comma-separated destination channel slice.")
    _add_json_output_argument(add_op)
    add_op.set_defaults(func=_run_pipeline_add_op)

    remove_op = pipeline_subparsers.add_parser("remove-op", help="Remove an operator by index.")
    remove_op.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    remove_op.add_argument("--index", type=int, required=True, help="Operator index from pipeline inspect.")
    _add_json_output_argument(remove_op)
    remove_op.set_defaults(func=_run_pipeline_remove_op)

    set_input = pipeline_subparsers.add_parser("set-input", help="Set top-level pipeline inputs.")
    set_input.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    set_input.add_argument("names", nargs="+", help="Input tensor names.")
    _add_json_output_argument(set_input)
    set_input.set_defaults(func=_run_pipeline_set_input)

    set_output = pipeline_subparsers.add_parser("set-output", help="Set top-level pipeline outputs.")
    set_output.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    set_output.add_argument("names", nargs="+", help="Output tensor names.")
    _add_json_output_argument(set_output)
    set_output.set_defaults(func=_run_pipeline_set_output)

    validate = pipeline_subparsers.add_parser("validate", help="Validate a pipeline JSON file.")
    validate.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    _add_json_output_argument(validate)
    validate.set_defaults(func=_run_pipeline_validate)

    inspect = pipeline_subparsers.add_parser("inspect", help="Print a pipeline summary.")
    inspect.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    _add_json_output_argument(inspect)
    inspect.set_defaults(func=_run_pipeline_inspect)

    trace = pipeline_subparsers.add_parser("trace", help="Trace a Python function into pipeline JSON.")
    trace.add_argument("source", type=Path, help="Python source file containing a @trace-decorated function.")
    trace.add_argument("--function", required=True, help="Function name to trace.")
    trace.add_argument("--input", action="append", default=[], help="Trace input in name=path format. Repeatable.")
    trace.add_argument("--output", type=Path, required=True, help="Output pipeline JSON path.")
    _add_json_output_argument(trace)
    trace.set_defaults(func=_run_pipeline_trace)


def _add_package_parser(subparsers: argparse._SubParsersAction) -> None:
    package = subparsers.add_parser("package", help="Create, validate, and inspect SpatialML packages.")
    package_subparsers = package.add_subparsers(dest="package_command", required=True)

    package_create = package_subparsers.add_parser("create", help="Create a package directory or zip.")
    package_create.add_argument("source", type=Path, nargs="?", help="Source package root/zip, or asset root for loose pipelines.")
    package_create.add_argument("--id", dest="package_id", help="Package id.")
    package_create.add_argument("--pipeline", action="append", default=[], help="Pipeline in id=path format. Repeatable.")
    package_create.add_argument("--output", type=Path, required=True, help="Output package directory or .zip path.")
    package_create.add_argument("--supported-mode", action="append", default=[], help="Supported mode: xr or spatial. Repeatable.")
    package_create.add_argument("--asset-root", type=Path, action="append", default=[], help="Additional root for resolving assets. Repeatable.")
    package_create.add_argument("--zip", action="store_true", help="Also create a zip archive; implied when --output ends with .zip.")
    package_create.add_argument("--force", action="store_true", help="Overwrite existing output without prompting.")
    package_create.add_argument("--yes", action="store_true", help="Answer yes to overwrite prompts.")
    _add_json_output_argument(package_create)
    package_create.set_defaults(func=_run_package_create)

    package_validate = package_subparsers.add_parser("validate", help="Validate a package directory or zip.")
    package_validate.add_argument("package", type=Path, help="Package directory or .zip path.")
    _add_json_output_argument(package_validate)
    package_validate.set_defaults(func=_run_package_validate)

    package_inspect = package_subparsers.add_parser("inspect", help="Print package summary.")
    package_inspect.add_argument("package", type=Path, help="Package directory or .zip path.")
    _add_json_output_argument(package_inspect)
    package_inspect.set_defaults(func=_run_package_inspect)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Run pipelines and packages.")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    run_host = run_subparsers.add_parser("host", help="Run a pipeline or package on the host Python executor.")
    run_host.add_argument("target", type=Path, help="Pipeline JSON, package directory, or package zip.")
    run_host.add_argument("--pipeline", action="append", default=[], help="Pipeline id to run from a package. Repeatable.")
    run_host.add_argument("--input", action="append", default=[], help="Input tensor in name=path format. Repeatable.")
    run_host.add_argument("--dump", action="append", default=[], help="Tensor name to dump, or 'all'. Repeatable.")
    run_host.add_argument("--output-dir", type=Path, help="Directory for .npy output tensors.")
    run_host.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Accepted for parity with device runs; host execution currently runs one pass.",
    )
    _add_json_output_argument(run_host)
    run_host.set_defaults(func=_run_host)

    run_device = run_subparsers.add_parser("device", help="Run a package on a device through a runner APK.")
    run_device.add_argument("target", type=Path, help="Package directory or package zip.")
    run_device.add_argument(
        "--mode",
        choices=["xr", "spatial"],
        help="Runner mode. Defaults to manifest runtime.supported_modes, preferring spatial.",
    )
    run_device.add_argument(
        "--input",
        action="append",
        default=[],
        help="Image/raw input path or tensor=path binding. Repeatable.",
    )
    run_device.add_argument("--pipeline", action="append", default=[], help="Pipeline id to run from a package. Repeatable.")
    run_device.add_argument("--output-dir", type=Path, help="Local directory for pulled device outputs.")
    run_device.add_argument("--dump", action="append", default=[], help="Use 'all' to dump every readable numeric device tensor.")
    run_device.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Max seconds to wait for one-shot output; fixed run time for loop/keep-running.",
    )
    run_device.add_argument("--loop", action="store_true", help="Run the pipeline repeatedly on device.")
    run_device.add_argument("--keep-running", action="store_true", help="Leave the runner app running after pulling outputs.")
    run_device.add_argument("--use-vst", action="store_true", help="Use device VST camera input instead of --input.")
    run_device.add_argument(
        "--backend",
        choices=["npu", "gpu", "cpu"],
        help="Override model backend for device run without changing the package.",
    )
    run_device.add_argument("--interval-ms", type=int, default=50, help="Loop interval in milliseconds.")
    run_device.add_argument("--apk", type=Path, help="Runner APK path.")
    run_device.add_argument("--device", help="ADB device serial.")
    _add_json_output_argument(run_device)
    run_device.set_defaults(func=_run_device)


def _add_json_output_argument(parser: argparse.ArgumentParser, *, help: str = "Print JSON output.") -> None:
    parser.add_argument("--json", action="store_true", help=help)
    parser.add_argument(
        "--format",
        choices=["json"],
        dest="output_format",
        help="Output format. Currently only 'json' is supported.",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run pySpatialML CLI."""
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if getattr(args, "command", None) == "pipeline" and getattr(args, "pipeline_command", None) == "add-op":
            try:
                args.operator_bindings = [
                    *(getattr(args, "operator_bindings", None) or []),
                    *_parse_dynamic_operator_inputs(unknown),
                ]
            except CliError as exc:
                parser.error(str(exc))
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if getattr(args, "output_format", None) == "json":
        args.json = True
    try:
        return args.func(args)
    except LiteRTToolError as exc:
        if _json_requested(args):
            _print_error_json("PSM_LITERT_UNAVAILABLE", "litert_tool", str(exc), exit_code=2)
            return 2
        print_litert_error(exc)
        return 2
    except subprocess.CalledProcessError as exc:
        if _json_requested(args):
            _print_error_json(
                "PSM_SUBPROCESS",
                "subprocess",
                str(exc),
                exit_code=exc.returncode or 3,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
            return exc.returncode or 3
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode or 3
    except Exception as exc:
        handled = _handle_domain_error(args, exc)
        if handled is not None:
            return handled
        raise


def _run_litert_tool(args: argparse.Namespace) -> int:
    if args.action in {"install", "repair"}:
        force = bool(args.force or args.action == "repair")
        install = repair_litert_cli if args.action == "repair" else install_litert_cli
        kwargs = {
            "cache_dir": args.tool_cache,
            "package": args.package,
            "version": args.version,
        }
        if args.action == "install":
            kwargs["force"] = force
        cli = install(**kwargs)
        if args.json:
            _print_json(
                {
                    "ok": True,
                    "command": f"tools litert {args.action}",
                    "path": str(cli.path),
                    "managed": cli.managed,
                    "package": args.package,
                    "version": args.version,
                    "recreated": force,
                }
            )
            return 0
        action = "repaired" if args.action == "repair" else "installed"
        print(f"LiteRT CLI {action}: {cli.path}")
        return 0

    cli = resolve_litert_cli(cache_dir=args.tool_cache)
    source = "managed" if cli.managed else "system"
    version = cli.version()
    if args.json:
        _print_json(
            {
                "ok": True,
                "command": "tools litert status",
                "path": str(cli.path),
                "source": source,
                "managed": cli.managed,
                "version": version,
                "managed_cache": str(args.tool_cache or default_tool_cache()),
                "managed_path": str(managed_litert_bin(args.tool_cache)),
            }
        )
        return 0
    print(f"LiteRT CLI: {cli.path}")
    print(f"Source: {source}")
    print(f"Version: {version}")
    print(f"Managed cache: {args.tool_cache or default_tool_cache()}")
    print(f"Managed path: {managed_litert_bin(args.tool_cache)}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    compare_cli = _domain_module("compare_cli")
    return compare_cli.compare_paths(
        args.expected,
        args.actual,
        rtol=args.rtol,
        atol=args.atol,
        as_json=args.json,
    )


def _run_model_info(args: argparse.Namespace) -> int:
    model_cli = _domain_module("model_cli")
    return model_cli.model_info(
        args.model,
        signature_index=args.signature_index,
        as_json=args.json,
    )


def _run_operator_list(args: argparse.Namespace) -> int:
    operator_cli = _domain_module("operator_cli")
    return operator_cli.list_operators(as_json=args.json)


def _run_operator_describe(args: argparse.Namespace) -> int:
    operator_cli = _domain_module("operator_cli")
    return operator_cli.describe_operator(args.name, as_json=args.json)


def _run_litert_model_command(args: argparse.Namespace) -> int:
    litert_args = _strip_remainder_separator(args.litert_args)
    if args.litert_command == "convert" and _help_requested(litert_args):
        return _run_model_convert_help(args, litert_args)
    if args.litert_command == "convert":
        _validate_model_convert_input(litert_args)
        onnx_model = _onnx_convert_input(litert_args)
        if onnx_model is not None:
            return _run_onnx_model_convert(args, onnx_model, litert_args)
    cli = resolve_litert_cli(ensure=True, cache_dir=args.tool_cache)
    env = dict(os.environ)
    env.setdefault("UV_SYSTEM_CERTS", "true")
    argv = [str(cli.path), args.litert_command, *litert_args]
    if _json_requested(args):
        result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _print_json(
            {
                "ok": result.returncode == 0,
                "command": f"model {args.litert_command}",
                "litert": str(cli.path),
                "argv": argv,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result.returncode
    result = subprocess.run(argv, env=env)
    return result.returncode


def _run_model_convert_help(args: argparse.Namespace, litert_args: Sequence[str]) -> int:
    cli = resolve_litert_cli(ensure=True, cache_dir=args.tool_cache)
    env = dict(os.environ)
    env.setdefault("UV_SYSTEM_CERTS", "true")
    argv = [str(cli.path), "convert", *litert_args]
    result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    help_text = _model_convert_help_text(result.stdout or result.stderr)
    if _json_requested(args):
        _print_json(
            {
                "ok": result.returncode == 0,
                "command": "model convert",
                "litert": str(cli.path),
                "argv": argv,
                "returncode": result.returncode,
                "stdout": help_text,
                "stderr": "" if result.stdout else result.stderr,
            }
        )
        return result.returncode
    print(help_text, end="" if help_text.endswith("\n") else "\n")
    return result.returncode


def _model_convert_help_text(litert_help: str) -> str:
    prefix = """pySpatialML model convert

ONNX input:
  .onnx files are converted with pySpatialML's managed onnx2tf environment.

  Example:
    pyspatialml model convert -- model.onnx --output ./converted_tflite

  ONNX aliases:
    --input-shape NAME:DIMS          -> onnx2tf --overwrite_input_shape
    --shape-hint NAME:DIMS           -> onnx2tf --shape_hints
    --no-large-tensor                -> onnx2tf --no_large_tensor
    --keep-nchw NAME                 -> onnx2tf --keep_ncw_or_nchw_or_ncdhw_input_names
    --keep-nhwc NAME                 -> onnx2tf --keep_nwc_or_nhwc_or_ndhwc_input_names
    --non-verbose                    -> onnx2tf --non_verbose
    --copy-input-output-names        -> onnx2tf --copy_onnx_input_output_names_to_tflite
    --dynamic-range-quantize         -> onnx2tf --output_dynamic_range_quantized_tflite
    --integer-quantize               -> onnx2tf --output_integer_quantized_tflite
    --onnx2tf-arg VALUE              -> append raw onnx2tf argument

  Other ONNX-specific flags after the model path are passed to onnx2tf, except
  --output, which pySpatialML maps to onnx2tf's output directory.

Other supported inputs:
  Non-ONNX inputs are delegated to LiteRT convert.

"""
    if not litert_help:
        return prefix
    return prefix + "LiteRT convert help:\n" + litert_help


def _help_requested(args: Sequence[str]) -> bool:
    return any(value in {"-h", "--help"} for value in args)


def _run_onnx_model_convert(args: argparse.Namespace, model: Path, litert_args: Sequence[str]) -> int:
    onnx_tools = _domain_module("onnx_tools")
    output = _convert_output_dir(litert_args)
    _validate_onnx_shape_aliases(litert_args)
    extra_args = _onnx_extra_args(litert_args)
    result = onnx_tools.convert_onnx_to_tflite(
        model=model,
        output=output,
        extra_args=extra_args,
        cache_dir=args.tool_cache,
        verbose=not _json_requested(args),
    )
    if _json_requested(args):
        _print_json(
            {
                "ok": True,
                "command": "model convert",
                "converter": "onnx2tf",
                "model": str(result.model),
                "output": str(result.output),
                "tflite_models": [str(path) for path in result.tflite_models],
                "tool": str(result.tool.path),
                "managed": result.tool.managed,
                "argv": result.argv,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return 0
    print(f"Converted ONNX model: {result.model}")
    print(f"Output directory: {result.output}")
    for path in result.tflite_models:
        print(f"TFLite model: {path}")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return 0


def _onnx_convert_input(litert_args: Sequence[str]) -> Optional[Path]:
    for value in litert_args:
        if value.startswith("-"):
            continue
        if Path(value).suffix.lower() == ".onnx":
            return Path(value)
    return None


def _validate_model_convert_input(litert_args: Sequence[str]) -> None:
    model_value = _model_convert_input(litert_args)
    if model_value is None:
        return
    model = Path(model_value)
    if _looks_like_local_model_path(model_value) and not model.exists():
        onnx_tools = _domain_module("onnx_tools")
        raise onnx_tools.OnnxToolError(
            f"Model input file not found: {model}. "
            "Pass an existing model file, such as .onnx, .tflite, .pth, or a conversion script. "
            "If this is a model name rather than a file path, use the format expected by the LiteRT CLI."
        )


def _model_convert_input(litert_args: Sequence[str]) -> Optional[str]:
    options_with_values = {
        "--output",
        "--input-shape",
        "--shape-hint",
        "--keep-nchw",
        "--keep-nhwc",
        "--onnx2tf-arg",
        "--model-func",
        "--input-func",
        "--target",
        "--export-aipack",
        "--quantize-recipe",
        "--quantize",
        "--model-args",
        "--prefill-lengths",
        "--cache-length",
        "--script",
    }
    index = 0
    while index < len(litert_args):
        value = litert_args[index]
        if value == "--":
            index += 1
            continue
        if value in options_with_values:
            index += 2
            continue
        if any(value.startswith(option + "=") for option in options_with_values):
            index += 1
            continue
        if value.startswith("-"):
            index += 1
            continue
        return value
    return None


def _looks_like_local_model_path(value: str) -> bool:
    path = Path(value)
    if path.is_absolute() or value.startswith(("./", "../", "~/")):
        return True
    return path.suffix.lower() in {
        ".onn",
        ".onnx",
        ".tflite",
        ".lite",
        ".pt",
        ".pth",
        ".ckpt",
        ".py",
    }


def _convert_output_dir(litert_args: Sequence[str]) -> Path:
    onnx_tools = _domain_module("onnx_tools")
    for index, value in enumerate(litert_args):
        if value == "--output":
            if index + 1 >= len(litert_args):
                raise onnx_tools.OnnxToolError("--output requires a directory for ONNX conversion")
            return Path(litert_args[index + 1])
        if value.startswith("--output="):
            return Path(value.split("=", 1)[1])
    raise onnx_tools.OnnxToolError("ONNX conversion requires --output DIRECTORY")


def _validate_onnx_shape_aliases(litert_args: Sequence[str]) -> None:
    onnx_tools = _domain_module("onnx_tools")
    input_shapes = set(_onnx_alias_names(litert_args, "--input-shape"))
    shape_hints = set(_onnx_alias_names(litert_args, "--shape-hint"))
    overlap = sorted(input_shapes & shape_hints)
    if overlap:
        names = ", ".join(overlap)
        raise onnx_tools.OnnxToolError(
            f"Do not pass both --input-shape and --shape-hint for the same ONNX input: {names}. "
            "--input-shape fixes or overwrites the model input shape; --shape-hint is only for "
            "dynamic input dimensions when you are not overwriting that same input."
        )


def _onnx_alias_names(args: Sequence[str], alias: str) -> list[str]:
    names = []
    index = 0
    while index < len(args):
        value = args[index]
        raw = None
        if value == alias and index + 1 < len(args):
            raw = args[index + 1]
            index += 2
        elif value.startswith(alias + "="):
            raw = value.split("=", 1)[1]
            index += 1
        else:
            index += 1
        if raw:
            names.append(raw.split(":", 1)[0])
    return names


def _onnx_extra_args(litert_args: Sequence[str]) -> list[str]:
    extra_args = []
    index = 0
    skipped_model = False
    while index < len(litert_args):
        value = litert_args[index]
        if not skipped_model and not value.startswith("-") and Path(value).suffix.lower() == ".onnx":
            skipped_model = True
            index += 1
            continue
        if value == "--output":
            index += 2
            continue
        if value.startswith("--output="):
            index += 1
            continue
        consumed = _append_onnx_extra_arg_alias(litert_args, index, extra_args)
        if consumed:
            index += consumed
            continue
        extra_args.append(value)
        index += 1
    return extra_args


def _append_onnx_extra_arg_alias(args: Sequence[str], index: int, extra_args: list[str]) -> int:
    onnx_tools = _domain_module("onnx_tools")
    value = args[index]
    alias_flags = {
        "--no-large-tensor": "--no_large_tensor",
        "--non-verbose": "--non_verbose",
        "--copy-input-output-names": "--copy_onnx_input_output_names_to_tflite",
        "--dynamic-range-quantize": "--output_dynamic_range_quantized_tflite",
        "--integer-quantize": "--output_integer_quantized_tflite",
    }
    if value in alias_flags:
        extra_args.append(alias_flags[value])
        return 1

    value_aliases = {
        "--input-shape": "--overwrite_input_shape",
        "--shape-hint": "--shape_hints",
        "--keep-nchw": "--keep_ncw_or_nchw_or_ncdhw_input_names",
        "--keep-nhwc": "--keep_nwc_or_nhwc_or_ndhwc_input_names",
    }
    if value in value_aliases:
        if index + 1 >= len(args):
            raise onnx_tools.OnnxToolError(f"{value} requires a value")
        extra_args.extend([value_aliases[value], args[index + 1]])
        return 2
    for alias, target in value_aliases.items():
        if value.startswith(alias + "="):
            extra_args.extend([target, value.split("=", 1)[1]])
            return 1

    if value == "--onnx2tf-arg":
        if index + 1 >= len(args):
            raise onnx_tools.OnnxToolError("--onnx2tf-arg requires a value")
        extra_args.append(args[index + 1])
        return 2
    if value.startswith("--onnx2tf-arg="):
        extra_args.append(value.split("=", 1)[1])
        return 1

    return 0


def _strip_remainder_separator(args: Sequence[str]) -> list[str]:
    values = list(args)
    if values and values[0] == "--":
        return values[1:]
    return values


def _run_pipeline_init(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.init",
        lambda: pipeline_cli.init_pipeline(args.pipeline, force=args.force),
        {"pipeline": str(args.pipeline), "force": args.force},
    )


def _run_pipeline_add_tensor(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.add_tensor",
        lambda: pipeline_cli.add_tensor(
            args.pipeline,
            args.name,
            shape=args.shape,
            dtype=args.dtype,
            usage=args.usage,
            channels=args.channels,
            is_input=args.input,
            is_output=args.output,
            value=args.value,
            asset=args.asset,
        ),
        {
            "pipeline": str(args.pipeline),
            "tensor": args.name,
            "shape": args.shape,
            "dtype": args.dtype,
            "usage": args.usage,
            "channels": args.channels,
            "input": args.input,
            "output": args.output,
        },
    )


def _run_pipeline_remove_tensor(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.remove_tensor",
        lambda: pipeline_cli.remove_tensor(args.pipeline, args.name, force=args.force),
        {"pipeline": str(args.pipeline), "tensor": args.name, "force": args.force},
    )


def _run_pipeline_add_op(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    try:
        input_bindings, output_bindings = split_named_bindings(
            args.op_type, getattr(args, "operator_bindings", None) or []
        )
        inputs = order_named_inputs(
            args.op_type,
            input_bindings,
            args.named_input,
        )
        outputs = order_named_outputs(args.op_type, output_bindings, args.named_output)
    except OperatorInputError as exc:
        raise CliError(str(exc)) from exc
    return _json_action(
        args,
        "pipeline.add_op",
        lambda: pipeline_cli.add_op(
            args.pipeline,
            args.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=args.attr,
            expression=args.expression,
            dtype=args.dtype,
            flag=args.flag,
            threshold=args.threshold,
            comparison=args.comparison,
            script=args.script,
            config=args.config,
            text=None,
            update_type=args.update_type,
            mode=args.mode,
            model=args.model,
            model_name=args.model_name,
            model_target=args.model_target,
            cpu_target_num_threads=args.cpu_target_num_threads,
            scenegraph=None,
            entity_path=args.entity_path,
            property=args.property,
            target_property=args.target_property,
            data=None,
            src_slices=args.src_slices,
            dst_slices=args.dst_slices,
            src_channel_slice=args.src_channel_slice,
            dst_channel_slice=args.dst_channel_slice,
        ),
        {
            "pipeline": str(args.pipeline),
            "op_type": args.op_type,
            "inputs": inputs,
            "outputs": outputs,
        },
    )


def _parse_dynamic_operator_inputs(tokens: Sequence[str]) -> list[tuple[str, str]]:
    """Parse otherwise unknown ``--operand-name TENSOR`` pairs for dynamic ops."""
    values: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        switch = tokens[index]
        if not switch.startswith("--") or switch == "--":
            raise CliError(f"Unexpected add-op argument: {switch}")
        if "=" in switch:
            name, value = switch[2:].split("=", 1)
            if not name or not value:
                raise CliError(f"{switch} must use --NAME=TENSOR format")
            values.append((name, value))
            index += 1
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
            raise CliError(f"{switch} requires a tensor name")
        values.append((switch[2:], tokens[index + 1]))
        index += 2
    return values


def _run_pipeline_remove_op(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.remove_op",
        lambda: pipeline_cli.remove_op(args.pipeline, args.index),
        {"pipeline": str(args.pipeline), "index": args.index},
    )


def _run_pipeline_set_input(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.set_input",
        lambda: pipeline_cli.set_input(args.pipeline, args.names),
        {"pipeline": str(args.pipeline), "inputs": list(args.names)},
    )


def _run_pipeline_set_output(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.set_output",
        lambda: pipeline_cli.set_output(args.pipeline, args.names),
        {"pipeline": str(args.pipeline), "outputs": list(args.names)},
    )


def _run_pipeline_validate(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.validate",
        lambda: pipeline_cli.validate_pipeline(args.pipeline),
        {"pipeline": str(args.pipeline), "valid": True},
    )


def _run_pipeline_inspect(args: argparse.Namespace) -> int:
    if args.json:
        spec = _read_json(args.pipeline)
        operators = spec.get("operators", [])
        _print_json(
            {
                "ok": True,
                "command": "pipeline.inspect",
                "pipeline": str(args.pipeline),
                "tensors": len(spec.get("tensors", {})),
                "operators": len(operators),
                "inputs": spec.get("inputs", []),
                "outputs": spec.get("outputs", []),
                "operator_summaries": [
                    {
                        "index": index,
                        "type": op.get("type", "<unknown>") if isinstance(op, dict) else "<invalid>",
                        "inputs": op.get("inputs", []) if isinstance(op, dict) else [],
                        "outputs": op.get("outputs", []) if isinstance(op, dict) else [],
                    }
                    for index, op in enumerate(operators)
                ],
            }
        )
        return 0
    pipeline_cli = _domain_module("pipeline_cli")
    return pipeline_cli.inspect_pipeline(args.pipeline)


def _run_pipeline_trace(args: argparse.Namespace) -> int:
    pipeline_cli = _domain_module("pipeline_cli")
    return _json_action(
        args,
        "pipeline.trace",
        lambda: pipeline_cli.trace_pipeline(
            args.source,
            function_name=args.function,
            output=args.output,
            inputs=args.input,
        ),
        {"source": str(args.source), "function": args.function, "output": str(args.output)},
    )


def _run_package_create(args: argparse.Namespace) -> int:
    package_cli = _domain_module("package_cli")
    return _json_action(
        args,
        "package.create",
        lambda: package_cli.create_package(
            package_id=args.package_id or "",
            pipelines=args.pipeline,
            output=args.output,
            source=args.source,
            supported_modes=args.supported_mode,
            asset_roots=args.asset_root,
            force=args.force,
            zip_output=args.zip,
            assume_yes=args.yes,
        ),
        {
            "package_id": args.package_id,
            "pipelines": list(args.pipeline),
            "source": str(args.source) if args.source else None,
            "output": str(args.output),
            "zip": args.zip or args.output.suffix == ".zip",
        },
    )


def _run_package_validate(args: argparse.Namespace) -> int:
    package_cli = _domain_module("package_cli")
    return _json_action(
        args,
        "package.validate",
        lambda: package_cli.validate_package(args.package),
        {"package": str(args.package), "valid": True},
    )


def _run_package_inspect(args: argparse.Namespace) -> int:
    package_cli = _domain_module("package_cli")
    if args.json:
        root = package_cli._materialize_package(args.package)
        manifest = package_cli._load_validated_manifest(root)
        _print_json(
            {
                "ok": True,
                "command": "package.inspect",
                "package": str(args.package),
                "root": str(root),
                "manifest": manifest,
                "assets": package_cli._package_assets(root),
            }
        )
        return 0
    return package_cli.inspect_package(args.package)


def _run_host(args: argparse.Namespace) -> int:
    _preflight_run_target(args.target)
    run_cli = _domain_module("run_cli")
    if args.json:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_cli.run_host(
                args.target,
                pipeline_ids=args.pipeline,
                inputs=args.input,
                output_dir=args.output_dir,
                dumps=args.dump,
                duration=args.duration,
            )
        _print_json(
            {
                "ok": code == 0,
                "command": "run.host",
                "target": str(args.target),
                "pipelines": list(args.pipeline),
                "output_dir": str(args.output_dir) if args.output_dir else None,
                "dumps": list(args.dump),
                "duration": args.duration,
                "stdout": stdout.getvalue(),
            }
        )
        return code
    return run_cli.run_host(
        args.target,
        pipeline_ids=args.pipeline,
        inputs=args.input,
        output_dir=args.output_dir,
        dumps=args.dump,
        duration=args.duration,
    )


def _run_device(args: argparse.Namespace) -> int:
    _preflight_run_target(args.target)
    run_cli = _domain_module("run_cli")
    return run_cli.run_device(
        args.target,
        mode=args.mode,
        inputs=args.input,
        pipeline_ids=args.pipeline,
        output_dir=args.output_dir,
        dumps=args.dump,
        duration=args.duration,
        loop=args.loop,
        keep_running=args.keep_running,
        use_vst=args.use_vst,
        backend=args.backend,
        interval_ms=args.interval_ms,
        apk=args.apk,
        device=args.device,
        as_json=args.json,
    )


def _json_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False) or getattr(args, "output_format", None) == "json")


def _domain_module(name: str) -> ModuleType:
    module = globals().get(name)
    if isinstance(module, ModuleType):
        return module
    module = importlib.import_module(_DOMAIN_MODULES[name])
    globals()[name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    if name in _DOMAIN_MODULES:
        return _domain_module(name)
    raise AttributeError(name)


def _preflight_run_target(target: Path) -> None:
    if target.is_file() and target.suffix.lower() == ".json":
        raise RunTargetCliError(f"Raw pipeline JSON is not a valid run target: {target}. {_PACKAGE_TARGET_HINT}")


def _handle_domain_error(args: argparse.Namespace, exc: Exception) -> Optional[int]:
    error = _DOMAIN_ERRORS.get(type(exc).__name__)
    if error is None:
        return None
    code, category, prefix = error
    if _json_requested(args):
        _print_error_json(code, category, str(exc), exit_code=1)
        return 1
    print(f"{prefix}: {exc}", file=sys.stderr)
    return 1


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _print_error_json(
    code: str,
    category: str,
    message: str,
    *,
    exit_code: int,
    stdout: object = None,
    stderr: object = None,
) -> None:
    payload = {
        "ok": False,
        "error": {
            "code": code,
            "category": category,
            "message": message,
            "exit_code": exit_code,
        },
    }
    if stdout:
        payload["stdout"] = stdout
    if stderr:
        payload["stderr"] = stderr
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=sys.stderr)


def _json_action(args: argparse.Namespace, command: str, func, payload: dict) -> int:
    if not _json_requested(args):
        return func()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = func()
    result = {"ok": code == 0, "command": command, **payload}
    captured = stdout.getvalue()
    if captured:
        result["stdout"] = captured
    _print_json(result)
    return code


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise CliError(f"JSON file must contain an object: {path}")
    return payload


if __name__ == "__main__":
    sys.exit(main())
