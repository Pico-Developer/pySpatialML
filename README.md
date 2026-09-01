# pySpatialML

<p align="center">
  <a  alt="python version">
      <img src="https://img.shields.io/badge/python-3.13-blue?logo=python" /></a>
  <a> <img src="https://img.shields.io/badge/spatial-ml-green" /></a>
  <a> <img src="https://img.shields.io/badge/os-linux-yellow" /></a>
  <a> <img src="https://img.shields.io/badge/os-windows(wsl2)-yellow" /></a>
</p>

SpatialML pipeline package and LiteRT/TFLite workflow tools.

SecureMR pipeline debugging can be difficult because intermediate operator
outputs are not normally easy to inspect from an app. pySpatialML provides CLI
tools to build, run, inspect, and compare pipelines without spending every
iteration inside the app runtime.

## Table of Contents

  * [Why pySpatialML](#why-pyspatialml)
  * [Install](#install)
     * [Pip](#pip)
     * [Manual install](#manual-install)
  * [Command summary](#command-summary)
  * [LiteRT CLI](#litert-cli)
  * [Pipeline builder](#pipeline-builder)
  * [Package authoring](#package-authoring)
  * [Run pipelines](#run-pipelines)
  * [Run test](#run-test)
  * [Supported operators](#supported-operators)

## Why pySpatialML?

pySpatialML focuses on SpatialML pipeline package authoring, pipeline JSON, and LiteRT/TFLite
model workflows. The existing `securemr` Python modules still expose SecureMR
operator bindings and py2smr tracing helpers, while the user-facing CLI is `pyspatialml`.

## Install

Use Python 3.13 for pySpatialML. LiteRT currently requires Python 3.13 for the
managed runtime and CLI environment.

### Pip

```bash
python3.13 -m pip install pyspatialml-pico
```

The distribution is published on PyPI as `pyspatialml-pico`, but the CLI is
still `pyspatialml` and the import packages are still `securemr` and
`pyspatialml`.

### Manual install
```bash
git clone https://github.com/Pico-Developer/pySpatialML
cd pySpatialML
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e "."
```
Check installation:
```bash
pyspatialml --version
python -c "import securemr"
```

## Command summary

Use `pyspatialml --help` or any subcommand `--help` for the full option list.

| Command | Purpose |
| --- | --- |
| `pyspatialml tools litert status` | Show which LiteRT CLI executable pySpatialML will use. |
| `pyspatialml tools litert install` | Install the managed LiteRT CLI into the pySpatialML tool cache. |
| `pyspatialml tools litert install --force` | Recreate and reinstall the managed LiteRT CLI. |
| `pyspatialml tools litert repair` | Repair a missing or corrupted managed LiteRT CLI install. |
| `pyspatialml model info MODEL.tflite` | Print LiteRT/TFLite model input and output metadata. |
| `pyspatialml model run -- ...` | Delegate `litert run` through pySpatialML tool resolution. |
| `pyspatialml model convert -- ...` | Convert supported models. |
| `pyspatialml model quantize -- ...` | Delegate `litert quantize`. |
| `pyspatialml model benchmark -- ...` | Delegate `litert benchmark`. |
| `pyspatialml model visualize -- ...` | Delegate `litert visualize`. |
| `pyspatialml visualize model -- ...` | Run LiteRT model visualization, currently Model Explorer. |
| `pyspatialml operator list` | List supported SecureMR/SpatialML operators. |
| `pyspatialml operator describe-op NAME` | Show one operator's enum name, JSON type, creator, and signature. |
| `pyspatialml pipeline init PIPELINE.json` | Create an empty pipeline JSON file. |
| `pyspatialml pipeline add-tensor PIPELINE.json NAME` | Add a tensor descriptor. |
| `pyspatialml pipeline add-op PIPELINE.json OP_TYPE` | Append an operator. |
| `pyspatialml pipeline set-input PIPELINE.json NAME...` | Set top-level pipeline input tensor names. |
| `pyspatialml pipeline set-output PIPELINE.json NAME...` | Set top-level pipeline output tensor names. |
| `pyspatialml pipeline validate PIPELINE.json` | Validate pipeline JSON. |
| `pyspatialml pipeline inspect PIPELINE.json` | Print a pipeline summary. |
| `pyspatialml pipeline trace SOURCE.py --function NAME --output PIPELINE.json` | Trace a decorated Python function into pipeline JSON. |
| `pyspatialml package create ...` | Create a SpatialML pipeline package directory or zip from one or more pipeline JSON files. |
| `pyspatialml package validate PACKAGE` | Validate a package directory or zip. |
| `pyspatialml package inspect PACKAGE` | Print a package summary. |
| `pyspatialml run host PACKAGE` | Run a package directory or zip on the host Python executor. |
| `pyspatialml run device PACKAGE` | Run a package directory or zip on a connected XR device through the bundled runner APK. |
| `pyspatialml compare EXPECTED ACTUAL` | Compare tensor output `.npy` files or directories. |

## LiteRT CLI

Model-level commands are delegated to Google AI Edge LiteRT CLI where possible.
PyTorch conversion is handled by the LiteRT CLI, while ONNX conversion is
handled by a managed `onnx2tf` install when an `.onnx` input is passed to
`pyspatialml model convert`. Host-side model inspection and pipeline execution
use the Python LiteRT runtime installed with pySpatialML. Delegated LiteRT CLI
commands first use an existing `litert` executable on `PATH`; otherwise
pySpatialML installs a managed copy into the tool cache when needed.

Traditional PyTorch checkpoints may need a small conversion script that
constructs the model, loads weights, and returns sample inputs. See the
[LiteRT CLI troubleshooting and tips](https://github.com/google-ai-edge/LiteRT-CLI#troubleshooting-and-tips)
for script-based conversion guidance.

```bash
pyspatialml tools litert status
pyspatialml tools litert install
pyspatialml tools litert repair
pyspatialml model convert -- <model> --output ./converted_tflite
pyspatialml model run -- --help
pyspatialml model benchmark -- --help
```

Set `PYSPATIALML_LITERT=/path/to/litert` to force a specific executable, or
`PYSPATIALML_TOOL_CACHE=/path/to/cache` to move the managed install location.

## Pipeline builder

Build and validate pipeline JSON files directly from the CLI:

```bash
pyspatialml pipeline init pipeline.json
pyspatialml pipeline add-tensor pipeline.json image --shape 128,128,3 --dtype uint8 --input
pyspatialml pipeline add-tensor pipeline.json image_f32 --shape 128,128,3 --dtype float32
pyspatialml pipeline add-tensor pipeline.json normalized --shape 128,128,3 --dtype float32 --output
pyspatialml pipeline add-op pipeline.json assignment --input image --output image_f32
pyspatialml pipeline add-op pipeline.json arithmetic --input image_f32 --output normalized --expression "{0} / 255.0"
pyspatialml pipeline validate pipeline.json
pyspatialml pipeline inspect pipeline.json
```

Trace a Python function decorated with `securemr.py2smr.trace`:

```bash
pyspatialml pipeline trace preprocess.py --function preprocess --input image=sample.npy --output pipeline.json
```

## Package authoring

Create a package from one or more pipeline JSON files. The command copies
pipelines into `pipeline/`, copies referenced `.tflite` models into `model/`,
copies referenced glTF assets into `gltf/`, and rewrites packaged pipeline paths
to match. Packages keep model metadata inline on model inference operators; they
do not use manifest-level `model` entries or external `model/model.json`
metadata files.

```bash
pyspatialml package create \
  --id face-demo \
  --pipeline detection=./detection.json \
  --pipeline display=./display.json \
  --supported-mode xr \
  --output ./face-demo-package

pyspatialml package validate ./face-demo-package
pyspatialml package inspect ./face-demo-package
```

Use `--asset-root` when referenced assets are not next to the source pipeline
or relative to the current working directory.

## Run pipelines

Run commands require a SpatialML pipeline package directory or zip containing
`manifest.json`; the manifest must point to valid pipeline JSON files. If you
only have pipeline JSON, create a package with `pyspatialml package create`
first.

Run a pipeline package on the host Python executor:

```bash
pyspatialml run host ./face-demo-package \
  --input ./face.jpg \
  --output-dir ./outputs
```

Host runs always execute LiteRT/TFLite model operators on CPU. Inputs can be
specified as `tensor=path` bindings, or as one bare image path for VST-style
left/right image inputs:

```bash
pyspatialml run host ./face-demo-package \
  --input vst_left_image=./left.jpg \
  --input vst_right_image=./right.jpg \
  --output-dir ./outputs
```

Package outputs are written under a per-pipeline directory:

```text
outputs/
  detection/
    post_det.npy
    post_det.json
  display/
    frame_pose.npy
    display_summary.json
```

Use `--dump all` to also write every host tensor under each pipeline's
`all_tensors/` directory.

Run a package on a connected XR device through the bundled runner APK:

```bash
pyspatialml run device ./face-demo-package.zip \
  --input ./face.jpg \
  --backend npu \
  --output-dir ./device-outputs
```

Device inputs support both forms:

```bash
pyspatialml run device ./face-demo-package.zip \
  --input vst_left_image=./left.jpg \
  --input vst_right_image=./right.jpg
```

`--backend {npu,gpu,cpu}` overrides model operator backends in the staged
device package only; it does not modify the source package. Device summaries
include LiteRT/Secure MR log lines, tensor shape/dtype/stat previews, and the
package runtime mode. Device outputs use the same per-pipeline layout:

```text
device-outputs/
  status.json
  detection/
    detection_post_det_1.bin
  display/
    display_frame_pose_1.bin
```

With `--dump all`, dump-only tensors are written to
`<pipeline>/all_tensors/` but are not printed in the terminal summary:

```text
device-outputs/
  display/
    all_tensors/
      display_post_det_1.bin
```

## Run test

```bash
python3.13 -m pytest
```
Refer to [test code](./tests) to learn more about the usage.

## Supported operators

Use the CLI for the current operator list and per-operator details:

```bash
pyspatialml operator list
pyspatialml operator describe-op RUN_MODEL_INFERENCE
```

The pipeline JSON reference is in
`skills/spatialml/reference/pipeline_json_spec.md`. Use full
`XR_SECURE_MR_OPERATOR_TYPE_*_PICO` names unless a specific runtime documents
additional aliases. `UNKNOWN` is a Python fallback/testing operator and should
not be used in production packages.

Visualize pipeline json

```
python3 -m securemr.viz.pipeline_viz path-to-pipeline.json
```

## How to contribute

Before coding, please install develop related tools by:
```
make env
```

For new features, unittest is required.
