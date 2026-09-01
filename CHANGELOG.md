# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.5.0] - 2026-09-01

First release under the new **pySpatialML** name (formerly `pySecureMR`),
published on PyPI as `pyspatialml-pico`. Ground-up rework focused on SpatialML
pipeline authoring and a LiteRT/TFLite model workflow behind a single unified
`pyspatialml` CLI, with host and on-device pipeline execution.

### Added

- Unified `pyspatialml` CLI covering the full workflow: model, pipeline,
  package, run, and compare.
- LiteRT/TFLite model workflow: `pyspatialml tools litert {status,install,repair}`
  and `pyspatialml model {info,run,convert,quantize,benchmark,visualize}`,
  delegating to the Google AI Edge LiteRT CLI with managed on-demand install.
  ONNX conversion via a managed `onnx2tf`.
- Pipeline JSON builder and py2smr tracing:
  `pyspatialml pipeline {init,add-tensor,add-op,set-input,set-output,validate,inspect,trace}`.
- SpatialML pipeline package authoring:
  `pyspatialml package {create,validate,inspect}` with inline model metadata.
- Pipeline execution: `pyspatialml run host` (host CPU) and
  `pyspatialml run device` (connected XR device via bundled runner APK, with
  `--backend {npu,gpu,cpu}` override).
- `pyspatialml operator {list,describe-op}` and
  `pyspatialml compare EXPECTED ACTUAL`.
- Dual `xr` and `spatial` execution modes with dedicated `xr_pipeline_runner`
  and new `spatial_pipeline_runner` runner apps.
- Bundled runner APKs (`pyspatialml_xr_runner`, `pyspatialml_spatial_runner`)
  as package data.

### Changed

- Renamed project from `pySecureMR` to `pySpatialML`; console entry point is
  now `pyspatialml`. PyPI distribution name is `pyspatialml-pico`; import
  packages remain `securemr` and `pyspatialml`.
- Pipeline spec updated to schema version 2; rewritten `README.md`, SpatialML
  skill docs, and PyPI-ready `pyproject.toml`.
- Substantial `py2smr` improvements (converter, ops, tracer, verifier) with a
  much larger operator test suite.

### Removed

- QNN toolchain: `securemr/qnn/*`, `install_qnn`, `onnx_to_qnn`,
  `pytorch_to_qnn`, `qnn_model*`, and `qnn_to_pipeline`, replaced by the
  LiteRT/TFLite path.
- Native bindings: `securemr/bindings/*` (including `_securemr` and bundled
  OpenCV/SNPE `.so` files). The package is now pure Python.
- Legacy `securemr` inspect CLIs and large serialized example/model binaries.
