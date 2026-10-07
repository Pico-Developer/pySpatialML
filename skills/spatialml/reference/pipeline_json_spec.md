# SpatialML Pipeline JSON Spec

This document consolidates how SecureMR pipelines are recorded to JSON by the Python tools and reconstructed by package deserializers.
Use it as a reference when authoring or reviewing pipeline specs.

## Package Manifest

Pipeline packages include a root `manifest.json` that points at pipeline JSON and package assets. Model metadata is carried inline by model inference operators. Use `runtime.supported_modes` to indicate whether the package is valid in XR mode, Spatial mode, or both:

```json
{
  "schema_version": "2",
  "id": "example-package",
  "pipelines": [{ "id": "main", "path": "pipeline/main.json" }],
  "runtime": { "supported_modes": ["xr", "spatial"] }
}
```

Allowed execution mode values are `xr` and `spatial`. Include only the modes supported by the operators and assets in that package.
Schema version 2 removes manifest-level `model` / `models` entries and external model metadata JSON files.

### Package-Relative Paths

Paths stored in a packaged pipeline must be relative to the package root. This
includes manifest pipeline entries, model `bin_path` values, and tensor `asset`
values. Paths must not be absolute or escape the package with `..` traversal.
Source files supplied to packaging tools may use host filesystem paths; the
generated package records the copied assets using package-relative paths.

Most operators are valid in both modes. The current mode-specific exceptions are:

| Operator type | XR mode | Spatial mode | Notes |
|---------------|---------|--------------|-------|
| `XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO` | yes | no | GLTF rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO` | yes | no | GLTF rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO` | yes | no | GLTF/text rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_UPLOAD_TEXTURE_TO_GLTF_PICO` | yes | no | GLTF texture upload path. |
| `XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO` | no | yes | Spatial scenegraph path. |
| `XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO` | no | yes | Spatial component update path. |

## Tensor Descriptors

Every tensor entry under `tensors` is an object with the following fields (`securemr/serialization.py:279-320`, `SecureMR_Samples/base/securemr_utils/serialization.cpp:479-499`):

| Key            | Type                 | Notes |
|----------------|---------------------|-------|
| `dimensions`   | array\<int>          | Spatial dimensions; flattened order follows SecureMR expectations. |
| `channels`     | int                  | Number of channels per element. |
| `data_type`    | int                  | Encodes `XrSecureMrTensorDataTypePICO`; see table below. |
| `is_placeholder` | bool              | Placeholders are allocated externally and become pipeline IO. |
| `usage`        | int                  | Numeric `XrSecureMrTensorTypePICO` value (e.g. 6 for MAT, 2 for OUTPUT). String usage names are unsupported. |
| `flag`         | int (optional)       | Bitmask combining data type with `smr.BaseType` modifiers (`smr.BaseType.MAT`, channel bits, etc.). |
| `data`        | array\<number> (optional) | Flattened tensor contents for preload. |
| `is_gltf`      | bool (optional)      | Marks GLTF placeholders that skip numeric attributes (`serialization.cpp:483-488`). |
| `asset`        | string (optional)    | Package-relative asset path for GLTF tensors. Package loaders use it to materialize scene graph tensors from GLTF assets. |

Usage-specific shape and channel rules are part of the descriptor contract: scalar tensors use one dimension and one channel; point tensors use one dimension and two or three channels; color tensors use one dimension, three or four channels, and UINT8 data; timestamp tensors use `dimensions: [1]`, four channels, and INT32 data; slice tensors use one dimension, two or three channels, and INT32 data.

`usage` must be an integer code. For MAT/matrix tensors (`usage: 6`), the
tensor must have at least two entries in `dimensions`. Use `[1, N]` or `[N, 1]`
for vector-shaped matrix data; use the appropriate numeric `usage`,
`dimensions`, and `channels` values for true one-dimensional scalar or point
arrays.

### Data Type Codes

Python exposes explicit mappings (`securemr/serialization.py:44-73`):

| Code | NumPy dtype | `smr.EDataType` |
|------|-------------|-----------------|
| 1    | `np.uint8`  | `UINT8` |
| 2    | `np.int8`   | `INT8` |
| 3    | `np.uint16` | `UINT16` |
| 4    | `np.int16`  | `INT16` |
| 5    | `np.int32`  | `INT32` |
| 6    | `np.float32`| `FLOAT32` |
| 7    | `np.float64`| `FLOAT64` |

`data_type` must use one of the numeric codes in the table above.

### Placeholder Semantics

- `is_placeholder: true` declares a pipeline tensor whose storage must be supplied by a compatible global tensor binding before execution. The loader rejects a submitted pipeline if any placeholder is left unbound.
- Tensors listed in the top-level `inputs` or `outputs` arrays must therefore be declared as placeholders. These arrays also define package-boundary ordering for runners and multi-pipeline composition.
- A placeholder does not necessarily need to appear in `inputs` or `outputs` when the loader provides another binding mechanism. For example, the XR loader binds GLTF placeholders from their package `asset`, and may bind tensors identified by runtime metadata.
- `is_placeholder: false` declares pipeline-local storage. A local numeric tensor may be initialized using its `data` field.

## Inline Model Metadata

Model inference operators carry their model metadata inline under the operator `model` key. For LiteRT/TFLite packages, use the shape produced by `securemr.pipeline_zoo.create_litert_model_spec`:

```json
{
  "bin_path": "model/model.tflite",
  "model_name": "main",
  "model_type": "tflite",
  "model_target": "npu",
  "cpu_target_num_threads": 1,
  "input": [
    {
      "name": "image",
      "shape": [1, 256, 256, 3],
      "encoding_type": "FP32"
    }
  ],
  "output": [
    {
      "name": "output",
      "shape": [1, 1],
      "encoding_type": "FP32"
    }
  ]
}
```

Model metadata fields are:

| Key | Type | Notes |
|-----|------|-------|
| `bin_path` | string | Package-relative model binary path. |
| `model_name` | string | Logical model name referenced by inference operators; package loaders commonly default to `main` when omitted. |
| `model_type` | string | Runtime family. New TFLite packages should use `tflite`. |
| `model_target` | string | Runtime backend. Use `npu` for NPU execution; `cpu` and `gpu` are also recognized where supported. |
| `cpu_target_num_threads` | int (optional) | CPU thread count, only meaningful when `model_target` is `cpu`. Omit it for NPU packages. |
| `input` / `output` | array\<object> | Model tensor metadata arrays. |
| `input[].name` / `output[].name` | string | Model graph node name. |
| `input[].shape` / `output[].shape` | array\<int> | Model tensor shape in model-runtime order. |
| `input[].encoding_type` / `output[].encoding_type` | string | Encoding such as `FP32`, `UINT8`, or other runtime-supported encodings. |

## Operator Entries

Common fields for every operator entry:

| Key        | Type          | Notes |
|------------|---------------|-------|
| `type`     | string        | Canonical `XR_SECURE_MR_OPERATOR_TYPE_*_PICO` enumerant name. |
| `inputs`   | array         | Positional tensor-reference objects. Each element must contain a `tensor` key and may contain an optional `name` binding. |
| `outputs`  | array         | Same rules as `inputs`. |
| `attrs`    | array\<string> (optional) | Operator configuration strings. Use this field for any operator-specific configuration. |

Every `inputs` and `outputs` entry uses the same object form:

```json
{
  "tensor": "package_tensor",
  "name": "logical_name"
}
```

`tensor` identifies the package tensor. `name` is optional and identifies a
model or JavaScript input/output when that logical name differs from the
package tensor name. Array position still determines the operator slot.

The short string form such as `"image"` is not part of the package JSON form.

### Operator Dictionary by `type`

The bullets below describe the package-facing operator contract. Operator-specific
configuration is provided through `attrs`.

#### `XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO`
- Positional signature: 0 inputs and 4 results. The results are ordered and named `right image`, `left image`, `timestamp`, and `camera matrix`.
- `right image` and `left image` are UINT8, 3- or 4-channel, 2D MAT tensors. Their dimensions normally match the VST session; a mismatch can cause runtime resizing.
- `timestamp` is a 4-channel INT32 VEC with shape `{1}`.
- `camera matrix` is a 1-channel FLOAT32/FLOAT64 3×3 MAT.
- It is Android-only and requires camera access.

#### `XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO`
- Positional signature: 1 input and 2 result slots. Input 0 is the timestamp operand, a 4-channel INT32 VEC with shape `{1}`.
- Input 0 is the timestamp tensor; JSON package connections remain positional through `inputs[0]`.
- Result 0 is `right`; result 1 is optional `left`. Each provided result must be a 1-channel floating-point 4×4 MAT.
- Result names are `right` and `left`; JSON package connections remain positional through `outputs[0]` and optional `outputs[1]`.

#### `XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO`
- Positional signature: 5 inputs and 1 result: `uv`, `timestamp`, `camera intrinsic`, `left image`, `right image`.
- `uv` is a 2-channel Point tensor. `timestamp` is a 4-channel VEC. `camera intrinsic` is a 3×3 MAT. Both image inputs are MAT tensors.
- The result is floating-point and must represent one 3D point per UV. Accepted layouts include a Point3 tensor, a 1-channel MAT shaped `{N,3}` or `{3,1}` for `N == 1`, or a 3-channel MAT shaped `{N,1}` or `{1,N}`.
- This operator is unavailable on emulators because depth data is unavailable.

#### `XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO`
- Positional signature: 2 inputs (`src`, `dst`) and 1 result (`result`).
- `src` and `dst` are floating-point, 2-channel point data with exactly three total points, represented as a one-dimensional `{3}` tensor or a 2D tensor with three total elements.
- `result` is a 1-channel MAT with shape `2×3`.
- Inline point arrays and tensor/value conversion are loader behavior, not part of the operator contract.

#### `XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO`
- Positional signature: 2 inputs (`affine`, `src image`) and 1 result (`dst image`).
- `affine` is a 1-channel MAT of shape `2×3`. `src image` and `dst image` are MAT tensors with 2D shapes; the result dimensions are the destination image dimensions.

#### `XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO`
- Positional signature: 2 inputs (`affine`, `src points`) and 1 result (`dst points`).
- `affine` is a 1-channel MAT of shape `2×3`. Point tensors are floating-point, 2-channel, and one-dimensional or 2D vector-shaped. The result uses the corresponding floating-point 2-channel point layout.

#### `XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO`
- The only supported JSON form uses `inputs[0]` for `src` and `outputs[0]` for `dst`.
- Slicing, when used, must be specified with the named fields `src_slices`, `dst_slices`, `src_channel_slice`, and `dst_channel_slice`. Dimension slice fields are arrays containing one to three integers per dimension (`start`, `end`, optional `step`); channel slice fields contain one to three integers.
- For example:
  ```json
  {
    "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
    "inputs": [{"tensor": "src"}],
    "outputs": [{"tensor": "dst"}],
    "src_slices": [[0, 10]],
    "dst_slices": [[5, 15]]
  }
  ```
- Omitting the slice fields performs a full copy. Assignment also performs element-wise type conversion when the selected source and destination regions differ in data type.
- Bracketed tensor references such as `src[0:10]` and `dst[5:15]`, and separate positional slice-tensor inputs, are unsupported and must not be used.

#### `XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO`
- Positional signature: 2 inputs and 1 result. Inputs 0 and 1 are the compared tensors; result 0 is the integer comparison result.
- Inputs and result must have matching dimensions and channel counts. The result must be an integer tensor.
- `attrs[0]` must contain one comparator: `==`, `!=`, `>`, `>=`, `<`, or `<=`.

#### `XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO`
- Positional signature: 1 input and 1 result. The input may be any tensor.
- The result must be an integer, 1-channel VEC with shape `{1}`. It is a whole-tensor logical AND reduction; there is no per-channel result form.

#### `XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO`
- Positional signature and result requirements are the same as `all`: one input and one integer, 1-channel VEC result with shape `{1}`.
- The operation is a whole-tensor logical OR/non-zero reduction; there is no optional per-channel result form.

#### `XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO`
- Positional signature: 1 input and 1 result.
- The input may use the supported numeric data types. The result must be an integer tensor.
- The result channel count must equal the input dimensionality, and the total result element count must equal the input channel count. This is a channel-wise argmax, not simply a one-channel MAT result.

#### `XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO`
- Positional signature: 1 input (`src`) and 1 result (`dst`), with no nullable slots.
- Both tensors must be MAT tensors. The conversion is performed by OpenCV `cvtColor`.
- `attrs[0]` must contain the numeric OpenCV color-conversion enum.

#### `XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO`
- Positional signature: 2 input slots (`source`/input 0 and optional `alpha_beta`/input 1) and 1 result.
- `alpha_beta`, when present, contains exactly two floating-point values: alpha and beta. It may be a floating-point VEC or another compatible two-value tensor.
- The source and result must have the same tensor type, rank, and shape. MAT, tensor-array MAT, and compatible scalar/point/tensor forms are supported by the implementation.
- `attrs` may be omitted or contain one value: `L1`, `L2` (default), `INF`, or `MINMAX`. Attribute matching is case-sensitive.

#### `XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO`
- Positional signature: 1 to 10 input slots and 1 result. Only inputs referenced by the expression need to be connected; inputs are MAT tensors.
- `attrs[0]` contains the expression. Supported syntax includes tensor references `{0}`, `{1}`, and so on, constants, parentheses, `+`, `-`, `*`, `/`, `^`, transpose, inverse, and trigonometric/hyperbolic functions (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`).
- The result is a MAT whose shape, OpenCV depth, and channel count match the evaluated expression.

#### `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MAX_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_OR_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_AND_PICO`
- Each operator has exactly 2 inputs and 1 result.
- Both inputs and the result must have identical ranks, dimensions, and channel counts.
- `or` and `and` require integer inputs and an integer result. `min`, `max`, and `multiply` accept numeric tensors.

#### `XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO`
- Positional signature: 2 inputs (`scores`, `boxes`) and 3 optional result slots (`scores`, `boxes`, `indices`).
- `scores` is a floating-point MAT shaped `N×1` or `1×N`. `boxes` is floating-point and may be a 1-channel `N×4` MAT or a 4-channel `N×1` MAT.
- Result 0 is floating-point scores, result 1 is floating-point boxes, and result 2 is integer indices. Provided result sizes must agree.
- `attrs` may be omitted or contain one numeric IoU threshold; the default is `0.95`.

#### `XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO`
- Positional signature: exactly 3 inputs (`object points`, `image points`, `camera matrix`) and 2 results (`rotation`, `translation`); optional result slots are not defined.
- Object points are floating-point 3-channel point data, image points are floating-point 2-channel point data, and the camera matrix is a 1-channel floating-point 3×3 MAT.
- Both results are 1-channel FLOAT64 MATs with three total elements.
- Any JSON point-array conversion is loader behavior and must produce these tensor forms before connection.

#### `XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO`
- Positional signature: 1 input (`input`) and 2 optional results (`sorted`, `indices`).
- The input and `sorted` result are one-channel VEC tensors with one-dimensional shape. `indices`, when present, is also one-channel, one-dimensional, and integer.
- Any provided result must have the same shape as the input.

#### `XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO`
- Positional signature: 1 input (`input`) and 2 optional results (`sorted`, `indices`).
- Input and results are 2D, one-channel MAT tensors. `indices` must be integer and all provided result shapes must match the input shape.
- `attrs` may be omitted or contain one value: uppercase `ROW` (default) or `COLUMN`.

#### `XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO`
- Positional signature: 1 input (`src`) and 3 result slots (`w`, `u`, `vt`). Each result slot may be null; provided results must be 1-channel floating-point 2D MAT tensors with the corresponding SVD shape.

#### `XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO`
- Positional signature: 1 input (`operand0`) and 1 result (`result0`).
- The input is a MAT tensor. The result must be a one-channel FLOAT32/FLOAT64 scalar tensor: a one-dimensional shape `{1}` or a shape whose every dimension is 1. It is not a per-channel reduction.
- `attrs` may be omitted or contain one value: `L1`, `L2` (default), or `INF`; more than one attribute is rejected.

#### `XR_SECURE_MR_OPERATOR_TYPE_CHW_HWC_PICO`
- Positional signature: 1 input and 1 result.
- One side must be an HWC 2D MAT with shape `[H,W]` and `C` channels. The other side must be a CHW 3D MAT tensor array with shape `[C,H,W]` and one channel.
- The input and result data types must match. The operator supports both HWC→CHW and CHW→HWC.

#### `XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO`
- Positional signature: 1 input (`operand`) and 1 result (`result`). Both are MAT tensors with 2D shapes.
- The result shape must equal the matrix inverse shape; the input must therefore be square for a normal inverse operation.

#### `XR_SECURE_MR_OPERATOR_TYPE_MAKE_TRANSFORM_MAT_PICO`
- Positional signature: 3 inputs (`rotation`, `translation`, optional `scale`) and 1 result (`result`).
- Rotation, translation, and provided scale are one-channel floating-point MAT tensors containing three values. The scale slot is nullable and defaults to identity.
- The result is a one-channel floating-point 4×4 MAT. Translation may be shaped 3×1 or 1×3.

#### `XR_SECURE_MR_OPERATOR_TYPE_UPLOAD_TEXTURE_TO_GLTF_PICO`
- Positional signature: 2 inputs (`gltf`, `rgb image`) and 1 result (`texture ID`).
- The GLTF input must be a GLTF tensor. The image is a UINT8, 3- or 4-channel MAT, either a 2D image or an array of such images.
- The result is a 1-channel UINT16 VEC containing one texture ID per uploaded image.

#### `XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO`
- Positional signature: exactly 4 inputs and 0 results: `gltf`, `world pose`, `visible`, and `view locked`.
- `gltf` is required and must be a GLTF tensor. `world pose` is nullable; when provided it must be a 1-channel FLOAT32/FLOAT64 4×4 MAT.
- `visible` is nullable and has no type restriction. Null means visible; a false/zero-valued tensor ends rendering, while a truthy tensor permits rendering to begin when the pose is valid.
- `view locked` is nullable and truthiness controls the renderer's `viewLocked` argument. It does not provide a pose.
- The operator begins rendering only when the pose is valid and `visible` is null or truthy; otherwise it ends rendering.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO`
- Positional signature: 3 input slots and 0 results. Input 0 is always `gltf`; the other slots depend on the required `attrs[0]` value.
- Supported `attrs[0]` values are `local`, `animation`, `world pose`, `texture`, `material::metallic_factor`, `material::roughness_factor`, `material::metallic_roughness_texture`, `material::base_color_factor`, `material::base_color_texture`, `material::normal_map_texture`, `material::occlusion_texture`, `material::emissive_factor`, `material::emissive_strength`, and `material::emissive_texture`.
- Operand names depend on `attrs[0]`: `node ID`/`transform`, `animation ID`/`animation timer`, `world pose`, `material ID`/`value`, or `texture ID`/`rgb image`. Unused slots remain null.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO`
- Positional signature: 6 inputs and 0 results: `text`, `start`, `colors`, `gltf`, `texture ID`, and `font size`.
- `text` is a required text-compatible tensor. `start` is a one-element FLOAT32/FLOAT64 Point2. `colors` is a two-element, 4-channel UINT8 VEC containing text RGBA and background RGBA.
- `gltf` is the target GLTF tensor. `texture ID` is a one-element UINT16 VEC identifying an existing RGBA texture. `font size` is a one-element FLOAT32/FLOAT64 VEC.
- `attrs[0]` must use the form `typeface#language#width#height`. Text is an input tensor, not `attrs[1]`, and this operator has no output tensor.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_SSMR_SWITCH_VISIBILITY_PICO`
- The package also accepts `XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO` as an alias for this operator.
- Positional signature: 2 inputs (`scenegraph`, `visible`) and 0 results.
- `scenegraph` is required and must be a GLTF tensor. `visible` is a nullable boolean-like tensor; null means visible and a truthy value means visible.
- `visible` is a tensor input, not an operator attribute or constructor setting.
- Spatial mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_SSMR_UPDATE_COMPONENT_PICO`
- The package also accepts `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO` as an alias for this operator.
- Positional signature: 2 inputs (`scenegraph`, `data`) and 0 results. `attrs[0]` contains the component path used to identify the target component.
- `scenegraph` is required and must be a GLTF tensor. `data` is validated by the target component callback and can represent supported strings, scalar factors, 3D/4D vectors, points, colors, 3×3/4×4 matrices, or UINT8 enum values.
- `attrs[0]` has the component-path form `/entity/path:component.field`; it must start with `/` and contain `:`. The entity path is split on `/` before the colon, and the component identifier uses the text after the colon.
- Spatial mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO`
- Positional signature: 0 inputs and 1 or 2 results: required `stereo audio`, followed by optional `timestamp`.
- `timestamp` is a 4-channel INT32 VEC with shape `{1}`.
- The audio result data type must match the encoding selected by `attrs[0]`: `PCM_16BIT` accepts UINT16 or INT16, `PCM_32BIT` accepts INT32, and `PCM_FLOAT` accepts FLOAT32. Stereo may be represented by two channels or a one-channel tensor whose final dimension is 2; left/right results are one-channel.
- Exactly one attribute is required. `attrs[0]` must use the form `<SAMPLE_RATE>;<PCM_16BIT|PCM_32BIT|PCM_FLOAT>`, where `SAMPLE_RATE` is a positive value within the supported audio sample-rate range. `PCM_8BIT` is not supported.

#### `XR_SECURE_MR_OPERATOR_TYPE_SPEAKER_PICO`
- Positional signature: 1 input (`audio`) and 0 results.
- The input must contain one or two channels and use UINT16/INT16, INT32, or FLOAT32 data for PCM16, PCM32, or PCM float playback. A one-channel tensor with a trailing dimension of 2 is treated as interleaved stereo.
- `attrs[0]` must contain the positive sample rate, within the supported audio sample-rate range.

#### `XR_SECURE_MR_OPERATOR_TYPE_DEPTH_PICO`
- Positional signature: 0 inputs and 1 result named `depth map`.
- The result must be a 1-channel FLOAT32/FLOAT64 2D MAT. If its dimensions differ from the camera depth map, the depth map is resized into the result.
- Construction fails on an emulator because depth camera data is unavailable.

#### `XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO`
- The package also accepts `XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO` as an alias for this operator.
- `attrs[0]` contains the JavaScript source.
- Input and result slots are independent named bindings. Every input binding is made available to the script and every result binding is written by the script; their counts and names do not need to match. Zero inputs are allowed, but at least one connected result is required.
- Each input/output object uses `tensor` for the package tensor and may use `name` for the JavaScript variable name. The loader must place the source in `attrs[0]`.

#### `XR_SECURE_MR_OPERATOR_TYPE_UNKNOWN_PICO`
- Reserved for unknown or pass-through operator records. No executable implementation is defined for this enum; package loaders and tests must not assume it executes.

#### `XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO`
- Positional signature is model-defined: the number of inputs and results comes from the LiteRT model's input/output count. Input and output names come from the model metadata.
- The package loader must provide the model metadata and runtime selection needed to initialize the LiteRT/TFLite model. The supported runtime type is `tflite`; supported backends are `cpu`, `gpu`, and `npu`.
- Tensor compatibility is checked against the actual model buffers, including element type and dimensions; it is not determined only by the JSON tensor descriptor.
- Inline schema-v2 `model` metadata, `bin_path`, and `model_target` are package fields. Each model input/output reference uses the common `{name, tensor}` object form when the model name differs from the package tensor name.

## Inputs and Outputs Arrays

- Besides marking placeholders, the `inputs` / `outputs` lists at the top level preserve the ordering expected by task wrappers (`securemr/serialization.py:432-447`, `serialization.cpp:520-528`).
- When tensors appear in both lists the Python helper converts them to locals to avoid conflicts (`securemr/serialization.py:662-669`).

## Saving and Loading Workflow

1. The extended Python `Pipeline` records every allocation and connection into `self.spec` while user code builds the graph (`securemr/serialization.py:264-390`).
2. Calling `Pipeline.save` writes the spec with UTF-8 encoding and pretty formatting (`securemr/serialization.py:536-540`).
3. The package loader validates `tensors` and `operators`, validates `attrs` and nullable positional references, and wires the pipeline. Any mismatch should produce a descriptive error.
