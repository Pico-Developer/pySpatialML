# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Traceable operations for py2smr.

This module provides wrapped versions of common operations that automatically
record themselves to the current trace context when executed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from securemr.core.types import EOperatorType
from .tracer import get_current_trace

_OP_GET_TRANSFORM_MAT = EOperatorType.GET_TRANSFORM_MAT
_OP_LOAD_TEXTURE = getattr(EOperatorType, "LOAD_TEXTURE", getattr(EOperatorType, "UPLOAD_TEXTURE_TO_GLTF", None))
_OP_SWAP_HWC_CHW = getattr(EOperatorType, "SWAP_HWC_CHW", getattr(EOperatorType, "CHW_HWC", None))
_OP_JAVASCRIPT = EOperatorType.JAVASCRIPT
__all__ = [
    "arithmetic",
    "cvt_color",
    "normalize",
    "argmax",
    "elementwise_min",
    "elementwise_max",
    "elementwise_multiply",
    "elementwise_or",
    "elementwise_and",
    "customized_compare",
    "all",
    "any",
    "assignment",
    "nms",
    "get_affine",
    "get_transform_mat",
    "apply_affine",
    "apply_affine_point",
    "camera_space_to_world",
    "solve_p_n_p",
    "uv_to_3d_in_camera_space",
    "camera_access",
    "sort_vec",
    "sort_mat",
    "inversion",
    "svd",
    "norm",
    "swap_hwc_chw",
    "run_algorithm",
    "load_texture",
    "switch_gltf_render_status",
    "update_gltf",
    "render_text",
    "scenegraph_visibility",
    "update_component",
    "microphone",
    "speaker",
    "depth",
    "javascript",
    "unknown",
]


def _eval_arithmetic_expression(tensor: np.ndarray, expression: str) -> np.ndarray:
    """Evaluate an arithmetic expression on a tensor.

    Supports expressions like "{0} / 255.0", "{0} + {1}", etc.
    """
    tensors = list(tensor) if isinstance(tensor, (list, tuple)) else [tensor]
    expr = expression
    env = {"np": np}
    for index, value in enumerate(tensors):
        name = f"x{index}"
        expr = expr.replace("{" + str(index) + "}", name)
        env[name] = np.asarray(value).astype(np.float32)
    if "x0" in env:
        env["x"] = env["x0"]
    return eval(expr, {"__builtins__": {}}, env)


def arithmetic(
    tensor: np.ndarray,
    expression: str,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an arithmetic expression to a tensor.

    Corresponds to EOperatorType.ARITHMETIC_COMPOSE.

    Args:
        tensor: Input tensor.
        expression: Arithmetic expression with {0} as placeholder for input.
                   Examples: "{0} / 255.0", "{0} * 2.0 + 1.0"
        output_name: Optional name for the output tensor.

    Returns:
        Result tensor.
    """
    result = _eval_arithmetic_expression(tensor, expression)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ARITHMETIC_COMPOSE,
            attrs=[expression],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def cvt_color(
    tensor: np.ndarray,
    code: int,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Convert color space of an image tensor.

    Corresponds to EOperatorType.CONVERT_COLOR.

    Args:
        tensor: Input image tensor (H, W, C).
        code: OpenCV color conversion code (e.g., cv2.COLOR_BGR2RGB = 4).
        output_name: Optional name for the output tensor.

    Returns:
        Color-converted tensor.
    """
    try:
        import cv2
        result = cv2.cvtColor(tensor, code)
    except ImportError:
        # Fallback for common conversions without cv2
        if code == 4:  # COLOR_BGR2RGB
            result = tensor[..., ::-1].copy()
        elif code == 6:  # COLOR_BGR2GRAY
            # Simple grayscale conversion
            result = np.mean(tensor, axis=-1, keepdims=True).astype(tensor.dtype)
        else:
            raise ImportError("cv2 required for this color conversion")

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CONVERT_COLOR,
            attrs=[str(code)],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def normalize(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
    *,
    normalize_type: str = "L2",
    alpha_beta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize a tensor.

    Corresponds to EOperatorType.NORMALIZE.

    Args:
        tensor: Input tensor.
        normalize_type: Normalization mode: ``L1``, ``L2``, ``INF``, or
            ``MINMAX``. Defaults to ``L2``.
        alpha_beta: Optional floating-point tensor containing exactly two
            values, the alpha and beta parameters. Defaults to ``[1.0, 0.0]``.
        output_name: Optional name for the output tensor.

    Returns:
        Normalized tensor.
    """
    has_alpha_beta = alpha_beta is not None
    if has_alpha_beta:
        alpha_beta = np.asarray(alpha_beta)
        if alpha_beta.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("alpha_beta must have FLOAT32 or FLOAT64 dtype")
        if alpha_beta.size != 2:
            raise ValueError("alpha_beta must contain exactly two floating-point values")
        alpha, beta = (float(value) for value in alpha_beta.reshape(-1))
    else:
        alpha, beta = 1.0, 0.0

    mode = (normalize_type or "L2").upper()
    values = np.asarray(tensor, dtype=np.float32)
    if mode == "L2":
        scale = float(np.linalg.norm(values.astype(np.float64)))
        result = values.copy() if scale == 0.0 else values * (alpha / scale)
    elif mode == "L1":
        scale = float(np.sum(np.abs(values), dtype=np.float64))
        result = values.copy() if scale == 0.0 else values * (alpha / scale)
    elif mode == "INF":
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        result = values.copy() if scale == 0.0 else values * (alpha / scale)
    elif mode == "MINMAX" and has_alpha_beta:
        lower = float(np.min(values)) if values.size else 0.0
        upper = float(np.max(values)) if values.size else 0.0
        scale = upper - lower
        result = np.full_like(values, alpha) if scale == 0.0 else (values - lower) / scale * (beta - alpha) + alpha
    elif mode == "MINMAX":
        lower = float(np.min(values)) if values.size else 0.0
        upper = float(np.max(values)) if values.size else 0.0
        scale = upper - lower
        result = np.zeros_like(values) if scale == 0.0 else (values - lower) / scale
    else:
        raise ValueError("normalize_type must be L1, L2, INF, or MINMAX")

    ctx = get_current_trace()
    if ctx is not None:
        if alpha_beta is not None and ctx.get_tensor_name(alpha_beta) is None:
            # Constants need to be emitted as initialized local tensors before
            # record_op resolves the operand reference.
            ctx.register_tensor(alpha_beta, value=alpha_beta)
        ctx.record_op(
            op_type=EOperatorType.NORMALIZE,
            attrs=[mode] if mode != "L2" or has_alpha_beta else [],
            inputs=[tensor, alpha_beta] if alpha_beta is not None else [tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def argmax(
    tensor: np.ndarray,
    axis: int = -1,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute argmax along an axis.

    Corresponds to EOperatorType.ARGMAX.

    Args:
        tensor: Input tensor.
        axis: Axis along which to compute argmax.
        output_name: Optional name for the output tensor.

    Returns:
        Indices of maximum values.
    """
    # SecureMR argmax is channel-wise over spatial dimensions; axis is ignored.
    if tensor.ndim == 1:
        result = np.array([int(np.argmax(tensor))], dtype=np.int32)
    elif tensor.ndim == 2:
        idx = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
        result = np.array(idx, dtype=np.int32)
    elif tensor.ndim == 3:
        h, w, c = tensor.shape
        indices = []
        for ch in range(c):
            idx = np.unravel_index(int(np.argmax(tensor[:, :, ch])), (h, w))
            indices.extend(idx)
        result = np.array(indices, dtype=np.int32).reshape(c, 2)
    else:
        idx = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
        result = np.array(idx, dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ARGMAX,
            attrs=[str(axis)],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_min(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise minimum of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MIN.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise minimum.
    """
    result = np.minimum(a, b)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MIN,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_max(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise maximum of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MAX.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise maximum.
    """
    result = np.maximum(a, b)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MAX,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_multiply(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise multiplication of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MULTIPLY.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise product.
    """
    result = a * b

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MULTIPLY,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_or(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise logical OR of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_OR.
    """
    result = np.logical_or(a, b).astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_OR,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_and(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise logical AND of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_AND.
    """
    result = np.logical_and(a, b).astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_AND,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def customized_compare(
    a: np.ndarray,
    b: np.ndarray,
    compare: str,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise comparison between two tensors.

    Corresponds to EOperatorType.CUSTOMIZED_COMPARE.

    Args:
        a: First input tensor.
        b: Second input tensor.
        compare: Comparison operator string ("==", "!=", ">", ">=", "<", "<=").
        output_name: Optional name for the output tensor.

    Returns:
        Comparison result as int32 tensor (0 or 1).
    """
    if compare == "==":
        result = (a == b)
    elif compare == "!=":
        result = (a != b)
    elif compare == ">":
        result = (a > b)
    elif compare == ">=":
        result = (a >= b)
    elif compare == "<":
        result = (a < b)
    elif compare == "<=":
        result = (a <= b)
    else:
        raise ValueError(f"Unsupported compare operator: {compare}")

    result = result.astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CUSTOMIZED_COMPARE,
            attrs=[compare],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def all(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Check whether all elements are non-zero.

    Corresponds to EOperatorType.ALL.

    Args:
        tensor: Input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        A 1-element int32 vector (1 if all elements are non-zero, else 0).
    """
    value = 1 if np.all(tensor) else 0
    result = np.array([value], dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ALL,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def any(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Check whether any elements are non-zero.

    Corresponds to EOperatorType.ANY.

    Args:
        tensor: Input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        A 1-element int32 vector (1 if any element is non-zero, else 0).
    """
    value = 1 if np.any(tensor) else 0
    result = np.array([value], dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ANY,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def assignment(
    src: np.ndarray,
    dst: Optional[np.ndarray] = None,
    src_slices: Optional[List[List[int]]] = None,
    dst_slices: Optional[List[List[int]]] = None,
    src_channel_slice: Optional[Sequence[int]] = None,
    dst_channel_slice: Optional[Sequence[int]] = None,
    src_slices_tensor: Optional[Union[str, np.ndarray]] = None,
    dst_slices_tensor: Optional[Union[str, np.ndarray]] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Assign values from source to destination with optional slicing.

    Corresponds to EOperatorType.ASSIGNMENT.

    Args:
        src: Source tensor.
        dst: Destination tensor (will be copied).
        src_slices: Source slices as [[row_start, row_end], [col_start, col_end]].
        dst_slices: Destination slices as [[row_start, row_end], [col_start, col_end]].
        output_name: Optional name for the output tensor.

    Returns:
        Result tensor with assigned values.
    """
    def _slice(value: Sequence[int]) -> slice:
        if len(value) not in (2, 3):
            raise ValueError("slice descriptors must contain start, end, and optional step")
        start, end = int(value[0]), int(value[1])
        step = int(value[2]) if len(value) == 3 else 1
        if step == 0:
            raise ValueError("slice step must not be zero")
        # The pipeline format uses -1 as an open-ended end marker.
        return slice(start, None if end == -1 else end, step)

    def _index(
        descriptors: Optional[Sequence[Sequence[int]]],
        channel: Optional[Sequence[int]],
        rank: int,
    ) -> tuple[slice, ...]:
        index = [slice(None)] * rank
        if descriptors is not None:
            if len(descriptors) > rank:
                raise ValueError(f"slice rank {len(descriptors)} exceeds tensor rank {rank}")
            for axis, descriptor in enumerate(descriptors):
                index[axis] = _slice(descriptor)
        if channel is not None:
            if rank == 0:
                raise ValueError("channel slices require a tensor with channels")
            index[-1] = _slice(channel)
        return tuple(index)

    def _tensor_slice_value(value, inline_value, field):
        if value is None:
            return inline_value
        if inline_value is not None:
            raise ValueError(f"{field} cannot be combined with its inline slice")
        if isinstance(value, str):
            raise ValueError(f"{field} string references require a resolved tensor")
        return np.asarray(value).tolist()

    src_slices = _tensor_slice_value(src_slices_tensor, src_slices, "src_slices_tensor")
    dst_slices = _tensor_slice_value(dst_slices_tensor, dst_slices, "dst_slices_tensor")
    destination_was_omitted = dst is None
    if dst is None:
        dst = np.asarray(src).copy()
    src_index = _index(src_slices, src_channel_slice, src.ndim)
    dst_index = _index(dst_slices, dst_channel_slice, dst.ndim)
    src_data = np.asarray(src)[src_index]
    has_destination_slice = dst_slices is not None or dst_channel_slice is not None
    has_source_slice = src_slices is not None or src_channel_slice is not None
    if destination_was_omitted and not has_destination_slice:
        result = src_data.astype(src.dtype, copy=True)
    elif not has_source_slice and not has_destination_slice:
        result = src.astype(dst.dtype, copy=True)
    elif not has_destination_slice:
        result = src_data.astype(dst.dtype, copy=True)
    else:
        result = dst.copy()
        result[dst_index] = src_data

    ctx = get_current_trace()
    if ctx is not None:
        extra_info = {}
        if src_slices is not None and src_slices_tensor is None:
            extra_info["src_slices"] = src_slices
        if dst_slices is not None and dst_slices_tensor is None:
            extra_info["dst_slices"] = dst_slices
        if src_channel_slice is not None:
            extra_info["src_channel_slice"] = list(src_channel_slice)
        if dst_channel_slice is not None:
            extra_info["dst_channel_slice"] = list(dst_channel_slice)
        if isinstance(src_slices_tensor, str):
            extra_info["src_slices_tensor"] = src_slices_tensor
        elif isinstance(src_slices_tensor, np.ndarray):
            tensor_name = ctx.get_tensor_name(src_slices_tensor)
            if tensor_name is not None:
                extra_info["src_slices_tensor"] = tensor_name
        if isinstance(dst_slices_tensor, str):
            extra_info["dst_slices_tensor"] = dst_slices_tensor
        elif isinstance(dst_slices_tensor, np.ndarray):
            tensor_name = ctx.get_tensor_name(dst_slices_tensor)
            if tensor_name is not None:
                extra_info["dst_slices_tensor"] = tensor_name

        record_inputs = [src] if destination_was_omitted else [src, dst]
        ctx.record_op(
            op_type=EOperatorType.ASSIGNMENT,
            attrs=[],
            inputs=record_inputs,
            outputs=[result],
            output_names=[output_name] if output_name else None,
            extra_info=extra_info,
        )

    return result


def nms(
    scores: np.ndarray,
    boxes: np.ndarray,
    threshold: float = 0.5,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Non-maximum suppression.

    Corresponds to EOperatorType.NMS.

    Args:
        scores: Confidence scores of shape (N,).
        boxes: Bounding boxes of shape (N, 4) in format [x1, y1, x2, y2].
        threshold: IoU threshold for suppression.
        output_name: Optional name for the output tensor.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    result = np.array(keep, dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.NMS,
            attrs=[str(threshold)],
            inputs=[scores, boxes],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def apply_affine(
    affine: np.ndarray,
    image: np.ndarray,
    output_shape: Optional[Union[List[int], tuple]] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an affine transform to an image tensor.

    Corresponds to EOperatorType.APPLY_AFFINE.

    Args:
        affine: Affine matrix of shape (2, 3).
        image: Input image tensor (H, W) or (H, W, C).
        output_shape: Optional output shape (H, W) or (H, W, C). Defaults to input shape.
        output_name: Optional name for the output tensor.

    Returns:
        Affine-warped image tensor.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for apply_affine") from exc

    if output_shape is None:
        out_h, out_w = image.shape[:2]
    else:
        if len(output_shape) < 2:
            raise ValueError("output_shape must have at least 2 dimensions")
        out_h, out_w = int(output_shape[0]), int(output_shape[1])

    affine_mat = np.asarray(affine, dtype=np.float32)
    result = cv2.warpAffine(image, affine_mat, (out_w, out_h))

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.APPLY_AFFINE,
            attrs=[],
            inputs=[affine, image],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def apply_affine_point(
    affine: np.ndarray,
    points: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an affine transform to 2D points.

    Corresponds to EOperatorType.APPLY_AFFINE_POINT.

    Args:
        affine: Affine matrix of shape (2, 3).
        points: Input points tensor with 2 channels (e.g. (N, 1, 2) or (1, N, 2)).
        output_name: Optional name for the output tensor.

    Returns:
        Transformed points tensor with the same shape as input.
    """
    if affine.shape != (2, 3):
        raise ValueError("affine must be shape (2, 3)")

    try:
        import cv2
        result = cv2.transform(points, affine.astype(points.dtype, copy=False))
    except ImportError:
        pts = np.asarray(points, dtype=points.dtype)
        original_shape = pts.shape
        if pts.ndim == 2 and pts.shape[1] == 2:
            flat = pts
        elif pts.ndim == 3 and pts.shape[2] == 2:
            flat = pts.reshape(-1, 2)
        else:
            raise ValueError("points must have shape (N, 2) or (N, 1, 2)/(1, N, 2)")
        ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
        homo = np.concatenate([flat, ones], axis=1)
        transformed = homo @ affine.T
        result = transformed.reshape(original_shape)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.APPLY_AFFINE_POINT,
            attrs=[],
            inputs=[affine, points],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def camera_space_to_world(
    timestamp: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Query camera space to world transforms.

    Corresponds to EOperatorType.CAMERA_SPACE_TO_WORLD.

    Args:
        timestamp: Timestamp tensor (int32, 4-channel).
        output_names: Optional names for the right/left output tensors.

    Returns:
        Tuple of (right_eye_transform, left_eye_transform), each 4x4 float32.
    """
    _ = timestamp

    if np.all(np.asarray(timestamp) == 0):
        right = np.zeros((4, 4), dtype=np.float32)
        left = np.zeros((4, 4), dtype=np.float32)
    else:
        right = np.array(
            [
                [0.99999573, 0.0008423, 0.00279751, 0.0312826],
                [-0.00081392, 0.99994834, -0.01013152, 0.0397733],
                [-0.0028059, 0.0101292, 0.99994476, -0.03535056],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        left = np.array(
            [
                [0.99997533, 0.00673646, -0.00199215, -0.03277511],
                [-0.00674351, 0.99997094, -0.0035545, 0.03963064],
                [0.00196815, 0.00356784, 0.9999917, -0.03536495],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CAMERA_SPACE_TO_WORLD,
            attrs=[],
            inputs=[timestamp],
            outputs=[right, left],
            output_names=output_names,
        )

    return right, left


def get_affine(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute affine transform matrix from 3 source and destination points.

    Corresponds to EOperatorType.GET_AFFINE.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for get_affine") from exc

    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    if src.shape[0] != 3 or dst.shape[0] != 3:
        raise ValueError("get_affine expects exactly 3 (x, y) points for src and dst")

    result = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.GET_AFFINE,
            attrs=[],
            inputs=[src_points, dst_points],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def get_transform_mat(
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: Optional[np.ndarray] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Build a 4x4 transform matrix from rotation, translation, and optional scale.

    Corresponds to EOperatorType.GET_TRANSFORM_MAT.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for get_transform_mat") from exc

    rvec = np.asarray(rotation, dtype=np.float64).reshape(-1)
    tvec = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rvec.size != 3 or tvec.size != 3:
        raise ValueError("rotation and translation must have 3 elements")
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    rmat, _ = cv2.Rodrigues(rvec)

    if scale is None:
        scale_vec = np.ones(3, dtype=rmat.dtype)
    else:
        scale_vec = np.asarray(scale, dtype=rmat.dtype).reshape(-1)
        if scale_vec.size != 3:
            raise ValueError("scale must have 3 elements")
    scale_mat = np.diag(scale_vec.astype(rmat.dtype, copy=False))
    rs = rmat @ scale_mat

    result = np.eye(4, dtype=rs.dtype)
    result[:3, :3] = rs
    result[:3, 3] = tvec.reshape(3)

    ctx = get_current_trace()
    if ctx is not None:
        inputs = [rotation, translation]
        if scale is not None:
            inputs.append(scale)
        ctx.record_op(
            op_type=_OP_GET_TRANSFORM_MAT,
            attrs=[],
            inputs=inputs,
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result.astype(np.float32)


def inversion(
    mat: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Matrix inversion.

    Corresponds to EOperatorType.INVERSION.
    """
    result = np.linalg.inv(mat)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.INVERSION,
            attrs=[],
            inputs=[mat],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def norm(
    tensor: np.ndarray,
    norm_type: str = "L2",
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute tensor norm.

    Corresponds to EOperatorType.NORM.
    """
    norm_type = (norm_type or "L2").upper()
    if norm_type == "L2":
        value = np.linalg.norm(tensor.astype(np.float64))
    elif norm_type == "L1":
        value = np.linalg.norm(tensor.astype(np.float64), ord=1)
    elif norm_type == "INF":
        value = np.linalg.norm(tensor.astype(np.float64), ord=np.inf)
    else:
        raise ValueError("norm_type must be L1, L2, or INF")

    result = np.array([value], dtype=np.float32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.NORM,
            attrs=[norm_type],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def solve_p_n_p(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve PnP problem using OpenCV.

    Corresponds to EOperatorType.SOLVE_P_N_P.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for solve_p_n_p") from exc

    obj_pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    cam = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("cv2.solvePnP failed")

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SOLVE_P_N_P,
            attrs=[],
            inputs=[object_points, image_points, camera_matrix],
            outputs=[rvec, tvec],
            output_names=output_names,
        )

    return rvec, tvec


def sort_vec(
    vec: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort a 1D vector descending and return indices.

    Corresponds to EOperatorType.SORT_VEC.
    """
    flat = np.asarray(vec).reshape(-1)
    order = np.argsort(-flat)
    sorted_vec = flat[order]
    indices = order.astype(np.int32)

    sorted_vec = sorted_vec.reshape(vec.shape)
    indices = indices.reshape(vec.shape)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SORT_VEC,
            attrs=[],
            inputs=[vec],
            outputs=[sorted_vec, indices],
            output_names=output_names,
        )

    return sorted_vec, indices


def sort_mat(
    mat: np.ndarray,
    axis: str = "ROW",
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort matrix rows or columns in descending order.

    Corresponds to EOperatorType.SORT_MAT.
    """
    axis = (axis or "ROW").upper()
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError("sort_mat expects a 2D matrix")

    if axis == "ROW":
        indices = np.argsort(-arr, axis=1)
        sorted_mat = np.take_along_axis(arr, indices, axis=1)
    elif axis == "COLUMN":
        indices = np.argsort(-arr, axis=0)
        sorted_mat = np.take_along_axis(arr, indices, axis=0)
    else:
        raise ValueError("axis must be ROW or COLUMN")

    indices = indices.astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SORT_MAT,
            attrs=[axis],
            inputs=[mat],
            outputs=[sorted_mat, indices],
            output_names=output_names,
        )

    return sorted_mat, indices


def svd(
    mat: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition.

    Corresponds to EOperatorType.SVD.
    """
    try:
        import cv2
        w, u, vt = cv2.SVDecomp(np.asarray(mat, dtype=np.float64))
    except Exception:
        u, s, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64), full_matrices=False)
        w = s.reshape(-1, 1)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SVD,
            attrs=[],
            inputs=[mat],
            outputs=[w, u, vt],
            output_names=output_names,
        )

    return w.astype(np.float32), u.astype(np.float32), vt.astype(np.float32)


def swap_hwc_chw(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Swap between HWC and CHW layouts.

    Corresponds to EOperatorType.SWAP_HWC_CHW.
    """
    arr = np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError("swap_hwc_chw expects a 3D tensor")

    # SecureMR image tensors are normally HWC. For the inverse CHW form,
    # preserve the existing shape heuristic and transpose the data itself.
    if arr.shape[-1] <= 4 or arr.shape[0] > 4:
        result = np.transpose(arr, (2, 0, 1)).copy()
    else:
        result = np.transpose(arr, (1, 2, 0)).copy()

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=_OP_SWAP_HWC_CHW,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def uv_to_3d_in_camera_space(
    uv: np.ndarray,
    timestamp: np.ndarray,
    camera_matrix: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Estimate rectified-stereo UV points in camera space.

    Corresponds to EOperatorType.UV_TO_3D_IN_CAM_SPACE.
    """
    _ = timestamp
    uv_array = np.asarray(uv)
    if uv_array.size % 2 != 0:
        uv_flat = np.zeros((1, 2), dtype=np.float32)
        result = np.zeros((1, 3), dtype=np.float32)
    else:
        uv_flat = uv_array.reshape(-1, 2)
        camera = np.asarray(camera_matrix, dtype=np.float32)
        if camera.shape not in {(3, 3), (3, 4)}:
            raise ValueError("camera_matrix must have shape (3, 3) or (3, 4)")
        left = np.asarray(left_image)
        right = np.asarray(right_image)
        if left.ndim < 2 or right.ndim < 2 or left.shape[:2] != right.shape[:2]:
            raise ValueError("left_image and right_image must have matching image dimensions")
        fx, fy = float(camera[0, 0]), float(camera[1, 1])
        cx, cy = float(camera[0, 2]), float(camera[1, 2])
        if fx == 0.0 or fy == 0.0:
            raise ValueError("camera_matrix focal lengths must be non-zero")
        baseline = abs(float(camera[0, 3] / fx)) if camera.shape == (3, 4) else 0.06
        baseline = baseline or 0.06
        height, width = left.shape[:2]
        result = np.empty((uv_flat.shape[0], 3), dtype=np.float32)

        def pixel(image: np.ndarray, y: int, x: int) -> np.ndarray:
            return np.asarray(image[y, x], dtype=np.float32).reshape(-1)

        for index, (u, v) in enumerate(uv_flat.astype(np.float32)):
            x = int(np.clip(round(float(u)), 0, width - 1))
            y = int(np.clip(round(float(v)), 0, height - 1))
            reference = pixel(left, y, x)
            best_x = x
            best_error = float("inf")
            # Rectified stereo correspondence lies on the same image row.
            for candidate in range(max(x - 1, 0), -1, -1):
                error = float(np.mean(np.abs(reference - pixel(right, y, candidate))))
                if error < best_error:
                    best_error, best_x = error, candidate
                    if error == 0.0:
                        break
            disparity = float(x - best_x)
            # A zero disparity has no finite stereo depth. Use a documented,
            # deterministic unit-depth fallback for host-only execution.
            depth = fx * baseline / disparity if disparity > 0.0 else 1.0
            result[index] = (
                (float(u) - cx) * depth / fx,
                (float(v) - cy) * depth / fy,
                depth,
            )

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UV_TO_3D_IN_CAM_SPACE,
            attrs=[],
            inputs=[uv, timestamp, camera_matrix, left_image, right_image],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def camera_access(
    output_shapes: Optional[List[tuple]] = None,
    output_names: Optional[List[str]] = None,
    image_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stub for rectified VST access.

    Corresponds to EOperatorType.RECTIFIED_VST_ACCESS.
    """
    right = None
    left = None
    if image_path:
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                if output_shapes and len(output_shapes) >= 2:
                    target_right = output_shapes[0]
                    target_left = output_shapes[1]
                    if target_right[:2] != img.shape[:2]:
                        img = cv2.resize(img, (int(target_right[1]), int(target_right[0])))
                right = img.copy()
                left = img.copy()
        except Exception:
            right = None
            left = None

    if right is None or left is None:
        if output_shapes is None or len(output_shapes) < 4:
            right = np.zeros((1, 1, 3), dtype=np.uint8)
            left = np.zeros((1, 1, 3), dtype=np.uint8)
            timestamp = np.zeros((1, 4), dtype=np.int32)
            cam_mat = np.zeros((3, 3), dtype=np.float32)
        else:
            right = np.zeros(output_shapes[0], dtype=np.uint8)
            left = np.zeros(output_shapes[1], dtype=np.uint8)
            timestamp = np.zeros(output_shapes[2], dtype=np.int32)
            cam_mat = np.zeros(output_shapes[3], dtype=np.float32)
    else:
        if output_shapes and len(output_shapes) >= 4:
            timestamp = np.zeros(output_shapes[2], dtype=np.int32)
            cam_mat = np.zeros(output_shapes[3], dtype=np.float32)
        else:
            timestamp = np.zeros((1, 4), dtype=np.int32)
            cam_mat = np.zeros((3, 3), dtype=np.float32)

    # The host implementation has no physical camera calibration.  Still
    # provide a usable pinhole intrinsic matrix so downstream host operators
    # such as UV_TO_3D_IN_CAM_SPACE can execute against a supplied image.  A
    # device VST stream supplies its real calibration through this operator.
    if np.asarray(cam_mat).shape == (3, 3):
        image_shape = np.asarray(left).shape if left is not None else ()
        if len(image_shape) >= 2:
            height, width = float(image_shape[0]), float(image_shape[1])
        elif output_shapes and output_shapes[0] and len(output_shapes[0]) >= 2:
            height, width = float(output_shapes[0][0]), float(output_shapes[0][1])
        else:
            height, width = 1.0, 1.0
        cam_array = np.asarray(cam_mat, dtype=np.float32)
        if not np.isfinite(cam_array).all() or cam_array[0, 0] == 0.0 or cam_array[1, 1] == 0.0:
            cam_mat = np.array(
                [[width, 0.0, width * 0.5],
                 [0.0, height, height * 0.5],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.RECTIFIED_VST_ACCESS,
            attrs=[],
            inputs=[],
            outputs=[right, left, timestamp, cam_mat],
            output_names=output_names,
        )

    return right, left, timestamp, cam_mat


def run_algorithm(
    inputs: Dict[str, np.ndarray],
    model_file: str,
    model_name: str,
    output_names: List[str],
    output_shapes: Optional[List[tuple]] = None,
    output_dtypes: Optional[List[np.dtype]] = None,
    input_aliasing: Optional[Dict[str, str]] = None,
    output_aliasing: Optional[Dict[str, str]] = None,
    model_type: str = "tflite",
    model: Optional[Dict[str, Any]] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
) -> Dict[str, np.ndarray]:
    """Record a LiteRT/TFLite model inference operator.

    Corresponds to EOperatorType.RUN_MODEL_INFERENCE.

    Args:
        inputs: Dictionary of input tensors.
        model_file: Package-relative or local TFLite model path.
        model_name: Name of the model.
        output_names: List of output tensor names.
        output_shapes: Required list of output shapes for trace-time placeholder tensors.
        output_dtypes: Optional list of output dtypes. Defaults to float32.
        input_aliasing: Optional mapping from ``inputs`` keys to model node names.
        output_aliasing: Optional mapping from output tensor names to model node names.
        model_type: Serialized model runtime type. Only ``tflite`` is supported.
        model: Optional inline model spec to serialize on the operator.
        model_target: Serialized backend target for SpatialML packages.
        cpu_target_num_threads: Serialized CPU thread count when ``model_target`` is CPU.

    Returns:
        Dictionary of trace-time placeholder output tensors.
    """
    if not inputs:
        raise ValueError("run_algorithm requires at least one input tensor")
    if model_type != "tflite":
        raise ValueError("run_algorithm only supports LiteRT/TFLite model_type='tflite'")
    if not model_file.endswith(".tflite"):
        raise ValueError(f"Unsupported model format: {model_file}. Use a .tflite model file")
    if not output_names:
        raise ValueError("run_algorithm requires at least one output name")
    if not output_shapes or len(output_shapes) != len(output_names):
        raise ValueError("run_algorithm requires one output shape per output name")

    output_dtypes = output_dtypes or [np.float32] * len(output_names)
    if len(output_dtypes) != len(output_names):
        raise ValueError("run_algorithm requires one output dtype per output name")

    def _validate_aliasing(
        aliases: Optional[Dict[str, str]], available_names: List[str], field: str
    ) -> Dict[str, str]:
        aliases = dict(aliases or {})
        unknown = sorted(set(aliases) - set(available_names))
        if unknown:
            raise ValueError(f"{field} contains unknown tensor name(s): {', '.join(unknown)}")
        for source_name, model_node_name in aliases.items():
            if not isinstance(model_node_name, str) or not model_node_name:
                raise ValueError(f"{field}[{source_name!r}] must be a non-empty model node name")
        return aliases

    input_aliasing = _validate_aliasing(input_aliasing, list(inputs), "input_aliasing")
    output_aliasing = _validate_aliasing(output_aliasing, output_names, "output_aliasing")

    outputs: Dict[str, np.ndarray] = {}
    for name, shape, dtype in zip(output_names, output_shapes, output_dtypes):
        outputs[name] = np.zeros(tuple(shape), dtype=np.dtype(dtype))

    if model is None:
        def _model_encoding(dtype):
            return {
                np.dtype(np.uint8): "uint8",
                np.dtype(np.int8): "int8",
                np.dtype(np.uint16): "uint16",
                np.dtype(np.int16): "int16",
                np.dtype(np.int32): "int32",
                np.dtype(np.float32): "float32",
                np.dtype(np.float64): "float64",
            }[np.dtype(dtype)]

        model = {
            "bin_path": model_file,
            "model_name": model_name,
            "model_type": "tflite",
            "model_target": model_target,
            "input": [
                {"name": alias, "shape": list(np.asarray(tensor).shape),
                 "encoding_type": _model_encoding(np.asarray(tensor).dtype)}
                for alias, tensor in inputs.items()
            ],
            "output": [
                {"name": name, "shape": list(shape),
                 "encoding_type": _model_encoding(dtype)}
                for name, shape, dtype in zip(output_names, output_shapes, output_dtypes or [np.float32] * len(output_names))
            ],
            **({"cpu_target_num_threads": int(cpu_target_num_threads)} if model_target == "cpu" else {}),
        }
    else:
        model = dict(model)
        model.setdefault("bin_path", model_file)
        model.setdefault("model_name", model_name)
        model.setdefault("model_type", "tflite")
        model.setdefault("model_target", model_target)
        if model.get("model_target") == "cpu":
            model.setdefault("cpu_target_num_threads", int(cpu_target_num_threads))

    ctx = get_current_trace()
    if ctx is not None:
        input_refs = []
        for alias, tensor in inputs.items():
            tensor_name = ctx.get_tensor_name(tensor)
            if tensor_name is not None:
                input_refs.append({
                    "name": input_aliasing.get(alias, alias),
                    "tensor": tensor_name,
                })
        output_refs = [
            {"name": output_aliasing.get(name, name), "tensor": name}
            for name in output_names
        ]
        if model is not None:
            for field, refs in (("input", input_refs), ("output", output_refs)):
                metadata = model.get(field)
                if isinstance(metadata, list):
                    model[field] = [
                        {**item, "name": refs[index]["name"]}
                        if index < len(refs) and isinstance(item, dict) else item
                        for index, item in enumerate(metadata)
                    ]
        extra_info = {
            "model_name": model_name,
            "model_type": model_type,
            "model_target": model_target,
            "cpu_target_num_threads": int(cpu_target_num_threads),
            "input_aliasing": input_aliasing,
            "output_aliasing": output_aliasing,
            "output_names": output_names,
            "input_refs": input_refs,
            "output_refs": output_refs,
            "model": model,
            "output_shapes": [tuple(shape) for shape in output_shapes],
        }
        ctx.record_op(
            op_type=EOperatorType.RUN_MODEL_INFERENCE,
            attrs=[],
            inputs=list(inputs.values()),
            outputs=list(outputs.values()),
            output_names=output_names,
            extra_info=extra_info,
        )

    return outputs


def load_texture(
    gltf_placeholder: np.ndarray,
    texture_src: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Stub for loading a texture into glTF.

    Corresponds to EOperatorType.LOAD_TEXTURE.
    """
    _ = (gltf_placeholder, texture_src)
    result = np.array([0], dtype=np.uint16)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=_OP_LOAD_TEXTURE,
            attrs=[],
            inputs=[gltf_placeholder, texture_src],
            outputs=[result],
            output_names=[output_name] if output_name else None,
            extra_info={"gltf_input_index": 0, "texture_input_index": 1},
        )

    return result


def switch_gltf_render_status(
    gltf_placeholder: np.ndarray,
    pose: Optional[np.ndarray] = None,
) -> None:
    """Stub for toggling glTF render status.

    Corresponds to EOperatorType.SWITCH_GLTF_RENDER_STATUS.
    """
    _ = (gltf_placeholder, pose)
    ctx = get_current_trace()
    if ctx is not None:
        inputs = [gltf_placeholder]
        if pose is not None:
            inputs.append(pose)
        ctx.record_op(
            op_type=EOperatorType.SWITCH_GLTF_RENDER_STATUS,
            attrs=[],
            inputs=inputs,
            outputs=[],
            extra_info={
                "gltf_input_index": 0,
                "pose_input_index": 1 if pose is not None else None,
            },
        )


def update_gltf(
    gltf_placeholder: np.ndarray,
    update_type: str,
    operands: Optional[List[np.ndarray]] = None,
) -> None:
    """Stub for updating glTF attributes.

    Corresponds to EOperatorType.UPDATE_GLTF.
    """
    operands = list(operands or [])
    ctx = get_current_trace()
    material_texture_types = {
        "material::metallic_roughness_texture", "material::base_color_texture",
        "material::normal_map_texture", "material::occlusion_texture",
        "material::emissive_texture",
    }
    material_color_types = {"material::base_color_factor", "material::emissive_factor"}
    if ctx is not None and len(operands) < {
        "texture": 2, "animation": 2, "world pose": 1, "local": 2,
    }.get(update_type, 2):
        defaults = {
            "texture": [np.array([0], dtype=np.uint16), np.zeros((1, 1, 3), dtype=np.uint8)],
            "animation": [np.array([0], dtype=np.uint16), np.array([0.0], dtype=np.float32)],
            "world pose": [np.eye(4, dtype=np.float32)],
            "local": [np.array([0], dtype=np.uint16), np.eye(4, dtype=np.float32)],
        }.get(
            update_type,
            [
                np.array([0], dtype=np.uint16),
                np.array(
                    [0] if update_type in material_texture_types else
                    [0.0, 0.0, 0.0, 0.0] if update_type in material_color_types else [0.0],
                    dtype=np.uint16 if update_type in material_texture_types else np.float32,
                ),
            ],
        )
        operands.extend(defaults[len(operands):])
    required_operands = {
        "texture": 2,
        "animation": 2,
        "world pose": 1,
        "local": 2,
    }.get(update_type, 2)
    if len(operands) < required_operands:
        raise ValueError(f"update_gltf {update_type!r} requires {required_operands} operand tensor(s)")
    if ctx is not None:
        for operand in operands:
            ctx.register_tensor(np.asarray(operand), value=np.asarray(operand))
        ctx.record_op(
            op_type=EOperatorType.UPDATE_GLTF,
            attrs=[update_type],
            inputs=[gltf_placeholder, *operands[:2]],
            outputs=[],
            extra_info={
                "update_type": update_type,
            },
        )


def render_text(
    gltf_placeholder: np.ndarray,
    config: str,
    text: str,
    start: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    texture_id: Optional[np.ndarray] = None,
    font_size: Optional[np.ndarray] = None,
) -> None:
    """Stub for rendering text into a glTF texture.

    Corresponds to EOperatorType.RENDER_TEXT.
    """
    ctx = get_current_trace()
    if ctx is not None:
        def constant(value: np.ndarray) -> np.ndarray:
            value = np.asarray(value)
            ctx.register_tensor(value, value=value)
            return value

        text_tensor = constant(np.frombuffer(text.encode("utf-8"), dtype=np.uint8))
        start_tensor = constant(np.asarray(start if start is not None else [0.0, 0.0], dtype=np.float32))
        colors_tensor = constant(np.asarray(
            colors if colors is not None else [[255, 255, 255, 255], [0, 0, 0, 0]], dtype=np.uint8
        ))
        texture_tensor = constant(np.asarray(texture_id if texture_id is not None else [0], dtype=np.uint16))
        font_tensor = constant(np.asarray(font_size if font_size is not None else [16.0], dtype=np.float32))
        ctx.record_op(
            op_type=EOperatorType.RENDER_TEXT,
            attrs=[config],
            inputs=[text_tensor, start_tensor, colors_tensor, gltf_placeholder, texture_tensor, font_tensor],
            outputs=[],
            extra_info={
                "config": config,
                "text": text,
            },
        )


def scenegraph_visibility(
    scenegraph: np.ndarray,
    visible: Union[np.ndarray, bool] = True,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Stub for scenegraph visibility updates.

    Corresponds to EOperatorType.SCENEGRAPH_VISIBILITY.
    """
    _ = scenegraph
    visible_value = bool(np.asarray(visible).reshape(-1)[0]) if isinstance(visible, np.ndarray) else bool(visible)
    result = np.array([1 if visible_value else 0], dtype=np.uint8)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SCENEGRAPH_VISIBILITY,
            attrs=[],
            inputs=[scenegraph, visible] if isinstance(visible, np.ndarray) else [
                scenegraph, np.asarray([1 if visible_value else 0], dtype=np.uint8)
            ],
            outputs=[result],
            output_names=[output_name] if output_name else None,
            extra_info={
                "visible_input_index": 1,
            },
        )

    return result


def update_component(
    scenegraph: np.ndarray,
    data: np.ndarray,
    entity_path: str,
    property: str,
) -> np.ndarray:
    """Record a Spatial scene-graph component property update.

    Corresponds to EOperatorType.UPDATE_COMPONENT.
    """
    if not entity_path.startswith("/"):
        raise ValueError("entity_path must start with '/'")
    if not property or not property.strip():
        raise ValueError("property is required")
    result = np.asarray(data).copy()

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UPDATE_COMPONENT,
            attrs=[],
            inputs=[scenegraph, data],
            outputs=[],
            extra_info={
                "entity_path": entity_path,
                "property": property,
            },
        )

    return result


def microphone(
    output_shape: tuple = (1, 2),
    output_name: Optional[str] = None,
    *,
    sample_rate: int = 16000,
    encoding: str = "PCM_FLOAT",
) -> np.ndarray:
    """Stub for microphone capture.

    Corresponds to EOperatorType.MICROPHONE.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 8000 <= sample_rate <= 96000:
        raise ValueError("sample_rate must be an integer between 8000 and 96000 Hz")
    encoding_dtypes = {
        "PCM_16BIT": np.int16,
        "PCM_32BIT": np.int32,
        "PCM_FLOAT": np.float32,
    }
    if encoding not in encoding_dtypes:
        raise ValueError("encoding must be PCM_16BIT, PCM_32BIT, or PCM_FLOAT")

    result = np.zeros(output_shape, dtype=encoding_dtypes[encoding])

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.MICROPHONE,
            attrs=[f"{sample_rate};{encoding}"],
            inputs=[],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def speaker(
    audio: np.ndarray,
    output_name: Optional[str] = None,
    *,
    sample_rate: int = 16000,
) -> None:
    """Stub for speaker playback.

    Corresponds to EOperatorType.SPEAKER.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 8000 <= sample_rate <= 96000:
        raise ValueError("sample_rate must be an integer between 8000 and 96000 Hz")

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SPEAKER,
            attrs=[str(sample_rate)],
            inputs=[audio],
            outputs=[],
            output_names=[],
        )


def depth(output_shape: tuple = (1, 1), output_name: Optional[str] = None) -> np.ndarray:
    """Stub for depth capture.

    Corresponds to EOperatorType.DEPTH.
    """
    result = np.zeros(output_shape, dtype=np.float32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.DEPTH,
            attrs=[],
            inputs=[],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


class _JsArray:
    def __init__(self, array: np.ndarray):
        self._array = array.reshape(-1)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, idx: int) -> Any:
        return self._array[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._array[idx] = value


def _translate_js_expr(expr: str) -> str:
    expr = expr.replace("===", "==").replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"\b([A-Za-z_]\w*)\.length\b", r"len(\1)", expr)
    return expr


def _exec_js_lines(lines: List[str], env: Dict[str, Any]) -> None:
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("for"):
            header = line
            i += 1
            if i >= n or lines[i] != "{":
                raise ValueError("Malformed for-loop in JS code")
            block_start = i + 1
            depth = 1
            j = block_start
            while j < n and depth > 0:
                if lines[j] == "{":
                    depth += 1
                elif lines[j] == "}":
                    depth -= 1
                j += 1
            block_end = j - 1
            body = lines[block_start:block_end]
            header_inner = header[header.find("(") + 1: header.rfind(")")]
            init, cond, incr = [part.strip() for part in header_inner.split(";")]
            _exec_js_statement(init, env)
            while _eval_js_condition(cond, env):
                _exec_js_lines(body, env)
                _exec_js_statement(incr, env)
            i = block_end + 1
            continue
        if line.startswith("if"):
            i = _exec_js_if_chain(lines, i, env)
            continue
        if line in ("{", "}"):
            i += 1
            continue
        _exec_js_statement(line, env)
        i += 1


def _exec_js_if_chain(lines: List[str], start: int, env: Dict[str, Any]) -> int:
    i = start
    executed = False
    while i < len(lines):
        line = lines[i]
        is_if = line.startswith("if") or line.startswith("else if")
        is_else = line == "else"
        if not (is_if or is_else):
            break
        if is_if:
            cond = line[line.find("(") + 1: line.rfind(")")]
            cond_value = _eval_js_condition(cond, env)
        else:
            cond_value = True
        i += 1
        if i >= len(lines) or lines[i] != "{":
            raise ValueError("Malformed if-block in JS code")
        block_start = i + 1
        depth = 1
        j = block_start
        while j < len(lines) and depth > 0:
            if lines[j] == "{":
                depth += 1
            elif lines[j] == "}":
                depth -= 1
            j += 1
        block_end = j - 1
        if not executed and cond_value:
            _exec_js_lines(lines[block_start:block_end], env)
            executed = True
        i = block_end + 1
        if i < len(lines) and lines[i].startswith("else"):
            continue
        break
    return i


def _eval_js_condition(expr: str, env: Dict[str, Any]) -> bool:
    translated = _translate_js_expr(expr)
    return bool(eval(translated, {"__builtins__": {}}, {**env, "len": len}))


def _exec_js_statement(stmt: str, env: Dict[str, Any]) -> None:
    stmt = stmt.strip()
    if not stmt:
        return
    if stmt.startswith("var "):
        stmt = stmt[len("var "):].strip()
    if stmt.startswith("let "):
        stmt = stmt[len("let "):].strip()
    if stmt.startswith("throw "):
        raise RuntimeError("JS exception")
    if stmt.endswith("++"):
        name = stmt[:-2].strip()
        env[name] = env.get(name, 0) + 1
        return
    if "+=" in stmt:
        name, expr = stmt.split("+=", 1)
        name = name.strip()
        value = _eval_js_expr(expr, env)
        env[name] = env.get(name, 0) + value
        return
    if "*=" in stmt:
        name, expr = stmt.split("*=", 1)
        name = name.strip()
        value = _eval_js_expr(expr, env)
        env[name] = env.get(name, 0) * value
        return
    if "=" in stmt:
        left, expr = stmt.split("=", 1)
        left = left.strip()
        value = _eval_js_expr(expr, env)
        if "[" in left and left.endswith("]"):
            var_name = left[:left.find("[")].strip()
            idx_expr = left[left.find("[") + 1: left.rfind("]")]
            idx = int(_eval_js_expr(idx_expr, env))
            env[var_name][idx] = value
        else:
            env[left] = value
        return


def _eval_js_expr(expr: str, env: Dict[str, Any]) -> Any:
    translated = _translate_js_expr(expr)
    return eval(translated, {"__builtins__": {}}, {**env, "len": len})


def javascript(
    js_code: str,
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
    output_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, np.ndarray]:
    """Execute JavaScript-like code over tensors.

    Corresponds to EOperatorType.JAVASCRIPT.
    """
    env: Dict[str, Any] = {}
    for name, tensor in inputs.items():
        env[name] = _JsArray(np.asarray(tensor))
    outputs: Dict[str, np.ndarray] = {}
    template = None
    if inputs:
        template = next(iter(inputs.values()))

    def output_shape(name: str) -> tuple[int, ...]:
        spec = (output_specs or {}).get(name, {})
        dimensions = spec.get("dimensions", [])
        channels = int(spec.get("channels", 1) or 1)
        if isinstance(dimensions, list) and dimensions:
            shape = tuple(int(item) for item in dimensions)
            if channels > 1:
                shape += (channels,)
            return shape
        if template is not None:
            return tuple(np.asarray(template).shape)
        return (1,)

    def output_dtype(name: str) -> np.dtype:
        code = int((output_specs or {}).get(name, {}).get("data_type", 6) or 6)
        return {
            1: np.dtype(np.uint8),
            2: np.dtype(np.int8),
            3: np.dtype(np.uint16),
            4: np.dtype(np.int16),
            5: np.dtype(np.int32),
            6: np.dtype(np.float32),
            7: np.dtype(np.float64),
        }.get(code, np.dtype(np.float32))

    for name in output_names:
        if name in inputs:
            outputs[name] = inputs[name]
        else:
            outputs[name] = np.zeros(output_shape(name), dtype=output_dtype(name))
        env[name] = _JsArray(outputs[name])

    def _tokenize_js(code: str) -> List[str]:
        tokens: List[str] = []
        buf = ""
        paren_depth = 0
        for ch in code:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            if ch == "{" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                tokens.append("{")
                buf = ""
                continue
            if ch == "}" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                tokens.append("}")
                buf = ""
                continue
            if ch == ";" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            tokens.append(buf.strip())
        return tokens

    lines = _tokenize_js(js_code)

    try:
        _exec_js_lines(lines, env)
    except Exception:
        pass

    ctx = get_current_trace()
    if ctx is not None:
        input_refs = []
        for alias, tensor in inputs.items():
            tensor_name = ctx.get_tensor_name(tensor)
            if tensor_name is not None:
                input_refs.append({"name": alias, "tensor": tensor_name})
        output_refs = [{"name": name, "tensor": name} for name in output_names]
        ctx.record_op(
            op_type=_OP_JAVASCRIPT,
            attrs=[js_code],
            inputs=list(inputs.values()),
            outputs=list(outputs.values()),
            output_names=output_names,
            extra_info={"input_refs": input_refs, "output_refs": output_refs},
        )

    return outputs


def unknown(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Fallback unknown operator (pass-through)."""
    result = np.asarray(tensor).copy()
    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UNKNOWN,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )
    return result
