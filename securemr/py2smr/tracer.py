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
"""Execution tracer for py2smr."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from securemr.core.types import EOperatorType

__all__ = ["trace", "TraceContext", "TracedOp", "TensorInfo", "get_current_trace"]


# Thread-local storage for trace context
_trace_local = threading.local()


def get_current_trace() -> Optional["TraceContext"]:
    """Get the current trace context for this thread."""
    return getattr(_trace_local, "context", None)


@contextlib.contextmanager
def _trace_context(ctx: "TraceContext"):
    """Context manager to set the current trace context."""
    old_ctx = getattr(_trace_local, "context", None)
    _trace_local.context = ctx
    try:
        yield ctx
    finally:
        _trace_local.context = old_ctx


@dataclass
class TensorInfo:
    """Information about a tensor in the trace."""
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    value: Optional[np.ndarray] = None
    is_input: bool = False
    is_output: bool = False


@dataclass
class TracedOp:
    """Record of a traced operation."""
    op_type: EOperatorType
    attrs: List[str]
    input_names: List[str]
    output_names: List[str]
    extra_info: Dict[str, Any] = field(default_factory=dict)


class TraceContext:
    """Context for tracing function execution."""

    def __init__(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
    ):
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.operations: List[TracedOp] = []
        self.tensors: Dict[str, TensorInfo] = {}
        self._tensor_counter = 0
        self._tensor_id_map: Dict[int, str] = {}  # id(ndarray) -> tensor_name
        # Keep the actual ndarray objects alive for the duration of tracing.
        # The id-based lookup is only safe while those objects remain alive;
        # helper operators commonly create temporary constant arrays, and
        # Python may otherwise reuse their ids for a later tensor.
        self._tensor_objects: Dict[int, np.ndarray] = {}

    def _generate_tensor_name(self, prefix: str = "tensor") -> str:
        """Generate a unique tensor name."""
        name = f"{prefix}_{self._tensor_counter}"
        self._tensor_counter += 1
        return name

    def register_input(self, name: str, tensor: np.ndarray) -> str:
        """Register an input tensor."""
        tensor_name = name
        self.tensors[tensor_name] = TensorInfo(
            name=tensor_name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            value=tensor.copy(),
            is_input=True,
        )
        self._tensor_id_map[id(tensor)] = tensor_name
        self._tensor_objects[id(tensor)] = tensor
        return tensor_name

    def register_tensor(
        self,
        tensor: np.ndarray,
        name: Optional[str] = None,
        is_output: bool = False,
        value: Optional[np.ndarray] = None,
    ) -> str:
        """Register an intermediate or output tensor."""
        # Check if tensor is already registered
        tensor_id = id(tensor)
        if tensor_id in self._tensor_id_map:
            existing_name = self._tensor_id_map[tensor_id]
            if is_output:
                self.tensors[existing_name].is_output = True
            return existing_name

        tensor_name = name or self._generate_tensor_name()
        self.tensors[tensor_name] = TensorInfo(
            name=tensor_name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            value=tensor.copy() if is_output or value is not None else None,
            is_output=is_output,
        )
        self._tensor_id_map[tensor_id] = tensor_name
        self._tensor_objects[tensor_id] = tensor
        return tensor_name

    def get_tensor_name(self, tensor: np.ndarray) -> Optional[str]:
        """Get the name of a registered tensor."""
        return self._tensor_id_map.get(id(tensor))

    def record_op(
        self,
        op_type: EOperatorType,
        attrs: List[str],
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
        output_names: Optional[List[str]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation."""
        input_names = []
        for inp in inputs:
            name = self.get_tensor_name(inp)
            if name is None:
                name = self.register_tensor(inp)
            input_names.append(name)

        out_names = []
        for i, out in enumerate(outputs):
            if output_names and i < len(output_names):
                name = output_names[i]
            else:
                name = self._generate_tensor_name("out")
            self.register_tensor(out, name=name)
            out_names.append(name)

        self.operations.append(TracedOp(
            op_type=op_type,
            attrs=attrs,
            input_names=input_names,
            output_names=out_names,
            extra_info=extra_info or {},
        ))

    def mark_outputs(self, outputs: Dict[str, np.ndarray]) -> None:
        """Mark tensors as outputs and store their values."""
        for name, tensor in outputs.items():
            tensor_name = self.get_tensor_name(tensor)
            if tensor_name is None:
                tensor_name = self.register_tensor(tensor, name=name, is_output=True)
            elif self.tensors[tensor_name].is_input and tensor_name != name:
                # Keep the declared input tensor intact.  An identity or
                # pass-through function may return an input under a distinct
                # output name; renaming the input would make the top-level
                # inputs/outputs lists point at the same tensor.
                output_tensor = np.asarray(tensor).copy()
                self.tensors[name] = TensorInfo(
                    name=name,
                    shape=output_tensor.shape,
                    dtype=output_tensor.dtype,
                    value=output_tensor,
                    is_output=True,
                )
                self._tensor_id_map[id(tensor)] = tensor_name
            else:
                self.tensors[tensor_name].is_output = True
                self.tensors[tensor_name].value = tensor.copy()
                # Rename if needed
                if tensor_name != name and name not in self.tensors:
                    old_info = self.tensors.pop(tensor_name)
                    old_info.name = name
                    self.tensors[name] = old_info
                    self._tensor_id_map[id(tensor)] = name
                    # Update operation references
                    for op in self.operations:
                        op.input_names = [
                            name if n == tensor_name else n
                            for n in op.input_names
                        ]
                        op.output_names = [
                            name if n == tensor_name else n
                            for n in op.output_names
                        ]


class TracedFunction:
    """Wrapper for a traced function."""

    def __init__(
        self,
        func: Callable,
        input_names: Sequence[str],
        output_names: Sequence[str],
    ):
        self._func = func
        self._input_names = list(input_names)
        self._output_names = list(output_names)
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        """Call the function without tracing."""
        return self._func(*args, **kwargs)

    def trace(self, **kwargs) -> Tuple[Any, TraceContext]:
        """Execute the function with tracing enabled.

        Args:
            **kwargs: Input tensors as keyword arguments matching input_names.

        Returns:
            Tuple of (function result, TraceContext with recorded operations).
        """
        # Validate inputs
        for name in self._input_names:
            if name not in kwargs:
                raise ValueError(f"Missing required input: {name}")
            if not isinstance(kwargs[name], np.ndarray):
                raise TypeError(f"Input '{name}' must be a numpy array")

        ctx = TraceContext(self._input_names, self._output_names)

        # Register inputs
        for name in self._input_names:
            ctx.register_input(name, kwargs[name])

        # Execute with tracing
        with _trace_context(ctx):
            result = self._func(**kwargs)

        # Handle outputs
        if isinstance(result, np.ndarray):
            if len(self._output_names) != 1:
                raise ValueError(
                    f"Function returned single array but {len(self._output_names)} "
                    "outputs were declared"
                )
            ctx.mark_outputs({self._output_names[0]: result})
        elif isinstance(result, (tuple, list)):
            if len(result) != len(self._output_names):
                raise ValueError(
                    f"Function returned {len(result)} values but "
                    f"{len(self._output_names)} outputs were declared"
                )
            outputs = {
                name: arr for name, arr in zip(self._output_names, result)
            }
            ctx.mark_outputs(outputs)
        elif isinstance(result, dict):
            ctx.mark_outputs(result)
        elif result is None and not self._output_names:
            pass
        else:
            raise TypeError(
                f"Function must return ndarray, tuple/list of ndarrays, or dict. "
                f"Got {type(result)}"
            )

        return result, ctx


def trace(
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> Callable[[Callable], TracedFunction]:
    """Decorator to enable tracing for a function.

    Args:
        inputs: Names of input tensors.
        outputs: Names of output tensors.

    Returns:
        Decorator that wraps the function with tracing capability.

    Example:
        @trace(inputs=["image"], outputs=["result"])
        def preprocess(image):
            return ops.arithmetic(image, "{0} / 255.0")

        result, ctx = preprocess.trace(image=input_array)
    """
    def decorator(func: Callable) -> TracedFunction:
        return TracedFunction(func, inputs, outputs)
    return decorator
