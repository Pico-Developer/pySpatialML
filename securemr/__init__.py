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

"""SecureMR package for SpatialML pipeline representation and deployment."""

from .core.types import BaseType, EDataType, EOperatorType


def __getattr__(name: str):  # noqa: D401
    """Lazy import optional utilities."""
    if name == "TORCH_INSTALLED":
        from .core.utils import TORCH_INSTALLED as _ti

        return _ti
    if name in {
        "PipelinePackageEntry",
        "PipelineZooPackageSpec",
        "SUPPORTED_EXECUTION_MODES",
        "configure_litert_inference_operator",
        "create_litert_model_spec",
        "load_pipeline_zoo_manifest",
        "validate_pipeline_zoo_manifest",
        "write_pipeline_zoo_package",
    }:
        from . import pipeline_zoo as _pipeline_zoo

        return getattr(_pipeline_zoo, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__version__ = "0.5.0"
__all__ = [
    "BaseType",
    "EDataType",
    "EOperatorType",
    "TORCH_INSTALLED",
    "PipelinePackageEntry",
    "PipelineZooPackageSpec",
    "SUPPORTED_EXECUTION_MODES",
    "configure_litert_inference_operator",
    "create_litert_model_spec",
    "load_pipeline_zoo_manifest",
    "validate_pipeline_zoo_manifest",
    "write_pipeline_zoo_package",
]
