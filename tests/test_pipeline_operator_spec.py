from securemr.py2smr.verifier import _get_operator_type, run_pipeline_python
from pyspatialml import operator_cli
import numpy as np


def test_verifier_rejects_noncanonical_operator_names():
    for name in (
        "not_an_operator", "XR_SECURE_MR_OPERATOR_TYPE_UNKNOWN_PICO",
        "UNKNOWN", "not_a_type", "draw_text", "render_gltf",
    ):
        assert _get_operator_type(name) is None


def test_canonical_schema_operator_types_are_discoverable():
    operators = operator_cli.discover_operators()
    by_name = {item.enum_name: item for item in operators}

    assert by_name
    assert all(item.supported for item in operators)
    assert all(item.type_name.startswith("XR_SECURE_MR_OPERATOR_TYPE_") for item in operators)


def test_python_consumer_accepts_schema_comparison_field():
    spec = {
        "tensors": {
            "a": {"dimensions": [1, 3], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            "b": {"dimensions": [1, 3], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            "out": {"dimensions": [1, 3], "channels": 1, "data_type": 5, "is_placeholder": True, "usage": 6},
        },
        "operators": [{
            "type": "XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO",
            "inputs": [{"tensor": "a"}, {"tensor": "b"}],
            "outputs": [{"tensor": "out"}],
            "attrs": ["<"],
        }],
        "inputs": ["a", "b"],
        "outputs": ["out"],
    }

    result = run_pipeline_python(
        spec,
        {"a": np.array([1.0, 3.0, 2.0], dtype=np.float32),
         "b": np.array([2.0, 2.0, 2.0], dtype=np.float32)},
    )
    np.testing.assert_array_equal(result["out"], [1, 0, 0])
