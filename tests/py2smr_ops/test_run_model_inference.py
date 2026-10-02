import numpy as np
import pytest

from securemr.py2smr import ops, trace, convert, verify


@trace(inputs=["input_1"], outputs=["_538", "_539"])
def traced_run_algorithm(input_1):
    return ops.run_algorithm(
        inputs={"input": input_1},
        model_file="model/mnist.tflite",
        model_name="mnist",
        output_names=["_538", "_539"],
        output_shapes=[(1,), (1,)],
        output_dtypes=[np.float32, np.int32],
    )


def test_run_algorithm_records_inline_litert_metadata():
    input_shape = (28, 28, 1)
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_map, ctx = traced_run_algorithm.trace(input_1=input_tensor)
    spec = convert(ctx)
    op = spec["operators"][0]

    assert out_map["_538"].shape == (1,)
    assert out_map["_538"].dtype == np.float32
    assert out_map["_539"].shape == (1,)
    assert out_map["_539"].dtype == np.int32
    assert op["inputs"] == [{"name": "input", "tensor": "input_1"}]
    assert op["outputs"] == [
        {"name": "_538", "tensor": "_538"},
        {"name": "_539", "tensor": "_539"},
    ]
    assert op["model"]["bin_path"] == "model/mnist.tflite"
    assert op["model"]["model_type"] == "tflite"
    assert op["model"]["model_name"] == "mnist"
    assert "model_file" not in op
    assert "model_asset" not in op
    assert "model_id" not in op

    expected = {"_538": out_map["_538"], "_539": out_map["_539"]}
    verification = verify(spec, {"input_1": input_tensor}, expected_outputs=expected)
    assert verification.success


def test_run_algorithm_applies_model_io_aliases_to_serialized_bindings():
    @trace(inputs=["input_1"], outputs=["result"])
    def aliased(input_1):
        return ops.run_algorithm(
            inputs={"pipeline_input": input_1},
            model_file="model/aliased.tflite",
            model_name="aliased",
            output_names=["result"],
            output_shapes=[(1,)],
            input_aliasing={"pipeline_input": "serving_default_image"},
            output_aliasing={"result": "StatefulPartitionedCall"},
        )["result"]

    _, ctx = aliased.trace(input_1=np.ones((1,), dtype=np.float32))
    op = convert(ctx)["operators"][0]

    assert op["inputs"] == [{"name": "serving_default_image", "tensor": "input_1"}]
    assert op["outputs"] == [{"name": "StatefulPartitionedCall", "tensor": "result"}]
    assert op["model"]["input"][0]["name"] == "serving_default_image"
    assert op["model"]["output"][0]["name"] == "StatefulPartitionedCall"


@pytest.mark.parametrize(
    ("input_aliasing", "output_aliasing", "message"),
    [
        ({"missing": "model_input"}, None, "input_aliasing contains unknown"),
        (None, {"missing": "model_output"}, "output_aliasing contains unknown"),
        ({"input": ""}, None, "must be a non-empty model node name"),
    ],
)
def test_run_algorithm_rejects_invalid_model_io_aliases(input_aliasing, output_aliasing, message):
    with pytest.raises(ValueError, match=message):
        ops.run_algorithm(
            inputs={"input": np.ones((1,), dtype=np.float32)},
            model_file="model/aliased.tflite",
            model_name="aliased",
            output_names=["result"],
            output_shapes=[(1,)],
            input_aliasing=input_aliasing,
            output_aliasing=output_aliasing,
        )


def test_run_algorithm_rejects_non_tflite_models():
    with pytest.raises(ValueError, match="\\.tflite"):
        ops.run_algorithm(
            inputs={"input": np.zeros((1,), dtype=np.float32)},
            model_file="model.bin",
            model_name="invalid",
            output_names=["output"],
            output_shapes=[(1,)],
        )


def test_run_algorithm_requires_output_shapes():
    with pytest.raises(ValueError, match="one output shape"):
        ops.run_algorithm(
            inputs={"input": np.zeros((1,), dtype=np.float32)},
            model_file="model.tflite",
            model_name="model",
            output_names=["output"],
        )


def test_run_algorithm_rejects_non_tflite_model_type():
    with pytest.raises(ValueError, match="model_type='tflite'"):
        ops.run_algorithm(
            inputs={"input": np.zeros((1,), dtype=np.float32)},
            model_file="model.tflite",
            model_name="model",
            output_names=["output"],
            output_shapes=[(1,)],
            model_type="onnx",
        )
