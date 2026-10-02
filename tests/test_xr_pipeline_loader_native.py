import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XR_UTILS = REPO_ROOT / "external" / "SpatialML-XR-Utils"
PIPELINE = REPO_ROOT / "examples" / "all_xr_operators_pipeline.json"


def test_xr_loader_parses_all_xr_operators_pipeline(tmp_path):
    source = REPO_ROOT / "tests" / "native" / "xr_pipeline_loader_test.cpp"
    generated_json = next(
        (REPO_ROOT / "pyspatialml" / "xr_pipeline_runner" / "app" / ".cxx").glob(
            "**/_deps/nlohmann_json-src/include"
        ),
        None,
    )
    assert generated_json is not None, "Build the XR runner once to provision nlohmann_json headers"

    executable = tmp_path / "xr_pipeline_loader_test"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "-I",
            str(XR_UTILS / "base"),
            "-I",
            str(XR_UTILS / "base" / "securemr_utils"),
            "-I",
            str(XR_UTILS / "base" / "oxr_utils"),
            "-I",
            str(XR_UTILS / "external" / "openxr" / "include"),
            "-I",
            str(generated_json),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable), str(PIPELINE)], check=True)
