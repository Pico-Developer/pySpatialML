import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_xr_model_io_aliasing_and_bounds(tmp_path):
    source = REPO_ROOT / "tests" / "native" / "xr_model_io_test.cpp"
    include_dir = (
        REPO_ROOT / "external" / "SpatialML-XR-Utils" / "base" / "securemr_utils"
    )
    executable = tmp_path / "xr_model_io_test"

    subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
         "-I", str(include_dir), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
