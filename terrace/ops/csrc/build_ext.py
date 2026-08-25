"""Cluster-side build of the torch binding: terrace_ops.cpp -> terrace/ops/lib/terrace_ops.so.

Runs on the cluster only (needs torch_npu and the installed opp vendor package;
without CANN locally it sys.exit's outright). Uses torch.utils.cpp_extension to
avoid hand-written compile commands -- it automatically carries the compile
flags matching the torch ABI (-D_GLIBCXX_USE_CXX11_ABI etc.; a hand-written
makefile trips over exactly that most easily).

Usage (source set_env.sh first, then enter the training venv):
    python terrace/ops/csrc/build_ext.py
Tunable environment:
    VENDOR_NAME       opp vendor name, matching ascendc/build.sh (default terrace)
    ASCEND_HOME_PATH  exported by set_env.sh; CANN header/library root
    TERRACE_NINJA     directory holding the ninja executable (default:
                      auto-detect, see _ensure_ninja)
Artifact:
    terrace/ops/lib/terrace_ops.so   (the default search location of terrace/ops/__init__.py)

**Must be built with the venv used for training**: the .so links that venv's
libtorch/libtorch_npu; the ABI is pinned to the torch version. Build in the
same environment you train in (measured baseline: your venv
(torch 2.9.0 + torch_npu 2.9.0.post2)).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_DIR = os.path.dirname(HERE)
LIB_DIR = os.path.join(OPS_DIR, "lib")


def _fail(msg: str) -> None:
    print(f"[build_ext] {msg}", file=sys.stderr)
    sys.exit(1)


def _ensure_ninja() -> None:
    """Put the ninja executable on PATH.

    torch.utils.cpp_extension.load() hard-requires the ninja **executable** (it
    subprocesses `ninja --version`); having only the python package does not
    count. Measured on the cluster: the training venv is layered on a conda
    env, the ninja package imports fine (yet BIN_DIR is an empty string) while
    the executable sits in the underlying conda env's bin/, not the venv's
    bin/ -- so verify_ninja_availability() reports "Ninja is required to load
    C++ extensions", which looks like a missing install but is really just a
    PATH miss.
    """
    from torch.utils.cpp_extension import verify_ninja_availability

    try:
        verify_ninja_availability()
        return
    except Exception:
        pass

    candidates = []
    explicit = os.environ.get("TERRACE_NINJA")
    if explicit:
        candidates.append(explicit)
    try:
        import ninja  # noqa: F401  (pip package; BIN_DIR may be an empty string on old versions)
        bin_dir = getattr(ninja, "BIN_DIR", "") or ""
        if bin_dir:
            candidates.append(bin_dir)
        candidates.append(os.path.join(os.path.dirname(ninja.__file__), "data", "bin"))
    except ImportError:
        pass
    # The venv's bin, plus the bin of the base interpreter (the conda env) it
    # is layered on.
    candidates.append(os.path.join(sys.prefix, "bin"))
    candidates.append(os.path.join(sys.base_prefix, "bin"))
    candidates.append(os.path.dirname(sys.executable))

    for d in candidates:
        exe = os.path.join(d, "ninja")
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            print(f"[build_ext] ninja was not on PATH, added: {exe}")
            try:
                verify_ninja_availability()
            except Exception as e:                       # still broken after the PATH fix, fail loud
                _fail(f"ninja still unusable after patching PATH: {e}")
            return
    _fail("ninja executable not found -- set TERRACE_NINJA=<directory containing "
          f"ninja>, or pip install ninja. Tried: {candidates}")


def main() -> None:
    try:
        import torch  # noqa: F401
        import torch_npu
    except ImportError as e:
        _fail(f"torch + torch_npu required (cluster training venv): {e}")

    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann or not os.path.isdir(cann):
        _fail("ASCEND_HOME_PATH unset -- source ascend-toolkit/set_env.sh first")

    vendor = os.environ.get("VENDOR_NAME", "terrace")
    opp = os.environ.get("ASCEND_OPP_PATH", os.path.join(cann, "opp"))
    op_api = os.path.join(opp, "vendors", vendor, "op_api")
    if not os.path.isfile(os.path.join(op_api, "lib", "libcust_opapi.so")):
        _fail(f"{op_api}/lib/libcust_opapi.so does not exist -- run ascendc/build.sh "
              f"first to install the opp package (or VENDOR_NAME differs from the "
              f"one used at install time)")

    npu_root = os.path.dirname(os.path.abspath(torch_npu.__file__))

    _ensure_ninja()
    from torch.utils.cpp_extension import load

    os.makedirs(LIB_DIR, exist_ok=True)
    module = load(
        name="terrace_ops",
        sources=[os.path.join(HERE, "terrace_ops.cpp")],
        extra_include_paths=[
            os.path.join(cann, "include"),
            os.path.join(op_api, "include"),
            os.path.join(npu_root, "include"),
        ],
        extra_ldflags=[
            f"-L{os.path.join(cann, 'lib64')}",
            f"-L{os.path.join(op_api, 'lib')}",
            f"-L{os.path.join(npu_root, 'lib')}",
            "-lascendcl", "-lnnopbase", "-lcust_opapi", "-ltorch_npu",
            # rpath: find the vendor package and CANN libraries at runtime
            # without relying on LD_LIBRARY_PATH
            f"-Wl,-rpath,{os.path.join(op_api, 'lib')}",
            f"-Wl,-rpath,{os.path.join(cann, 'lib64')}",
            f"-Wl,-rpath,{os.path.join(npu_root, 'lib')}",
        ],
        build_directory=LIB_DIR,
        verbose=True,
        # **Must be False**. This .so is an op-registration library
        # (TORCH_LIBRARY), not a python extension module: there is no
        # PyInit_terrace_ops inside. The default is_python_module=True makes
        # load() die at the final import step **after** compiling and linking
        # fully succeed:
        #   ImportError: dynamic module does not define module export function (PyInit_terrace_ops)
        # With False, torch loads it via torch.ops.load_library() instead.
        is_python_module=False,
    )
    del module   # With is_python_module=False the return value varies by torch
                 # version (not necessarily None); do not treat it as evidence
                 # -- the artifact read-back below is authoritative.
    # With is_python_module=False the artifact lands at
    # build_directory/<name>.so, already named terrace_ops.so per the loader
    # convention; no extra copy needed. Read back to verify it exists.
    target = os.path.join(LIB_DIR, "terrace_ops.so")
    if not os.path.isfile(target):
        _fail(f"build finished but {target} is missing -- did the "
              f"build_directory layout change?")
    print(f"[build_ext] OK -> {target}")
    print("[build_ext] smoke test: TERRACE_CUSTOM_OPS=require python -c "
          "\"import torch,torch_npu,terrace.ops as o; x=torch.randn(8,2048,"
          "dtype=torch.bfloat16).npu(); y=o.passthrough(x); "
          "assert torch.equal(y.cpu(), x.cpu()); print('bitwise OK', o.status())\"")


if __name__ == "__main__":
    main()
