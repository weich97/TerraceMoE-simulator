"""集群侧编译 torch 绑定:terrace_ops.cpp -> terrace/ops/lib/terrace_ops.so。

只在集群跑(需 torch_npu 与已安装的 opp vendor 包;本地无 CANN 直接 sys.exit)。
用 torch.utils.cpp_extension 免手写编译命令 -- 它自动带对 torch ABI 正确的
编译旗标(-D_GLIBCXX_USE_CXX11_ABI 等,手写 makefile 最容易在这里翻车)。

用法(先 source set_env.sh,再进训练 venv):
    python terrace/ops/csrc/build_ext.py
可调环境:
    VENDOR_NAME       opp vendor 名,与 ascendc/build.sh 一致(缺省 terrace)
    ASCEND_HOME_PATH  set_env.sh 已导出;CANN 头/库根
    TERRACE_NINJA     ninja 可执行文件所在目录(缺省自动探测,见 _ensure_ninja)
产物:
    terrace/ops/lib/terrace_ops.so   (terrace/ops/__init__.py 的默认搜索位)

**必须用训练用的那个 venv 编**:.so 链的是该 venv 的 libtorch/libtorch_npu,
ABI 与 torch 版本绑死。用你训练用的同一个环境编译(实测口径:
你的 venv(torch 2.9.0 + torch_npu 2.9.0.post2)。
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
    """把 ninja 可执行文件塞进 PATH。

    torch.utils.cpp_extension.load() 硬性要求 ninja **可执行文件**(它 subprocess
    调 `ninja --version`),光有 python 包不算数。集群实测:训练 venv 是叠在 conda
    env 上的,ninja 包能 import(BIN_DIR 却是空串),而可执行文件躺在底下 conda env
    的 bin/ 里,不在 venv 的 bin/ 里 —— 于是 verify_ninja_availability() 报
    "Ninja is required to load C++ extensions",看着像没装,其实只是没在 PATH 上。
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
        import ninja  # noqa: F401  (pip 包;BIN_DIR 在老版本上可能是空串)
        bin_dir = getattr(ninja, "BIN_DIR", "") or ""
        if bin_dir:
            candidates.append(bin_dir)
        candidates.append(os.path.join(os.path.dirname(ninja.__file__), "data", "bin"))
    except ImportError:
        pass
    # venv 的 bin,以及它叠在其上的 base 解释器(conda env)的 bin。
    candidates.append(os.path.join(sys.prefix, "bin"))
    candidates.append(os.path.join(sys.base_prefix, "bin"))
    candidates.append(os.path.dirname(sys.executable))

    for d in candidates:
        exe = os.path.join(d, "ninja")
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            print(f"[build_ext] ninja 不在 PATH 上,已补入: {exe}")
            try:
                verify_ninja_availability()
            except Exception as e:                       # 补了还是不行,fail loud
                _fail(f"补 PATH 后 ninja 仍不可用: {e}")
            return
    _fail("找不到 ninja 可执行文件 —— 指定 TERRACE_NINJA=<含 ninja 的目录>,"
          f"或 pip install ninja。已试过: {candidates}")


def main() -> None:
    try:
        import torch  # noqa: F401
        import torch_npu
    except ImportError as e:
        _fail(f"需要 torch + torch_npu(集群训练 venv):{e}")

    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann or not os.path.isdir(cann):
        _fail("ASCEND_HOME_PATH 未设 -- 先 source ascend-toolkit/set_env.sh")

    vendor = os.environ.get("VENDOR_NAME", "terrace")
    opp = os.environ.get("ASCEND_OPP_PATH", os.path.join(cann, "opp"))
    op_api = os.path.join(opp, "vendors", vendor, "op_api")
    if not os.path.isfile(os.path.join(op_api, "lib", "libcust_opapi.so")):
        _fail(f"{op_api}/lib/libcust_opapi.so 不存在 -- 先跑 ascendc/build.sh "
              f"装 opp 包(或 VENDOR_NAME 与装包时不一致)")

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
            # rpath:运行期不靠 LD_LIBRARY_PATH 也能找到 vendor 包与 CANN 库
            f"-Wl,-rpath,{os.path.join(op_api, 'lib')}",
            f"-Wl,-rpath,{os.path.join(cann, 'lib64')}",
            f"-Wl,-rpath,{os.path.join(npu_root, 'lib')}",
        ],
        build_directory=LIB_DIR,
        verbose=True,
        # **必须 False**。本 .so 是算子注册库（TORCH_LIBRARY），不是 python 扩展模块：
        # 里面没有 PyInit_terrace_ops。缺省 is_python_module=True 会让 load() 在编译
        # 链接**全部成功之后**才死在最后一步 import 上：
        #   ImportError: dynamic module does not define module export function (PyInit_terrace_ops)
        # 置 False 后 torch 改用 torch.ops.load_library() 加载。
        is_python_module=False,
    )
    del module   # is_python_module=False 时返回值因 torch 版本而异（不一定是 None），
                 # 不拿它当凭据 —— 以下面的产物回读为准。
    # is_python_module=False 时产物就落在 build_directory/<name>.so，名字已经是
    # 加载器约定的 terrace_ops.so，不用再拷一份。回读校验存在。
    target = os.path.join(LIB_DIR, "terrace_ops.so")
    if not os.path.isfile(target):
        _fail(f"编完了但找不到 {target} —— build_directory 布局变了?")
    print(f"[build_ext] OK -> {target}")
    print("[build_ext] 冒烟: TERRACE_CUSTOM_OPS=require python -c "
          "\"import torch,torch_npu,terrace.ops as o; x=torch.randn(8,2048,"
          "dtype=torch.bfloat16).npu(); y=o.passthrough(x); "
          "assert torch.equal(y.cpu(), x.cpu()); print('bitwise OK', o.status())\"")


if __name__ == "__main__":
    main()
