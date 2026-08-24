"""terrace.ops 脚手架的行为契约:无 .so 环境(本地/CI,无 CANN)下的全部语义。

为什么单独锁这些:AscendC 工程今晚只搭链路(kernel 逻辑未实现,见
本目录各文件的头注),但**降级语义从第一天就是生产语义** —— 将来任何一台机器
.so 没编好/编错,训练都要走到这里的路径。三条硬契约:

  1. 加载失败 -> 恰好一行 WARNING(fail-loud 不静默)-> 视同 TERRACE_CUSTOM_OPS=0,
     现组合链结果**逐位**不变;
  2. 开关语义:"0" 显式关(不尝试加载、不告警);"require" 加载失败直接炸
     (防 bench 读数被静默偷换成组合链读数);
  3. 判定进程内只做一次(缓存),不因反复调用重复告警/重复尝试 dlopen。

外加工程纪律锁:本目录全部文件 LF(build.sh 要上集群 bash,内部守护脚本(未随仓发布) 的 CRLF
事故同款纪律)+ py_compile。
"""
from __future__ import annotations

import logging
import py_compile
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import terrace.ops as tops  # noqa: E402


@pytest.fixture()
def clean_ops(monkeypatch):
    """每例独立判定:清相关环境、清缓存;用完再清缓存,不污染别的测试。"""
    monkeypatch.delenv("TERRACE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("TERRACE_OPS_LIB", raising=False)
    tops.reset()
    yield monkeypatch
    tops.reset()


def _warnings(caplog):
    return [r for r in caplog.records
            if r.name == "terrace.ops" and r.levelno == logging.WARNING]


# ======================================================================================
# 契约 1:无 .so + 自动档 -> 一行告警降级,组合链结果逐位不变
# ======================================================================================

def test_auto_mode_degrades_loud_and_falls_back(clean_ops, caplog):
    """**显式索取**(=1)但加载不上时:降级一行 WARNING,不炸。

    2026-08-24:这条原来用「未设环境」当"开",因为默认曾是 "1"。
    默认已改成 "0" —— `.so` 第一次编出来那刻,未过逐位校验的 K1 kernel
    自动进了训练路径,判决床白烧两发(见 terrace/ops/__init__.py 的说明)。
    契约本身没变(降级要出声、不静默、不刷屏),变的是它由**谁**触发:
    现在必须有人显式写 =1。
    """
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.loaded is False
    assert st.requested == "1"
    assert st.lib is None
    assert st.reason, "降级必须给出人话原因"
    warns = _warnings(caplog)
    assert len(warns) == 1, "fail-loud 契约:恰好一行 WARNING,不静默也不刷屏"
    assert "降级" in warns[0].getMessage()
    assert tops.custom_ops_enabled() is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fallback_is_bitwise_identity_fresh_tensor(clean_ops, dtype):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(16, 64, generator=g).to(dtype)
    y = tops.passthrough(x)
    assert torch.equal(y, x), "降级路径必须与现链(原样拷出)逐位相等"
    assert y.data_ptr() != x.data_ptr(), "契约是拷出,不是别名"


def test_fallback_gradient_is_identity(clean_ops):
    x = torch.randn(8, 8, requires_grad=True)
    tops.passthrough(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ======================================================================================
# 契约 2:开关语义
# ======================================================================================

def test_switch_off_means_no_attempt_no_warning(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.loaded is False
    assert st.requested == "0"
    assert "显式关闭" in st.reason, "=0 是显式关,不是加载失败"
    assert _warnings(caplog) == [], "显式关不是降级,不许告警"
    x = torch.randn(4, 4)
    assert torch.equal(tops.passthrough(x), x)


@pytest.mark.parametrize("value", ["require", "2"])
def test_switch_require_fails_hard_without_so(clean_ops, value):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", value)
    with pytest.raises(RuntimeError, match="TERRACE_CUSTOM_OPS"):
        tops.status()


def test_explicit_lib_path_missing_fails_hard_under_require(clean_ops, tmp_path):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "require")
    ghost = str(tmp_path / "no_such_terrace_ops.so")
    clean_ops.setenv("TERRACE_OPS_LIB", ghost)
    with pytest.raises(RuntimeError, match="no_such_terrace_ops"):
        tops.status()


# ======================================================================================
# 契约 3:判定缓存 —— 一次告警,此后不再尝试
# ======================================================================================

def test_degradation_decided_once_not_per_call(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")   # 默认已是关,要显式索取才会尝试加载
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        tops.status()
        for _ in range(5):
            tops.passthrough(torch.randn(4, 4))
        assert tops.status() is tops.status(), "判定对象应缓存复用"
    assert len(_warnings(caplog)) == 1, "反复调用不得重复告警/重复尝试加载"


def test_reset_re_reads_environment(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        assert tops.status().requested == "1"
        clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
        assert tops.status().requested == "1", "无 reset 不重读环境(热路径契约)"
        tops.reset()
        assert tops.status().requested == "0"


def test_unset_env_means_off_not_auto(clean_ops, caplog):
    """**未设环境 = 关**,而且连加载都不尝试。

    这是 2026-08-24 事故的正面钉子(另一面在
    tests/test_terrace_k1_arrival.py::test_custom_ops_default_is_off_not_on)。
    事故:默认为 "1" 时,`.so` 一编出来 `custom_ops_enabled()` 就变真,
    未过逐位校验的 K1 kernel 直接进训练 dispatch 路径 ——
    全 128 rank 在第 0 步同炸 `Split sizes dosen't match total dim 0 size`。
    bitcheck 判决行写着「K1 不得上床」,却没有任何机制执行它。

    **连 WARNING 都不该有** —— 没人索取算子,就不存在"降级",也就没什么可告警的。
    这一点区分了"关"与"想开但开不了"。
    """
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.requested == "0", "未设环境必须归一化成 '0'(关)"
    assert st.loaded is False
    assert tops.custom_ops_enabled() is False
    assert _warnings(caplog) == [], (
        "没人索取算子时不该有降级告警 —— 那会把'本来就关'说成'想开没开成'")


# ======================================================================================
# 工程纪律:LF + py_compile(build.sh 要上集群 bash;CRLF 即事故,test_no_crlf 同款)
# ======================================================================================

OPS_DIR = ROOT / "terrace" / "ops"
_SKIP_DIRS = {"build_gen", "lib", "__pycache__"}


def _scaffold_files():
    out = [Path(__file__)]
    for p in sorted(OPS_DIR.rglob("*")):
        if p.is_file() and not (_SKIP_DIRS & set(part.name for part in p.parents)):
            out.append(p)
    return out


@pytest.mark.parametrize("path", _scaffold_files(), ids=lambda p: p.name)
def test_scaffold_file_has_no_crlf(path):
    assert b"\r\n" not in path.read_bytes(), (
        f"{path} 有 CRLF;build.sh/源文件要原样上集群,bash 会死在 $'\\r'")


@pytest.mark.parametrize("rel", ["terrace/ops/__init__.py",
                                 "terrace/ops/csrc/build_ext.py"])
def test_scaffold_python_compiles(rel):
    py_compile.compile(str(ROOT / rel), doraise=True)
