"""Behavior contract of the terrace.ops scaffold: the full semantics of a no-.so
environment (local/CI, no CANN).

Why these get locked on their own: tonight the AscendC work only wires up the build
chain (kernel logic unimplemented, see the header notes of the files in that
directory), but **the fallback semantics are production semantics from day one** --
on any future machine whose .so failed to build or built wrong, training walks
exactly this path. Three hard contracts:

  1. Load failure -> exactly one WARNING line (fail-loud, not silent) -> treated as
     TERRACE_CUSTOM_OPS=0; the live composed-chain result stays **bitwise** unchanged;
  2. Switch semantics: "0" is an explicit off (no load attempt, no warning);
     "require" dies hard on load failure (so a bench readout cannot be silently
     swapped for a composed-chain readout);
  3. The decision is made exactly once per process (cached); repeated calls must not
     re-warn or re-attempt dlopen.

Plus one engineering-discipline lock: every file in that directory is LF (build.sh
must run under cluster bash; same discipline as the CRLF incident in an internal
watchdog script (not shipped with this repo)) + py_compile.
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
    """Independent decision per test: clear the relevant env vars and the cache;
    clear the cache again afterwards so other tests stay unpolluted."""
    monkeypatch.delenv("TERRACE_CUSTOM_OPS", raising=False)
    monkeypatch.delenv("TERRACE_OPS_LIB", raising=False)
    tops.reset()
    yield monkeypatch
    tops.reset()


def _warnings(caplog):
    return [r for r in caplog.records
            if r.name == "terrace.ops" and r.levelno == logging.WARNING]


# ======================================================================================
# Contract 1: no .so + auto mode -> degrade with one warning line, composed-chain
# result bitwise unchanged
# ======================================================================================

def test_auto_mode_degrades_loud_and_falls_back(clean_ops, caplog):
    """**Explicitly requested** (=1) but the load fails: degrade with one WARNING,
    no crash.

    2026-08-24: this test used to treat "env unset" as "on", because the default
    used to be "1". The default is now "0" -- the moment the `.so` first built, the
    K1 kernel, which had not passed bitwise validation, walked straight into the
    training path and wasted two runs on the verdict testbed (see the note in
    terrace/ops/__init__.py). The contract itself is unchanged (degradation must be
    loud, not silent, not spammy); what changed is **who** triggers it: someone now
    has to write =1 explicitly.
    """
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.loaded is False
    assert st.requested == "1"
    assert st.lib is None
    assert st.reason, "degradation must state a human-readable reason"
    warns = _warnings(caplog)
    assert len(warns) == 1, "fail-loud contract: exactly one WARNING, neither silent nor spammy"
    # Substring must match the wording produced in terrace/ops/__init__.py.
    assert "downgrad" in warns[0].getMessage()
    assert tops.custom_ops_enabled() is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fallback_is_bitwise_identity_fresh_tensor(clean_ops, dtype):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(16, 64, generator=g).to(dtype)
    y = tops.passthrough(x)
    assert torch.equal(y, x), "the fallback path must equal the live chain (verbatim copy-out) bitwise"
    assert y.data_ptr() != x.data_ptr(), "the contract is a copy-out, not an alias"


def test_fallback_gradient_is_identity(clean_ops):
    x = torch.randn(8, 8, requires_grad=True)
    tops.passthrough(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ======================================================================================
# Contract 2: switch semantics
# ======================================================================================

def test_switch_off_means_no_attempt_no_warning(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.loaded is False
    assert st.requested == "0"
    assert "explicit" in st.reason, "=0 is an explicit off, not a load failure"
    assert _warnings(caplog) == [], "an explicit off is not a degradation; no warning allowed"
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
# Contract 3: decision caching -- warn once, never attempt again
# ======================================================================================

def test_degradation_decided_once_not_per_call(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")   # default is already off; the load is only attempted on explicit request
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        tops.status()
        for _ in range(5):
            tops.passthrough(torch.randn(4, 4))
        assert tops.status() is tops.status(), "the decision object should be cached and reused"
    assert len(_warnings(caplog)) == 1, "repeated calls must not re-warn or re-attempt the load"


def test_reset_re_reads_environment(clean_ops, caplog):
    clean_ops.setenv("TERRACE_CUSTOM_OPS", "1")
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        assert tops.status().requested == "1"
        clean_ops.setenv("TERRACE_CUSTOM_OPS", "0")
        assert tops.status().requested == "1", "without reset the env is not re-read (hot-path contract)"
        tops.reset()
        assert tops.status().requested == "0"


def test_unset_env_means_off_not_auto(clean_ops, caplog):
    """**Env unset = off**, and no load is even attempted.

    This is the positive-side nail for the 2026-08-24 incident (the other side
    lives in tests/test_terrace_k1_arrival.py::test_custom_ops_default_is_off_not_on).
    The incident: with the default at "1", `custom_ops_enabled()` flipped true the
    moment the `.so` got built, and the K1 kernel -- which had not passed bitwise
    validation -- went straight into the training dispatch path: all 128 ranks
    crashed together at step 0 with `Split sizes dosen't match total dim 0 size`.
    The bitcheck verdict line said "K1 must not go on the testbed", yet no
    mechanism enforced it.

    **There must not even be a WARNING** -- if nobody requested the ops, there is
    no "degradation", hence nothing to warn about. This is what separates "off"
    from "wanted on but could not".
    """
    with caplog.at_level(logging.INFO, logger="terrace.ops"):
        st = tops.status()
    assert st.requested == "0", "env unset must normalize to '0' (off)"
    assert st.loaded is False
    assert tops.custom_ops_enabled() is False
    assert _warnings(caplog) == [], (
        "no degradation warning when nobody requested the ops -- that would recast "
        "'was off all along' as 'wanted on but failed to'")


# ======================================================================================
# Engineering discipline: LF + py_compile (build.sh runs under cluster bash;
# CRLF means an incident, same as test_no_crlf)
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
        f"{path} has CRLF; build.sh/source files go to the cluster verbatim, "
        f"and bash dies on $'\\r'")


@pytest.mark.parametrize("rel", ["terrace/ops/__init__.py",
                                 "terrace/ops/csrc/build_ext.py"])
def test_scaffold_python_compiles(rel):
    py_compile.compile(str(ROOT / rel), doraise=True)
