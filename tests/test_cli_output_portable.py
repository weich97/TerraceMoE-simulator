# -*- coding: utf-8 -*-
"""Every command this repository tells someone to run must survive their console.

The README's first instruction to an adopter is ``python tools/breakeven.py --ratio
<your measured ratio>``, and docs/05 lists a dozen ``python -m sim.*`` entry points as
the way to recompute the tables. Until 2026-09-05 several of them raised
``UnicodeEncodeError`` part-way through their output on any console that cannot encode
the characters they print -- a POSIX locale, a container with ``LANG`` unset, a plain
Windows code page. ``tools/breakeven.py`` failed on every one of its three verdict
branches, so the applicability criterion, the single thing this repository asks people
to run first, crashed instead of answering.

Python encodes to the console's encoding, not to UTF-8, so this is a property of the
reader's machine and not of ours: it cannot be found by running the tools here. The
test therefore does what the unlucky console does -- it replaces stdout with a strict
ASCII writer and runs the real entry point.

Prose keeps its unicode. This is about the bytes that reach a terminal.
"""
import io
import runpy
import sys
import warnings

import pytest

# The entry points README and docs/05 hand to a reader, each run as ``__main__`` so
# the test exercises the same path the instruction does.
CLI_MODULES = [
    "sim.validate_micro",
    "sim.validate_sweep",
    "sim.compute",
    "sim.imbalance",
    "sim.platforms",
    "sim.profile",
    "sim.phase",
    "sim.validate",
    "sim.overlap",
    "sim.sweep",
    "sim.codesign",
    "sim.envelope",
    "sim.archsearch",
    "sim.record",
    "sim.hostregime",
    "sim.hierarchy",
    "sim.tiers",
]

# tools/breakeven.py branches on the ratio it is given, and each branch prints its own
# verdict sentence, so one ratio exercises only one third of the output. 0.8 is below
# the criterion's own floor of 1 - 1/R, so no quota clears; 1.03 is platform A, which
# scrapes past at large q and is reported as marginal; 9.0 clears outright.
BREAKEVEN_RATIOS = ["0.8", "1.03", "9.0"]


def _run(fn):
    """Run ``fn`` with stdout as an unlucky console sees it, and return what it wrote."""
    buf = io.BytesIO()
    out = io.TextIOWrapper(buf, encoding="ascii", errors="strict", newline="")
    real = sys.stdout
    sys.stdout = out
    try:
        with warnings.catch_warnings():
            # Re-running a module that this session already imported is what runpy
            # warns about, and it is an artifact of testing the entry point in
            # process, not something a reader of the command ever sees.
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message=r".*found in sys\.modules.*")
            fn()
        out.flush()
        return buf.getvalue()
    finally:
        sys.stdout = real
        out.detach()      # leave buf open; a collected wrapper would close it


@pytest.mark.parametrize("modname", CLI_MODULES)
def test_module_entry_point_prints_ascii(modname):
    try:
        _run(lambda: runpy.run_module(modname, run_name="__main__"))
    except UnicodeEncodeError as e:
        pytest.fail(
            "python -m %s crashes on a console that cannot encode %r.\n"
            "Printed output has to be ASCII: use '--' for an em dash, '+-' for the "
            "plus-minus sign, '<=' and '>=' for the inequalities, and spell out Greek "
            "letters. Docstrings and markdown are unaffected." % (modname, e.object[e.start:e.end]))


@pytest.mark.parametrize("ratio", BREAKEVEN_RATIOS)
def test_breakeven_verdict_branches_print_ascii(ratio):
    argv = sys.argv
    sys.argv = ["breakeven.py", "--ratio", ratio]
    try:
        _run(lambda: runpy.run_path("tools/breakeven.py", run_name="__main__"))
    except UnicodeEncodeError as e:
        pytest.fail(
            "tools/breakeven.py --ratio %s crashes on a console that cannot encode "
            "%r. This is the first command the README gives an adopter; it has to "
            "produce a verdict on any terminal." % (ratio, e.object[e.start:e.end]))
    finally:
        sys.argv = argv


def test_the_three_ratios_cover_all_three_verdicts():
    """The ratios above must actually reach the three different verdict branches.

    Without this the test could pass by exercising one branch three times, which is how
    the crash survived: the tool was only ever run at the ratio of the machine in front
    of us.
    """
    argv = sys.argv
    seen = set()
    try:
        for ratio in BREAKEVEN_RATIOS:
            sys.argv = ["breakeven.py", "--ratio", ratio]
            text = _run(lambda: runpy.run_path("tools/breakeven.py",
                                               run_name="__main__"))
            if b"does not clear the breakeven line" in text:
                seen.add("none")
            elif b"marginal case" in text:
                seen.add("marginal")
            elif b"worth an integration experiment" in text:
                seen.add("clears")
    finally:
        sys.argv = argv
    assert seen == {"none", "marginal", "clears"}, (
        "the ratios only reach %s; every verdict branch prints its own sentence and "
        "each one needs a reader" % sorted(seen))
