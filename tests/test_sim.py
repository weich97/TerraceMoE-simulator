# -*- coding: utf-8 -*-
"""terrace-sim 的契约测试:几何账、标定一致性、两层验证门的当前状态。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.core import MoEGeometry, _interp, one_hop_call, two_hop_call  # noqa: E402
from sim.calibrate import aug_flat, synthetic                          # noqa: E402


def test_geometry_call_counts_match_ledger():
    """flag 几何的每步调用数必须等于账本钉死的 152/76(内部实测记录)。"""
    g = MoEGeometry(name="flag", n_groups=16, R=8, k=6, M=2, mbs=1)
    assert g.microbatches == 4
    assert g.calls_per_step_fwd() == 152
    assert g.calls_per_step_bwd() == 76
    # n8:EP=64 -> microbatch 翻倍
    n8 = MoEGeometry(name="n8", n_groups=8, R=8, k=6, M=2, mbs=1)
    assert n8.microbatches == 8


def test_interp_clamps_do_not_extrapolate():
    assert _interp([(1, 10.0), (2, 20.0)], 0.5) == 10.0
    assert _interp([(1, 10.0), (2, 20.0)], 99) == 20.0


def test_calibration_matches_independent_ledger_numbers():
    """拟合出的 α(128)/β 必须落在账本独立记过的数附近 —— 标定自检。

    内部实测记录:Aug α(128)≈0.47(64→192MB 斜率反推);β≈118 GB/s。
    拟合走的是另一条路(33 份扫描全尺寸最小二乘),两者独立。
    """
    c = aug_flat()
    a128 = c.flat.alpha_ms(128)
    b = c.flat.beta_gbps(8e6)
    assert 0.35 <= a128 <= 0.60, "α(128)=%.3f 偏离账本的 ~0.47" % a128
    assert 100 <= b <= 140, "β=%.1f 偏离账本的 ~118" % b


def test_breakeven_consistency_with_analysis():
    """仿真核与解析判据(analysis/hier_breakeven)必须同向:

    比值 8、q=3、零实现开销 -> 两跳赢;扁平(1.03)、q=3、PyTorch 链 -> 两跳输。
    这是两条独立实现的同一套账,方向拧了就是有一边算错。
    """
    g = MoEGeometry(name="x", n_groups=16, R=8, k=6, M=2, mbs=1)
    hier = synthetic(8.0, chain_us_per_row=0.0)
    assert one_hop_call(hier, g) > two_hop_call(hier, g)
    flat = synthetic(1.03, chain_us_per_row=2.15 * 1000 / 24576)
    assert one_hop_call(flat, g) < two_hop_call(flat, g)


def test_tier1_micro_gate_passes():
    """Tier-1(通信微观)当前必须通过 —— 它是一切外推的前提。"""
    from sim.validate_micro import validate_micro
    ok, info = validate_micro(aug_flat(), verbose=False)
    assert ok, "Tier-1 掉下去了:median=%.3f worst=%.3f cross=%s" % (
        info["median"], info["worst"], info["cross"])


def test_tier2_step_gate_currently_fails_documented():
    """Tier-2(步级)当前**必须不过** —— 这是文档化的已知状态,不是期望。

    它被内部实测记录 的相位账/步账矛盾挡住(相位差 x 次数 ≈ 5x 步级差,
    两臂双流重叠不同,相位跨度加不出步时)。
    **若这条测试哪天红了(门突然通过),不是好消息自动成立** —— 说明有人改了
    模型或数据,必须人工核查是真解决了 内部实测记录,还是把门改松了。
    """
    from sim.validate import validate
    ok, info = validate(aug_flat(), verbose=False)
    assert not ok, (
        "Tier-2 突然通过了(MAE=%.4f)。先别庆祝:去核查是真解决了 内部实测记录"
        "(重叠感知合成 + 自己的保留点),还是门被改松了。" % info["mae"])


def test_sweep_internal_external_consistency():
    """外推的两条锚:扁平列复现判负(≤1),8x 列复现公开正面结果(>1)。"""
    from sim.sweep import comm_speedup
    flat = synthetic(1.03)
    hier = synthetic(8.0)
    assert comm_speedup(flat, 3, 4096) <= 1.0
    assert comm_speedup(hier, 3, 4096) > 1.0
