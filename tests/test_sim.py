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

    内部实测记录:本机 α(128)≈0.47(64→192MB 斜率反推);β≈118 GB/s。
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


# ---------------------------------------------------------------- 重叠模型族


def test_overlap_families_all_fail_documented():
    """六个单参数重叠模型族当前**全部不过门** —— 文档化的负结果(docs/07)。

    与 Tier-2 门同款逻辑:哪天有族突然回溯通过,不构成解锁 —— 六族在同一组
    保留点上串行评估本身就是模型选择,解锁只认 docs/07 协议新采的保留点 +
    人工核查。这条测试红了 = 有人改了模型/数据,先查再说。
    """
    from sim.overlap import evaluate
    from sim.calibrate import flat_supernode
    res = evaluate(flat_supernode(), verbose=False)
    assert len(res) == 6
    passed = [k for k, r in res.items() if r["gate"]]
    assert not passed, (
        "族 %s 突然回溯过门。这不是解锁:回溯通过 = 七点上的模型选择;"
        "去按 docs/07 协议采新保留点,人工核查后才动 Tier-2 的门。" % passed)


def test_overlap_family_structure_pins():
    """钉住负结果的两个结构性事实(docs/07 §1 的读法依据):

    - M4(隐藏∝计算)方向全对(5/5)但 MAE 差门约 2 倍;
    - M2(每调用隐藏)量级最近却在规模轴翻符号(负号 <5)。
    这两个数变了 = 标定或模型动了,docs/07 的表要跟着重发。
    """
    from sim.overlap import evaluate
    from sim.calibrate import flat_supernode
    res = evaluate(flat_supernode(), verbose=False)
    assert res["M4"]["signs_ok"] == 5 and 0.035 <= res["M4"]["mae"] <= 0.07
    assert res["M2"]["signs_ok"] < 5 and res["M2"]["mae"] <= 0.08
    assert res["M0"]["mae"] >= 0.10   # naive 基线的失败量级(~0.14)
    # docs/07 §1 表的 MAE 快照 pin(±0.005):标定常数一动这里就红,
    # 提醒同步重发 docs/07 的表(审查发现到达链常数漂 20% 时宽 pin 抓不住)
    for fam, doc in (("M0", 0.141), ("M1", 0.149), ("M2", 0.065),
                     ("M3", 0.140), ("M4", 0.048), ("M5", 0.087)):
        assert abs(res[fam]["mae"] - doc) <= 0.005,             "%s MAE=%.4f 偏离 docs/07 快照 %.3f" % (fam, res[fam]["mae"], doc)


# ---------------------------------------------------------------- 不确定度


def test_mc_bands_reproducible_and_anchor_robust():
    """固定种子的蒙特卡洛必须逐位可复现;两条鲁棒性锚必须成立:

    - 扁平列最有利情形(零实现开销)p95 ≤ 1:判负对标定误差鲁棒;
    - 8x 列最不利情形(PyTorch 链)p5 > 1:层级机方向鲁棒。
    锚破了 = 标定常数或扰动口径变了,docs/05 的结论句要跟着改。
    """
    from sim.uncertainty import mc_band
    from sim.sweep import CHAIN_SCENARIOS
    a = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    b = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    assert a == b, "同种子两次结果不同 —— 可复现性破了"
    _, _, flat_p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])
    assert flat_p95 <= 1.0, "扁平列 p95=%.3f > 1:判负结论不再鲁棒" % flat_p95
    hier_p5, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])
    assert hier_p5 > 1.0, "8x 列 p5=%.3f ≤ 1:方向结论不再鲁棒" % hier_p5


def test_breakeven_ordering_and_snapshot():
    """breakeven 层级比必须随实现档单调下降,且与 docs/05 的快照一致(±0.1)。"""
    from sim.uncertainty import breakeven_ratio
    from sim.sweep import CHAIN_SCENARIOS
    bes = [breakeven_ratio(chain) for _, chain in CHAIN_SCENARIOS]
    assert bes[0] > bes[1] > bes[2] >= 1.0
    for got, doc in zip(bes, (4.20, 1.57, 1.16)):
        assert abs(got - doc) <= 0.1, "breakeven %.2f 偏离 docs/05 快照 %.2f" % (got, doc)


def test_heatmap_monotone_in_ratio():
    """同一 token 档,层级比越大两跳越有利 —— 比值对 ratio 单调不减。"""
    from sim.uncertainty import heatmap
    from sim.sweep import CHAIN_SCENARIOS
    mat, ratios, toks = heatmap(CHAIN_SCENARIOS[1][1])
    for row in mat:
        assert all(row[j + 1] >= row[j] - 1e-9 for j in range(len(row) - 1))


def test_geometry_grid_bounds():
    """54 组几何网格的 breakeven 全部落在 [1, 32] 且不出现荒谬值(>3)。"""
    from sim.uncertainty import geometry_grid
    from sim.sweep import CHAIN_SCENARIOS
    rows = geometry_grid(CHAIN_SCENARIOS[1][1])
    assert len(rows) == 54
    assert all(1.0 <= be <= 3.0 for *_, be in rows),         "有几何的 breakeven 出圈:%s" % [r for r in rows if not 1.0 <= r[-1] <= 3.0]
