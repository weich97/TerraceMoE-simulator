# -*- coding: utf-8 -*-
"""外推的不确定度:蒙特卡洛标定扰动 + breakeven(盈亏平衡层级比)。

## 为什么要这个模块(多数仿真工作缺的第二环)

标定常数不是真值,是带散布的测量。机台发间漂移是实测过的
(同基准三发 t(2048) = 0.648/0.772/0.711,±9% 左右;α 档漂移 ~20%)。
**一张没有误差带的外推表,读者没法判断"2.18x"和"1.9-2.5x"的区别** ——
而结论稳不稳恰恰取决于带的两端,不是中位数。

## 扰动口径(每个都注明出处;拍脑袋的明确标注「假设」)

  α 曲线    x U[0.80, 1.20]   整条曲线乘同一因子(发间漂移 ~20% 实测;
                              漂移是机台状态,整条曲线同涨同落)
  β(慢边)  x U[0.90, 1.10]   同基准三发 ±9% 实测,取整为 ±10%
  β(快边)  x U[0.995,1.005]  节点内档发间散布 <0.3% 实测(物理背书档)
  splits    x U[0.95, 1.05]   实测区间 0.042-0.046
  到达链    x U[0.90, 1.10]   「假设」:张量操作链,漂移应小于通信;±10% 保守

固定种子,任何人重跑得到逐位相同的带(`python -m sim.uncertainty`)。

## 判读纪律

带是**标定不确定度**的传播,不含模型结构误差(那由验证门管)。
两者叠加时以更宽者为准。
"""
from __future__ import annotations

import random

from .calibrate import synthetic
from .core import MoEGeometry, one_hop_call, two_hop_call

SEED = 20260825
N_DRAWS = 400

# (名字, 乘性扰动下界, 上界, 出处)
PERTURB = [
    ("alpha", 0.80, 1.20, "发间漂移 ~20%(实测)"),
    ("beta_slow", 0.90, 1.10, "三发 ±9%(实测,取整)"),
    ("beta_fast", 0.995, 1.005, "发间散布 <0.3%(实测,物理背书档)"),
    ("splits", 0.95, 1.05, "实测区间 0.042-0.046"),
    ("chain", 0.90, 1.10, "假设:张量链漂移小于通信"),
]


def _perturbed(ratio: float, chain: float, f: dict):
    """按一组因子建扰动后的合成集群。"""
    c = synthetic(ratio, chain_us_per_row=chain * f["chain"])
    # α:整条曲线同因子;β:快/慢各自因子(见模块 docstring 的口径表)
    for lvl, bf in ((c.fast, f["beta_fast"]), (c.slow, f["beta_slow"]),
                    (c.flat, f["beta_slow"])):
        lvl.alpha_pts = [(w, a * f["alpha"]) for w, a in lvl.alpha_pts]
        lvl.beta_pts = [(x, b * bf) for x, b in lvl.beta_pts]
    c.splits_sync_ms *= f["splits"]
    return c


def _speedup(cluster, q: int, tok: int, k_base: int = 6) -> float:
    k = k_base if k_base % q == 0 and k_base >= q else q
    m = max(k // q, 1)
    g = MoEGeometry(name="mc", n_groups=16, R=cluster.R, k=k, M=m,
                    seq=tok, mbs=1, gbs=16 * cluster.R * tok)
    return one_hop_call(cluster, g) / two_hop_call(cluster, g)


def mc_band(ratio: float, chain: float, q: int = 3, tok: int = 4096,
            n: int = N_DRAWS, seed: int = SEED):
    """返回 (p5, 中位, p95):一跳/两跳比在标定扰动下的分布分位。"""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        f = {name: rng.uniform(lo, hi) for name, lo, hi, _ in PERTURB}
        vals.append(_speedup(_perturbed(ratio, chain, f), q, tok))
    vals.sort()
    return vals[int(0.05 * n)], vals[n // 2], vals[int(0.95 * n)]


def breakeven_ratio(chain: float, q: int = 3, tok: int = 4096,
                    lo: float = 1.0, hi: float = 32.0) -> float:
    """两跳开始赢(比值=1)的最小层级比;区间内不穿越返回边界值。"""
    def s(r):
        return _speedup(synthetic(r, chain_us_per_row=chain), q, tok)
    if s(lo) >= 1.0:
        return lo
    if s(hi) < 1.0:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if s(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def heatmap(chain: float, q: int = 3,
            ratios=(1.03, 1.5, 2, 3, 4.5, 6, 8, 11, 16),
            toks=(512, 1024, 2048, 4096, 8192, 16384)):
    """(ratios x toks) 的比值矩阵,给图用。行 = tok,列 = ratio。"""
    return [[_speedup(synthetic(r, chain_us_per_row=chain), q, t)
             for r in ratios] for t in toks], list(ratios), list(toks)


def geometry_grid(chain: float, tok: int = 4096):
    """几何敏感性:breakeven 层级比随 (组数, R, k, M) 怎么走。

    (k, M) 显式枚举 —— 每行的实际 q = k/M 各不相同,标注按 (k, M) 走,
    不冒充同一个 q(第一版把 k=4/8 行都标成 q=3,是错的)。
    """
    rows = []
    for ng in (8, 16, 32):
        for R in (4, 8, 16):
            for k, m in ((4, 1), (4, 2), (6, 2), (6, 3), (8, 2), (8, 4)):
                if m > ng:
                    continue

                def s(r):
                    c = synthetic(r, R=R, chain_us_per_row=chain)
                    g = MoEGeometry(name="grid", n_groups=ng, R=R, k=k, M=m,
                                    seq=tok, mbs=1, gbs=ng * R * tok)
                    return one_hop_call(c, g) / two_hop_call(c, g)

                lo, hi = 1.0, 32.0
                if s(lo) >= 1.0:
                    be = lo
                elif s(hi) < 1.0:
                    be = hi
                else:
                    for _ in range(40):
                        mid = (lo + hi) / 2.0
                        if s(mid) >= 1.0:
                            hi = mid
                        else:
                            lo = mid
                    be = hi
                rows.append((ng, R, k, m, be))
    return rows


def main() -> None:
    from .sweep import CHAIN_SCENARIOS
    print("蒙特卡洛误差带(%d 次抽样,种子 %d;扰动口径见模块 docstring)" %
          (N_DRAWS, SEED))
    print("几何:16 组 x 8,k=6/M=2(q=3),T=4096;值 = 一跳/两跳(>1 两跳快)")
    ratios = [1.03, 2.0, 3.2, 4.5, 8.0, 15.7]
    for name, chain in CHAIN_SCENARIOS:
        print("\n-- %s --" % name)
        print("%-10s %10s %18s" % ("层级比", "中位", "[p5, p95]"))
        for r in ratios:
            p5, med, p95 = mc_band(r, chain)
            print("%-10.2f %10.2f       [%.2f, %.2f]" % (r, med, p5, p95))
    print("\nbreakeven 层级比(比值=1 的最小层级比,q=3,T=4096):")
    for name, chain in CHAIN_SCENARIOS:
        print("  %-24s %.2f" % (name, breakeven_ratio(chain)))
    print("\n两条鲁棒性锚(结论稳不稳看带的两端):")
    p5, _, p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])   # 零开销 + 扁平
    print("  扁平列最有利情形(零实现开销)p95 = %.2f -> %s" %
          (p95, "≤1,判负结论对标定误差鲁棒" if p95 <= 1.0 else "!! 越线,写结论要收敛"))
    p5b, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])     # PyTorch 链 + 8x
    print("  8x 列最不利情形(PyTorch 链)p5 = %.2f -> %s" %
          (p5b, "仍 >1,方向结论鲁棒" if p5b > 1.0 else "<1:8x 列的赢面依赖实现档,只报融合档以上"))


if __name__ == "__main__":
    main()
