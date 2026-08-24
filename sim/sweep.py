# -*- coding: utf-8 -*-
"""外推:换集群参数,比较一跳 vs 两跳。**门在入口处,不是在注释里。**

- 通信级外推:需要 Tier-1 通过(sim/validate_micro.py,当前:通过)。
- 步级(端到端 G)外推:需要 Tier-2 通过(sim/validate.py,当前:**不过**,
  被内部实测记录 的相位账/步账矛盾挡住 —— 相位差 x 次数与步级差相差 ~5x,
  两臂在双流上的重叠不同,事件计时的相位跨度加不出步时。要解锁 Tier-2,
  需要重叠感知的合成模型 + 它自己的保留点,不是调参)。

跑法:
    python -m sim.sweep
"""
from __future__ import annotations

from .calibrate import aug_flat, synthetic
from .core import MoEGeometry, one_hop_call, two_hop_call
from .validate import validate
from .validate_micro import validate_micro

CHAIN_SCENARIOS = [
    ("PyTorch 链(实测)", 2.15 * 1000.0 / 24576.0),
    ("融合 kernel(估)", 0.012),
    ("零实现开销(上界)", 0.0),
]


def comm_speedup(cluster, q: int, tok: int, k_base: int = 6) -> float:
    """一跳/两跳 的通信耗时比(>1 = 两跳快)。q 通过 M=k/q 实现。"""
    k = k_base if k_base % q == 0 and k_base >= q else q
    m = max(k // q, 1)
    g = MoEGeometry(name="sweep", n_groups=16, R=cluster.R, k=k, M=m,
                    seq=tok, mbs=1, gbs=16 * cluster.R * tok)
    return one_hop_call(cluster, g) / two_hop_call(cluster, g)


def main() -> None:
    aug = aug_flat()
    t1_ok, _ = validate_micro(aug, verbose=False)
    print("Tier-1(通信级):%s" % ("通过" if t1_ok else "不过"))
    t2_ok, _ = validate(aug, verbose=False)
    print("Tier-2(步级):%s" % ("通过" if t2_ok else
                                "不过 —— 步级外推封禁(内部实测记录 相位/步账矛盾未解)"))
    if not t1_ok:
        raise SystemExit("Tier-1 不过,一切外推禁止。")

    print()
    print("=" * 74)
    print("通信级外推(**仿真**,标注纪律:这些不是实测)")
    print("横轴 = 快/慢带宽比;表值 = 一跳耗时/两跳耗时(>1 两跳快)")
    print("几何:16 组 x R=8,k=6,T=4096 tok/rank(对照床操作点)")
    print("=" * 74)
    ratios = [1.03, 2.0, 3.2, 4.5, 8.0, 15.7]
    labels = ["扁平超节点", "2x", "NVLink/IB 3.2x", "跨超节点 4.5x",
              "HCCS/RoCE 8x", "CM 内/间 15.7x"]
    for chain_name, chain in CHAIN_SCENARIOS:
        print("\n-- 实现档:%s(%.4f µs/行)--" % (chain_name, chain))
        print("%-16s" % "q \\ 比值", "".join("%12s" % l[:10] for l in labels))
        for q in (2, 3, 6):
            cells = []
            for rt in ratios:
                c = synthetic(rt, chain_us_per_row=chain)
                cells.append("%12.2f" % comm_speedup(c, q, 4096))
            print("%-16s" % ("q=%d" % q), "".join(cells))
    print()
    print("读法:")
    print("  · 扁平列(1.03)在任何实现档都 <=1 —— 与我们的实测判负一致(内部一致性)。")
    print("  · 层级比 >=3.2 的列全面 >1 —— 与公开工作(DeepSeek/TeleChat3/Pangu)的")
    print("    正面结果方向一致(外部一致性)。")
    print("  · 实现档的影响与层级比同量级 —— 「先修实现再谈拓扑」在层级机上同样成立。")


if __name__ == "__main__":
    main()
