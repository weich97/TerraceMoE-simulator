# -*- coding: utf-8 -*-
"""两跳(分层)all-to-all 的盈亏判据:你的集群到底该不该用这套方法。

这是本仓库最重要的一个文件。T-Route + T-A2A 是**面向层级化特征明显的集群**
设计的;在带宽扁平的互联上它不划算 —— 我们自己就在一台扁平超节点上把它测到了负收益
(见 docs/03-applicability.md)。**先跑这个判据,再决定要不要接入。**

------------------------------------------------------------------------------
模型

每 token、每个目标组(组 = 服务器 / 机柜 / 超节点,取决于你的层级边界),
组内 R 个加速卡、该 token 在组内命中 q 个专家(q = top-k / 组数上限 M):

    一跳(平坦 a2a)   q 行载荷全部跨**慢**链路                     慢侧 q
    两跳(T-A2A)      1 行跨慢链路到代表卡,再在**快**侧散开        慢侧 1 + 快侧 q(1-1/R)

    省下的慢侧 = q - 1          加出的快侧 = q(1 - 1/R)

    **盈亏平衡比  r_be = (1 - 1/R) * q / (q - 1)**

你的集群快/慢两级的带宽比(beta_fast / beta_slow)必须**大于** r_be,
两跳在字节上才净赚。R=8 时 r_be 随 q 单调下降、极限 0.875:

    q=2 -> 1.750    q=3 -> 1.3125    q=4 -> 1.167    q=6 -> 1.050    q=8 -> 1.000

注意两点:
  * 这个式子假设两侧都按 (token, expert) 逐份发载荷 —— 与主流实现一致。
    若两跳侧再做卡级去重(同一 token 发往同卡的多个专家只发一行),
    r_be 还能再降(用 --dedup 看)。
  * 字节只是账的一半:两跳多付一次集合通信的固定开销(发射/同步),
    小消息、深流水的场景要单独核 α 侧;判据通过只说明"值得做实验",
    不等于"必然更快"。

------------------------------------------------------------------------------
公开参照(数量级,出处见 docs/03;以你自己的实测为准):

    NVLink 域内 vs 跨节点 IB/RoCE      ~3-18x   —— 判据轻松通过,同族方法已被
                                                  多个公开工作验证有效
    服务器内 HCCS vs 跨服务器 RoCE      ~8x     —— 同上
    扁平超节点内部(统一交换)           ~1.0    —— **判据不通过,别用两跳**

用法:
    python tools/breakeven.py --ratio 8          # 你的集群的快/慢带宽比
    python tools/breakeven.py --ratio 8 --dies 8 --dedup
"""
from __future__ import annotations

import argparse
from math import comb


def D_of(q: int, R: int, E: int) -> float:
    """token 的 q 个专家落在几张不同卡上(期望)。只在卡级去重时用到。

    组内 S = R*E 个专家均匀可选;某张卡一个专家都没被选中的概率是
    C(S-E, q)/C(S, q),取补再乘 R 即期望命中卡数。
    """
    S = R * E
    if q <= 0 or q > S:
        return 0.0
    return R * (1.0 - comb(S - E, q) / comb(S, q))


def breakeven(q: int, R: int, dedup: bool = False, E: int = 4) -> float:
    """两跳净赚所需的 beta_fast/beta_slow 下限。q=1 时两跳纯亏(省 0),返回 inf。"""
    if q <= 1:
        return float("inf")
    added = (D_of(q, R, E) if dedup else q) * (1.0 - 1.0 / R)
    return added / (q - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratio", type=float, default=None,
                    help="你的集群的 快侧带宽 / 慢侧带宽(自己实测,别用标称)")
    ap.add_argument("--dies", type=int, default=8, help="每组(快侧域)内的卡数 R")
    ap.add_argument("--epr", type=int, default=4, help="每卡专家数(仅 --dedup 用)")
    ap.add_argument("--dedup", action="store_true",
                    help="两跳侧做卡级去重时的更宽松判据")
    args = ap.parse_args()
    R, E = args.dies, args.epr

    print("两跳 all-to-all 的盈亏平衡带宽比(组内 R=%d 卡)" % R)
    print()
    print("  q(每组专家数)   r_be%s" % ("(卡级去重)" if args.dedup else ""))
    rows = []
    for q in range(2, 13):
        r = breakeven(q, R, args.dedup, E)
        rows.append((q, r))
        mark = ""
        if args.ratio is not None:
            mark = "   <- 你的 %.2f %s" % (args.ratio,
                                           "**够**" if args.ratio > r else "不够")
        print("  %2d               %7.4f%s" % (q, r, mark))
    print()
    if args.ratio is None:
        print("加 --ratio <你的实测比值> 得到判定。**用实测,不要用标称**:")
        print("标称带宽与集合通信的有效带宽经常差 2 倍以上。")
        return
    good = [q for q, r in rows if args.ratio > r]
    if good:
        # **边缘警示**:字节侧余量太薄时,α 侧(两跳多付的一次集合通信固定开销)
        # 几乎必然把它吃掉。扁平互联(比值 ~1.0)在大 q 下也能"数学上通过"
        # (r_be 的极限是 1-1/R < 1),但那 1-3% 的余量不是工程上可兑现的信号。
        margin = max(args.ratio / r - 1.0 for q, r in rows if q in good)
        if margin < 0.10:
            print("比值 %.2f 只在 q ∈ %s 勉强过线,最大余量 %.1f%% —— **边缘情况,"
                  "几乎必被 α 侧吃掉,不建议接入**。" % (args.ratio, good, margin * 100))
            print("这套方法要的是余量倍数级的层级(见 docs/03 §2),不是百分位的擦线。")
        else:
            print("比值 %.2f 下,q ∈ %s 时两跳在字节上净赚(最大余量 %.0f%%)—— "
                  "值得接入实验。" % (args.ratio, good, margin * 100))
            print("下一步:核 α 侧(两跳多一次集合通信的固定开销),见 docs/03 §4。")
    else:
        print("比值 %.2f 不超过任何配额的平衡线 —— **你的互联对这套方法太扁平,"
              "别用两跳**。T-Route 的路由约束部分(负载均衡)仍可独立评估。"
              % args.ratio)


if __name__ == "__main__":
    main()
