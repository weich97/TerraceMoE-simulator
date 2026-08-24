# -*- coding: utf-8 -*-
"""Tier-1 验证门:通信微观层 —— 模型 vs 同机直测的两种策略耗时。

## 为什么门的目标是「多发合并中位」而不是单发

同一基准在同一台机上连跑,发间漂移 ~20%(例:两跳 @2048 tok 三发分别
0.648 / 0.772 / 0.711 ms)。单发目标的分辨率低于机台漂移,门的红绿等于掷骰 ——
我们在这上面栽过(第一版恰好用了一发"顺手"的数据,过了门;换成另一发就不过)。
**量具的分辨率必须优于要测的效应**,对验证门同样成立。

## 预注册门(阈值先于目标值写死)

  相对误差:中位 ≤ 20%,最大 ≤ 35%;
  交叉点(一跳/两跳 从 <1 变 >1 再回落的换向位置)落在实测窗口 [2048, 16384]。

Tier-1 过 = 允许**通信级**外推。步级外推另由 Tier-2 把守(sim/validate.py)。

## 目标值(4 发合并中位;基准为纯通信形态:定长等分、无 counts 交换、无到达链)

外部使用者换机器时:用 bench 同款口径重测,替换这张表,重新过门。
"""
from __future__ import annotations

import dataclasses

from .core import MoEGeometry, one_hop_call, two_hop_call

# (token/rank, 一跳 ms, 两跳 ms, 参与合并的 run 数)
# 口径:16 组 x 8 卡,k=6/M=2 的行数结构;<256 token 的档发间散布 1.8-2.4,不作目标
MICRO_TARGETS = [
    (256, 0.394, 0.340, 4),
    (1024, 0.639, 0.510, 2),
    (2048, 0.856, 0.670, 4),
    (4096, 1.366, 1.217, 4),
    (8192, 2.119, 2.360, 2),
]


def validate_micro(cluster, verbose: bool = True):
    micro = dataclasses.replace(cluster, splits_sync_ms=0.0, chain_us_per_row=0.0)
    rows, rel = [], []
    for tok, mv, mt, n_runs in MICRO_TARGETS:
        g = MoEGeometry(name="micro", n_groups=16, R=8, k=6, M=2,
                        seq=tok, mbs=1, gbs=16 * 8 * tok)
        pv, pt = one_hop_call(micro, g), two_hop_call(micro, g)
        ev, et = (pv - mv) / mv, (pt - mt) / mt
        rel += [abs(ev), abs(et)]
        rows.append((tok, mv, pv, ev, mt, pt, et, n_runs))
    rel_sorted = sorted(rel)
    med = rel_sorted[len(rel_sorted) // 2]
    worst = rel_sorted[-1]

    cross_lo = cross_hi = None
    prev = None
    for tok in (1024, 2048, 4096, 8192, 16384):
        g = MoEGeometry(name="x", n_groups=16, R=8, k=6, M=2,
                        seq=tok, mbs=1, gbs=16 * 8 * tok)
        ratio = one_hop_call(micro, g) / two_hop_call(micro, g)
        if prev is not None and (prev[1] - 1.0) * (ratio - 1.0) < 0:
            cross_lo, cross_hi = prev[0], tok
        prev = (tok, ratio)
    cross_ok = cross_lo is not None and cross_lo >= 2048 and cross_hi <= 16384

    ok = (med <= 0.20) and (worst <= 0.35) and cross_ok
    if verbose:
        print("Tier-1(通信微观)vs 多发合并中位")
        print("%6s %4s  %8s %8s %7s   %8s %8s %7s" %
              ("tok", "runs", "一跳实测", "预测", "误差", "两跳实测", "预测", "误差"))
        for tok, mv, pv, ev, mt, pt, et, n in rows:
            print("%6d %4d  %8.3f %8.3f %+6.1f%%   %8.3f %8.3f %+6.1f%%" %
                  (tok, n, mv, pv, ev * 100, mt, pt, et * 100))
        print("相对误差:中位 %.1f%%(门 20)最大 %.1f%%(门 35);交叉点 %s(门 [2048,16384])"
              % (med * 100, worst * 100,
                 "%s-%s" % (cross_lo, cross_hi) if cross_lo else "未穿过"))
        print("**Tier-1 %s**" % ("通过 —— 允许通信级外推" if ok else "不通过"))
    return ok, {"median": med, "worst": worst,
                "cross": (cross_lo, cross_hi), "rows": rows}


if __name__ == "__main__":
    from .calibrate import flat_supernode
    validate_micro(flat_supernode())
