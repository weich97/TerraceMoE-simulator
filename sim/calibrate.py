# -*- coding: utf-8 -*-
"""标定:把实测蒸馏成 ClusterSpec。**公开版嵌入常数,不带原始扫描数据。**

每个常数标注口径与出处性质;原始测量(几十份扫描 JSON、逐对探针)属内部数据,
不随仓发布 —— 但**验证门的目标值**(validate_micro)同样来自独立测量,
外部使用者可以用自己的机器重走同一套流程:测原语 -> 填 ClusterSpec ->
过验证门 -> 才允许外推。

产出两类集群:
  flat_supernode()  我们实测过的那台带宽扁平超节点(16 节点 x 8 卡)
  synthetic(...)    合成层级集群 —— 验证门通过后才允许用于外推
"""
from __future__ import annotations

from .core import ClusterSpec, Level

# ---------------------------------------------------------------------------
# 蒸馏常数(全部为实测,口径注明;机台发间漂移 ~20%,以中位为准)
# ---------------------------------------------------------------------------

# α(world) [ms]:每次集合通信的固定开销。
#   8 / 16 / 128 三点为本机直测(1 token/rank 档,两发中位);
#   256 / 512 借另一台机器的曲线形状、按本机 128 点重标(只有合成大集群用到,
#   使用时报告敏感性)。直测同时给出一个重要事实:**本机 α(16)+α(8) ≈ α(128)**,
#   两跳在 α 上几乎不省 —— α 的可省性是机器性质,不可跨机假设。
ALPHA_PTS = [(2, 0.09), (8, 0.111), (16, 0.157), (128, 0.378),
             (256, 0.735), (512, 1.859)]

# β [GB/s](对齐口径:每对端字节为真实行宽的整数倍;不对齐会掉进
# 按 2 的幂阶梯变化的实现行为,测的是对齐效应不是链路 —— 我们踩过)。
BETA_FLAT = 122.2    # 128 卡全域 a2a:33 份扫描、6 尺寸、最小二乘拟合的纯 β
BETA_FAST = 122.4    # 节点内 8 卡 a2a:**物理背书** ——
                     #   实测 88.08 MB / 0.719 ms = 122.6,
                     #   物理聚合出口 (6x节点内链路 112.1 + 1x封装内直连 185)/7 = 122.4
                     #   两者差 0.2%;该档发间散布 <0.3%,是全部数据里最稳的
CROSS_NODE_RATIO = 0.974   # 跨节点/节点内(逐对探针,360 对,CV<0.4%)——平的

SPLITS_SYNC_MS = 0.044     # 变长 a2a 的 splits 主机取回,每次(实测 0.042-0.046)
CHAIN_US_PER_ROW = 2.15 * 1000.0 / 24576.0   # 到达链 PyTorch 组合链,按行
                                             # (2.15 ms/次 @ 24576 行实测)


def flat_supernode() -> ClusterSpec:
    """我们实测过的那台机器:带宽扁平超节点(跨节点/节点内 = 0.974)。"""
    beta_flat = [(1e5, BETA_FLAT), (1e9, BETA_FLAT)]
    beta_fast = [(1e5, BETA_FAST), (1e9, BETA_FAST)]
    beta_slow = [(x, b * CROSS_NODE_RATIO) for x, b in beta_flat]
    return ClusterSpec(
        name="flat_supernode(实测标定)", R=8,
        fast=Level("node-internal", ALPHA_PTS, beta_fast),
        slow=Level("cross-node", ALPHA_PTS, beta_slow),
        flat=Level("full-fabric", ALPHA_PTS, beta_flat),
        splits_sync_ms=SPLITS_SYNC_MS,
        chain_us_per_row=CHAIN_US_PER_ROW,
    )


# 兼容别名(内部代码/测试用这个名字)
aug_flat = flat_supernode


def synthetic(ratio: float, name: str = "", R: int = 8,
              base_beta_gbps: float = 100.0, alpha_like_measured: bool = True,
              chain_us_per_row: float = CHAIN_US_PER_ROW,
              **_compat) -> ClusterSpec:
    """合成层级集群:快侧 β = base,慢侧 β = base/ratio,α 借实测表。

    **只允许在验证门通过之后使用**(sim/validate_micro.py / sim/validate.py 把守);
    产出一律标注「仿真外推」。α 表借用带着上面的警告:α 的形状是机器性质。
    """
    ap = ALPHA_PTS if alpha_like_measured else [(2, 0.05), (512, 0.05)]
    flat_b = [(x, base_beta_gbps / ratio) for x in (1e5, 1e6, 1e7, 1e8)]
    fast_b = [(x, base_beta_gbps) for x in (1e5, 1e6, 1e7, 1e8)]
    return ClusterSpec(
        name=name or ("synthetic(ratio=%.2f)" % ratio), R=R,
        fast=Level("fast", ap, fast_b),
        slow=Level("slow", ap, flat_b),
        flat=Level("flat", ap, flat_b),   # 一跳的有效带宽由慢边决定
        splits_sync_ms=SPLITS_SYNC_MS, chain_us_per_row=chain_us_per_row,
    )
