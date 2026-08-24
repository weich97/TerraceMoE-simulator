# -*- coding: utf-8 -*-
"""仿真核心:集群规格、MoE 几何 → 流量,两种策略的每次调用成本,步级合成。

每条公式旁标注它的实测出处(内部实测记录)。**公式全部用 2026-08-24
红队更正后的版本**(内部实测记录):字节账按「两侧都逐份发、不做卡级去重」,
Hop A 发往自己节点那份是纯自拷贝(_send_index 恒等于发送方,不过链路)。
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# 集群规格
# ----------------------------------------------------------------------------

@dataclass
class Level:
    """一层通信域:能做 a2a 的一组卡。

    alpha_ms(world) 与 beta_gbps(per_peer_bytes) 都是可调用对象或插值表 ——
    标定层(sim/calibrate.py)负责把实测点装进来;合成集群直接给解析形式。
    """
    name: str
    # α(world) [ms]:每次集合通信的固定开销。实测形状:world>=16 近似线性
    # (另一测试框,内部实测记录:斜率 ~0.0107 ms/rank,本机整体重标)
    alpha_pts: list = field(default_factory=list)     # [(world, ms)]
    # β 有效带宽 [GB/s],按每对端字节插值(对齐尺寸口径,内部实测记录:真实行宽下平坦)
    beta_pts: list = field(default_factory=list)      # [(per_peer_bytes, GB/s)]

    def alpha_ms(self, world: int) -> float:
        return _interp(self.alpha_pts, float(world))

    def beta_gbps(self, per_peer_bytes: float) -> float:
        return _interp(self.beta_pts, per_peer_bytes, logx=True)


def _interp(pts, x, logx=False):
    """分段线性插值,端点外夹紧(外推是标定的事,不是插值的事)。"""
    assert pts, "空插值表 —— 标定没跑"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if logx:
        t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    else:
        t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@dataclass
class ClusterSpec:
    """一台(可能是假想的)集群。

    fast  = 组内域(节点内 R 张卡)
    slow  = 跨组域(节点间;跨超节点场景下 = 超节点间)
    flat  = 一跳 a2a 实际走的「全域」(world = n_groups*R;扁平机上它与 slow
            同速,层级机上它的有效带宽由慢边决定 —— 标定/合成时显式给)

    knobs(全部有实测出处,可按场景开关):
      splits_sync_ms    变长 a2a 的 splits 主机取回,每次(内部实测记录)
      chain_us_per_row  两跳到达链的本地张量操作,**按行**。
                        第一版把它当常数 —— 验证门当场不过(负载轴 MAE 0.084):
                        实测惩罚随 T 涨得比字节快,而到达链是张量操作,随行数线性。
                        内部实测记录, k=6 = 24576 行)上测的
                        => 0.0875 µs/行。融合 kernel 情景 ~0.012;完美情景 0。
    """
    name: str
    R: int
    fast: Level
    slow: Level
    flat: Level
    splits_sync_ms: float = 0.044
    chain_us_per_row: float = 2.15 * 1000.0 / 24576.0   # = 0.0875 µs/行,内部实测记录/flag

    def ratio(self) -> float:
        """快/慢带宽比(在 8 MB 对齐操作点上取,报告用)。"""
        p = 8 * 2 ** 20
        return self.fast.beta_gbps(p) / self.slow.beta_gbps(p)


# ----------------------------------------------------------------------------
# MoE 几何 → 每次调用的流量
# ----------------------------------------------------------------------------

@dataclass
class MoEGeometry:
    """对照床几何的参数化(内部实测记录 的几何表)。"""
    name: str
    n_groups: int          # 组数 = 节点数(组即层级边界)
    R: int                 # 组内卡数
    k: int                 # top-k
    M: int                 # 组数上限
    H: int = 2048          # 隐藏维
    seq: int = 4096
    mbs: int = 1
    gbs: int = 512
    moe_layers: int = 19   # 20 层 - 1 dense 首层
    bytes_per_elem: int = 2   # bf16

    @property
    def q(self) -> int:
        assert self.k % self.M == 0
        return self.k // self.M

    @property
    def ep(self) -> int:
        return self.n_groups * self.R

    @property
    def tokens_per_rank(self) -> int:
        return self.seq * self.mbs

    @property
    def microbatches(self) -> int:
        # DP = EP(对照床恒真);GBS 均摊到 DP,再按 MBS 切
        assert self.gbs % (self.ep * self.mbs) == 0, "GBS 不整除,床上会直接拦"
        return self.gbs // (self.ep * self.mbs)

    def calls_per_step_fwd(self) -> int:
        # 内部实测记录:每步前向 dispatch 次数 = MoE 层数 x microbatch x 2(重算重放)
        return self.moe_layers * self.microbatches * 2

    def calls_per_step_bwd(self) -> int:
        return self.moe_layers * self.microbatches

    def row_bytes(self) -> int:
        return self.H * self.bytes_per_elem

    # ---- 每 rank 每次调用发出的行数(内部实测记录 更正后的账;两侧都逐份发)----
    def rows_one_hop(self) -> int:
        return self.tokens_per_rank * self.k

    def rows_hop_a(self) -> int:
        # 每 (token, 入选组) 1 行;发往自己组那份是纯自拷贝、不过链路,
        # 但缓冲里仍占一行(_send_index 布局)。链路账在 cost 侧扣。
        return self.tokens_per_rank * self.M

    def rows_hop_b(self) -> int:
        return self.tokens_per_rank * self.k


# ----------------------------------------------------------------------------
# 策略成本
# ----------------------------------------------------------------------------

def _a2a_ms(level: Level, world: int, total_rows: int, row_bytes: int,
            self_fraction: float) -> float:
    """一次 a2a 的墙钟 [ms]。

    total_rows 里 self_fraction 是自拷贝(不过链路);其余按每对端均匀,
    时间 = α(world) + 过链路字节 / β(每对端字节)。
    β 按**每对端字节**查表 —— 内部实测记录/内部实测记录:带宽对每对端消息尺寸敏感,
    对齐(真实行宽的整数倍)口径下平坦,标定表就是这个口径。
    """
    wire_rows = total_rows * (1.0 - self_fraction)
    wire_bytes = wire_rows * row_bytes
    per_peer = wire_bytes / max(world - 1, 1)
    beta = level.beta_gbps(per_peer)                       # GB/s
    return level.alpha_ms(world) + wire_bytes / (beta * 1e6)   # bytes/(GB/s)=ns*... -> ms


def one_hop_call(c: ClusterSpec, g: MoEGeometry) -> float:
    """厂商一跳:一次全域 a2a。自拷贝份额 = 1/EP(均匀路由期望)。"""
    return _a2a_ms(c.flat, g.ep, g.rows_one_hop(), g.row_bytes(),
                   self_fraction=1.0 / g.ep)


def two_hop_call(c: ClusterSpec, g: MoEGeometry) -> float:
    """T-A2A 两跳(串行,与现实现一致;内部实测记录:无流水)。

    Hop A:跨组域,world = n_groups;自拷贝份额 = 选中自己组的期望 M/n_groups
           (那份 _send_index 恒等于发送方,纯本地,内部实测记录)。
    Hop B:组内域,world = R;自拷贝 1/R。
    额外:一次 splits 主机同步(两跳比一跳多一次变长交换)+ 本地到达链(按行)。
    """
    a = _a2a_ms(c.slow, g.n_groups, g.rows_hop_a(), g.row_bytes(),
                self_fraction=g.M / g.n_groups if g.n_groups > 1 else 1.0)
    b = _a2a_ms(c.fast, g.R, g.rows_hop_b(), g.row_bytes(),
                self_fraction=1.0 / g.R)
    chain = c.chain_us_per_row * g.rows_hop_b() / 1000.0
    return a + b + c.splits_sync_ms + chain


def step_delta(c: ClusterSpec, g: MoEGeometry) -> dict:
    """每步的通信差(on − off)[ms] 与 G 预测所需的分量。

    dispatch 与 combine 镜像(同字节、同结构,内部实测记录);反向重放前向的
    splits(内部实测记录),次数减半。G 的合成在 validate 侧做(需要 off 臂实测步时)。
    """
    per_call = two_hop_call(c, g) - one_hop_call(c, g)
    calls = 2 * (g.calls_per_step_fwd() + g.calls_per_step_bwd())   # x2: dispatch+combine
    return {
        "per_call_on_ms": two_hop_call(c, g),
        "per_call_off_ms": one_hop_call(c, g),
        "per_call_delta_ms": per_call,
        "calls_per_step": calls,
        "step_delta_ms": per_call * calls,
    }
