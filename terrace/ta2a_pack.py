"""T-A2A 集合通信并包(A1 / A2,2026-08-21):同一跳里共用 splits 的多条集合通信
合成一条。**纯字节重排** —— 不动任何归约序、配对序、排序键、dtype 与圆整点。

为什么这是当前性价比最高的一刀(内部设计记录(未随仓发布)/§2/§5):
w128 a2a 曲线拆成 `α + β` 后,α₁₂₈ = 0.45 ms、α₈ = 0.058 ms,β₈ ≈ β₁₂₈ ≈ 110 GB/s
—— **字节几乎不要钱、条数很贵**。一次 dispatch 原走 8 条(inter 4 + intra 4),
固定开销 4α₁₂₈ + 4α₈ ≈ 2.03 ms,而厂商 alltoall_seq 侧只有约 3 条 ≈ 1.35 ms。
独立佐证:combine 侧只有 2 条,实测相位与厂商打平;dispatch 8 条,实测差 +6 ms。

## 并包后每次 dispatch 的条数(本模块闸开时)

| 跳 | 并包前 | 并包后 | 省 α | 多付(HBM+线上) | 净 |
|---|---|---|---|---|---|
| Hop A(inter,α₁₂₈) | counts + payload + id + gate = 4 | counts + **[id‖payload‖gate]** = 2 | 0.900 ms | 0.106 ms | **+0.794 ms** |
| Hop B(intra,α₈)   | counts + exp_rx + slot + gate = 4 | counts + exp_rx + **[slot‖gate]** = 3 | 0.058 ms | 0.002 ms | **+0.056 ms** |
| combine            | 2 | 2 | — | — | — |

合计 dispatch **8 -> 5** 条,判决床几何(H=2048 / M=2 / quota=3 / n_rows=8192 /
pairs=24576 / bf16 / HBM 有效 1275 GB/s,内部硬件画像实测)**净 −0.85 ms/次 dispatch**;
拆半接缝前向 10 -> 7,**反向 6 -> 6 不变**(见「可微性」一节)。

**为什么 Hop B 只并两个标量面、不并载荷面**:并包省的是 α,但每并进一个**大**平面
就多付两趟它自己的 HBM 拷贝。Hop A 的 α 贵 7.8× 而载荷小 3.2 倍 ⇒ 划算;Hop B 反过来
⇒ 三条并一条要 +0.317 ms 去省 0.116 ms,**净亏 0.202 ms**。算式见下方 Hop B 一节。
要拿回那 1α₈ 只有零拷贝一条路(让 K1 直接写进合并缓冲),已排在 A4
(`npu_alltoallv_gmm` 吃掉 Hop B)之后 —— A4 一旦落地这条自动消失。

**活路径证据的观测点换了(A1′,2026-08-21)**:`tests/test_ta2a_quota_wire_bitparity.py`
原先靠「Hop A 线上有没有 `[n_rows, quota]` int64 槽号表 / 1 维位掩码 / gate 宽浮点表」
判 C1 有没有静默回退。并包后三个平面共用一条集合通信,该判据改成**在打包器
`hopa_pack` 的入参与 `HopALayout.id_w` 上判** —— 平面还是那三个、宽度还是那几个数,
只是读早了一步(并包只搬字节、不改平面)。判别力等价的完整论证写在那个文件的文件头。

## 行内布局与对齐

- **Hop A(int64 容器)**:行 = `[id(id_w word) | payload(H) | gate(gw) | pad]`,
  行宽 `W = id_w + ceil((H + gw) / (8 // itemsize))` word。id 面放**行首**,浮点区
  起始字节恒为 8 的倍数(反过来会随 H/dtype 漂,int64 视图取不出来)。
  判决床 quota 臂:`id_w=3`、`W=516` word `=4128 B/行`,对三条原字节 4126 B 冗余
  **2 B(+0.05%)**。**splits 原样是行数,不缩放** —— dim 0 就是 n_rows。
- **Hop B(int64 容器)**:行 = `[slot(1 word) | gate(1 float) | pad]` = 2 word。
  同样 int64 面在行首。16 B/行 对两条原字节 10 B(bf16)冗余 6 B —— 相对同一跳
  4096 B 的载荷行是 +0.15%。splits 同样原样是行数。

## 可微性:一条前向,两条**独立**的反向边

两条 float 路仍各挂一条独立的 `autograd.Function` 边(`_PackedEdge`),反向就是
并包前 `ep_dist._A2A.backward` 的同一句 `_a2a_raw(g, out_splits, in_splits, group)`
—— 逐字节同一调用,梯度逐位不变,**反向条数不变**。

不把两路焊进一枚融合节点,理由与 K1 同一条(内部工程记录 2026-08-20「融合 kernel 进
手写 backward 的段图要拆边」/ `terrace/ops/__init__.py::_K1SendEdge`):厂商 gmm 的
手写 backward 对 `permute2_graph` 与 `permute2_prob_graph` 分**两次** `.backward()`
进同一段,融合节点第二次会撞 *backward through the graph a second time*,且第一次
会把 materialize 出的零梯度先写进另一路的 `.grad`。所以**数据**在图外并包一次收下,
**图**由两条互不相交的边重建,与并包前的两条 `_A2A` 子图同构。

## 闸门

`TERRACE_TA2A_PACK`:未设 / 非 "0" = 开(默认);"0" = 关,两条接缝走并包前的
现链**原文**(零行为变化),供床上 A/B 与一键回滚。进程启动读一次(热路径不查环境),
测试翻开关用 `reset()` 或直接 monkeypatch `pack_enabled`。
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .ep_dist import _a2a_raw

_ENV_SWITCH = "TERRACE_TA2A_PACK"
_MODE: str | None = None

# 三种形态。**默认已于 2026-08-22 从 full 改成 small**,依据是判决床实测:
#
#   形态    Hop A 集合通信   Hop A 并进容器的平面        判决床实测 dispatch/次
#   off     4 条             —                          12.171 ms(基线)
#   full    2 条             id + 载荷 + gate           14.222 ms(**+2.051,亏**)
#   small   3 条             id + gate(载荷单走)      预期 −0.43(待实测)
#
# 为什么 full 会亏:并包省的是 α,但每并进一个**大**平面就多付两趟它自己的 HBM 拷贝。
# full 省 2α₁₂₈ + 1α₈ = 0.96 ms,却要把 [n, 2048] bf16 的载荷抄进容器、再从容器抄出来
# —— 判决床上反推出来的真实拷贝成本 ≈ 3.0 ms(原估 0.106 ms,差 28 倍)。
#
# 这条道理本模块的 Hop B 一节**早就写对了**(「三条并一条要 +0.317 ms 去省 0.116 ms,
# 净亏 0.202 ms」),只是当时把 Hop A 的载荷拷贝估小了 28 倍,于是对同一个问题得出了
# 相反的结论。small 就是把 Hop B 那条正确的取舍原样搬到 Hop A:
# **只并每行几十字节的标量面,大载荷永远自己走一条。**
_MODE_OFF, _MODE_SMALL, _MODE_FULL = "off", "small", "full"
_ALIASES = {"0": _MODE_OFF, "off": _MODE_OFF, "no": _MODE_OFF,
            "1": _MODE_SMALL, "small": _MODE_SMALL, "meta": _MODE_SMALL,
            "full": _MODE_FULL, "2": _MODE_FULL, "all": _MODE_FULL}


def _env_mode() -> str:
    """环境里写的形态。进程生命周期内只读一次(dispatch 是热路径)。"""
    global _MODE
    if _MODE is None:
        raw = os.environ.get(_ENV_SWITCH)
        if raw is None:
            _MODE = _MODE_SMALL          # 未设 = 默认形态,这是有意的
        else:
            key = raw.strip().lower()
            if key not in _ALIASES:
                # **未知值一律炸,不许静默落回默认。**
                # 2026-08-23 审计实测:原来 `smal` / `fulll` / `false` / `true` / ""
                # 全都静默变成 small。最险的是 `false` —— 写它的人想关并包,
                # 实际把并包**打开了**;而档位护栏只查"赋值语句出现过"、不查值,
                # 于是档表里一个字符的笔误可以让某一档跑成另一档,全套测试还是绿的。
                raise RuntimeError(
                    "%s=%r 不是合法形态。合法值:%s。"
                    "**不静默落回默认** —— 档表里一个笔误会让整档跑成另一档,"
                    "而读数看起来完全正常。" % (_ENV_SWITCH, raw, sorted(_ALIASES)))
            _MODE = _ALIASES[key]
    return _MODE


def pack_enabled() -> bool:
    """并包开关。**测试翻 A/B 对照臂的历史钩子就是 monkeypatch 这个函数。**"""
    return _env_mode() != _MODE_OFF


def pack_mode() -> str:
    """"off" / "small" / "full"。热路径读它。

    `TERRACE_TA2A_PACK=1` 沿用为 **small**(而不是历史上的 full)—— 床上脚本与
    既有 runner 都写的 1,让它们自动吃到修正后的形态;要跑回亏的那一档得显式写 full。

    **必须先问 `pack_enabled()`**:一大批既有测试是 monkeypatch 那个函数来造未并包
    对照臂的。如果这里绕过它直接读环境,那些 monkeypatch 会**静默失效** —— 对照臂
    也在并包,而 A/B 逐位对账两边都是并包臂,等价断言全变空转,测试照样绿。
    (2026-08-22 接 small 形态时真踩了一次,靠 hopb 计数 {'hopb': 2} 才发现。)
    """
    if not pack_enabled():
        return _MODE_OFF
    return _env_mode()


def reset() -> None:
    """忘掉缓存判定,下次 pack_mode() 重读环境。测试/调试钩子,训练代码不得调用。"""
    global _MODE
    _MODE = None


def _ceil_mul(n: int, unit: int) -> int:
    return -(-n // unit) * unit


def _own(t: torch.Tensor) -> torch.Tensor:
    """连续化并**保证独立存储**。解包出来的每个平面都必须走这里。

    为什么不能用 `.contiguous()`(2026-08-21 判决床 rank61 的死因):
    `.contiguous()` 在张量"已经连续"时把**原视图**原样交回。而一个 `[R, W]` 缓冲的
    列切片 `buf[:, a:b]`,当 **R <= 1** 时 dim0 的 stride 在连续性判定里被忽略,
    于是切片被判为已连续 —— `.contiguous()` 成了空操作,解包结果与并包缓冲共享存储。
    调用方紧接着 `_rbuf.untyped_storage().resize_(0)` 立刻还内存,三个平面当场悬空:

        RuntimeError: The tensor has a non-zero number of elements,
                      but its data is not allocated yet.
        (报在 ta2a_dispatch.py `owner = slot_idx // epr`,slot_idx 从 rmask 展开)

    R==1 在对齐床(4 节点)几乎撞不上,在判决床(16 节点、行被切得更碎、
    load_cv 随训练从 1.0 爬到 1.4)会撞上 —— 08-21 pack-on 两发都死在 iter 30 附近。

    成本:R>1 时 `.contiguous()` 本来就要拷一次,`clone` 也是一次,**等价**;
    只有在那个危险的退化形状上才多一次拷贝(1 行,可忽略)。
    """
    return t.clone(memory_format=torch.contiguous_format)


# --------------------------------------------------------------------------------------
# 可微性:并包后重建图的独立边
# --------------------------------------------------------------------------------------

class _PackedEdge(torch.autograd.Function):
    """把并包后收下来的**某一路**挂回它自己的发送端张量。

    前向零工作(数据已在图外由一次并包 a2a 收到,`recvd` 是它的解包结果);
    反向 = 该路自己的一次反向 a2a,与并包前 `ep_dist._A2A.backward` 逐字节同一句:
    `_a2a_raw(g, out_splits, in_splits, group)`。所以**梯度逐位不变、反向集合通信
    条数不变**;省下的是前向的条数。

    每一路一条边、互不相交 —— 厂商 gmm 手写 backward 对 permute2 段分两次
    `.backward()`,融合节点会撞车(模块头「可微性」一节 / _K1SendEdge)。
    """

    @staticmethod
    def forward(ctx, src, recvd, in_splits, out_splits, group):
        ctx.a2a_splits = (in_splits, out_splits, group)
        return recvd            # 数据已产出;autograd 自动别名并挂 grad_fn

    @staticmethod
    def backward(ctx, g):
        in_splits, out_splits, group = ctx.a2a_splits
        if g is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        # 反向对 = (out, in),与 _A2A.backward 同序。交换的 in/out 写死在这里而不是
        # 由 caller 传,免得未来某处把方向传反 —— 那会静默错路由而前向仍然像样。
        return _a2a_raw(g, out_splits, in_splits, group), None, None, None, None


def attach_edge(src, recvd, in_splits, out_splits, group):
    """把 `recvd` 挂回 `src`。grad 关时零开销直接交出(与 `_a2a` 的分支同构)。"""
    if torch.is_grad_enabled():
        return _PackedEdge.apply(src, recvd, in_splits, out_splits, group)
    return recvd


# --------------------------------------------------------------------------------------
# Hop A:id ‖ payload ‖ gate(int64 容器,id 面放行首保证 8 字节对齐)
# --------------------------------------------------------------------------------------
#
# A1′(2026-08-21,协调者拍板):id 平面也并进来,Hop A 由 4 条降到 **2 条**
# (counts + 一条并包)。此前 id 面单走一条,是因为
# `tests/test_ta2a_quota_wire_bitparity.py::_wire_flags` 用「线上有没有这个张量形状」
# 判 C1 有没有静默回退;该判据已改成**在打包器入参上**判(见那个文件的文件头),
# 判别力等价而不再依赖"每个平面各占一条集合通信"。
#
# 布局理由(与 Hop B 同一条):int64 面放**行首**,行宽以 int64 word 计,浮点区从
# word `id_w` 起 —— 起始字节恒为 8 的倍数。反过来(float 在前)行内偏移会随 H 与
# dtype 漂,int64 视图取不出来,还会让每行的 int64 面落在 4 字节边界上。
#
# splits **原样是行数**,不缩放(dim 0 就是 n_rows)—— 这比 A1′ 之前那版还干净:
# 那版为了让线上张量宽度等于 gate 宽度而按 `F/gw` 缩放,现在整个缩放没有了。
# `st.send_l / st.recv_l`(交给厂商 `disp.input_splits/output_splits` 的那两份)
# 因此与并包前逐项相同,厂商 backward 的两次手工重放照跑不误、仍各是一次 a2a。


class HopALayout:
    """Hop A 并包缓冲的行内布局。`pack` 产出,`unpack` 照它读回。

    `id_w` 就是活路径证据现在盯的那个数:C1 quota 线格式 = quota,
    改前的位掩码格式 = 1。
    """

    __slots__ = ("hidden", "gate_w", "id_w", "id_1d", "words", "dtype", "per_word")

    def __init__(self, hidden, gate_w, id_w, id_1d, dtype):
        self.hidden, self.gate_w = hidden, gate_w
        self.id_w, self.id_1d = id_w, id_1d
        self.dtype = dtype
        self.per_word = 8 // dtype.itemsize
        self.words = id_w + -(-(hidden + gate_w) // self.per_word)

    def __repr__(self):                                   # 报错信息里要看得见
        return (f"HopALayout(H={self.hidden}, gw={self.gate_w}, id_w={self.id_w}, "
                f"id_1d={self.id_1d}, words={self.words}, dtype={self.dtype})")


def hopa_layout(payload: torch.Tensor, gate_rows: torch.Tensor,
                ids: torch.Tensor) -> HopALayout:
    """三个平面 -> 行内布局。`ids` 是 `[n]` 位掩码(通用)或 `[n, quota]` 槽号表(C1)。"""
    if payload.dtype != gate_rows.dtype:
        raise RuntimeError(
            f"Hop A 并包要求 payload/gate 同 dtype,收到 {payload.dtype} vs "
            f"{gate_rows.dtype} —— gate 平面按契约派生自 payload(见 _pack_quota_wire)")
    if ids.dtype != torch.int64:
        raise RuntimeError(f"Hop A 并包的 id 面必须是 int64,收到 {ids.dtype}")
    if ids.dim() not in (1, 2):
        raise RuntimeError(f"Hop A 的 id 面只能是 [n] 或 [n, quota],收到 {ids.shape}")
    id_1d = ids.dim() == 1
    id_w = 1 if id_1d else int(ids.shape[1])
    n = payload.shape[0]
    if ids.shape[0] != n or gate_rows.shape[0] != n:
        raise RuntimeError(
            f"Hop A 三个平面的行数必须一致(共用 splits),收到 payload={n} "
            f"gate={gate_rows.shape[0]} id={ids.shape[0]}")
    return HopALayout(int(payload.shape[1]), int(gate_rows.shape[1]), id_w, id_1d,
                      payload.dtype)


def hopa_pack(payload: torch.Tensor, gate_rows: torch.Tensor, ids: torch.Tensor):
    """(`[n, H]` float, `[n, gw]` float, `[n]`/`[n, quota]` int64) -> (`[n, W]` int64, 布局)。

    只做拷贝,不做任何转换:三段的每一个比特原样落进缓冲。
    """
    lay = hopa_layout(payload, gate_rows, ids)
    n, pw, iw = payload.shape[0], lay.per_word, lay.id_w
    buf = torch.empty(n, lay.words, dtype=torch.int64, device=payload.device)
    buf[:, :iw] = ids.unsqueeze(1) if lay.id_1d else ids
    fv = buf.view(lay.dtype)                      # 连续缓冲的按位重解释,元数据操作
    base = iw * pw
    fv[:, base:base + lay.hidden] = payload
    fv[:, base + lay.hidden:base + lay.hidden + lay.gate_w] = gate_rows
    tail = base + lay.hidden + lay.gate_w
    if tail < lay.words * pw:
        fv[:, tail:] = 0                          # 收侧永不读;清零只为不把未初始化位上线
    return buf, lay


def hopa_unpack(buf: torch.Tensor, lay: HopALayout):
    """`[n, W]` int64 -> (`[n, H]` payload, `[n, gw]` gate, id 面原形状),三块都连续。

    刻意不返回 strided 视图:2026-08-01 的 `torch.cat` 门控并包正是**下游 gather 吃
    非连续切片**而慢了几十 ms(ta2a_fwd.py 的「DO NOT FUSE」注释),不是并包本身慢。
    这里多付一次到达载荷的拷贝(判决床 ≈34 MB,约 0.05 ms),换下游算子(含 K1
    kernel)与并包前见到**完全相同的连续张量**。
    """
    pw, iw = lay.per_word, lay.id_w
    fv = buf.view(lay.dtype)
    base = iw * pw
    # `_own` 而不是 `.contiguous()`:R<=1 时列切片被判为已连续,contiguous 是空操作,
    # 交回去的是 buf 的视图;调用方随后把 buf 的存储 resize(0),三个平面全部悬空。
    ids = _own(buf[:, 0]) if lay.id_1d else _own(buf[:, :iw])
    return (_own(fv[:, base:base + lay.hidden]),
            _own(fv[:, base + lay.hidden:base + lay.hidden + lay.gate_w]),
            ids)


class HopASmallLayout:
    """A1'' 小平面容器的行内布局:`[id(id_w word) | gate(gw) | pad]`,载荷不在其中。

    与 `HopALayout` 同一套对齐规矩(int64 面在行首,浮点区起始字节恒为 8 的倍数),
    只是少了 hidden 那一段 —— 行宽从 516 word(4128 B)掉到 4 word(32 B)。
    """

    __slots__ = ("gate_w", "id_w", "id_1d", "words", "dtype", "per_word")

    def __init__(self, gate_w, id_w, id_1d, dtype):
        self.gate_w, self.id_w, self.id_1d = gate_w, id_w, id_1d
        self.dtype = dtype
        self.per_word = 8 // dtype.itemsize
        self.words = id_w + -(-gate_w // self.per_word)

    def __repr__(self):
        return (f"HopASmallLayout(gw={self.gate_w}, id_w={self.id_w}, "
                f"id_1d={self.id_1d}, words={self.words}, dtype={self.dtype})")


def hopa_small_layout(gate_rows: torch.Tensor, ids: torch.Tensor) -> HopASmallLayout:
    if ids.dtype != torch.int64:
        raise RuntimeError(f"Hop A 小并包的 id 面必须是 int64,收到 {ids.dtype}")
    if not gate_rows.is_floating_point():
        raise RuntimeError(f"Hop A 小并包的 gate 面必须是浮点,收到 {gate_rows.dtype}")
    if ids.dim() not in (1, 2):
        raise RuntimeError(f"Hop A 的 id 面只能是 [n] 或 [n, quota],收到 {ids.shape}")
    if ids.shape[0] != gate_rows.shape[0]:
        raise RuntimeError(
            f"Hop A 小并包两面行数必须一致(共用 splits),收到 gate={gate_rows.shape[0]} "
            f"id={ids.shape[0]}")
    id_1d = ids.dim() == 1
    return HopASmallLayout(int(gate_rows.shape[1]), 1 if id_1d else int(ids.shape[1]),
                           id_1d, gate_rows.dtype)


def hopa_pack_small(gate_rows: torch.Tensor, ids: torch.Tensor):
    """(`[n, gw]` float, `[n]`/`[n, quota]` int64) -> (`[n, Ws]` int64, 布局)。

    判决床几何:gw=quota=3、id_w=3、per_word=4 ⇒ Ws=4 word=**32 B/行**
    (对比 full 形态的 4128 B/行,拷贝量 129 倍之差 —— 这正是 A1'' 的全部理由)。
    """
    lay = hopa_small_layout(gate_rows, ids)
    n, pw, iw = gate_rows.shape[0], lay.per_word, lay.id_w
    buf = torch.empty(n, lay.words, dtype=torch.int64, device=gate_rows.device)
    buf[:, :iw] = ids.unsqueeze(1) if lay.id_1d else ids
    fv = buf.view(lay.dtype)
    base = iw * pw
    fv[:, base:base + lay.gate_w] = gate_rows
    if base + lay.gate_w < lay.words * pw:
        fv[:, base + lay.gate_w:] = 0        # 收侧永不读;清零只为不把未初始化位上线
    return buf, lay


def hopa_unpack_small(buf: torch.Tensor, lay: HopASmallLayout):
    """`[n, Ws]` int64 -> (`[n, gw]` gate, id 面原形状)。两块都是独立存储(见 `_own`)。"""
    pw, iw = lay.per_word, lay.id_w
    fv = buf.view(lay.dtype)
    base = iw * pw
    ids = _own(buf[:, 0]) if lay.id_1d else _own(buf[:, :iw])
    return _own(fv[:, base:base + lay.gate_w]), ids


def hopa_exchange_raw(buf, in_splits, out_splits, group=None):
    """一次裸 a2a。splits 原样是行数 —— dim 0 就是 n_rows,不需要任何缩放。"""
    return _a2a_raw(buf, in_splits, out_splits, group)


def hopa_exchange_async(buf, in_splits, out_splits, group=None):
    """异步版(overlap 接缝的 Hop A 在段外发,18c 用飞行时间算共享专家)。

    返回 (recv_buf, handle);caller wait 之后直接 `hopa_unpack`。
    """
    out = buf.new_empty((sum(out_splits), buf.shape[1]))
    handle = dist.all_to_all_single(out, buf, out_splits, in_splits, group=group,
                                    async_op=True)
    return out, handle


# --------------------------------------------------------------------------------------
# Hop B:slot ‖ gate(两个**每配对一个标量**的元数据面,int64 容器)
# --------------------------------------------------------------------------------------
#
# 为什么 Hop B **不**把 exp_rx 也并进来(拍板前算过账,判决床几何 H=2048 / quota=3 /
# P=24576 / bf16 / HBM 有效 1275 GB/s(内部硬件画像实测)/ α₈=0.058 ms):
#
#   方案①(三条并一条,slot‖exp_rx‖gate):行宽 514 word = 4112 B,线上冗余只有
#     6 B/行(+0.15%)—— **字节不是问题**;问题是打包+解包多付 4 趟 100 MB 级的
#     HBM 拷贝 = 403.9 MB ⇒ **+0.317 ms**,而省下的是 2α₈ = 0.116 ms ⇒ **净 −0.202 ms**。
#   方案②(只并两个标量面):多付 1.28 MB HBM + 129 KB 线上 ⇒ +0.002 ms,
#     省 1α₈ = 0.058 ms ⇒ **净 +0.056 ms**。
#
# 取②。这正是 2026-08-01「torch.cat 门控并包」翻车的同一条物理:**并包省的是 α,
# 但每并进一个大平面就多付两趟它自己的 HBM 拷贝**。Hop A 的账反过来(α₁₂₈ 贵 7.8×、
# 载荷小 3 倍:+0.105 ms 换 0.450 ms,净 +0.344 ms),所以那边并、这边不并。
#
# 要把 Hop B 剩下的 1α₈ 也拿回来,唯一正确的做法是**零拷贝**:让产出 exp_rx 的那一步
# (K1 kernel,或 else 分支的 gather)直接写进合并缓冲的载荷视图,而不是先落一个
# [pairs, H] 再拷。那要动 K1 的输出契约与 permute2 段图的边(_K1SendEdge),
# 属独立一件,值 0.058 ms —— 不在本次刀内。


def hopb_meta_words() -> int:
    """Hop B 元数据并包缓冲的行宽:1 word 槽号 + 1 word 装下 1 个 gate 浮点。"""
    return 2


def hopb_pack_meta(slot_idx: torch.Tensor, gate_pairs: torch.Tensor) -> torch.Tensor:
    """(`[P]` int64 槽号, `[P]` float gate) -> `[P, 2]` int64 连续缓冲。

    int64 面放**行首**:行宽以 int64 word 计,浮点区从 word 1 起,起始字节恒为 8 的
    倍数 —— 反过来(float 在前)行内偏移会随 dtype 漂,int64 视图取不出来。
    """
    if slot_idx.dtype != torch.int64:
        raise RuntimeError(f"Hop B 并包的 id 面必须是 int64,收到 {slot_idx.dtype}")
    if slot_idx.shape != gate_pairs.shape:
        raise RuntimeError(
            f"Hop B 并包要求两面同形状(每配对一个标量),收到 {tuple(slot_idx.shape)} "
            f"vs {tuple(gate_pairs.shape)}")
    if not gate_pairs.is_floating_point():
        raise RuntimeError(f"Hop B 并包的 gate 面必须是浮点,收到 {gate_pairs.dtype}")
    P = slot_idx.numel()
    per_word = 8 // gate_pairs.element_size()
    buf = torch.empty(P, hopb_meta_words(), dtype=torch.int64, device=slot_idx.device)
    buf[:, 0] = slot_idx
    fv = buf.view(gate_pairs.dtype)               # 连续缓冲的按位重解释,元数据操作
    fv[:, per_word] = gate_pairs
    fv[:, per_word + 1:] = 0                      # 收侧永不读
    return buf


def hopb_unpack_meta(buf: torch.Tensor, dtype):
    """`[P, 2]` int64 -> (`[P]` int64 槽号, `[P]` float gate)。"""
    per_word = 8 // dtype.itemsize
    # 同 hopa_unpack:P<=1 时 `.contiguous()` 是空操作,两个面会共享 buf 的存储。
    # Hop B 今天没有紧跟其后的 resize(0),但保持同一条契约,免得将来加了才发现。
    return _own(buf[:, 0]), _own(buf.view(dtype)[:, per_word])


def assert_not_aliased(buf: torch.Tensor, *planes: torch.Tensor) -> None:
    """在把 `buf` 的存储 resize(0) 之前调一次:任何解包平面都不许还指着 buf。

    三次指针比较,热路径上可忽略。留着它是因为这个坑一旦复发就是**静默的**
    (悬空要到下游第一次真读才炸,而那时栈已经离现场很远 —— 08-21 就炸在
    `owner = slot_idx // epr`,离并包点隔了十几个算子)。
    """
    sp = buf.untyped_storage().data_ptr()
    for i, p in enumerate(planes):
        if p is not None and p.numel() and p.untyped_storage().data_ptr() == sp:
            raise RuntimeError(
                f"并包解包第 {i} 个平面仍与缓冲共享存储(shape={tuple(p.shape)}, "
                f"buf={tuple(buf.shape)})—— 还内存会让它悬空。见 ta2a_pack._own")
