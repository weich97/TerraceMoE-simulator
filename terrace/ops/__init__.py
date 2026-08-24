"""terrace.ops: torch 侧封装 —— T-A2A 定制 AscendC 算子的加载器与降级开关。

工程脚手架(2026-08-20 连夜搭建,见 本目录各文件的头注 与
内部设计记录(未随仓发布))。注册的算子:
  - terrace_passthrough:输入原样拷出,全链路(msopgen 工程 → opp 包 → aclnn →
    torch.library → autograd.Function → 降级开关)冒烟基准;
  - terrace_k1_arrival(2026-08-20 落地):到达侧融合链(展开 + owner 稳定桶排 +
    i_send 直方图 + 发送缓冲 gather + gate gather),C1 quota 线格式,与现组合链
    逐位一致 —— 见下方 k1_arrival_ref 的可执行规格与
    ascendc/op_kernel/terrace_k1_arrival.cpp 的两遍法论证。
K2(发送侧打包链)接口草案见文件底部注释与 README §5。

开关语义(TERRACE_CUSTOM_OPS,进程启动时读一次):
  **未设 / "0"** -> 关。不尝试加载 .so,一切走现组合链(位级正确的已验路径)。
  "1"            -> 开。尝试加载;失败则**打一行 WARNING 后**视同 "0"。
                    fail-loud 不静默:降级必须在日志里可见,但不炸训练。
  "require"/"2"  -> 硬性。加载失败直接 RuntimeError。给「今晚必须跑在 kernel 上」
                    的 bench/验收场景用,防止降级把 kernel 读数偷换成组合链读数。

**默认从 "1" 改成 "0" 是 2026-08-24 的事故修**:`.so` 第一次编译成功那一刻,
未过逐位校验的 K1 kernel 自动进了训练路径,此后每一发 T-A2A on 臂在第 0 步
全 128 rank 同炸(K1 索引错 -> i_send 错 -> Hop B splits 对不上),白烧两发判决床。
「编译成功」不等于「算对了」,而当时唯一的闸就是「.so 能不能 dlopen」。
现在启用一个 kernel 必须有人显式签字。

为什么读一次而不是每次调用读环境:dispatch 每层每 microbatch 都要过这里,热路径上
一次 os.environ 查询 + 字符串比较是纯浪费;训练进程也从不中途翻开关。测试要翻开关
用 reset()(见下)。
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

import torch

_LOG = logging.getLogger("terrace.ops")

_ENV_SWITCH = "TERRACE_CUSTOM_OPS"
_ENV_LIB = "TERRACE_OPS_LIB"          # 显式指定 .so 路径,绕过默认搜索
_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")

# 加载成功后必须存在的算子名单。少一个都算加载失败(fail loud):宁可整体降级,
# 也不要「.so 加载了但 schema 没注册」这种半死状态在首次调用时才炸。
# k1_arrival 落地(2026-08-20)后,旧的仅含 passthrough 的 .so 会整体降级 ——
# 有意如此:半新半旧的算子集在训练中途炸,比慢更糟。集群按 README §3 重编。
_REQUIRED_OPS = ("passthrough", "k1_arrival")


class OpsLoadError(RuntimeError):
    """定制算子库加载失败(找不到 .so / dlopen 失败 / schema 缺失)。"""


@dataclass(frozen=True)
class OpsState:
    """加载器的一次性判定结果。requested 是环境开关原文,loaded 是最终事实。"""
    requested: str        # "0" | "1" | "require"(归一化后)
    loaded: bool          # torch.ops.terrace.* 可用
    lib: str | None       # 实际加载的 .so 路径(未加载则 None)
    reason: str           # loaded=False 时的人话原因;loaded=True 时为 "ok"


_STATE: OpsState | None = None


def _normalized_switch() -> str:
    """开关归一化。**默认是 "0"(关),不是 "1"。**

    2026-08-24 的事故:`.so` 于 07:25 第一次编译成功;因为默认是 "1",
    `custom_ops_enabled()` 当场变真,**未过逐位校验的 K1 kernel 直接进了训练路径**。
    此后每一发 T-A2A on 臂在第 0 步全 128 rank 同时炸
    `RuntimeError: Split sizes dosen't match total dim 0 size`
    (K1 的 slot_idx 索引算错 -> i_send 错 -> Hop B 的 splits 对不上),
    r4 与 isub 两发判决床白烧,而 runner 照样报"收工"。

    bitcheck 的判决行早就写着「K1 不得上床」—— **但没有任何机制执行它**,
    唯一的闸是「.so 能不能 dlopen」。「编译成功」不等于「算对了」。

    改成 fail-safe:**没人明确要,算子就不上路径**。
        未设 / "0"      -> 关(参考实现,零行为变化)
        "1"             -> 开(显式索取)
        "require" / "2" -> 开,且加载不上就炸

    代价是每个想用算子的地方都得显式写一次 —— 这正是要的:
    让"启用一个 kernel"变成一个**有人签字**的动作。
    """
    raw = os.environ.get(_ENV_SWITCH, "0").strip()
    if raw in ("2", "require"):
        return "require"
    if raw == "1":
        return "1"
    return "0"


def _find_library() -> str:
    """定位编译产物。显式 env 优先;默认搜 terrace/ops/lib/*.so(集群编译落点)。"""
    explicit = os.environ.get(_ENV_LIB)
    if explicit:
        if not os.path.isfile(explicit):
            raise OpsLoadError(f"{_ENV_LIB}={explicit} 不存在")
        return explicit
    hits = sorted(glob.glob(os.path.join(_LIB_DIR, "*.so")))
    if not hits:
        raise OpsLoadError(
            f"{_LIB_DIR} 下没有 .so(本地无 CANN 属预期;集群按 README §3 编译)")
    return hits[0]


def _try_load() -> str:
    """加载 .so 并验证 schema 齐全。返回库路径;任何一步失败抛 OpsLoadError。"""
    path = _find_library()
    try:
        torch.ops.load_library(path)
    except Exception as e:                          # dlopen/注册失败原因五花八门
        raise OpsLoadError(f"load_library({path}) 失败: {e}") from e
    ns = getattr(torch.ops, "terrace", None)
    missing = [op for op in _REQUIRED_OPS
               if ns is None or not hasattr(ns, op)]
    if missing:
        raise OpsLoadError(
            f"{path} 已加载但缺 torch.ops.terrace.{{{','.join(missing)}}} —— "
            f"TORCH_LIBRARY 注册没生效,检查 csrc/terrace_ops.cpp")
    return path


def _initialize() -> OpsState:
    switch = _normalized_switch()
    if switch == "0":
        return OpsState(requested="0", loaded=False, lib=None,
                        reason=f"{_ENV_SWITCH}=0(显式关闭,不尝试加载)")
    try:
        path = _try_load()
        _LOG.info("terrace 定制算子已加载: %s", path)
        return OpsState(requested=switch, loaded=True, lib=path, reason="ok")
    except OpsLoadError as e:
        if switch == "require":
            raise RuntimeError(
                f"{_ENV_SWITCH}={os.environ.get(_ENV_SWITCH)} 要求定制算子,"
                f"但加载失败: {e}") from e
        # fail-loud 降级:恰好一行 WARNING(logging 未配置时 lastResort 也会打到
        # stderr),然后视同 TERRACE_CUSTOM_OPS=0。
        _LOG.warning("TERRACE_CUSTOM_OPS 降级为 0(%s)—— 走现组合链", e)
        return OpsState(requested=switch, loaded=False, lib=None, reason=str(e))


def status() -> OpsState:
    """惰性初始化并缓存。训练进程整个生命周期只判定一次。"""
    global _STATE
    if _STATE is None:
        _STATE = _initialize()
    return _STATE


def custom_ops_enabled() -> bool:
    """dispatch 侧将来的闸门:True 才许调 torch.ops.terrace.*。"""
    return status().loaded


def reset() -> None:
    """忘掉缓存判定,下次 status() 重读环境。测试/调试钩子,训练代码不得调用。

    注意 torch.ops.load_library 是进程级不可逆的:reset() 只重置**本模块**的判定,
    已注册的 schema 不会消失。测试用它翻的是开关语义,不是卸载 .so。
    """
    global _STATE
    _STATE = None


# --------------------------------------------------------------------------------------
# autograd.Function 样板 + 函数式入口
# --------------------------------------------------------------------------------------

class TerracePassthroughFn(torch.autograd.Function):
    """样板:forward 调定制 kernel,backward 用组合链。

    passthrough 是恒等拷贝,它的"组合链 backward"恰好也是恒等 —— 但样板照全套写,
    因为 K1/K2 长这个形状(内部设计记录(未随仓发布):反向继续用现组合链):

      K1 已按此形状落地(见下方 TerraceK1ArrivalFn):forward 调
      torch.ops.terrace.k1_arrival,backward 是现组合链的 index_add_/gather。
      K2 正式实现时改这里:
        forward:  payload, mask, gate_rows, ... = torch.ops.terrace.k2_pack(...)
        backward: payload 的反向 = index_add_(0, u_src, grad)(去重 gather 的
                  scatter-add 伴随),gate_rows 的反向 = grad[inverse, slot_flat]
                  的 gather —— 全部现成组合链原语,位级语义与今天逐位相同。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # K1 正式实现时改这里:换成对应 torch.ops.terrace.* 调用与 ctx.save
        return torch.ops.terrace.passthrough(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # 恒等拷贝的伴随是恒等。K1/K2 的 backward 见类 docstring —— 继续组合链,
        # 不写反向 kernel(路线图拍板:两枚 kernel 均纯置换/拷贝,反向语义现成)。
        return grad_out


def passthrough(x: torch.Tensor) -> torch.Tensor:
    """恒等拷贝,链路验证专用。kernel 可用走定制算子,否则走组合链等价式。

    契约(tests/test_terrace_ops_scaffold.py 把守):两条路径输出与输入逐位相等、
    是新张量(不与输入共存储)、梯度恒等回传。
    """
    if custom_ops_enabled():
        return TerracePassthroughFn.apply(x)
    # 组合链等价式:clone 即「原样拷出」的现有原语写法(可微,恒等伴随)。
    return x.clone()


# --------------------------------------------------------------------------------------
# K1:到达侧融合链(C1 quota 线格式,2026-08-20 落地)
# --------------------------------------------------------------------------------------

def _stable_ordo(owner: torch.Tensor, rpn: int) -> torch.Tensor:
    """现链的稳定桶排序原语(延迟导入避免 ops <-> ta2a_fwd 模块级环)。"""
    from ..ta2a_fwd import _stable_argsort_small
    return _stable_argsort_small(owner, rpn)


def k1_arrival_ref(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
                   quota: int, epr: int, rpn: int, my_local: int = 0):
    """K1 的 CPU/组合链参考实现 —— kernel 语义的可执行规格,与现链逐位同。

    逐位复刻 ta2a_fwd.ta2a_moe_forward / ta2a_dispatch.ta2a_permute 的到达段
    (quota 快路径分支,C1 之后、Hop B 之前):

        r_idx, slot_idx = _expand_arrival_quota(rslot)
        owner = slot_idx // epr
        ordo  = _stable_argsort_small(owner, rpn)
        r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
        from ..ta2a_fwd import fixed_hist   # 延迟导入,避开模块级环
        i_send = fixed_hist(owner, rpn)     # 与现链同一个定长直方图
        send_buf, gate_pairs = rx[r_idx], rgate.reshape(-1)[ordo]

    与 AscendC kernel 的两遍法(计数 -> 前缀游标 -> 展开写行)是同一个数学对象:
    稳定计数排序 == 稳定升序 argsort(桶间按 owner 升序,桶内按平铺位 p =
    r*quota + i 升序),论证全文见 ascendc/op_kernel/terrace_k1_arrival.cpp
    文件头。r_idx 用 ordo // quota 而非查表:_expand_arrival_quota 的预排 r_idx
    本就是 arange(R) 按 quota 展开,第 p 个配对的行号恒等于 p // quota。

    可微性与现链一致:send_buf(rx 的 gather)与 gate_pairs(rgate 平铺 gather)
    载梯度,r_idx/slot_idx/i_send 是索引/计数平面(整型,天然无梯度)。

    my_local 本段数学不使用 —— K1 接口按规格预留给 Hop B 之后的专家序整理半段
    (exp_j = my_slot - my_local*epr 的同构桶排序),kernel/tiling 已携带。
    """
    R, q = rslot.shape
    assert q == quota, f"rslot 第 1 维 {q} != quota {quota}(C1 线格式契约)"
    slot_flat = rslot.reshape(-1)
    owner = slot_flat // epr
    ordo = _stable_ordo(owner, rpn)
    r_idx = torch.div(ordo, quota, rounding_mode="floor")
    slot_idx = slot_flat[ordo]
    from ..ta2a_fwd import fixed_hist   # 延迟导入,避开模块级环
    i_send = fixed_hist(owner, rpn)     # 与现链同一个定长直方图
    send_buf = rx[r_idx]
    gate_pairs = rgate.reshape(-1)[ordo]
    return send_buf, gate_pairs, r_idx, slot_idx, i_send


def _k1_grad_rx(g_send, r_idx: torch.Tensor, rx_shape) -> torch.Tensor:
    """现链 `send_buf = rx[r_idx]` 的伴随。r_idx 有重复行(每行 quota 个配对),
    加法归约序 = 索引枚举序,与 index 的 autograd 伴随逐位相同
    (tests/test_terrace_k1_arrival.py::test_fn_backward_formula_* 把守)。"""
    grad_rx = g_send.new_zeros(rx_shape)
    grad_rx.index_add_(0, r_idx, g_send)
    return grad_rx


def _k1_grad_rgate(g_gate, rslot: torch.Tensor, epr: int, rpn: int,
                   rgate_shape) -> torch.Tensor:
    """现链 `gate_pairs = rgate平铺[ordo]` 的伴随。ordo 不是 kernel 输出
    (下游不需要它),这里用现链原语从 rslot 重算(_stable_argsort_small 是
    0.107ms 级的 float32 复合键排序,不是 5.32ms 的 int64 稳定排序;见
    ta2a_fwd._stable_argsort_small 的量测注释)。ordo 是置换,散射无重复,
    加法不引入归约序分歧。"""
    ordo = _stable_ordo(rslot.reshape(-1) // epr, rpn)
    flat = g_gate.new_zeros(rgate_shape[0] * rgate_shape[1])
    flat.index_add_(0, ordo, g_gate)
    return flat.view(rgate_shape)


class TerraceK1ArrivalFn(torch.autograd.Function):
    """K1:forward 调 AscendC kernel,backward 用现组合链(路线图拍板)。

    反向语义 = 现链两处 gather 的伴随,全部现成原语、位级(公式见上面两个
    _k1_grad_* 辅助函数,段图版接入点 k1_arrival_segment 与本类共用同一份):
      - send_buf = rx[r_idx]      的伴随:grad_rx = zeros.index_add_(0, r_idx, g);
      - gate_pairs = rgate平铺[ordo] 的伴随:grad_rgate平铺.index_add_(0, ordo, g)。

    适用范围:反向**一次性**走完整段图的调用方(融合前向 ta2a_moe_forward 与
    legacy 3 参接缝 ta2a_permute)。厂商 overlap 接缝分两次 .backward() 进
    permute2 段,不能用这枚融合节点 —— 见 _K1SendEdge 的 docstring。
    """

    @staticmethod
    def forward(ctx, rx, rslot, rgate, quota, epr, rpn, my_local):
        send_buf, gate_pairs, r_idx, slot_idx, i_send = torch.ops.terrace.k1_arrival(
            rx, rslot, rgate, quota, epr, rpn, my_local)
        ctx.save_for_backward(rslot, r_idx)
        ctx.k1_geom = (quota, epr, rpn, rx.shape, rgate.shape)
        ctx.mark_non_differentiable(r_idx, slot_idx, i_send)
        return send_buf, gate_pairs, r_idx, slot_idx, i_send

    @staticmethod
    def backward(ctx, g_send, g_gate, _g_r, _g_s, _g_i):
        rslot, r_idx = ctx.saved_tensors
        quota, epr, rpn, rx_shape, rgate_shape = ctx.k1_geom
        grad_rx = None
        if ctx.needs_input_grad[0] and g_send is not None:
            grad_rx = _k1_grad_rx(g_send, r_idx, rx_shape)
        grad_rgate = None
        if ctx.needs_input_grad[2] and g_gate is not None:
            grad_rgate = _k1_grad_rgate(g_gate, rslot, epr, rpn, rgate_shape)
        return grad_rx, None, grad_rgate, None, None, None, None


def k1_arrival(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
               quota: int, epr: int, rpn: int, my_local: int = 0):
    """到达侧融合链:(send_buf, gate_pairs, r_idx, slot_idx, i_send)。

    kernel 可用走定制算子(NPU),否则走组合链参考实现 —— 两条路径逐位同
    (tests/test_terrace_k1_arrival.py 把守 CPU 侧;NPU 位级由集群
    bench/machine 冒烟把守,命令见 README §3.4)。调用方(ta2a_fwd /
    ta2a_dispatch 的接入点)自带 custom_ops_enabled() 闸,降级时走现链原文,
    不经此函数 —— 这里的回退是给直接调用/测试用的。
    """
    if custom_ops_enabled():
        return TerraceK1ArrivalFn.apply(rx, rslot, rgate, quota, epr, rpn, my_local)
    return k1_arrival_ref(rx, rslot, rgate, quota, epr, rpn, my_local)


# --------------------------------------------------------------------------------------
# K1 段图版:overlap 6 参接缝专用(2026-08-20)
# --------------------------------------------------------------------------------------

class _K1SendEdge(torch.autograd.Function):
    """把 kernel 已算好的 send_buf 挂回 rx 叶子:前向零工作,反向 = rx[r_idx] 的伴随。

    为什么 overlap 接缝不能直接用 TerraceK1ArrivalFn:那是一枚**融合节点**,token
    路(permute2_graph)与 gate 路(permute2_prob_graph)共用它。厂商 gmm 的手写
    backward 对这两根分**两次** .backward() 进 permute2 段(先 prob 后 token,逐步
    复刻见 tests/test_ta2a_overlap_seam.py)——
      1. 第二次会撞 "Trying to backward through the graph a second time"
         (第一次已释放该节点的 saved tensors),厂商代码里没有 retain_graph 可给;
      2. 就算不炸,第一次还会把 materialize 出来的**零梯度**先写进另一路的 .grad,
         多一次 [pairs, H] 规模的散射,且 -0.0 + x 的符号位不再逐位安全。
    现链没有这个问题:两条 gather 子图互不相交,各自只挂在自己的 detach 叶下。
    所以段图版把 kernel 的**数据**产出与**图**分开:kernel 跑一次(no_grad),两个
    float 输出各挂一条独立的边,根分别是 rx_d / rgate_d —— 与现链同构,厂商编排
    照跑不误,席位契约(7+3)、detach 边界、splits 交接一律不动。
    """

    @staticmethod
    def forward(ctx, rx, r_idx, send_buf):
        ctx.save_for_backward(r_idx)
        ctx.rx_shape = rx.shape
        return send_buf            # kernel 已产出;autograd 自动别名并挂 grad_fn

    @staticmethod
    def backward(ctx, g_send):
        (r_idx,) = ctx.saved_tensors
        grad_rx = None
        if ctx.needs_input_grad[0] and g_send is not None:
            grad_rx = _k1_grad_rx(g_send, r_idx, ctx.rx_shape)
        return grad_rx, None, None


class _K1GateEdge(torch.autograd.Function):
    """把 kernel 已算好的 gate_pairs 挂回 rgate 叶子。见 _K1SendEdge 的 docstring。

    梯度落回 [R, quota] 的 rgate 形状(现链 reshape 反向的同一形状),厂商按行数
    splits 沿 Hop A 重放,对布局无感知。
    """

    @staticmethod
    def forward(ctx, rgate, rslot, gate_pairs, epr, rpn):
        ctx.save_for_backward(rslot)
        ctx.k1_gate_geom = (epr, rpn, rgate.shape)
        return gate_pairs

    @staticmethod
    def backward(ctx, g_gate):
        (rslot,) = ctx.saved_tensors
        epr, rpn, rgate_shape = ctx.k1_gate_geom
        grad_rgate = None
        if ctx.needs_input_grad[0] and g_gate is not None:
            grad_rgate = _k1_grad_rgate(g_gate, rslot, epr, rpn, rgate_shape)
        return grad_rgate, None, None, None, None


def k1_arrival_segment(rx: torch.Tensor, rslot: torch.Tensor, rgate: torch.Tensor,
                       quota: int, epr: int, rpn: int, my_local: int = 0):
    """K1 的**段图版**:返回值与 k1_arrival 逐位相同,图的形状与现链相同。

    kernel(降级时是参考实现)只跑一次,在 no_grad 下对 detach 过的入参求**数据**;
    图由两条互不相交的边重建 —— send_buf 挂回 rx,gate_pairs 挂回 rgate。
    r_idx / slot_idx / i_send 是整数索引/计数平面,天然不参与梯度,原样交出
    (不经任何 Function,也就不会被 materialize 出假梯度)。

    调用方是 ta2a_dispatch.ta2a_permute_overlap 的到达段(permute2 段图内部,
    detach 叶 rx_d / rgate_d 之下);一次性反向的调用方继续用 k1_arrival。
    """
    with torch.no_grad():
        send_buf, gate_pairs, r_idx, slot_idx, i_send = k1_arrival(
            rx.detach(), rslot, rgate.detach(), quota, epr, rpn, my_local)
    send_buf = _K1SendEdge.apply(rx, r_idx, send_buf)
    gate_pairs = _K1GateEdge.apply(rgate, rslot, gate_pairs, epr, rpn)
    return send_buf, gate_pairs, r_idx, slot_idx, i_send


# --------------------------------------------------------------------------------------
# K2 接口草案(仅注释,按 C1 落地后的链定稿 —— 勿据此写调用方)
#
#   terrace::k2_pack(Tensor hidden, Tensor expert_idx, Tensor gates,
#                    int world, int n_experts, int rpn, int groups_m)
#       -> (Tensor payload, Tensor mask, Tensor gate_rows,
#           Tensor u_src, Tensor node_counts)
#     替换 plan_ta2a(...) 到 _pack_quota_wire(...) 的整段(ta2a_permute
#     :106-:121;overlap 半边同段)。等配额快路径全形状静态;u_src/node_counts
#     仍需交回(combine 半边与 Hop A 计数交换要用)。C1 打包侧
#     (_pack_quota_wire)是 K2 的地盘,K1 不碰。
# --------------------------------------------------------------------------------------

__all__ = [
    "OpsLoadError", "OpsState", "TerracePassthroughFn", "TerraceK1ArrivalFn",
    "custom_ops_enabled", "k1_arrival", "k1_arrival_ref", "k1_arrival_segment",
    "passthrough", "reset", "status",
]
