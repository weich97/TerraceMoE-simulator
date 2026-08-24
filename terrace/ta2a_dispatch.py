"""T-A2A 拆成 dispatcher 的两半,好挂进厂商真实训练步(#18(a))。

`ta2a_fwd.ta2a_moe_forward` 是一个**闭合**的前向:调度、专家计算、回收全在里面。
Megatron 的 MoE 层要的是另一种形状 —— 一对方法,中间夹着**厂商自己的**专家计算:

    permuted, tokens_per_expert, permuted_probs = token_permutation(h, probs, routing_map)
    expert_out = <厂商的 grouped GEMM / aux loss / expert bias / 共享专家>
    output, _  = token_unpermutation(expert_out)

所以这里把那个闭合前向按专家计算切开,中间量挂在一个 state 上传递。
选这条路(而不是整层替换 moe_layer.forward)的理由见 内部设计记录(未随仓发布):
整层替换会让我们和 `--moe-router-enable-expert-bias`、共享专家、seq-aux-loss 这些
**已在四档消融里验证过的**厂商特性分家,把一条已验证的路重新变成未验证的路。

三个必须说清的口径:

1. **gate 不在这里乘**。厂商在 experts.py:241 用 `permuted_probs.unsqueeze(-1) *` 施加,
   我们只负责把每个 (row, slot) 的 gate 按同样顺序交出去。原融合前向是在专家处自己乘的
   (`wgt = my_gate[order]`),照搬会**乘两次**。
2. **`tokens_per_expert` 是数学必然量**:T-A2A 只改载荷去重与路由方式,不改「哪个专家要处理
   哪些 (token, expert) 对」,所以我们算出的 counts 必须与厂商 `preprocess(routing_map)`
   逐元素相等。这是判「接对了没有」最硬的探针(TERRACE_TA2A_ASSERT=1 打开)。
3. **EP=8 时按构造是 no-op**:一个节点 8 die 正好装下一个 EP 组,没有跨节点跳,
   此时直接交还厂商实现。EP 档位扫描里 EP=8 的增量必须 ≈0,不通过说明接错了。
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from .ep_dist import _a2a, _a2a_raw
from .ta2a import plan_ta2a
from .ta2a_fwd import (build_expansion, _stable_argsort_small, _expand_arrival,
                       _expand_arrival_quota, _pack_quota_wire, _send_index,
                       _splits_to_lists, fixed_hist)
# K1 kernel 闸(terrace.ops 只在函数体内延迟导入 ta2a_fwd,模块级不成环)。
from . import ops as _tops
# 集合通信并包(A1/A2,2026-08-21):Hop A 的 [payload‖gate]、Hop B 的 [slot‖gate]
# 各并成一条,dispatch 8 -> 6 条。纯字节重排,splits 语义与厂商契约零改动 —— 逐项
# 收益/代价账与「哪些面有意不并、为什么」见 terrace/ta2a_pack.py 模块头。
# 闸门 TERRACE_TA2A_PACK=0 走并包前的现链原文(含集合通信次序),零行为变化。
from . import ta2a_pack as _pk
# 漂移探针(TERRACE_DRIFT_PROBE=1 才生效,否则每处只多一次已缓存的 bool 读)。
# 两条接缝用**同名**点位:on 臂同床跑 legacy 与 overlap 两次,即可把
# tests/test_ta2a_seam_bitparity.py 的 CPU 逐位契约在 NPU 上对差一遍
# (python -m terrace.drift_probe compare ov.log lg.log)。
from . import drift_probe as _dp


class TA2AState:
    """两半之间要传递的中间量。每层每个 microbatch 一个。

    为什么不做成模块级全局:一个 transformer 里有 N 个 MoE 层,前向是层层递进的
    (层 i 的 unpermutation 发生在层 i+1 的 permutation 之前),但重计算(recompute)
    会让同一层的 permutation 跑两次 —— 全局变量在这种交错下会被后写的层覆盖,
    表现为「某些层的 combine 用了别层的 plan」,而且只在开重计算时出现。
    挂在 dispatcher 实例上则天然与层一一对应。
    """

    __slots__ = ("u_src", "r_idx", "order", "R", "T", "hidden",
                 "send_l", "recv_l", "is_l", "ir_l", "intra", "inter", "dtype")


# --------------------------------------------------------------------------------------
# A3-lite(2026-08-21):Hop A 的 counts 交换提前发、异步等
# --------------------------------------------------------------------------------------
#
# A3(计划常量化,干掉两条 counts 交换)已判净亏 −7.03 ms 不实施
# (内部设计记录(未随仓发布):等配额只固定每 token 扇出 M、不固定每目的地
# 负载,容量上界只能取几何最坏值 n_nodes/M = 8x 与 rpn = 8x)。**唯一可救的残片**是
# 这条:inter 的 counts 交换不需要等本地打包做完 —— 它只依赖 plan_ta2a 交出的
# node_counts。提前到 plan 之后异步发出,用本地 gather + 打包的时间去盖它的 α₁₂₈。
#
#   零字节冗余、零数值改动、不需要任何容量假设;拿回多少 = min(α₁₂₈, 本地打包耗时),
#   须床上直测(判决床 α₁₂₈ = 0.45 ms,打包段是 gather 33 MB + 并包拷贝 ~0.05 ms,
#   案头预期只能盖住一小部分 —— 所以它是独立一刀、独立读数,不许和 A1′/A2 混算)。
#
# intra 那条(:459 附近)没有同样的机会:i_send 来自到达展开,和 Hop B 之间没有可盖的
# 本地工作。
#
# 闸门 TERRACE_TA2A_ASYNC_COUNTS=0 回到原位置同步发,便于床上单独 A/B 归因。

_ENV_ASYNC_COUNTS = "TERRACE_TA2A_ASYNC_COUNTS"
_ASYNC_COUNTS: bool | None = None


_SYNC_PROBE_ENV = "TERRACE_TA2A_SYNC_PROBE"
_SYNC_PROBE = None


def sync_probe_enabled() -> bool:
    """判别探针:在 i_send 算完之后插一次**纯粹被丢弃**的 .tolist()。默认关。

    这不是优化,是**量具**。内部记录 把我原来对 fixed_hist 的机理解释证伪了:
    主机同步实测只值 0.042-0.046 ms,而被换掉的 bincount 是 0.797 ——
    同步最多解释 6%。可 fixed_hist 上机确实拿到 -1.166(两轮复现)。
    机理不明的收益不能拿去支撑下一把刀,所以要判别:

      开着这个探针跑一档,dispatch 若回到 ~9.2 => 那 1.166 主要是"在那个位置做一次
      主机同步"的价钱,同步家族(护栏、两次 _splits_to_lists)拿到真锚;
      若仍在 ~8.1 => 那 1.166 是 bincount 自身实现的代价,同步家族没有锚,
      人力立刻转向别处。

    **两种结果都有用**,这正是判别实验该有的样子。
    """
    global _SYNC_PROBE
    if _SYNC_PROBE is None:
        _SYNC_PROBE = os.environ.get(_SYNC_PROBE_ENV, "0").strip() == "1"
    return _SYNC_PROBE


def reset_sync_probe() -> None:
    """测试/调试钩子。"""
    global _SYNC_PROBE
    _SYNC_PROBE = None


_EARLY_HOPB_ENV = "TERRACE_TA2A_EARLY_HOPB"
_EARLY_HOPB = None


def early_hopb_counts_enabled() -> bool:
    """A6 闸门:Hop B 的 counts 交换提前异步发。默认开。

    为什么默认就开:这是**纯调度重排**,i_send 的值、alltoall 的语义、i_recv 的内容、
    以及下游的每一个比特都不变 —— 只有发起点提前了。不像 A1'/A5 那样动布局或归约序,
    不需要 eq 门。留闸门只为床上做 A/B(TERRACE_TA2A_EARLY_HOPB=0 走原文)。

    进程生命周期内只读一次:dispatch 是每层每 microbatch 的热路径。
    """
    global _EARLY_HOPB
    if _EARLY_HOPB is None:
        _EARLY_HOPB = os.environ.get(_EARLY_HOPB_ENV, "1").strip() != "0"
    return _EARLY_HOPB


def reset_early_hopb() -> None:
    """忘掉 A6 闸门的缓存判定。测试/调试钩子,训练代码不得调用。"""
    global _EARLY_HOPB
    _EARLY_HOPB = None


def async_counts_enabled() -> bool:
    """A3-lite 闸门。进程生命周期内只读一次环境(dispatch 是热路径)。"""
    global _ASYNC_COUNTS
    if _ASYNC_COUNTS is None:
        _ASYNC_COUNTS = os.environ.get(_ENV_ASYNC_COUNTS, "1").strip() != "0"
    return _ASYNC_COUNTS


def reset_async_counts() -> None:
    """忘掉缓存判定。测试/调试钩子,训练代码不得调用。"""
    global _ASYNC_COUNTS
    _ASYNC_COUNTS = None


def _group_world(group) -> int:
    """通信域的 world size;取不到就返回 0(调用方据此退回旧行为)。

    Hop A 的通信域从 2026-08-24 起可以是**跨节点子组**(n_nodes 个 rank)而不是
    整个 EP 组(world 个)。counts 与 splits 的长度必须跟着它走 —— 长度不匹配
    HCCL 会当场炸,不会静默出错,这一点是好的;但两条路径的长度都得算对。
    """
    if group is None:
        return 0
    try:
        return int(dist.get_world_size(group=group))
    except Exception:                                       # noqa: BLE001
        return 0


def _hopa_counts(world: int, n_nodes: int, rpn: int, my_local: int, dev,
                 node_counts: torch.Tensor, inter_world: int = 0):
    """Hop A counts 交换的两个缓冲。散射索引是几何的纯函数(已缓存,见 _send_index)。

    `inter_world == n_nodes` 时走**跨节点子组**:组内 rank i 恰是节点 i,
    于是 send 直接就是 node_counts —— 那 112 个补零正是要消掉的东西。
    否则退回整个 EP 组(长度 world、只有 n_nodes 个非零),行为与 2026-08-24 之前逐字相同。
    """
    if inter_world == n_nodes:
        send = node_counts.to(torch.long)
        return send, torch.empty_like(send)
    send = torch.zeros(world, dtype=torch.long, device=dev)
    send[_send_index(n_nodes, rpn, my_local, dev)] = node_counts
    return send, torch.empty_like(send)


def _per(src: torch.Tensor, n_out: int):
    """每个归约目标恰好几份贡献(整除才有,否则 None)。等配额下按构造成立:
    combine 第一级每行恰 quota 份、第二级每 token 恰 M 份 —— 有它,漂移探针才能
    拿一个**定形** reshape-sum 当确定性参考去量设备 index_add 的累加器宽度。"""
    n = int(src.shape[0])
    return (n // n_out) if (n_out and n % n_out == 0) else None


def routing_map_to_topk(routing_map: torch.Tensor, probs: torch.Tensor):
    """[T, E] 的布尔图 + [T, E] 的概率 -> [T, k] 的专家号与 gate。每行升序。

    厂商的 routing_map 是稠密布尔图,而 plan_ta2a 要的是 top-k 索引。
    这里**要求每行恰好 k 个**并 fail loud:drop-less 路由下这是恒真的,
    一旦不真(比如将来接了带容量丢弃的路由),形状会静默变化,
    而按 k 重排会把不同 token 的专家混进同一行 —— 那种错误在 loss 上看不出来。

    提取用 topk + 升序 sort 而非 nonzero(2026-08-20 铸刀B):
      - 位级同值:每行恰好 k 个已先验证,topk 在 {0,1} 值上选出的**集合**因此确定
        (k 个 1 全取,与并列序无关);升序排序后与 nonzero 的行主序逐位相同。
        索引值 < E << 2^24,在 float32 里排序精确(整数 sort 是此设备最贵的原语,
        见 _stable_argsort_small)。
      - 省两次主机同步:nonzero 的输出形状依赖数据,必须同步;guard 原来又付
        `.item()` 与 `.all()` 两次。现在 guard 的 min/max 并成一次 tolist,
        全函数 3 次同步 -> 1 次。计数在 float32 里求和(k <= E < 2^24,精确;
        int64 求和是 37 倍慢的原语,内部基准脚本(未随仓发布))。
    """
    rm_f = routing_map.to(torch.float32)
    counts = rm_f.sum(dim=1)                     # 精确:每行至多 E 个 1,E << 2^24
    mn_mx = torch.stack((counts.min(), counts.max())).tolist()   # 本函数唯一一次同步
    if mn_mx[0] != mn_mx[1]:
        raise RuntimeError(
            "T-A2A 要求每个 token 恰好 k 个专家(drop-less);实测每行专家数不一致")
    k = int(mn_mx[0])
    if k == 0:
        raise RuntimeError("T-A2A 收到全空 routing_map(k=0);drop-less 路由不可能")
    sel = torch.topk(rm_f, k, dim=1).indices                     # 恰为每行的 k 个置位
    if routing_map.shape[1] < (1 << 24):
        expert_idx = torch.sort(sel.to(torch.float32), dim=1).values.to(torch.int64)
    else:                                        # 不可达的护栏:E >= 2^24 才需整数排序
        expert_idx = torch.sort(sel, dim=1).values
    gates = probs.gather(1, expert_idx)
    return expert_idx, gates


def ta2a_permute(hidden_states, probs, routing_map, *, world, rank, rpn,
                 n_experts, intra_group, inter_group=None, groups_m=None):
    """dispatch 半边:返回 (按本地专家排好序的行, 每个本地专家的行数, 每行的 gate, state)。

    与 `ta2a_moe_forward` 的前半段逐行同构(dispatch 段,到专家序整理为止),差别只有:
    不乘 gate、把中间量装进 state、集合通信一律走传入的 group(厂商的 EP 组),
    而不是默认的全局组。
    """
    epr = n_experts // world
    slots = epr * rpn
    dev = hidden_states.device
    my_local = rank % rpn
    # 闸门只读一次:同一次 dispatch 的两跳必须走同一条路(半并包是没人验证过的第三种
    # 形态),而 pack_enabled 是可被测试 monkeypatch 的模块属性。
    pack_mode = _pk.pack_mode()
    packing = pack_mode != "off"

    expert_idx, gates = routing_map_to_topk(routing_map, probs)
    T, k = expert_idx.shape

    u_src, u_node, node_counts, inverse = plan_ta2a(
        expert_idx, world, n_experts, rpn, groups_m=groups_m)
    n_rows = u_src.numel()
    n_nodes = world // rpn
    # A3-lite:counts 交换只依赖 node_counts,提前发,用下面的 gather + 打包盖它的
    # α₁₂₈(见文件上方 A3-lite 一节)。闸关时保持在原位置同步发,零行为变化。
    h_cnt = send = recv = None
    if async_counts_enabled():
        send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
        h_cnt = dist.all_to_all_single(recv, send, group=inter_group, async_op=True)
    payload = hidden_states[u_src]
    if groups_m:
        assert k % groups_m == 0, f"k={k} not divisible by groups_m={groups_m}"
    quota = (k // groups_m) if groups_m else None
    if quota is not None:
        # C1 quota 线格式(2026-08-20):Hop A 的 id 平面从 int64 位掩码换成
        # [n_rows, quota] 升序槽号,gate 平面从 [n_rows, slots] 稀疏换成
        # [n_rows, quota] 致密 —— 升序槽号 == 旧到达侧 topk 的输出序,配对序列与
        # 每一处下游排序逐位不变(论证与穷举见 _pack_quota_wire)。sorted_rows:
        # routing_map_to_topk 按构造每行升序,打包侧因此一次排序都不需要。
        mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                           n_rows, slots, quota, n_experts,
                                           sorted_rows=True)
    else:
        # 一次取模,掩码平面与 gate 平面共用;sorted_rows:routing_map_to_topk
        # 按构造每行升序(掩码位逐位不变,见 build_expansion)。
        slot_flat = expert_idx.reshape(-1) % slots
        mask = build_expansion(expert_idx, inverse, n_rows, world, n_experts, rpn,
                               groups_m=groups_m, sorted_rows=True,
                               slot_flat=slot_flat)
        gate_rows = payload.new_zeros(n_rows, slots)
        gate_rows[inverse, slot_flat] = gates.reshape(-1)
    _dp.note("seam.payload", payload)
    _dp.note("seam.gate", gate_rows)
    _dp.note_int("seam.mask", mask)

    if h_cnt is None:
        send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
        dist.all_to_all_single(recv, send, group=inter_group)
    else:
        h_cnt.wait()
    send_l, recv_l = _splits_to_lists(send, recv)      # 一次同步,不是两次

    # mask 是任一线格式下的 id 平面:[n_rows] int64 位掩码(通用)或 [n_rows, quota]
    # 升序槽号表(C1 快路径)。三个平面共用 splits,A1′ 把它们并成一条。
    if pack_mode == "small":
        # A1''(2026-08-22,依判决床实测改的默认形态):**只并两个小平面**
        # (id ‖ gate,32 B/行),载荷 [n, 2048] 自己走一条 —— 与未并包臂**逐字同一句**
        # `_a2a(payload, ...)`,所以载荷路径的梯度与集合通信都零变化。
        # Hop A 集合通信 4 -> 3,省 1α₁₂₈ ≈ 0.45 ms;拷贝量比 full 形态少 129 倍。
        # 为什么不是 full(2 条):full 要把载荷抄进容器再抄出来,判决床实测那两趟
        # HBM 拷贝 ≈ 3.0 ms,把省下的 0.96 ms 吃光还倒亏 2.05 ms
        # (内部实测记录)。
        with torch.no_grad():
            _sbuf, _lay = _pk.hopa_pack_small(gate_rows.detach(), mask)
            _rbuf = _pk.hopa_exchange_raw(_sbuf, send_l, recv_l, group=inter_group)
            _rgate_raw, rmask = _pk.hopa_unpack_small(_rbuf, _lay)
        _pk.assert_not_aliased(_rbuf, _rgate_raw, rmask)
        rx = _a2a(payload, send_l, recv_l, group=inter_group)   # 与 off 臂逐字同一句
        rgate = _pk.attach_edge(gate_rows, _rgate_raw, send_l, recv_l, inter_group)
    elif pack_mode == "full":
        # A1′:id + payload + gate 三条并成一条(int64 容器,id 面在行首保 8 字节
        # 对齐)。数据在图外收下,图由两条独立的边重建 —— 反向仍是并包前的同两句
        # _a2a_raw,梯度逐位不变、反向条数不变。
        # **判决床实测净亏 2.05 ms/次**,保留只为可复现那次读数与做 A/B,不再是默认。
        with torch.no_grad():
            _sbuf, _lay = _pk.hopa_pack(payload.detach(), gate_rows.detach(), mask)
            _rbuf = _pk.hopa_exchange_raw(_sbuf, send_l, recv_l, group=inter_group)
            _rx_raw, _rgate_raw, rmask = _pk.hopa_unpack(_rbuf, _lay)
        _pk.assert_not_aliased(_rbuf, _rx_raw, _rgate_raw, rmask)
        rx = _pk.attach_edge(payload, _rx_raw, send_l, recv_l, inter_group)
        rgate = _pk.attach_edge(gate_rows, _rgate_raw, send_l, recv_l, inter_group)
    else:
        rx = _a2a(payload, send_l, recv_l, group=inter_group)
        rmask = _a2a_raw(mask, send_l, recv_l, group=inter_group)
        rgate = _a2a(gate_rows, send_l, recv_l, group=inter_group)
    _dp.note("seam.rx", rx)
    _dp.note("seam.rgate", rgate)
    _dp.note_int("seam.rmask", rmask)

    R = rx.shape[0]
    # K1(AscendC kernel,2026-08-20):quota 快路径下,下面整段到达链(配对展开、
    # owner 稳定桶排、i_send 直方图、[pairs, H] 发送 gather、gate 平铺 gather)融成
    # 一枚 kernel。两遍法与现链稳定序逐位一致的论证见
    # terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp 文件头;CPU 可执行规格
    # terrace.ops.k1_arrival_ref。else 分支是 K1 前的现链**原文**,kernel 缺席时
    # (TERRACE_CUSTOM_OPS=0 / 无 .so)是唯一路径:零行为变化。exp_rx/gate_pairs
    # 提到集合通信之前,只为两分支以同五张量进 Hop B —— 纯数据流重排,算子与操作数
    # 不变。
    _h_i = _i_recv_early = None          # A6 的句柄;K1 分支不提前发,保持 None
    if quota is not None and _tops.custom_ops_enabled():
        exp_rx, gate_pairs, r_idx, slot_idx, i_send = _tops.k1_arrival(
            rx, rmask, rgate, quota, epr, rpn, my_local)
    else:
        if quota is not None:
            r_idx, slot_idx = _expand_arrival_quota(rmask)  # 槽号表即配对表,免位抽取+topk
        else:
            r_idx, slot_idx = _expand_arrival(rmask, slots, quota)
        owner = slot_idx // epr
        # A6:直方图对置换盲,owner 一出来就能算 —— 不必等排序。算完立刻把 Hop B 的
        # counts 交换**异步**发出去,用下面的排序 + 两次索引 gather + [pairs,H] 大 gather
        # 盖住它。纯调度重排:i_send 的值、alltoall 语义、i_recv 内容、下游逐位全不变。
        # (依据:§7 证明"少发一条"价值为零,"挪到能被真实计算盖住的位置"才有收益。)
        i_send = fixed_hist(owner, rpn)          # 定长直方图,避开 bincount 的隐藏主机同步
        if sync_probe_enabled():
            _ = i_send.tolist()   # 判别探针:纯粹丢弃,只为付一次主机同步(见 sync_probe_enabled)
        if early_hopb_counts_enabled():
            _i_recv_early = torch.empty_like(i_send)
            _h_i = dist.all_to_all_single(_i_recv_early, i_send, group=intra_group,
                                          async_op=True)
        ordo = _stable_argsort_small(owner, rpn)
        r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
        exp_rx = rx[r_idx]
        # C1 快路径:致密 gate 表与配对枚举同构,二维 gather 退化为平铺 gather ——
        # rgate.reshape(-1)[ordo] 与 rgate[r_idx, slot_idx] 逐元素同一(平铺位
        # row*quota+i 即该配对的 ordo 前位置),逐位相等。
        gate_pairs = (rgate.reshape(-1)[ordo] if quota is not None
                      else rgate[r_idx, slot_idx])
    _dp.note("seam.exprx", exp_rx)
    _dp.note("seam.gpairs", gate_pairs)
    _dp.note_int("seam.slot", slot_idx)
    _dp.note_int("seam.isend", i_send)
    # A6 开且走的是非 K1 分支时,上面已经异步发过了 —— 这里只等它落地。
    # K1 分支的 i_send 从 kernel 出来,来不及提前,仍在原位同步发。
    if _h_i is not None:
        _h_i.wait()
        i_recv = _i_recv_early
    else:
        i_recv = torch.empty_like(i_send)
        dist.all_to_all_single(i_recv, i_send, group=intra_group)
    is_l, ir_l = _splits_to_lists(i_send, i_recv)

    # Step 2(融合前向 2026-08-20 已删 my_gate 交换、改在到达 rank 施门)**不适用于
    # 接缝**:这里的 my_gate 交换就是厂商契约 permuted_probs 的交付本身 —— 乘点在
    # 厂商手里(experts.py:241),不归我们选;删它 = 换接缝语义,不是省工。
    node_rx = _a2a(exp_rx, is_l, ir_l, group=intra_group)
    if packing:
        # A2:Hop B 的两个**每配对一个标量**的面(槽号 int64 + gate)并成一条。
        # 载荷面 exp_rx 有意**不并** —— 并进来要多付两趟 100 MB 级 HBM 拷贝
        # (+0.317 ms)去省 1α₈(0.058 ms),判决床上净亏 0.20 ms;算式与零拷贝
        # 后续件见 terrace/ta2a_pack.py 的 Hop B 一节。
        with torch.no_grad():
            _bbuf = _pk.hopb_pack_meta(slot_idx, gate_pairs.detach())
            _rbb = _a2a_raw(_bbuf, is_l, ir_l, group=intra_group)
            my_slot, _mg_raw = _pk.hopb_unpack_meta(_rbb, gate_pairs.dtype)
        my_gate = _pk.attach_edge(gate_pairs, _mg_raw, is_l, ir_l, intra_group)
    else:
        my_slot = _a2a_raw(slot_idx, is_l, ir_l, group=intra_group)
        my_gate = _a2a(gate_pairs, is_l, ir_l, group=intra_group)

    exp_j = my_slot - my_local * epr
    order = _stable_argsort_small(exp_j, epr)
    tokens_per_expert = fixed_hist(exp_j, epr)                 # 置换盲:免 exp_j[order]

    st = TA2AState()
    st.u_src, st.r_idx, st.order = u_src, r_idx, order
    st.R, st.T, st.hidden = R, T, hidden_states.shape[1]
    st.send_l, st.recv_l, st.is_l, st.ir_l = send_l, recv_l, is_l, ir_l
    st.intra, st.inter, st.dtype = intra_group, inter_group, hidden_states.dtype
    permuted, pprobs = node_rx[order], my_gate[order]
    _dp.note("seam.permuted", permuted)
    _dp.note("seam.pprobs", pprobs)
    _dp.note_int("seam.tpe", tokens_per_expert)
    return permuted, tokens_per_expert, pprobs, st


def ta2a_unpermute(expert_out, st: TA2AState, out_like):
    """combine 半边:把专家输出沿 dispatch 的反向送回原 token。

    **不乘 gate** —— 厂商已在 experts.py 里用 permuted_probs 乘过了。
    每个返回行携带的是「目的节点上所有相关专家的加权和」,所以原点按 (token, node)
    只加一次(u_src 命名的就是那个 token);按 (token, expert) 加是早期的 bug,
    会把行重复 M 遍并重新施加已消费过的 gate。
    """
    back_pairs = expert_out.new_empty(expert_out.shape)
    back_pairs[st.order] = expert_out
    ret = _a2a(back_pairs, st.ir_l, st.is_l, group=st.intra)   # 反向:ir, is
    red = ret.new_zeros(st.R, ret.shape[1])
    red.index_add_(0, st.r_idx, ret)
    _dp.note("seam.eout", expert_out)
    _dp.note("seam.ret", ret)
    _dp.note("seam.red", red)
    _dp.check_reduction("seam.red", red, ret, st.r_idx, st.R, _per(ret, st.R))
    back = _a2a(red, st.recv_l, st.send_l, group=st.inter)     # 反向:recv, send
    y = out_like.new_zeros(st.T, st.hidden)
    y = y.index_add(0, st.u_src, back)
    _dp.note("seam.back", back)
    _dp.note("seam.y", y)
    _dp.check_reduction("seam.y", y, back, st.u_src, st.T, _per(back, st.T))
    return y


# --------------------------------------------------------------------------------------
# overlap 族(--moe-alltoall-overlap-comm + alltoall_seq)6 参接缝的两半(#18a/18c,Phase B)
#
# 厂商接缝(上游训练栈(版本固定) moe_feature/overlap,只读研读结论;行号以 2607 解包件为准):
#
#   MoELayerOverlapAllToAllSeq.forward(moe_layer_overlap_all2allseq.py:69)调
#     (share_experts_output, dispatched_input, tokens_per_expert, global_probs) =
#         token_dispatcher.token_permutation(
#             hidden_states, scores, routing_map, shared_experts, save_tensors, moe_ctx)
#   —— 这就是"6 参":3 个张量 + 共享专家模块(可为 None)+ save_tensors 列表 + 该层
#   autograd.Function 的 ctx。combine 侧(:83)是 3 参且**返回单张量**(不是 legacy 的
#   二元组):output = token_dispatcher.token_unpermutation(expert_output, mlp_bias,
#   save_tensors)。
#
#   关键结构:整层是一个**手写 backward** 的 autograd.Function。前向把计算切成若干
#   独立子图("段",厂商用 forward_func 建,段的输入是 detach 出来的叶子),段与段
#   之间的 EP 组 all_to_all 故意放在段外、不进 autograd,backward 里用 dispatcher
#   身上的 input_splits/output_splits 手工重放。backward 按**固定位置**解包
#   save_tensors(moe_layer_overlap_all2allseq.py:157-168),所以 dispatcher 两半必须
#   恰好按下述顺序追加 7 + 3 个条目:
#     permutation 7 个:permute1_graph, permuted_probs_graph,
#         num_global_tokens_per_local_expert_cpu, permute2_input_detach, permute2_graph,
#         permute2_prob_detach, permute2_prob_graph
#     unpermutation 3 个:unpermute1_input_detach, unpermute1_graph,
#         unpermute2_input_detach
#   专家侧 gmm 也是手写 backward(grouped_mlp_with_comp_and_comm_overlap_all2allseq.py):
#   它对 permute2_graph / permute2_prob_graph 调 backward_func,再把两个 detach 叶子的
#   .grad 沿 EP 组用 (input_splits, output_splits) 反向重放交还 moe 层;moe 层 backward
#   对 unpermute2_input_detach.grad 做同样的手工重放。
#
# T-A2A 拆半怎么插(段边界与厂商一一对应,这是本适配的全部要点):
#   - 厂商手工重放的那一跳 EP a2a == T-A2A 的 Hop A(跨节点 fabric 跳,同一 EP 组)。
#     把 disp.input_splits=send_l、disp.output_splits=recv_l 交出去,厂商 backward 的
#     两处手工重放就**恰好是 Hop A 的反向**,一行厂商代码都不用改。
#   - Hop B(节点内 a2a)整段藏进 permute2 / unpermute1 子图内部,用 ep_dist._A2A
#     (自带 backward 的可微 all_to_all)。手写 backward 对这两段只调 .backward(),
#     autograd 自动把 Hop B 反着跑,厂商代码不需要知道它存在。
#   - 段边界即 detach 点:落地张量 rx / rgate(= permute2_input_detach /
#     permute2_prob_detach)、回程张量 back(= unpermute2_input_detach)。
#   - gate 精度走 payload.dtype,与 legacy 半程**同一圆整点**(dispatch 侧进 gate_rows
#     时)。一次内部提交 曾按「忠实厂商 overlap 的 probs 平面(router 输出精度)」走
#     probs.dtype;eqov 对齐床(2026-08-20,slots=16 噪声地板 1.3e-5)证伪了该选择:
#     fp32 gate 平面经厂商 gmm 的 probs 乘法把 expert_out 提升到 fp32,连带 combine
#     的两跳回程与两级 index_add 归约全部在 fp32 里做,与 legacy 已验路径(bf16 平面)
#     每一处归约的舍入都不同 —— 低于 dispatch 断言容差(出口 token 平面仍逐位相等),
#     但逐步放大:1e-5@20 → 1.38e-4@100,校准比 10.6×(界 3×)。改回 payload.dtype 后
#     两条接缝在床口径(bf16 载荷 + fp32 router probs)下前向与全部梯度逐位相等
#     (tests/test_ta2a_seam_bitparity.py)。probs 梯度仍回 fp32 叶子:.to 的反向是
#     精确 upcast,厂商手写编排对该 cast 节点无感知,席位契约不变。
#
# 18c(共享专家 x A2A 重叠)清单:
#   免费获得(随厂商调度,不需要我们写一行):
#     - 共享专家前向与 Hop A 重叠:本函数在 Hop A 异步发出后、wait 之前回调
#       run_shared_experts,与厂商 token_permutation 的段位相同;
#     - 共享专家 backward 的排布(moe 层 backward 统一处理 share_experts_graph);
#     - 专家 dW 与 dispatch 反向 a2a 的重叠(gmm 手写 backward 固有,对两半透明);
#     - 激活重算(should_recompute_activation)只涉及 gmm 内部,不碰 dispatch。
#   明确不做(留给后续件,包装器闸门回退、REQUIRE 下 raise):
#     - 相位计时(dispatch/combine 分相位打点)—— 单独立项;
#       (groups_m 穿透已由 Phase B 第二件完成:包装器在首个 batch 判几何后把 M
#       传进本文件两条接缝的 groups_m 形参,见 usercustomize._ta2a_groups_m_for)
#     - moe_zero_memory level0/level1:那两档的 backward 会按【厂商的】permute 重算
#       dispatch,与 T-A2A 不同构;
#     - TP>1 / moe_tp_extend_ep(厂商换 tp-ep 混合通信组)/ 容量丢弃路由;
#     - alltoall(非 seq)5 参 overlap、mc2moe、fb_overlap、balanced_moe 族 —— 未适配。
#
# 18c 的核实结论(2026-08-21,零机时,仓内 + 2607 只读):
#   `moe_shared_expert_overlap` 这个**开关与 alltoall_seq 无关**,它是 Megatron 原生
#   alltoall dispatcher 的四段式(pre_forward_comm / linear_fc1_forward_and_act /
#   linear_fc2_forward+post_forward_comm / get_output)重叠开关,校验在
#   `megatron/core/transformer/transformer_config.py:646-651`(0.12.1)。alltoall_seq
#   overlap 族**自带一套 seq 版重叠**,不看这个开关:层无条件把 shared_experts 交给
#   `token_permutation`(moe_layer_overlap_all2allseq.py:59-70),dispatcher 在两条
#   async a2a 发出后、wait 之前执行它(overlap/token_dispatcher.py:179-206);反过来
#   该层**显式拒绝**这个开关(overlap/moe_layer.py:118-121 raise ValueError)。
#   ⇒「moe_shared_expert_overlap=False ⇒ 共享专家串行」的推断不成立;我们这条接缝
#   在同一段位回调 run_shared_experts,重叠**已经兑现**。判决床几何(16 节点/EP=128/
#   h=2048/d_shared=768/T=4096 tok/rank)下共享专家 GEMM 只有 6·T·h·d/F = 0.118 ms,
#   而 Hop A 是 0.726 ms(α₁₂₈ 0.45 + 线上 0.276)—— 可盖 16%,而且它是**微批内唯一**
#   与 dispatch 无数据依赖的计算,盖不满的 0.61 ms 只能靠减通信(A1′/A2/A4/K1/K2)。
#   完整论证与逐条替代方案:内部设计记录(未随仓发布)。


_ENV_SHARED_OVERLAP = "TERRACE_SHARED_OVERLAP"
_SHARED_OVERLAP: bool | None = None


def shared_overlap_enabled() -> bool:
    """18c 的 A/B 闸门(默认开)。=0 时共享专家挪到 dispatch 全部集合通信之后执行。

    为什么要这个闸门:重叠是**设备侧**的事(HCCL 在通信流、GEMM 在计算流),案头
    读不出来,而"我们相信它重叠了"不是读数。关掉它 = 同一批集合通信、同一顺序、
    同一数值,只把共享专家的 kernel 入队点挪到所有 dispatch 通信之后 —— 两臂步时差
    就是 18c 在这张床上的真实价值。位级安全:共享专家子图与 dispatch 无任何数据
    依赖,算子、操作数、内部次序全同,`torch.equal` 级相等
    (tests/test_ta2a_shared_overlap.py)。
    """
    global _SHARED_OVERLAP
    if _SHARED_OVERLAP is None:
        _SHARED_OVERLAP = os.environ.get(_ENV_SHARED_OVERLAP, "1").strip() != "0"
    return _SHARED_OVERLAP


def reset_shared_overlap() -> None:
    """忘掉缓存判定。测试/调试钩子,训练代码不得调用。"""
    global _SHARED_OVERLAP
    _SHARED_OVERLAP = None


def _a2a_async(x, in_splits, out_splits, group=None):
    """异步裸 all_to_all(Hop A 专用):返回 (out, handle),caller 负责 wait。

    放段外、不进 autograd —— 它的反向由厂商 backward 手工重放,建图反而是错的。
    异步是 18c 的载体:发出后先算共享专家,再 wait。
    """
    out = x.new_empty((sum(out_splits), *x.shape[1:]))
    handle = dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits,
                                    group=group, async_op=True)
    return out, handle


def ta2a_permute_overlap(hidden_states, probs, routing_map, *, world, rank, rpn,
                         n_experts, intra_group, inter_group=None, groups_m=None,
                         save_tensors, run_shared_experts=None):
    """overlap 6 参接缝的 dispatch 半边。

    与 `ta2a_permute`(legacy 3 参半边)同一套数学(plan/expansion/两跳),差别只有
    三件事,全部由厂商手写 backward 的契约决定(见文件尾部大注释):
      1. 段边界:rx/rgate 处 detach,Hop A 放段外异步发;
      2. save_tensors:按厂商固定位置追加 7 个条目;
      3. 共享专家回调在 Hop A 发出后、wait 前执行(18c 重叠免费获得;
         `TERRACE_SHARED_OVERLAP=0` 把它挪到全部 dispatch 集合通信之后,
         给床上量"重叠到底值多少"的对照臂,数值逐位不变)。

    返回 (permuted, tokens_per_expert, permuted_probs, share_experts_output, state)。
    caller(usercustomize 包装器)负责重排成厂商的 4 元组并设置 disp 上的 splits。
    """
    epr = n_experts // world
    slots = epr * rpn
    n_nodes = world // rpn
    dev = hidden_states.device
    my_local = rank % rpn
    H = hidden_states.shape[-1]
    pack_mode = _pk.pack_mode()       # 只读一次,理由见 legacy 半边同位注释
    packing = pack_mode != "off"
    overlap_shared = shared_overlap_enabled()   # 同上,热路径只读一次
    share_experts_output = None

    def _shared_now():
        """在当前位置执行共享专家回调(没配共享专家时返回 None)。

        回调本体(usercustomize._ta2a_overlap_seam_permute.run_shared)自带
        enable_grad,图直接根在厂商造好的 hidden_states 叶子上 —— 与厂商
        `forward_func(shared_experts, (hidden_states))` 同构(其 detach_tensor 对
        "已是 requires_grad 叶子"的入参原样返回,不新建叶子)。
        """
        return run_shared_experts() if run_shared_experts is not None else None

    # 整个函数体显式 enable_grad:接缝在 autograd.Function.forward 内被调,梯度默认
    # 是关的,而段图必须建起来 —— 厂商 forward_func 同款。漏了它不是报错,是
    # backward_func 见 grad_fn is None 直接 return,梯度整段静默消失。
    with torch.enable_grad():
        # ---- 段1(= 厂商 permute1 段位):纯本地,根在 hidden/probs 两个叶子上 ----
        expert_idx, gates = routing_map_to_topk(routing_map, probs)
        T, k = expert_idx.shape
        u_src, _, node_counts, inverse = plan_ta2a(
            expert_idx, world, n_experts, rpn, groups_m=groups_m)
        n_rows = int(u_src.numel())
        # A3-lite:同 legacy 半边 —— counts 交换提前发,用 gather + 打包盖它的 α₁₂₈。
        h_cnt = send = recv = None
        if async_counts_enabled():
            send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
            h_cnt = dist.all_to_all_single(recv, send, group=inter_group,
                                           async_op=True)
        h = hidden_states.view(-1, H)
        payload = h[u_src]
        if groups_m:
            assert k % groups_m == 0, f"k={k} not divisible by groups_m={groups_m}"
        quota = (k // groups_m) if groups_m else None
        # gate 一律 payload.dtype,与 legacy 半程同一圆整点 —— 见文件尾注释"gate
        # 精度"一条(一次内部提交 曾走 probs.dtype,eqov 对齐床证伪,2026-08-20)。
        # cast 与 gather/reshape 逐元素可交换,与 legacy 路径(厂商层在 dispatch
        # 上游把 probs 转 hidden dtype)逐位同值。图仍根在原精度的 probs 叶子上,
        # .to 的反向是精确 upcast,厂商编排不感知。probs 本就是 payload.dtype 时
        # .to 返回自身,不加图节点,逐位零变化。
        if quota is not None:
            # C1 quota 线格式:同 legacy 半程(见 ta2a_permute 该分支注释与
            # _pack_quota_wire 的逐位论证);cast 在打包前施加 == 旧稀疏平面在
            # index_put 处的同一圆整点。
            mask, gate_rows = _pack_quota_wire(
                expert_idx, gates.to(payload.dtype), inverse, payload,
                n_rows, slots, quota, n_experts, sorted_rows=True)
        else:
            # 同 legacy 半程:一次取模两平面共用;sorted_rows 由 routing_map_to_topk
            # 的升序构造担保(掩码位逐位不变,见 build_expansion)。
            slot_flat = expert_idx.reshape(-1) % slots
            mask = build_expansion(expert_idx, inverse, n_rows, world, n_experts,
                                   rpn, groups_m=groups_m, sorted_rows=True,
                                   slot_flat=slot_flat)
            gate_rows = payload.new_zeros(n_rows, slots)
            gate_rows[inverse, slot_flat] = gates.reshape(-1).to(payload.dtype)
        _dp.note("seam.payload", payload)
        _dp.note("seam.gate", gate_rows)
        _dp.note_int("seam.mask", mask)
        save_tensors.append(payload)      # ↔ permute1_graph
        save_tensors.append(gate_rows)    # ↔ permuted_probs_graph

        # ---- Hop A(段外,异步):厂商 backward 手工重放的就是这一跳的反向 ----
        if h_cnt is None:
            send, recv = _hopa_counts(world, n_nodes, rpn, my_local, dev, node_counts,
                                  inter_world=_group_world(inter_group))
            dist.all_to_all_single(recv, send, group=inter_group)
        else:
            h_cnt.wait()
        send_l, recv_l = _splits_to_lists(send, recv)      # 一次同步,不是两次

        # mask 是任一线格式下的 id 平面(位掩码或 C1 槽号表),见 legacy 半程同位注释。
        if pack_mode == "small":
            # A1''(默认形态,理由见 legacy 半边同位注释与
            # 内部实测记录):只并 id ‖ gate,载荷自己走一条。
            # 两条都用异步发,共同盖住共享专家那一段 —— 与 full 形态的重叠窗口等长。
            with torch.no_grad():
                _sbuf, _lay = _pk.hopa_pack_small(gate_rows.detach(), mask)
            rx, h_rx = _a2a_async(payload.detach(), send_l, recv_l, group=inter_group)
            _rbuf, h_sm = _pk.hopa_exchange_async(_sbuf, send_l, recv_l,
                                                  group=inter_group)
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            h_sm.wait()
            with torch.no_grad():
                rgate, rmask = _pk.hopa_unpack_small(_rbuf, _lay)
            _pk.assert_not_aliased(_rbuf, rgate, rmask)
            _sbuf.untyped_storage().resize_(0)
            _rbuf.untyped_storage().resize_(0)
        elif pack_mode == "full":
            # A1′:三条并成一条。Hop A 在段外、不进 autograd(厂商 backward 手工
            # 重放),所以这里只并**数据**;重放按 disp.input_splits/output_splits
            # = send_l/recv_l 走,而那两份仍是**行数**(并包缓冲的 dim 0 就是 n_rows,
            # 不需要任何缩放),语义一字未变 ⇒ 厂商那两次重放(rx_d.grad /
            # rgate_d.grad)照跑不误,仍各是一次 a2a。
            # **判决床实测净亏 2.05 ms/次**,保留只为可复现与 A/B,不再是默认。
            with torch.no_grad():
                _sbuf, _lay = _pk.hopa_pack(payload.detach(), gate_rows.detach(), mask)
            _rbuf, h_rx = _pk.hopa_exchange_async(_sbuf, send_l, recv_l,
                                                  group=inter_group)
            # 18c:Hop A 在飞,此刻算共享专家 —— 与厂商 token_permutation 同段位。
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            with torch.no_grad():
                rx, rgate, rmask = _pk.hopa_unpack(_rbuf, _lay)
            # 还内存之前先确认三个平面不再指着缓冲。2026-08-21 判决床 rank61 就死在这:
            # R==1 时 `.contiguous()` 是空操作,解包结果是 _rbuf 的视图,resize(0)
            # 之后它们悬空,直到十几个算子之后 `owner = slot_idx // epr` 才炸
            # ("non-zero number of elements, but its data is not allocated yet")。
            # 已在 ta2a_pack._own 处根治;这三次指针比较是防复发的绊线,热路径可忽略。
            _pk.assert_not_aliased(_rbuf, rx, rgate, rmask)
            _sbuf.untyped_storage().resize_(0)      # 已上线,收侧已解包 —— 立刻还内存
            _rbuf.untyped_storage().resize_(0)
        else:
            rx, h_rx = _a2a_async(payload.detach(), send_l, recv_l, group=inter_group)
            rmask, h_rm = _a2a_async(mask, send_l, recv_l, group=inter_group)
            rgate, h_rg = _a2a_async(gate_rows.detach(), send_l, recv_l,
                                     group=inter_group)
            if overlap_shared:
                share_experts_output = _shared_now()
            h_rx.wait()
            h_rm.wait()
            h_rg.wait()
        _dp.note("seam.rx", rx)
        _dp.note("seam.rgate", rgate)
        _dp.note_int("seam.rmask", rmask)
        # 数据已交给 fabric,图的 backward(gather/index_put)只要索引不要数据 ——
        # 照厂商同款释放存储(save_tensors 里只留图和 .grad 位)。
        payload.untyped_storage().resize_(0)
        gate_rows.untyped_storage().resize_(0)

        # ---- 段2(= 厂商 permute2 段位):根在 rx_d/rgate_d 两个 detach 叶子上 ----
        rx_d = rx.detach().requires_grad_(True)
        rgate_d = rgate.detach().requires_grad_(True)
        R = rx.shape[0]
        # K1(AscendC kernel,2026-08-20 第二接:overlap 6 参接缝)。同 legacy 半边
        # 的同一枚 kernel、同一套数学(论证见 ta2a_permute 同位注释与
        # terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp 文件头),差别只有它
        # 落在 permute2 段图**内部**:入参是 detach 叶 rx_d/rgate_d,所以走段图版
        # k1_arrival_segment —— kernel 跑一次出数据,exp_rx / gate_pairs 各挂一条
        # 独立的边回自己的叶子(融合成一个节点会让厂商 gmm 的两次 .backward()
        # 撞车,理由见 terrace/ops/__init__.py::_K1SendEdge)。r_idx/slot_idx/i_send
        # 是整数索引/计数平面,不参与梯度。席位契约(7+3)、detach 边界、splits
        # 交接、厂商手写反向可见的一切均不变。
        # else 分支是 K1 前的现链**原文**,kernel 缺席时(TERRACE_CUSTOM_OPS=0 /
        # 无 .so)是唯一路径:零行为变化。exp_rx/gate_pairs 提到集合通信之前,只为
        # 两分支以同五张量进 Hop B —— 纯数据流重排,算子与操作数不变。
        _h_i = _i_recv_early = None      # A6 的句柄;K1 分支不提前发,保持 None
        if quota is not None and _tops.custom_ops_enabled():
            exp_rx, gate_pairs, r_idx, slot_idx, i_send = _tops.k1_arrival_segment(
                rx_d, rmask, rgate_d, quota, epr, rpn, my_local)
        else:
            if quota is not None:
                r_idx, slot_idx = _expand_arrival_quota(rmask)  # 槽号表即配对表
            else:
                r_idx, slot_idx = _expand_arrival(rmask, slots, quota)
            owner = slot_idx // epr
            # A6:同 legacy 半边 —— 直方图对置换盲,算完立刻异步发 Hop B 的 counts,
            # 用下面的排序与两次 gather 盖住。纯调度重排,逐位不变。
            i_send = fixed_hist(owner, rpn)          # 定长直方图,避开 bincount 的隐藏主机同步
            if sync_probe_enabled():
                _ = i_send.tolist()   # 判别探针:纯粹丢弃,只为付一次主机同步(见 sync_probe_enabled)
            if early_hopb_counts_enabled():
                _i_recv_early = torch.empty_like(i_send)
                _h_i = dist.all_to_all_single(_i_recv_early, i_send, group=intra_group,
                                              async_op=True)
            ordo = _stable_argsort_small(owner, rpn)
            r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
            exp_rx = rx_d[r_idx]
            # C1 快路径:致密 gate 表与配对枚举同构,二维 gather 退化为平铺 gather
            # (逐位相等的论证见 ta2a_permute 同位注释)。梯度经 view 反向落回
            # [R, quota] 的 rgate_d.grad,厂商按行数 splits 重放,对布局变窄无感知。
            gate_pairs = (rgate_d.reshape(-1)[ordo] if quota is not None
                          else rgate_d[r_idx, slot_idx])
        _dp.note("seam.exprx", exp_rx)
        _dp.note("seam.gpairs", gate_pairs)
        _dp.note_int("seam.slot", slot_idx)
        _dp.note_int("seam.isend", i_send)
        if _h_i is not None:
            _h_i.wait()                  # A6:上面已异步发过,这里只等它落地
            i_recv = _i_recv_early
        else:
            i_recv = torch.empty_like(i_send)
            dist.all_to_all_single(i_recv, i_send, group=intra_group)
        is_l, ir_l = _splits_to_lists(i_send, i_recv)

        # A2:槽号面(int64,无梯度)与 gate 面并成一条,占据槽号交换的原位置;
        # 载荷面 exp_rx 有意**不并** —— 并进来要多付两趟 100 MB 级 HBM 拷贝
        # (+0.317 ms)去省 1α₈(0.058 ms),判决床上净亏 0.20 ms(算式与零拷贝后续件
        # 见 terrace/ta2a_pack.py 的 Hop B 一节)。gate 路仍挂**自己的一条**独立边:
        # 厂商 gmm 手写 backward 对 permute2_graph / permute2_prob_graph 分两次
        # .backward() 进本段,任何把两路焊进一枚节点的写法都会撞 "backward a second
        # time",且第一次会把零梯度写进另一路(内部工程记录 2026-08-20 / _K1SendEdge)。
        # else 分支是并包前的现链**原文**(含集合通信的先后次序):闸关零行为变化。
        if packing:
            with torch.no_grad():
                _bbuf = _pk.hopb_pack_meta(slot_idx, gate_pairs.detach())
                _rbb = _a2a_raw(_bbuf, is_l, ir_l, group=intra_group)
                my_slot, _mg_raw = _pk.hopb_unpack_meta(_rbb, gate_pairs.dtype)
            _bbuf.untyped_storage().resize_(0)
            _rbb.untyped_storage().resize_(0)
        else:
            # 元数据(int,无梯度)先走裸 a2a。
            my_slot = _a2a_raw(slot_idx, is_l, ir_l, group=intra_group)
        exp_j = my_slot - my_local * epr
        order = _stable_argsort_small(exp_j, epr)
        tokens_per_expert = fixed_hist(exp_j, epr)                 # 置换盲,免 gather

        # Hop B 的两条 float 路走可微 _A2A / _PackedEdge,留在段内,厂商对
        # permute2_(prob_)graph 调 .backward() 时 autograd 自动反跑。
        node_rx = _a2a(exp_rx, is_l, ir_l, group=intra_group)
        # 同 legacy 半程:my_gate 交换是 permuted_probs 的交付,Step 2 不适用。此处
        # 更硬:厂商 gmm 手写反向按固定席位对 permute2_prob_graph 调 backward 并沿
        # Hop A 重放 rgate_d.grad,门控梯度必须从专家 rank 经这条链回来。
        my_gate = (_pk.attach_edge(gate_pairs, _mg_raw, is_l, ir_l, intra_group)
                   if packing
                   else _a2a(gate_pairs, is_l, ir_l, group=intra_group))
        permuted = node_rx[order]
        pprobs = my_gate[order]
        _dp.note("seam.permuted", permuted)
        _dp.note("seam.pprobs", pprobs)
        _dp.note_int("seam.tpe", tokens_per_expert)
        # 落地副本的数据段内已消费完(gather 的 backward 只要 r_idx);释放。
        rx.untyped_storage().resize_(0)
        rgate.untyped_storage().resize_(0)

        # 固定位置契约的后 5 席。第 3 席在 zm=disable 下厂商 backward 只携带不使用,
        # 我们没有那个语义(它是厂商 permute2 的 [ep, num_local] 分块表),给 None:
        # 万一哪条未闸的厂商路径真去用它,None 会当场 AttributeError —— 比一个形状
        # 相近的假表安静地算错要好。
        save_tensors.append(None)         # ↔ num_global_tokens_per_local_expert_cpu
        save_tensors.append(rx_d)         # ↔ permute2_input_detach
        save_tensors.append(permuted)     # ↔ permute2_graph
        save_tensors.append(rgate_d)      # ↔ permute2_prob_detach
        save_tensors.append(pprobs)       # ↔ permute2_prob_graph

        # 18c 对照臂:闸门关时才走这里。位置刻意选在 dispatch **全部**集合通信
        # 之后 —— 此后的临界路径只剩专家 GEMM(计算),没有通信可被它遮住,才是
        # 真正的"串行"基线;若只挪到 Hop A 的 wait 之后,它仍可能被 Hop B 盖掉
        # 一部分,读数会低估重叠的价值。席位契约不受影响(共享专家不占席位),
        # 数值逐位不变(与 dispatch 无数据依赖)。
        if not overlap_shared:
            share_experts_output = _shared_now()

    st = TA2AState()
    st.u_src, st.r_idx, st.order = u_src, r_idx, order
    st.R, st.T, st.hidden = R, h.shape[0], H
    st.send_l, st.recv_l, st.is_l, st.ir_l = send_l, recv_l, is_l, ir_l
    st.intra, st.inter, st.dtype = intra_group, inter_group, hidden_states.dtype
    return permuted, tokens_per_expert, pprobs, share_experts_output, st


def ta2a_unpermute_overlap(expert_out, st: TA2AState, save_tensors, out_shape=None):
    """overlap 6 参接缝的 combine 半边。返回**单张量**(厂商此接缝的契约)。

    数学与 `ta2a_unpermute` 相同;段边界:Hop B 反向(可微)藏进 unpermute1 段,
    Hop A 反向放段外(back 处 detach),由厂商 moe 层 backward 用
    (output_splits, input_splits) 手工重放。追加固定位置契约的 3 席。
    """
    with torch.enable_grad():
        # ---- 段(= 厂商 unpermute1 段位):根在 eo_d ----
        eo_d = expert_out.detach().requires_grad_(True)
        back_pairs = eo_d.new_empty(eo_d.shape)
        back_pairs[st.order] = eo_d
        ret = _a2a(back_pairs, st.ir_l, st.is_l, group=st.intra)   # Hop B 反向,可微
        red = ret.new_zeros(st.R, ret.shape[1])
        red.index_add_(0, st.r_idx, ret)
        _dp.note("seam.eout", expert_out)
        _dp.note("seam.ret", ret)
        _dp.note("seam.red", red)
        _dp.check_reduction("seam.red", red, ret, st.r_idx, st.R, _per(ret, st.R))
        save_tensors.append(eo_d)         # ↔ unpermute1_input_detach
        save_tensors.append(red)          # ↔ unpermute1_graph
        eo_d.untyped_storage().resize_(0)  # 厂商同款:专家输出数据不再需要,图仍完整

        # ---- Hop A 反向(段外,同步):反向对 = (recv, send) ----
        back = _a2a_raw(red.detach(), st.recv_l, st.send_l, group=st.inter)
        red.untyped_storage().resize_(0)

        # ---- 段(= 厂商 unpermute2 段位):根在 back_d ----
        back_d = back.requires_grad_(True)   # _a2a_raw 出来的就是叶子
        y = back_d.new_zeros(st.T, st.hidden).index_add(0, st.u_src, back_d)
        _dp.note("seam.back", back_d)
        _dp.note("seam.y", y)
        _dp.check_reduction("seam.y", y, back_d, st.u_src, st.T, _per(back_d, st.T))
        if out_shape is not None:
            y = y.view(out_shape)
        save_tensors.append(back_d)       # ↔ unpermute2_input_detach
        back_d.untyped_storage().resize_(0)
    return y


def ta2a_enabled(ep_world: int, rpn: int = 8) -> bool:
    """EP 组跨不跨节点。不跨(EP<=rpn)就没有 fabric 跳,T-A2A 按构造是 no-op。"""
    if os.environ.get("TERRACE_TA2A") != "1":
        return False
    return ep_world // rpn >= 2
