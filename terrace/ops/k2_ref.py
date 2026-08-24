"""K2 的 CPU/纯 torch 参考实现 —— kernel 语义的可执行规格,与现链逐位同。

独立文件、只依赖 torch(**不 import terrace.ta2a / ta2a_fwd**):它是 AscendC
kernel(ascendc/op_kernel/terrace_k2_pack.cpp)两遍计数排序的直译,不是现链的
转写 —— 这样 tests/test_terrace_k2_ref.py 里「参考 == 现链
(plan_ta2a 快路径 + 去重 gather + _pack_quota_wire)」的逐位对账才是一条
有信息量的证明,而不是同一段代码自己对自己。K1 的参考(k1_arrival_ref,在
terrace/ops/__init__.py 里)复用了现链原语;K2 的参考按任务要求独立成文件,
嫁接时 __init__.py 只需 `from .k2_ref import k2_pack_ref`(内部嫁接记录(未随仓发布))。

复刻的现链段(quota 快路径,等额配额 groups_m=M;三处调用点同段同构:
ta2a_fwd.ta2a_moe_forward / ta2a_dispatch.ta2a_permute[_overlap]):

    u_src, u_node, node_counts, inverse = plan_ta2a(expert_idx, world,
                                                    n_experts, rpn, groups_m=M)
    payload = hidden[u_src]
    mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                       n_rows, slots, quota, n_experts)

逐位一致的论证(全文见 kernel 文件头,这里给可执行的形态):

  - plan 的行序 = 占用表节点主序平铺后置位的升序位置,即 (node 升序, token
    升序);等额配额下 token 的 k 个专家升序后恰分成 M 段(run),每段 quota 个、
    整段同节点、段间节点严格升序 —— run 按 (t, j) 枚举、按目的节点稳定排序,
    得到的 dst 置换与 plan 的 sel 枚举逐元素相等(每 token 每节点至多一个 run,
    桶内序 == token 升序)。这里的 argsort(dest, stable=True) 就是 kernel
    两遍法(直方图 -> 前缀游标 -> 顺序落位)的同一个数学对象。
  - 行内排序:专家号行内互异 => 升序置换唯一,torch.sort 与 kernel 的插入排序、
    现链 _pack_quota_wire(sorted_rows=False) 的 float32 键 argsort 给出同一个
    置换;行本就升序时(接缝入口,routing_map_to_topk 按构造)排序恒等,故与
    sorted_rows=True 的免排序分支也逐位同。
  - gate 全程纯搬运(gather/reshape,无算术),与现链的 gather 逐位同。

契约(与 torch 侧 csrc/kernel 的 TORCH_CHECK 一致,fail loud):
  - gates.dtype == hidden.dtype:C1 圆整点契约 —— gate 平面从 payload 派生,
    失配必须大声死(_pack_quota_wire 在 index_put 处的同一失效形态;一次内部提交
    漂移缺陷的进场口),调用方持有圆整点(overlap 接缝在打包前 cast)。
  - 专家号越界直接 raise:现链在 plan 的 scatter/gather 处大声死,参考实现
    同样不静默。kernel 侧不能 raise,以「跳过写 + zeros 收容」代之(不承诺
    损坏输入的位级,见 kernel 文件头「损坏输入的收容」)。
  - 等额配额不变量(每 token 恰 M 个节点、每节点恰 quota 个专家)由调用方
    保证(plan_ta2a 首调 + 每 256 调验证);参考实现在不变量下与现链逐位同,
    漂移输入下与现链一样不承诺(现链 searchsorted 越界大声死)。
"""
from __future__ import annotations

import torch


def k2_pack_ref(hidden: torch.Tensor, expert_idx: torch.Tensor, gates: torch.Tensor,
                world: int, n_experts: int, rpn: int, groups_m: int):
    """发送侧融合打包链的参考实现:(payload, mask, gate_rows, u_src, node_counts)。

    hidden [T, H];expert_idx [T, k] int64(行内互异,无升序要求);gates [T, k]
    与 hidden 同 dtype。返回 payload [T*M, H]、mask [T*M, quota](升序槽号表,
    int64)、gate_rows [T*M, quota]、u_src [T*M] int64、node_counts [n_nodes]
    int64。可微性与现链一致:payload(hidden 的去重 gather)与 gate_rows
    (gates 的置换 gather)载梯度,mask/u_src/node_counts 是索引/计数平面。
    """
    if expert_idx.dim() != 2 or hidden.dim() != 2 or gates.dim() != 2:
        raise ValueError("k2_pack_ref: hidden/expert_idx/gates must be 2-D")
    T, k = expert_idx.shape
    if hidden.shape[0] != T or gates.shape != expert_idx.shape:
        raise ValueError(
            f"k2_pack_ref: geometry mismatch (hidden {tuple(hidden.shape)}, "
            f"expert_idx {tuple(expert_idx.shape)}, gates {tuple(gates.shape)})")
    if gates.dtype != hidden.dtype:
        # RuntimeError,与 _pack_quota_wire 的 index_put 失效形态同类同响。
        raise RuntimeError(
            f"k2_pack_ref: gates dtype {gates.dtype} != hidden dtype {hidden.dtype}"
            f" -- the C1 gate plane derives from the payload; cast at the caller"
            f" (the caller owns the rounding point)")
    if world <= 0 or rpn <= 0 or groups_m <= 0 or n_experts <= 0:
        raise ValueError("k2_pack_ref: bad geometry scalars")
    if world % rpn or n_experts % world or k % groups_m:
        raise ValueError(
            f"k2_pack_ref: world={world} rpn={rpn} n_experts={n_experts} k={k} "
            f"groups_m={groups_m} not divisible (same asserts as the live chain)")
    epr = n_experts // world
    slots = epr * rpn
    n_nodes = world // rpn
    quota = k // groups_m
    if slots > 63:
        raise ValueError(f"{slots} expert slots per node exceeds the chain bound")
    if T and (int(expert_idx.min()) < 0 or int(expert_idx.max()) >= n_experts):
        raise ValueError("k2_pack_ref: expert id out of range (the live chain dies "
                         "loudly in plan_ta2a's scatter; so does the reference)")

    dev = expert_idx.device
    P = T * groups_m                                 # 发送行数,静态(免同步的本钱)

    # ---- 行内升序(kernel 第 2 遍的插入排序;行互异 => 置换唯一,已升序则恒等)
    e_sorted, order = torch.sort(expert_idx, dim=1, stable=True)
    g_sorted = torch.gather(gates, 1, order)

    # ---- run 枚举(t 升序、j 升序)与目的节点;稳定计数排序 == plan 的行序 ----
    dest = torch.div(e_sorted[:, ::quota], slots,
                     rounding_mode="floor").reshape(-1)              # [T*M]
    sigma = torch.argsort(dest, stable=True)                         # 行 -> run
    node_counts = torch.bincount(dest, minlength=n_nodes)            # run 直方图

    # ---- 五个输出:全部是 sigma 的 gather(kernel 里是游标顺序落位,同一置换)
    u_src = torch.div(sigma, groups_m, rounding_mode="floor")        # run 的 token
    payload = hidden[u_src]
    mask = (e_sorted % slots).reshape(P, quota)[sigma] if P else \
        torch.zeros(0, quota, dtype=torch.int64, device=dev)
    gate_rows = g_sorted.reshape(P, quota)[sigma] if P else \
        hidden.new_zeros(0, quota)
    return payload, mask, gate_rows, u_src, node_counts
