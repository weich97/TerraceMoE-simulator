"""A3-lite(2026-08-21):Hop A 的 counts 交换提前发、异步等 —— 位级零变化 + 活路径。

Why this file exists:A3(计划常量化,干掉两条 counts 交换)已判**净亏 −7.03 ms
不实施**(内部设计记录(未随仓发布):等配额只固定每 token 扇出 M、不固定
每目的地负载,容量填充的上界只能取几何最坏值 n_nodes/M = 8x 与 rpn = 8x,冗余字节
远超省下的 α;另有 tokens_per_expert 硬探针与 fail-loud 溢出两条结构性阻塞)。
**唯一可救的残片**就是本件:inter 的 counts 交换只依赖 plan_ta2a 交出的 node_counts,
不必等本地 gather + 打包做完 —— 提前到 plan 之后 `async_op=True` 发出,wait 挪到
真正要用 splits 的地方,用本地工作去盖它的 α₁₂₈。

**它必须什么都不改**:同一批集合通信、同一个顺序(inter 组上它仍是第一条)、同样的
splits、同样的数值。所以判据是:
  1. **位级**:开/关两臂同进程 A/B,两条接缝的前向、四路叶子梯度、厂商手工重放要读的
     四个 detach 叶 `.grad`,全部 `torch.equal`;
  2. **条数不变**:异步只改 issue/wait 的位置,不改条数 —— 两臂 fwd/bwd 逐项相等
     (绝对值归 test_ta2a_gate_at_arrival 把守,本文件锁「两臂相等」);
  3. **splits 不变**:`st.send_l / st.recv_l`(= 厂商 disp.input_splits/output_splits)
     两臂逐项相等 —— 提前发不许改交给厂商的那两份;
  4. **活路径**:开臂必须真的走了 `async_op=True`(记 counts 交换的 async 标志),
     关臂必须一次都没有。等价断言对「闸门从未打开」是盲的(内部工程记录)。

单独立文件、单独一个 commit:床上要能把 A3-lite 的读数与 A1′/A2 并包的读数分开归因
(闸门 TERRACE_TA2A_ASYNC_COUNTS=0 即回到原位置同步发)。

gloo / CPU,world 4、rpn 2(2 节点,有真实跨节点跳)。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import terrace.ta2a_dispatch as disp_mod  # noqa: E402

WORLD, RPN, T, K, M, E, H, D = 4, 2, 8, 4, 2, 8, 6, 4


def test_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """闸门:未设 / 非 "0" = 开(出厂配置);"0" = 关,回到原位置同步发。"""
    monkeypatch.delenv("TERRACE_TA2A_ASYNC_COUNTS", raising=False)
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_ASYNC_COUNTS", "0")
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_ASYNC_COUNTS", "1")
    disp_mod.reset_async_counts()
    assert disp_mod.async_counts_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_ASYNC_COUNTS", raising=False)
    disp_mod.reset_async_counts()


def test_counts_buffers_are_geometry_only():
    """_hopa_counts 只依赖几何 + node_counts:提前发之所以合法,全部理由在这一条。

    它不读 payload、不读 mask、不读 gate —— 所以把它挪到 plan_ta2a 之后、打包之前,
    不可能读到还没算出来的东西。
    """
    dev = torch.device("cpu")
    n_nodes = WORLD // RPN
    node_counts = torch.tensor([3, 5], dtype=torch.long)
    send, recv = disp_mod._hopa_counts(WORLD, n_nodes, RPN, 1, dev, node_counts)
    assert send.shape == (WORLD,) and recv.shape == (WORLD,)
    assert send.dtype == torch.long and recv.dtype == torch.long
    # 节点 n 的指定对端 = n*rpn + my_local,其余位置恒为 0
    assert send.tolist() == [0, 3, 0, 5]
    assert int(send.sum()) == int(node_counts.sum())


def _routing(gen, n_nodes, per, quota):
    rows = []
    for _ in range(T):
        gs = torch.randperm(n_nodes, generator=gen)[:M]
        rows.append(torch.cat([torch.arange(a * per, (a + 1) * per)[
            torch.randperm(per, generator=gen)[:quota]] for a in gs]))
    return torch.stack(rows)


def _bitdiff(a, b):
    if a is None or b is None:
        return ("missing", None)
    if a.dtype != b.dtype:
        return (f"{a.dtype} vs {b.dtype}", float((a.float() - b.float()).abs().max()))
    if torch.equal(a, b):
        return None
    return (str(a.dtype), float((a.float() - b.float()).abs().max()))


def _run(rank, world, q, dtype_name):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 已占用:29577/29591/29613/29623/29627/29641/29645/29661/29665/29677/
    #         29681/29685/29697/29709/29713。
    os.environ.setdefault(
        "MASTER_PORT", "29725" if dtype_name == "float32" else "29729")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import terrace.ta2a_dispatch as dm
        from terrace.layer import grouped_mm
        from terrace.ep_dist import _a2a_raw
        from terrace.ta2a_fwd import init_ta2a_groups
        from terrace.ta2a_dispatch import (ta2a_permute, ta2a_unpermute,
                                           ta2a_permute_overlap,
                                           ta2a_unpermute_overlap)

        pdt = getattr(torch, dtype_name)
        intra = init_ta2a_groups(world, RPN)
        epr = E // world
        n_nodes, per, quota = world // RPN, E // (world // RPN), K // M

        real_a2a = dist.all_to_all_single
        counter, seen = [0], {"async": 0, "sync": 0}
        real_switch = dm.async_counts_enabled

        def counting_a2a(*args, **kwargs):
            counter[0] += 1
            # counts 交换的签名:1 维 long、长度 = 组大小。记它是不是异步发的。
            x = args[1]
            if x.dim() == 1 and x.dtype == torch.long and x.shape[0] in (world, RPN):
                seen["async" if kwargs.get("async_op") else "sync"] += 1
            return real_a2a(*args, **kwargs)

        real_early = dm.early_hopb_counts_enabled

        def one_pass(async_on, gm):
            # **两个闸门一起翻**:A3-lite(inter counts 提前)与 A6(intra counts 提前)
            # 是同一个机制的两处应用。对照臂只关 A3 的话,A6 的两次异步还在,
            # "同步臂 async==0" 当场不成立,而那种失败长得像 A6 坏了,其实是对照臂没关干净。
            dm.async_counts_enabled = (lambda: True) if async_on else (lambda: False)
            dm.early_hopb_counts_enabled = (lambda: True) if async_on else (lambda: False)
            g = torch.Generator().manual_seed(29 + rank)
            x0 = torch.randn(T, H, generator=g).to(pdt)
            idx = _routing(g, n_nodes, per, quota)
            gates0 = torch.rand(T, K, generator=g)          # fp32:router 输出精度
            w13_0 = (torch.randn(epr, H, 2 * D, generator=g) / (H ** 0.5)).to(pdt)
            w2_0 = (torch.randn(epr, D, H, generator=g) / (D ** 0.5)).to(pdt)
            G = torch.randn(T, H, generator=g).to(pdt)
            routing_map = torch.zeros(T, E, dtype=torch.bool)
            routing_map[torch.arange(T).unsqueeze(1), idx] = True
            probs_dense = torch.zeros(T, E)
            probs_dense[torch.arange(T).unsqueeze(1), idx] = gates0

            dist.all_to_all_single = counting_a2a
            try:
                hidL = x0.clone().requires_grad_(True)
                probL = probs_dense.clone().to(pdt).requires_grad_(True)
                w13L = w13_0.clone().requires_grad_(True)
                w2L = w2_0.clone().requires_grad_(True)
                counter[0] = 0
                permL, tpeL, ppL, stL = ta2a_permute(
                    hidL, probL, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm)
                a, b = grouped_mm(permL, w13L, tpeL).chunk(2, dim=-1)
                eoL = grouped_mm(F.silu(a) * b, w2L, tpeL) * ppL.unsqueeze(-1)
                outL = ta2a_unpermute(eoL, stL, hidL)
                fwd = counter[0]
                counter[0] = 0
                outL.backward(G)
                bwd = counter[0]

                hidO = x0.clone().requires_grad_(True)
                probO = probs_dense.clone().requires_grad_(True)
                save = []
                permO, tpeO, ppO, _share, stO = ta2a_permute_overlap(
                    hidO, probO, routing_map, world=world, rank=rank, rpn=RPN,
                    n_experts=E, intra_group=intra, inter_group=None, groups_m=gm,
                    save_tensors=save, run_shared_experts=None)
                w13O = w13_0.clone().requires_grad_(True)
                w2O = w2_0.clone().requires_grad_(True)
                dI = permO.detach().requires_grad_(True)
                pI = ppO.detach().requires_grad_(True)
                a, b = grouped_mm(dI, w13O, tpeO).chunk(2, dim=-1)
                eoO = grouped_mm(F.silu(a) * b, w2O, tpeO) * pI.unsqueeze(-1)
                outO = ta2a_unpermute_overlap(eoO, stO, save)
                (p1g, p1p, _n, p2ind, p2g, p2pd, p2pg, u1ind, u1g, u2ind) = save

                outO.backward(G)
                grad_red = _a2a_raw(u2ind.grad, stO.send_l, stO.recv_l)
                u1g.backward(grad_red)
                eoO.backward(u1ind.grad)
                p2pg.backward(pI.grad)
                ggr = _a2a_raw(p2pd.grad, stO.recv_l, stO.send_l)
                p2g.backward(dI.grad)
                gpl = _a2a_raw(p2ind.grad, stO.recv_l, stO.send_l)
                torch.autograd.backward([p1g, p1p], grad_tensors=[gpl, ggr])
            finally:
                dist.all_to_all_single = real_a2a

            return {
                "L.out": outL.detach(), "L.gx": hidL.grad, "L.gp": probL.grad,
                "L.gw13": w13L.grad, "L.gw2": w2L.grad, "L.perm": permL.detach(),
                "O.out": outO.detach(), "O.gx": hidO.grad, "O.gp": probO.grad,
                "O.gw13": w13O.grad, "O.gw2": w2O.grad, "O.perm": permO.detach(),
                "O.p2in": p2ind.grad, "O.p2prob": p2pd.grad,
                "O.u1in": u1ind.grad, "O.u2in": u2ind.grad,
                "_counts": (fwd, bwd),
                "_splits": (list(stO.send_l), list(stO.recv_l),
                            list(stL.send_l), list(stL.recv_l)),
            }

        report = {}
        try:
            for gm in (M, None):
                seen["async"] = seen["sync"] = 0
                sync = one_pass(async_on=False, gm=gm)
                seen_sync = dict(seen)
                seen["async"] = seen["sync"] = 0
                asyn = one_pass(async_on=True, gm=gm)
                seen_async = dict(seen)
                diffs = {n: _bitdiff(asyn[n], sync[n])
                         for n in asyn if not n.startswith("_")}
                report[f"gm={gm}"] = {
                    "diffs": diffs,
                    "counts_same": asyn["_counts"] == sync["_counts"],
                    "counts": (asyn["_counts"], sync["_counts"]),
                    "splits_same": asyn["_splits"] == sync["_splits"],
                    "seen_async": seen_async, "seen_sync": seen_sync,
                }
        finally:
            dm.async_counts_enabled = real_switch
        dm.early_hopb_counts_enabled = real_early

        q.put({"rank": rank, "status": "ok", "report": report})
    except Exception:                                      # noqa: BLE001
        import traceback
        q.put({"rank": rank, "status": "err", "trace": traceback.format_exc()})
    finally:
        dist.destroy_process_group()


def _spawn(dtype_name):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run, args=(r, WORLD, q, dtype_name))
             for r in range(WORLD)]
    for p in procs:
        p.start()
    out = [q.get(timeout=240) for _ in range(WORLD)]
    for p in procs:
        p.join(timeout=60)
    for r in out:
        assert r["status"] == "ok", f"rank {r['rank']}:\n{r.get('trace')}"
    return out


@pytest.fixture(scope="module")
def ac_fp32():
    return _spawn("float32")


@pytest.fixture(scope="module")
def ac_bf16():
    return _spawn("bfloat16")


def _assert_bitwise(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            for name, d in rep["diffs"].items():
                assert d is None, (
                    f"rank {r['rank']} {label} {tag}: {name} 异步臂与同步臂不逐位"
                    f"相等 {d} —— A3-lite 只挪 issue/wait 的位置,不许改任何数值")


def _assert_contract(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            assert rep["counts_same"], (
                f"rank {r['rank']} {label} {tag}: 两臂集合通信条数不等 "
                f"{rep['counts']} —— 异步化不改条数,只改 issue/wait 的位置")
            assert rep["splits_same"], (
                f"rank {r['rank']} {label} {tag}: st.send_l/recv_l 两臂不同 —— "
                f"那两份就是交给厂商 disp.input_splits/output_splits 的行数")


def _assert_live(results, label):
    for r in results:
        for tag, rep in r["report"].items():
            sa, ss = rep["seen_async"], rep["seen_sync"]
            # 每条接缝两次 counts 交换(inter + intra),两条接缝共 4 次。
            # **2026-08-22 起全部异步**:A3-lite 管 inter 那两次,A6 管 intra 那两次。
            # A6 的依据见 内部实测记录 —— 五档归因证明
            # "少发一条"价值为零(A1'' 精确 0),只有"挪到能被真实计算盖住的位置"
            # 才有收益(A3-lite −0.34 ms/次,full/small 两档一致)。
            # A6 把 intra 的 counts 提到排序与 [pairs,H] 大 gather 之前,用它们做遮蔽。
            assert sa["async"] == 4 and sa["sync"] == 0, (
                f"rank {r['rank']} {label} {tag}: 异步臂 counts 交换 {sa},预期 "
                f"async=4(两条接缝 x inter+intra)/ sync=0 —— "
                f"某个闸门没打开或提前发没接上(等价断言对这个是盲的,内部工程记录)")
            assert ss["async"] == 0 and ss["sync"] == 4, (
                f"rank {r['rank']} {label} {tag}: 同步臂 counts 交换 {ss},预期 "
                f"全同步 —— A/B 的对照臂失守")


@pytest.mark.timeout(300)
def test_fp32_async_counts_is_bitwise(ac_fp32):
    """fp32:两条接缝的前向、四路叶子梯度、四个 detach 叶 .grad,异步 == 同步。"""
    _assert_bitwise(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_async_counts_keeps_contract(ac_fp32):
    _assert_contract(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_fp32_async_counts_is_live(ac_fp32):
    _assert_live(ac_fp32, "fp32")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_is_bitwise(ac_bf16):
    """床口径(bf16 载荷/权重 + fp32 router probs):同上,逐位。"""
    _assert_bitwise(ac_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_keeps_contract(ac_bf16):
    _assert_contract(ac_bf16, "bf16")


@pytest.mark.timeout(300)
def test_bed_dtype_async_counts_is_live(ac_bf16):
    _assert_live(ac_bf16, "bf16")


def test_this_file_compiles_and_is_lf():
    import py_compile
    py_compile.compile(__file__, doraise=True)
    with open(__file__, "rb") as f:
        assert b"\r\n" not in f.read()


def test_early_hopb_switch_defaults_on_and_can_be_turned_off(monkeypatch):
    """A6 闸门(TERRACE_TA2A_EARLY_HOPB)的三态自证。

    默认开 —— 它是**纯调度重排**:i_send 的值、alltoall 语义、i_recv 内容、
    下游每一个比特都不变,只有发起点提前了。不像 A1'/A5 那样动布局或归约序,
    不需要 eq 门。留闸门只为床上做 A/B。
    """
    monkeypatch.delenv("TERRACE_TA2A_EARLY_HOPB", raising=False)
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is True
    monkeypatch.setenv("TERRACE_TA2A_EARLY_HOPB", "0")
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is False
    monkeypatch.setenv("TERRACE_TA2A_EARLY_HOPB", "1")
    disp_mod.reset_early_hopb()
    assert disp_mod.early_hopb_counts_enabled() is True
    monkeypatch.delenv("TERRACE_TA2A_EARLY_HOPB", raising=False)
    disp_mod.reset_early_hopb()


def test_bincount_is_permutation_blind():
    """A6 成立的前提:i_send = bincount(owner) 不依赖排序,所以能提到排序之前算。

    这条前提原本只写在注释里(「直方图对置换盲:免 owner[ordo]」)。A6 把整个
    通信的发起点押在它上面了 —— 前提得是可执行的,不能只是一句话。
    """
    import torch
    g = torch.Generator().manual_seed(5)
    owner = torch.randint(0, 8, (1024,), generator=g)
    perm = torch.randperm(1024, generator=g)
    a = torch.bincount(owner, minlength=8)
    b = torch.bincount(owner[perm], minlength=8)
    assert torch.equal(a, b), "bincount 对置换不盲 —— A6 的前提不成立,必须回退"


def test_sync_probe_is_off_by_default_and_present_in_both_seams():
    """判别探针默认关,且**两条接缝都插了**(判决床走 overlap 那条)。

    内部记录 把我原来对 fixed_hist 的机理解释证伪了:主机同步实测只值 0.042-0.046 ms,
    而被换掉的 bincount 是 0.797 —— 同步最多解释 6%。可 fixed_hist 上机确实拿到
    -1.166(两轮复现)。机理不明的收益不能拿去支撑下一把刀,所以要判别。
    只插 legacy 那条会让整个实验空跑 —— 判决床根本不走它。
    """
    import re
    monkeypatch = None
    disp_mod.reset_sync_probe()
    assert disp_mod.sync_probe_enabled() is False, "判别探针默认必须是关的"
    src = open(disp_mod.__file__, encoding="utf-8").read()
    hits = [m.start() for m in re.finditer(r"if sync_probe_enabled\(\):", src)]
    assert len(hits) == 2, "探针应在 legacy 与 overlap 两条接缝各插一处,实际 %d 处" % len(hits)
    i_overlap = src.index("def ta2a_permute_overlap")
    assert any(h > i_overlap for h in hits), (
        "overlap 接缝里没有探针 —— 判决床走的就是它,只插 legacy 等于实验空跑")
