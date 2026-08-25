#!/usr/bin/env python3
"""**真正的单边**:用 CANN aclshmem 的 kernel 内单边写,对集合 a2a。

为什么普通手段测不了单边(先说量具,再说测什么):

  · PyTorch 的跨设备 `tensor.copy_()` 每次调用约 257-268 µs 的固定开销,
    盖过一切:载荷从 4 MB 涨到 40 MB 耗时纹丝不动,报出来的"带宽"只是
    「字节 ÷ 常数」。用它给单边下的任何结论都是废数 —— 我们撤回过一整组这样的表。
  · hyper-parallel 的 `put_mem` 走的是另一条路:拿当前流直接发一个 AscendC kernel,
    **没有 aclnn、没有 host tiling,offset/size/target_pe 全以 device 指针传入,
    host 一次都不回读**;kernel 内 `aclshmem_ptr(target, pe)` 拿到远端可直接寻址的
    GM 指针,MTE 走 GM→UB→远端 GM。这正是 `copy_()` 那 257 µs 里没有的东西。
  · 对照的靶子要先立住:对齐尺寸(4096 B 整数倍)下,本机节点内集合 a2a
    实测 ~104 GB/s,单 die 聚合出口物理值 122.4(docs/05)—— a2a 已吃到 ~85%。

------------------------------------------------------------------------------
判据(跑之前写死,不许事后改)

  **上限本身就只有 ~15%。** a2a 已在聚合出口物理值的 ~85%,任何节点内传输
  方式的理论上限就是那剩下的 ~15 个百分点。所以:

    单边 ≥ 1.15 × a2a(在 ≥2 个对齐尺寸档上)  -> 单边确实能吃到线速,值得往下做
    否则                                       -> 单边这条路关闭

  1.15 不是拍的:它等价于"从 85% 打到 100% 线速"。低于它就说明单边换的是协议开销,
  而协议开销在这台机器上已经不是瓶颈。

  **量具地板**:本发量的是**带宽**,地板是 launch 开销,两者是不同的量,
  地板闸才成立。任一档耗时不到地板 3 倍 -> 该档作废。(反例:若量的就是固定
  开销本身,地板等于信号,这道闸永远红 —— 同一个手法要看量的是什么。)

------------------------------------------------------------------------------
为什么不 import hyper_parallel

顶层 `hyper_parallel/__init__.py` 会 import DTensor / FSDP / PP / CP 一整套,
并**无条件** `override_functions()` 打 `BackwardHookFunction` 的 monkey patch。
与训练框架同进程共存的风险全在那里。
`platform/torch/symmetric_memory/symmetric_memory.py:29-35` 自己就演示了怎么绕:
直接 `torch.ops.load_library` + `torch.classes.SymmetricMemory.Manager/Ops`。

**顺带**:`attr_init` 里写死的 `tcp://127.0.0.1:8662` 对本发不构成问题 ——
Hop B 就是节点内 8 die,全在同一台主机上,本机地址正好够用。跨节点才需要改,
而 C++ 侧 `attr_init` 本来就收地址参数(README 那条"硬编码"只在 Python 封装里)。

用法(单节点 8 die):
  torchrun --nnodes=1 --nproc_per_node=8 tools/onesided/bench_onesided.py --out onesided.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

try:
    import torch_npu  # noqa: F401
except Exception as exc:                       # pragma: no cover - 只在集群上跑
    raise SystemExit("需要 torch_npu:%s" % exc)

H = 2048                 # 判决床隐藏维;每行 H*2 = 4096 B
_MGR = _OPS = None


def _load(lib: str):
    """加载 libaclshmem_torch.so 并拿到两个 TorchScript 类。

    路径优先级:显式 --lib > TERRACE_SHMEM_LIB > 装好的 wheel 里那份。
    找不到就**大声退出**,不做静默降级 —— 降级会把"没测到"伪装成"测到了很慢"。
    """
    global _MGR, _OPS
    cands = [lib] if lib else []
    env = os.environ.get("TERRACE_SHMEM_LIB")
    if env:
        cands.append(env)
    try:
        import hyper_parallel.platform.torch.symmetric_memory as _sm
        cands.append(os.path.join(os.path.dirname(os.path.abspath(_sm.__file__)),
                                  "libaclshmem_torch.so"))
    except Exception:                                       # noqa: BLE001
        pass
    for c in cands:
        if c and os.path.exists(c):
            torch.ops.load_library(c)
            _MGR = torch.classes.SymmetricMemory.Manager()
            _OPS = torch.classes.SymmetricMemory.Ops()
            return c
    raise SystemExit(
        "找不到 libaclshmem_torch.so。试过:%s\n"
        "先按 tools/onesided/build_shmem.sh 编译(离线集群需预放 gitcode.com/cann/shmem v1.3.0)。"
        % cands)


def timed(fn, iters=10) -> float:
    """与本仓其余微基准同口径:3 次热身丢弃 + iters 次取均值。"""
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    # 尺寸取 H 的整数倍 -> 每对端字节数是 4096 B 的整数倍,与真实 Hop B 同形。
    # 不对齐会掉进 2 进制阶造成的假锯齿(实现行为),测的是对齐效应不是传输方式。
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[512, 1024, 1536, 2048, 3072, 4096, 6144],
                    help="每对端行数(1 行 = H*2 = 4096 B);512 行 = 2 MB")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lib", default="")
    ap.add_argument("--heap-mb", type=int, default=4096)
    ap.add_argument("--out", default="onesided_shmem.json")
    args = ap.parse_args()

    os.environ.setdefault("SYMMETRIC_MEMORY_HEAP_SIZE", str(args.heap_mb * 1024 * 1024))
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.npu.set_device(rank % 8)
    lib = _load(args.lib)
    _MGR.attr_init(rank, world, args.heap_mb * 1024 * 1024, "tcp://127.0.0.1:8662")
    # hyper-parallel 官方 shmem_alltoall 形态用的每对端一条流
    streams = [torch.npu.Stream() for _ in range(world)]
    if rank == 0:
        print("[onesided] 加载 %s;world=%d heap=%d MB" % (lib, world, args.heap_mb), flush=True)

    dt = torch.bfloat16

    def T(x, d=torch.int64):
        return torch.tensor([x], dtype=d, device="npu")

    # **offset/size 的 tensor 提到循环外。** 审计发现参考实现每次调用现建 3-5 个
    # `torch.tensor([x], device='npu')` —— 每发一次 put 就多几次 H2D 小分配,
    # 那正是我们要打败的那类开销。
    zero, one_i32 = T(0), T(1, torch.int32)

    recs, floor = [], [None]
    for rows in args.rows:
        n_elem = rows * H
        per_peer_mb = n_elem * 2 / 1e6
        # 对称内存:发送区(每对端一份)与接收区(每对端一份)
        send = _MGR.malloc([world * n_elem], dt)
        recv = _MGR.malloc([world * n_elem], dt)
        torch.fill_(send, 1.0)
        sig = _MGR.malloc([1], torch.int32)
        torch.zero_(sig)
        offs = [T(p * n_elem) for p in range(world)]      # 也提到循环外
        size_t = T(n_elem)
        dist.barrier()

        def run_onesided():
            # 异或蝶形:第 r 轮与 r^rank 交换。每轮每 die 恰好一进一出,按构造无竞争
            # —— 经典无竞争 p2p 调度形态。
            for r in range(1, world):
                pe = rank ^ r
                _OPS.put_mem(recv, offs[rank], send, offs[pe], size_t, pe)
            torch.npu.synchronize()

        # 对照臂:同样载荷的集合 a2a(同机实测在这些尺寸上 ~76-104 GB/s)
        sv = torch.ones(world * n_elem, dtype=dt, device="npu")
        rv = torch.empty_like(sv)

        def run_a2a():
            dist.all_to_all_single(rv, sv)

        # 第三臂:官方 shmem_alltoall 形态(每对端一条流 + put_mem_signal +
        # signal_wait_until)。蝶形臂每轮只占一条链路,天生吃不到链路并行;
        # 这一臂是该库生产在用的并行形态 —— **不用最强合法实现测过,没资格下结论**。
        # 完成检测用 signal 累计计数(op=1 是加),免每轮清零、免 host 同步;
        # sig 每个尺寸档新分配清零,n_calls 同步归零,口径自洽。
        n_calls = [0]
        expect_t = T(-1, torch.int32)   # 首次 add_(world) 后 = world-1,即 GT 阈值

        def run_official():
            for pe in range(world):
                with torch.npu.stream(streams[pe]):
                    _OPS.put_mem_signal(recv, offs[rank], send, offs[pe], size_t,
                                        sig, zero, one_i32, 1, pe)
            n_calls[0] += 1
            # compare_op 只有 0=EQ/1=GT/2=LT(kernel 源码),没有 GE。
            # 用 EQ 会死:快的 rank 先进下一轮,把 +1 打进我的 sig,精确等于的
            # 窗口被冲过头 -> 永久自旋(实测:全 rank 设备同步超时挂在这)。
            # GT + (8k-1) 等价于 GE 8k,单调递增计数下免疫冲过头。
            # 阈值 tensor 设备侧原位递增(expect_t 由外层预建),不在热路径
            # 现建 tensor —— 与 offs/size_t 同一条纪律。
            expect_t.add_(world)
            _OPS.signal_wait_until(recv, sig, zero, expect_t, 1)
            torch.npu.synchronize()

        # **量具地板**:同样 world-1 次 put,每次 1 个元素。这里量的是带宽,
        # 地板是 launch 开销,两者是不同的量 —— 所以这道闸在本发是对的
        # (若量的就是固定开销本身,同一道闸永远红)。
        if floor[0] is None:
            s1, r1 = _MGR.malloc([world], dt), _MGR.malloc([world], dt)
            o1, z1 = [T(p) for p in range(world)], T(1)

            def run_floor():
                for r in range(1, world):
                    pe = rank ^ r
                    _OPS.put_mem(r1, o1[rank], s1, o1[pe], z1, pe)
                torch.npu.synchronize()

            floor[0] = timed(run_floor, args.iters) * 1e3
            _MGR.free(s1)
            _MGR.free(r1)

        to = sorted(timed(run_onesided, args.iters) for _ in range(args.reps))[args.reps // 2]
        ta = sorted(timed(run_a2a, args.iters) for _ in range(args.reps))[args.reps // 2]
        tf2 = sorted(timed(run_official, args.iters) for _ in range(args.reps))[args.reps // 2]
        nbytes = (world - 1) * n_elem * 2          # 本 die 实际搬出的字节
        rec = {"rows": rows, "per_peer_mb": per_peer_mb,
               "onesided_ms": to * 1e3, "a2a_ms": ta * 1e3,
               "onesided_gbps": nbytes / to / 1e9,
               "a2a_gbps": (world - 1) / world * (world * n_elem * 2) / ta / 1e9,
               "ratio": ta / to if to else 0.0,
               "official_ms": tf2 * 1e3,
               "official_gbps": nbytes / tf2 / 1e9,
               "ratio_official": ta / tf2 if tf2 else 0.0,
               "over_floor": (to * 1e3) / floor[0] if floor[0] else 0.0,
               "official_over_floor": (tf2 * 1e3) / floor[0] if floor[0] else 0.0}
        recs.append(rec)
        if rank == 0:
            print("%5d 行 (%6.2f MB/对端)  蝶形 %6.1f  官方 %6.1f  a2a %6.1f GB/s"
                  "  -> 蝶形 **%.2f×** 官方 **%.2f×**  (地板 %.1f×/%.1f×)"
                  % (rows, per_peer_mb, rec["onesided_gbps"], rec["official_gbps"],
                     rec["a2a_gbps"], rec["ratio"], rec["ratio_official"],
                     rec["over_floor"], rec["official_over_floor"]), flush=True)
        for t in (send, recv, sig):
            _MGR.free(t)
        del sv, rv

    if rank != 0:
        dist.destroy_process_group()
        return

    out = {"records": recs, "floor_ms": floor[0], "world": world, "lib": lib}
    for r in recs:
        cands = []
        if r["over_floor"] >= 3.0:
            cands.append(r["ratio"])
        if r["official_over_floor"] >= 3.0:
            cands.append(r["ratio_official"])
        r["best_ratio"] = max(cands) if cands else None
    voided = [r["rows"] for r in recs if r["best_ratio"] is None]
    if voided:
        print(flush=True)
        print("!! 以下档耗时不到量具地板(%.3f ms)的 3 倍,**读数不作数**:%s"
              % (floor[0], voided), flush=True)
    live = [r for r in recs if r["best_ratio"] is not None]

    print(flush=True)
    wins = [r for r in live if r["best_ratio"] >= 1.15]
    if len(live) < 3:
        out["verdict"] = "INVALID:有效档不足 3 个"
        print("!! 有效档只有 %d 个,**不出判据**。" % len(live), flush=True)
    elif len(wins) >= 2:
        out["verdict"] = "单边能吃到线速,值得往下做"
        print("**单边在 %d 个档上 ≥1.15× a2a** -> %s" % (len(wins), out["verdict"]), flush=True)
        print("   最好的一档:%.2f× @ %.2f MB/对端"
              % (max(w["best_ratio"] for w in wins),
                 max(wins, key=lambda w: w["best_ratio"])["per_peer_mb"]), flush=True)
    else:
        out["verdict"] = "单边这条路关闭(本机)"
        best = max(live, key=lambda r: r["best_ratio"])
        print("单边(两种实现取优)最好只有 **%.2f×**(@ %.2f MB/对端),达不到 1.15 -> **%s**"
              % (best["best_ratio"], best["per_peer_mb"], out["verdict"]), flush=True)
        print("   理由:a2a 已在聚合出口物理值的 ~85%%,上限本就只有 ~15 个点;", flush=True)
        print("   单边换的是协议开销,而协议开销在这台机器上不是瓶颈。", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("写出 %s" % os.path.abspath(args.out), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
