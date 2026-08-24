"""T-A2A 接缝残余漂移的**设备端**探针(TERRACE_DRIFT_PROBE=1 才生效)。

Why this file exists(2026-08-20 晚,判决床 eq 门阻塞项):
overlap 6 参接缝的 on 臂相对厂商 overlap 在对齐床上逐步漂移(早窗 2e-5 干净,
iter80 |Δ|=1.31e-4),而同床同方法的 legacy 接缝 on 臂全程 ≤2e-5。已排除 gate 平面
dtype(一次内部提交)、topk 并列序(1ec68687)、量尺错配;CPU 上 HEAD / C1 前 / f73c041e
三方四路梯度**逐位相等** ⇒ 若还存在可消除的实现差异,它只在设备端产生。
设备端没有任何现成的对照手段:所有位级契约(两条接缝同值、quota 支路 == 通用支路、
plan 快路径 sel == 通用支路 sel)都只在 CPU 单测里证过。本模块补的就是这一口:
把那些契约在 NPU 上**执行一遍**,并给两臂共同点位一份可离线对差的指纹。

三类探针,都在 `torch.no_grad()` 里、都不改任何张量:

  1. `note` / `note_int` —— 点位指纹。float 走 (numel, Σ|x|, Σx², max|x|),
     **对行置换不变**:两臂在同一点位上的行序天生不同(T-A2A 去重 + 按节点重排,
     厂商按专家重排),逐元素比较无从谈起,而置换不变量对「值集合变了没有」是灵敏的
     —— Σ|x| 无抵消,fp32 树形归约的自身误差 ~1e-10 相对,而一个 bf16 ulp 级的值差在
     千分之一的元素上就能顶出 1e-6 相对。int 走精确 int64 校验和 + min/max。
  2. `check_equal` —— 两个张量在设备上**逐位**相等否(CPU 契约的设备端执行)。
  3. `check_reduction` —— 归约点复核:①同一 index_add 重跑一遍是否逐位可复现
     (NPU 上 index_add 的原子/分块顺序是否稳定,无文档保证);②与一个**确定性**
     fp32 参考(按 index 排序后定形 reshape-sum)的偏差 —— 量的是「设备归约的
     累加器宽度与顺序」,这正是 on/off 两臂唯一的结构性差异所在。

纪律:
  - 开关不设 = **零行为**:每个点位只多一次已缓存的 bool 读;不建张量、不打一行。
  - 探针自身绝不改变数值:全部 detach + no_grad,只读。
  - 探针出错不静默:打一行 WARN 并**自我关闭**(诊断死了要在日志里看得见,
    但不能把训练带走);契约类探针(check_equal / plan.sel)命中不一致时,
    TERRACE_DRIFT_PROBE_STRICT=1 下直接 raise。

离线对差:`python -m terrace.drift_probe compare on.log off.log`,按点位/step/layer
配对,按日志出现顺序报**第一处**越限点 —— 「漂移注入的第一层/第一步」即由此读出。
"""
from __future__ import annotations

import os
import re
import sys

import torch

ENV_SWITCH = "TERRACE_DRIFT_PROBE"
ENV_EVERY = "TERRACE_DRIFT_PROBE_EVERY"      # 每 N 次 dispatch 打一轮(默认 1)
ENV_MAXCALL = "TERRACE_DRIFT_PROBE_MAXCALL"  # 打满 N 次 dispatch 后自停(默认 8000)
ENV_RANKS = "TERRACE_DRIFT_PROBE_RANKS"      # 逗号分隔的 rank 白名单(默认 "0")
ENV_STRICT = "TERRACE_DRIFT_PROBE_STRICT"    # 契约探针命中不一致时 raise
ENV_CHUNK = "TERRACE_DRIFT_PROBE_CHUNK"      # 指纹分块元素数(默认 4M,限峰值显存)

TAG = "[terrace-drift]"

# 进程级状态。`on` 是三态:None = 未判定,True/False = 判定结果(热路径只读一次)。
_S: dict = {"on": None, "every": 1, "maxcall": 8000, "chunk": 1 << 22,
            "strict": False, "rank": 0, "ranks_ok": True,
            "step": 0, "layer": -1, "call": 0, "dispatch": 0, "dead": False}


def reset():
    """重读环境并清计数。单测用;训练进程从不中途翻开关。"""
    _S.update({"on": None, "step": 0, "layer": -1, "call": 0, "dispatch": 0,
               "dead": False})


def _decide():
    _S["on"] = os.environ.get(ENV_SWITCH) == "1"
    if not _S["on"]:
        return
    _S["every"] = max(1, int(os.environ.get(ENV_EVERY, "1") or "1"))
    _S["maxcall"] = int(os.environ.get(ENV_MAXCALL, "8000") or "8000")
    _S["chunk"] = max(1, int(os.environ.get(ENV_CHUNK, str(1 << 22)) or (1 << 22)))
    _S["strict"] = os.environ.get(ENV_STRICT) == "1"
    _S["rank"] = int(os.environ.get("RANK", "0") or "0")
    allow = os.environ.get(ENV_RANKS, "0") or "0"
    _S["ranks_ok"] = (allow.strip() == "*"
                      or _S["rank"] in {int(x) for x in allow.split(",") if x.strip()})


def enabled() -> bool:
    """热路径唯一的开关读。未判定时判一次,之后是一个 dict 取值。"""
    if _S["on"] is None:
        _decide()
    return bool(_S["on"]) and not _S["dead"]


def _live() -> bool:
    """本次调用要不要真打点(开关 + rank 白名单 + 周期 + 上限)。"""
    if not enabled() or not _S["ranks_ok"]:
        return False
    if 0 <= _S["maxcall"] <= _S["dispatch"]:
        return False
    # tick 后 dispatch 是 1-based;-1 让**第一次**调用必打(周期再粗也不丢首帧,
    # 而首帧正是「漂移注入的第一步」最可能落点)。未接包装器时 dispatch=0,同样打。
    return (max(_S["dispatch"] - 1, 0) % _S["every"]) == 0


def set_where(step=None, layer=None, call=None):
    """点位坐标(离线对差的配对键)。包装器每次调用前设一次。"""
    if step is not None:
        _S["step"] = int(step)
    if layer is not None:
        _S["layer"] = int(layer)
    if call is not None:
        _S["call"] = int(call)


def tick_dispatch():
    """dispatch 调用计数(周期/上限的口径)。包装器在 dispatch 半边入口调。"""
    _S["dispatch"] += 1
    return _S["dispatch"]


def where() -> str:
    return "step=%d layer=%d call=%d rank=%d" % (
        _S["step"], _S["layer"], _S["call"], _S["rank"])


def _emit(line: str):
    print("%s %s" % (TAG, line), flush=True)


def _die(exc: Exception, what: str):
    """探针自身出错:打一行 WARN 并自我关闭。诊断可以死,训练不能被它带走。"""
    _S["dead"] = True
    _emit("WARN 探针在 %s 处出错(%r)—— 已自我关闭,后续不再打点;"
          "本行之后的缺失点位是探针死亡所致,不是数据相同" % (what, exc))


class DriftProbeMismatch(RuntimeError):
    """契约探针命中不一致(TERRACE_DRIFT_PROBE_STRICT=1)。

    自成一类,好让下面的 `except Exception`(探针自保)**放它过去** —— 把
    STRICT 的硬停当成「探针自己出错」吞掉,正是探针最不该有的失效形态。
    """


def _fail(point: str, msg: str):
    _emit("MISMATCH pt=%s %s %s" % (point, where(), msg))
    if _S["strict"]:
        raise DriftProbeMismatch("%s 契约探针不一致 pt=%s %s" % (TAG, point, msg))


# ---------------------------------------------------------------------------------
# 指纹
# ---------------------------------------------------------------------------------

def digest(x: torch.Tensor, chunk: int | None = None):
    """置换不变的三元指纹。返回 (numel, sum|x|, sum x^2, max|x|),float 输入。

    分块累加:峰值临时显存被 `chunk` 个元素封顶(判决床上 [pairs, 2048] 的 fp32
    整体上转是 0.5 GB 级,分块后是常数),而两臂形状相同 ⇒ 分块边界相同 ⇒ 归约树
    相同,指纹之间只差被求和的**值**,不差求和的**方式**。
    """
    xd = x.detach().reshape(-1)
    n = int(xd.numel())
    if n == 0:
        return (0, 0.0, 0.0, 0.0)
    step = int(chunk or _S["chunk"])
    dev = xd.device
    s_abs = torch.zeros((), dtype=torch.float32, device=dev)
    s_sq = torch.zeros((), dtype=torch.float32, device=dev)
    amax = torch.zeros((), dtype=torch.float32, device=dev)
    for i in range(0, n, step):
        c = xd[i:i + step].to(torch.float32)
        a = c.abs()
        s_abs = s_abs + a.sum()
        s_sq = s_sq + (c * c).sum()
        amax = torch.maximum(amax, a.max())
    vals = torch.stack([s_abs, s_sq, amax]).tolist()      # 本函数唯一一次同步
    return (n, vals[0], vals[1], vals[2])


def digest_int(x: torch.Tensor):
    """整数/布尔平面的精确指纹。返回 (numel, Σx, min, max),int64 精确。"""
    xd = x.detach().reshape(-1)
    n = int(xd.numel())
    if n == 0:
        return (0, 0, 0, 0)
    xi = xd.to(torch.int64)
    vals = torch.stack([xi.sum(), xi.min(), xi.max()]).tolist()
    return (n, int(vals[0]), int(vals[1]), int(vals[2]))


def _line(point: str, dt, fields: str) -> str:
    return "pt=%s %s dt=%s %s" % (point, where(), str(dt).replace("torch.", ""),
                                  fields)


def note(point: str, x, **_ignored):
    """float 点位指纹。开关关 = 零行为。"""
    if not _live() or x is None:
        return
    try:
        with torch.no_grad():
            if not torch.is_tensor(x):
                return
            if not x.is_floating_point():
                return note_int(point, x)
            n, sa, sq, am = digest(x)
            _emit(_line(point, x.dtype,
                        "n=%d sabs=%.17g ssq=%.17g amax=%.17g" % (n, sa, sq, am)))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note(%s)" % point)


def note_layout(point: str, x, **_ignored):
    """**布局**指纹:shape / dtype / device / 是否连续 / stride。开关关 = 零行为。

    2026-08-23 加。`note` 只记 dtype 与数值摘要(numel/绝对和/平方和/最大值),
    不记布局 —— 而 §33 的待查项恰恰是布局:

      `seam_gap`(两接缝之间,= 专家 GEMM 前向)两轮稳定在 −0.478 / −0.483 ms/次,
      合 −73 ms/步,占 base 档胜势的相当一部分,**没有解释**。主机同步那条假说
      已排除(那行 `.cpu()` 属于我们不用的 dispatcher)。剩下的候选是:两臂喂给
      分组 GEMM 的是同一批 (token, expert) 对,但**排列顺序与内存连续性可能不同**。

    这些全是元数据 —— **不读设备内存、不触发同步、不发 kernel**,所以它比 `note`
    还便宜,可以在两臂上长期开着。
    """
    if not _live() or x is None:
        return
    try:
        if not torch.is_tensor(x):
            return
        _emit(_line(point + "#layout", x.dtype,
                    "shape=%s dev=%s contig=%d stride=%s" % (
                        "x".join(str(d) for d in x.shape),
                        x.device.type, int(x.is_contiguous()),
                        ",".join(str(v) for v in x.stride()))))
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note_layout(%s)" % point)


def note_int(point: str, x, **_ignored):
    """整数/布尔点位指纹(精确)。开关关 = 零行为。"""
    if not _live() or x is None:
        return
    try:
        with torch.no_grad():
            if not torch.is_tensor(x):
                return
            n, s, lo, hi = digest_int(x)
            _emit(_line(point, x.dtype,
                        "n=%d isum=%d imin=%d imax=%d" % (n, s, lo, hi)))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note_int(%s)" % point)


# ---------------------------------------------------------------------------------
# 契约探针
# ---------------------------------------------------------------------------------

def check_equal(point: str, a, b, note_a: str = ""):
    """两张量在**设备上**逐位相等否。CPU 单测证过的契约,在这里被真正执行一次。"""
    if not _live() or a is None or b is None:
        return
    try:
        with torch.no_grad():
            ad, bd = a.detach(), b.detach()
            if ad.shape != bd.shape or ad.dtype != bd.dtype:
                _fail(point, "shape/dtype 不同 %s%s vs %s%s %s"
                      % (tuple(ad.shape), ad.dtype, tuple(bd.shape), bd.dtype, note_a))
                return
            if torch.equal(ad, bd):
                _emit(_line(point + ".eq", ad.dtype, "bitequal=1 n=%d" % ad.numel()))
                return
            d = (ad.to(torch.float32) - bd.to(torch.float32)).abs()
            _fail(point, "bitequal=0 n=%d ndiff=%d maxabs=%.17g %s"
                  % (ad.numel(), int((d > 0).sum()), float(d.max()), note_a))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "check_equal(%s)" % point)


def check_reduction(point: str, out, src, index, n_out: int, per=None):
    """归约点复核:可复现性 + 与确定性 fp32 参考的偏差。

    `out` 必须是 `zeros(n_out, W).index_add_(0, index, src)` 的结果。

      det=1/0      同一 index_add 重跑一次是否**逐位**相同。NPU 的 index_add 在重复
                   下标上的累加顺序没有任何文档保证(CUDA 上就是 atomic,不可复现);
                   det=0 意味着 on 臂自己每步的归约序都在变 —— 那不是 on/off 差异,
                   是 on 臂内部的不可控变异,eq 门的噪声地板必须包含它。
      maxabs/maxrel 与确定性参考(按 index 分组后定形 reshape-sum,fp32 累加)的偏差。
                   量的是设备归约的累加器宽度:bf16 累加器给出 ~2^-9 相对,
                   fp32 累加器给出 ~2^-24 相对 —— 相差 5 个数量级,一眼可辨。
                   仅当每个目标行恰好 `per` 个贡献时可算(quota 快路径按构造成立)。
    """
    if not _live() or out is None:
        return
    try:
        with torch.no_grad():
            o, s, idx = out.detach(), src.detach(), index.detach()
            again = torch.zeros_like(o).index_add_(0, idx, s)
            det = 1 if torch.equal(again, o) else 0
            extra = ""
            if per and idx.numel() == n_out * int(per):
                p = torch.argsort(idx, stable=True)
                ref = s[p].to(torch.float32).reshape(
                    n_out, int(per), -1).sum(1)
                d = (o.to(torch.float32) - ref).abs()
                scale = ref.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
                extra = " per=%d maxabs=%.17g maxrel=%.17g" % (
                    int(per), float(d.max()), float(d.max() / scale))
            _emit(_line(point + ".chk", o.dtype,
                        "det=%d n_out=%d npair=%d%s"
                        % (det, n_out, int(idx.numel()), extra)))
            if det == 0:
                _fail(point + ".chk", "index_add 在本设备上**不可复现**"
                                      "(同输入两次结果逐位不同)")
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "check_reduction(%s)" % point)


# ---------------------------------------------------------------------------------
# 离线对差
# ---------------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"\[terrace-drift\] pt=(?P<pt>\S+) step=(?P<step>\d+) layer=(?P<layer>-?\d+) "
    r"call=(?P<call>\d+) rank=(?P<rank>\d+) dt=(?P<dt>\S+) (?P<rest>.*)$")
_KV_RE = re.compile(r"(\w+)=(-?[0-9.eE+-]+)")

# 对差判据:相对偏差上界。fp32 树形归约自身在 1e-9 以下,给三个数量级余量。
CMP_RTOL = 1e-6


def parse_line(line: str):
    """一行 → (key, dict) 或 None。key = (pt, step, layer, call)。"""
    m = _LINE_RE.search(line)
    if m is None:
        return None
    d = {k: float(v) for k, v in _KV_RE.findall(m.group("rest"))}
    d["dt"] = m.group("dt")
    return ((m.group("pt"), int(m.group("step")), int(m.group("layer")),
             int(m.group("call"))), d)


def parse_log(path):
    """日志 → (dict[key] = fields, [key...] 按出现顺序)。"""
    out, order = {}, []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            got = parse_line(line)
            if got is None:
                continue
            key, fields = got
            if key not in out:
                order.append(key)
            out[key] = fields
    return out, order


def _reldiff(a, b):
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def compare(path_a, path_b, rtol=CMP_RTOL, limit=20):
    """按点位对差两臂日志,按 A 的出现顺序报第一处越限点。返回 exit code。"""
    A, order = parse_log(path_a)
    B, _ = parse_log(path_b)
    common = [k for k in order if k in B]
    if not common:
        print("NOCOMMON: 两份日志没有共同点位(pt/step/layer/call 全不重合)")
        return 2
    bad = []
    for k in common:
        fa, fb = A[k], B[k]
        if fa.get("dt") != fb.get("dt"):
            bad.append((k, "dt", fa.get("dt"), fb.get("dt"), float("inf")))
            continue
        for f in ("n", "sabs", "ssq", "amax", "isum", "imin", "imax", "det",
                  "bitequal", "maxabs"):
            if f in fa and f in fb:
                r = _reldiff(fa[f], fb[f])
                if r > rtol:
                    bad.append((k, f, fa[f], fb[f], r))
    print("共同点位 %d 个(A 独有 %d,B 独有 %d)"
          % (len(common), len(order) - len(common), len(B) - len(common)))
    if not bad:
        print("CLEAN: 全部共同点位在 rtol=%g 内一致 —— 漂移不在被插桩的点位上" % rtol)
        return 0
    k, f, va, vb, r = bad[0]
    print("FIRST-DIVERGENCE: pt=%s step=%d layer=%d call=%d 字段 %s "
          "A=%.17g B=%.17g rel=%.3e" % (k[0], k[1], k[2], k[3], f,
                                        va if isinstance(va, float) else float("nan"),
                                        vb if isinstance(vb, float) else float("nan"),
                                        r))
    print("越限点共 %d 处,前 %d 条:" % (len(bad), min(limit, len(bad))))
    for k, f, va, vb, r in bad[:limit]:
        print("  pt=%-22s step=%-4d layer=%-3d call=%-4d %-8s rel=%.3e"
              % (k[0], k[1], k[2], k[3], f, r))
    return 1


def main(argv):
    if len(argv) >= 3 and argv[0] == "compare":
        rtol = float(argv[3]) if len(argv) > 3 else CMP_RTOL
        return compare(argv[1], argv[2], rtol)
    print("用法:python -m terrace.drift_probe compare <A.log> <B.log> [rtol]")
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
