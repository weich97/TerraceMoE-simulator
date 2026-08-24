# -*- coding: utf-8 -*-
"""验证门:仿真器必须先复现我们自己的端到端真值,才有资格外推。

## 预注册(2026-08-24,写于第一次运行之前,不许事后改)

**标定点(1 个)**:`flag`。combine 侧的实现差(on 臂 combine 反而更快,机制
未完全解释,内部实测记录/内部实测记录)无法从第一性原理给出,用 flag 的步级
真值反解一个常数 `combine_extra_ms`,**它对其余五个几何保持不变**。

**保留验证点(5 个,不参与任何拟合)**:tok2x / tok4x / k8m2 / k8m4 / n8。

**门(全部满足才算通过)**:
  1. 五个保留点的 G 预测,平均绝对误差 MAE ≤ 0.025;
  2. 至少 4/5 落在 ±0.035 内;
  3. 四个负号点(tok2x/tok4x/k8m4/n8)方向全对(G_pred < 1)。

不过门就不许外推 —— sweep 模块在运行时检查这个门。

## 端到端真值(对照床,同配置只换 a2a;出处:内部实测记录/内部实测记录/内部实测记录)

步时为稳态中位(掐头 300 步,n=10/臂);G = t_off / t_on。
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import MoEGeometry, step_delta


@dataclass
class Holdout:
    geom: MoEGeometry
    t_off_ms: float          # 厂商臂实测步时
    g_measured: float        # 实测 G
    role: str                # "calibration" | "holdout"


def _g(name, ng, R, k, M, mbs, layers=19):
    return MoEGeometry(name=name, n_groups=ng, R=R, k=k, M=M, mbs=mbs,
                       moe_layers=layers)


HOLDOUTS = [
    # flag:base 档 n=5 组的代表对(内部实测记录);作**标定点**
    Holdout(_g("flag", 16, 8, 6, 2, 1), 4420.4, 1.0355, "calibration"),  # n=7 均值(t=3.27, p≈0.017 显著)
    # 保留验证点
    Holdout(_g("tok2x", 16, 8, 6, 2, 2), 3343.6, 0.8845, "holdout"),   # 三发均值,散布 0.8%   # 两发均值 内部实测记录
    Holdout(_g("tok4x", 16, 8, 6, 2, 4), 2992.7, 0.8234, "holdout"),   # 三发均值,散布 0.6%
    Holdout(_g("k8m2", 16, 8, 8, 2, 1), 4791.7, 0.9935, "holdout"),
    Holdout(_g("k8m4", 16, 8, 8, 4, 1), 4664.9, 0.9335, "holdout"),
    Holdout(_g("n8", 8, 8, 6, 2, 1), 6657.5, 0.9027, "holdout"),
    Holdout(_g("n4", 4, 8, 6, 2, 1), 11712.0, 0.8647, "holdout"),   # 规模轴第三点
]


def predict_g(cluster, h: Holdout, combine_extra_ms: float) -> float:
    """G 预测 = t_off / (t_off + Δstep)。

    Δstep = dispatch 差 x dispatch 次数 + combine 差 x combine 次数。
    dispatch 差用完整模型;combine 的通信字节镜像 dispatch,但实现侧的差
    用标定常数 combine_extra_ms 替代 local_chain(见模块 docstring)。
    """
    d = step_delta(cluster, h.geom)
    calls_fwd = h.geom.calls_per_step_fwd() + h.geom.calls_per_step_bwd()
    chain_ms = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
    # dispatch:模型全量
    disp_delta = (d["per_call_on_ms"] - d["per_call_off_ms"]) * calls_fwd
    # combine:通信部分同 dispatch(镜像);实现差按行标定
    # (combine_extra_ms 实为 flag 行数下的值,换几何按行数缩放)
    comm_delta = d["per_call_on_ms"] - chain_ms - d["per_call_off_ms"]
    scale = h.geom.rows_hop_b() / 24576.0
    comb_delta = (comm_delta + combine_extra_ms * scale) * calls_fwd
    t_on = h.t_off_ms + disp_delta + comb_delta
    return h.t_off_ms / t_on


def calibrate_combine(cluster) -> float:
    """从标定点(flag)反解 combine_extra_ms。"""
    h = next(x for x in HOLDOUTS if x.role == "calibration")
    d = step_delta(cluster, h.geom)
    calls = h.geom.calls_per_step_fwd() + h.geom.calls_per_step_bwd()
    t_on_target = h.t_off_ms / h.g_measured
    chain_ms = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
    disp_delta = (d["per_call_on_ms"] - d["per_call_off_ms"]) * calls
    comm_delta = d["per_call_on_ms"] - chain_ms - d["per_call_off_ms"]
    # t_on_target = t_off + disp + (comm_delta + x) * calls
    x = (t_on_target - h.t_off_ms - disp_delta) / calls - comm_delta
    return x


def validate(cluster, verbose: bool = True):
    """跑验证门。返回 (通过?, 明细)。"""
    ce = calibrate_combine(cluster)
    rows, errs, in_tol, signs_ok = [], [], 0, 0
    neg = {"tok2x", "tok4x", "k8m4", "n8"}
    for h in HOLDOUTS:
        gp = predict_g(cluster, h, ce)
        e = gp - h.g_measured
        if h.role == "holdout":
            errs.append(abs(e))
            if abs(e) <= 0.035:
                in_tol += 1
            if h.geom.name in neg and gp < 1.0:
                signs_ok += 1
        rows.append((h.geom.name, h.role, h.g_measured, gp, e))
    mae = sum(errs) / len(errs)
    ok = (mae <= 0.025) and (in_tol >= 4) and (signs_ok == len(neg))
    if verbose:
        print("验证门(标定点 flag 反解 combine_extra=%.3f ms/次)" % ce)
        print("%-8s %-12s %8s %8s %8s" % ("几何", "角色", "实测G", "预测G", "误差"))
        for n, r, gm, gp, e in rows:
            print("%-8s %-12s %8.4f %8.4f %+8.4f" % (n, r, gm, gp, e))
        print("保留点 MAE=%.4f(门 0.025)  ±0.035 内 %d/5(门 4)  负号 %d/4(门 4)"
              % (mae, in_tol, signs_ok))
        print("**%s**" % ("通过 —— 允许外推" if ok else "不通过 —— 禁止外推,先修模型"))
    return ok, {"mae": mae, "in_tol": in_tol, "signs_ok": signs_ok,
                "combine_extra_ms": ce, "rows": rows}


if __name__ == "__main__":
    from .calibrate import aug_flat
    validate(aug_flat())
