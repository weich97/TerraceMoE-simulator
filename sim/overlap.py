# -*- coding: utf-8 -*-
"""Tier-2 攻坚:重叠感知合成模型族 —— 系统化的回溯评估。

## 背景(为什么步级门红着)

相位级计时之和与步级差相差 ~5x:两臂在双流上的重叠不同,事件计时的相位
跨度加不出步时。数值侦察把矛盾摆得更尖锐(数据 = sim/validate.py 的 HOLDOUTS,全部实测):

  - 通信模型的步差(Δmodel)几乎全是「到达链 + splits 同步」的固定成本
    (flag:1018 ms 里 1000 ms 是它);纯通信差在这台带宽扁平机上接近零。
  - 实测步差(Δmeas)却从 **-151 ms(flag,两跳反而赢)** 到 +1833 ms(n4)
    不等;隐含的"暴露比" Δmeas/Δmodel 从 -0.15 到 +0.62,**连符号都不一致**。

结论先行:**任何单参数的全局重叠模型都不可能同时解释这七个点** ——
本模块把这句话变成可复现的表:六个结构不同的单参数族,统一在标定点
(flag)上拟合、在六个保留点上评估、全部报告(包括失败的,尤其是失败的)。

## 纪律

  - 拟合只用 flag(与 sim/validate.py 的预注册切分一致),保留点绝不参与。
  - 全族报告,不挑赢家:即使某族回溯通过,也只是「七个点上的模型选择」,
    **Tier-2 的门保持红色**,解锁需要新采的保留点(见 docs/07 的实验协议)。
  - 表中的每个数字由本模块直接产出(`python -m sim.overlap`),脚本即出处。

## 模型族(每族恰好 1 个自由参数,flag 一个方程定一个参数)

  M0  naive      Δ = Δmodel(无重叠;现行 Tier-2 失败基线,0 参数)
  M1  prop       Δ = φ·Δmodel(全局比例暴露)
  M2  hide/call  Δ = Δmodel - h·calls(每次调用隐藏固定毫秒数)
  M3  hide-fixed Δ = Δmodel - φ·fixed(链+splits 被隐藏 φ 比例;纯通信全暴露)
  M4  hide∝comp  Δ = Δmodel - c·T_comp(隐藏量与步内计算时长成比例;
                  T_comp ≈ t_off - 一跳通信总量,即 off 臂的非通信时间)
  M5  expose-mb  Δ = Δmodel·(1 - λ/mbs)(micro-batch 越多、流水越挤、
                  暴露越多 —— 侦察中暴露比随 mbs 单调上升的直译)

评估门与 Tier-2 相同:保留点 MAE ≤ 0.025、≥4 点进 ±0.035、负号点方向全对。
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import step_delta
from .validate import HOLDOUTS

NEG_NAMES = {"tok2x", "tok4x", "k8m4", "n8", "n4"}   # 预注册负号点 4 个 + 后加的 n4;
# k8m2(G=0.9935,n=1,距 1 在噪声内)不计方向 —— 与 validate.py 的负号口径一致


# ----------------------------------------------------------------------------
# 每个几何的分解量(全部由 Tier-1 已验证的通信模型 + 实测步时导出)
# ----------------------------------------------------------------------------

@dataclass
class Decomp:
    name: str
    role: str
    g_meas: float
    t_off: float
    calls: int            # 每步 a2a 调用数(dispatch+combine, fwd+bwd)
    mbs: int
    d_model: float        # 通信模型步差 [ms](>0 = 两跳慢)
    d_meas: float         # 实测步差 [ms]
    fixed: float          # 其中 链+splits 的固定成本总量 [ms]
    comm_off: float       # off 臂(一跳)通信总量 [ms]
    t_comp: float         # ≈ t_off - comm_off:off 臂的非通信时间 [ms]


def decompose(cluster) -> list:
    out = []
    for h in HOLDOUTS:
        d = step_delta(cluster, h.geom)
        calls = d["calls_per_step"]
        chain = cluster.chain_us_per_row * h.geom.rows_hop_b() / 1000.0
        fixed = (chain + cluster.splits_sync_ms) * calls
        comm_off = d["per_call_off_ms"] * calls
        t_on = h.t_off_ms / h.g_measured
        out.append(Decomp(
            name=h.geom.name, role=h.role, g_meas=h.g_measured,
            t_off=h.t_off_ms, calls=calls, mbs=h.geom.mbs,
            d_model=d["step_delta_ms"], d_meas=t_on - h.t_off_ms,
            fixed=fixed, comm_off=comm_off,
            t_comp=h.t_off_ms - comm_off))
    return out


# ----------------------------------------------------------------------------
# 模型族:predict(dec, param) -> Δ预测;fit 在 flag 上解闭式
# ----------------------------------------------------------------------------

@dataclass
class Family:
    key: str
    desc: str
    n_params: int
    predict: object       # (Decomp, float) -> Δms
    fit: object           # (Decomp) -> float(在标定点上解参数)


def _families() -> list:
    return [
        Family("M0", "naive:Δ=Δmodel(无重叠)", 0,
               lambda d, p: d.d_model,
               lambda d: 0.0),
        Family("M1", "prop:Δ=φ·Δmodel", 1,
               lambda d, p: p * d.d_model,
               lambda d: d.d_meas / d.d_model),
        Family("M2", "hide/call:Δ=Δmodel-h·calls", 1,
               lambda d, p: d.d_model - p * d.calls,
               lambda d: (d.d_model - d.d_meas) / d.calls),
        Family("M3", "hide-fixed:Δ=Δmodel-φ·fixed", 1,
               lambda d, p: d.d_model - p * d.fixed,
               lambda d: (d.d_model - d.d_meas) / d.fixed),
        Family("M4", "hide∝comp:Δ=Δmodel-c·T_comp", 1,
               lambda d, p: d.d_model - p * d.t_comp,
               lambda d: (d.d_model - d.d_meas) / d.t_comp),
        Family("M5", "expose-mb:Δ=Δmodel·(1-λ/mbs)", 1,
               lambda d, p: d.d_model * (1.0 - p / d.mbs),
               lambda d: (1.0 - d.d_meas / d.d_model) * d.mbs),
    ]


def evaluate(cluster, verbose: bool = True) -> dict:
    """全族拟合 + 保留点评估。返回 {family_key: metrics}。"""
    decs = decompose(cluster)
    cal = next(d for d in decs if d.role == "calibration")
    holds = [d for d in decs if d.role == "holdout"]
    results = {}
    for fam in _families():
        p = fam.fit(cal)
        rows, errs, in_tol, signs_ok = [], [], 0, 0
        for d in holds:
            delta_pred = fam.predict(d, p)
            g_pred = d.t_off / (d.t_off + delta_pred)
            e = g_pred - d.g_meas
            errs.append(abs(e))
            if abs(e) <= 0.035:
                in_tol += 1
            if d.name in NEG_NAMES and g_pred < 1.0:
                signs_ok += 1
            rows.append((d.name, d.g_meas, g_pred, e))
        mae = sum(errs) / len(errs)
        n_neg = len([d for d in holds if d.name in NEG_NAMES])
        gate = (mae <= 0.025) and (in_tol >= 4) and (signs_ok == n_neg)
        results[fam.key] = {"param": p, "mae": mae, "in_tol": in_tol,
                            "signs_ok": signs_ok, "n_neg": n_neg,
                            "gate": gate, "rows": rows, "desc": fam.desc}
    if verbose:
        _report(decs, results)
    return results


def _report(decs, results) -> None:
    print("侦察表(模型步差几乎全是固定成本;实测暴露比连符号都不一致):")
    print("%-7s %6s %9s %+11s %+11s %9s %7s" %
          ("几何", "mbs", "实测G", "Δ实测ms", "Δ模型ms", "固定成本", "暴露比"))
    for d in decs:
        print("%-7s %6d %9.4f %+11.1f %+11.1f %9.1f %7.3f" %
              (d.name, d.mbs, d.g_meas, d.d_meas, d.d_model, d.fixed,
               d.d_meas / d.d_model))
    print()
    print("模型族回溯评估(拟合仅用 flag;门 = MAE≤0.025 且 ≥4/6 进 ±0.035 且负号全对):")
    print("%-4s %-34s %10s %8s %7s %7s %6s" %
          ("族", "结构", "参数值", "MAE", "±0.035", "负号", "门"))
    for k, r in results.items():
        print("%-4s %-34s %10.4f %8.4f %5d/6 %5d/%d %6s" %
              (k, r["desc"], r["param"], r["mae"], r["in_tol"],
               r["signs_ok"], r["n_neg"], "通过" if r["gate"] else "不过"))
    passed = [k for k, r in results.items() if r["gate"]]
    print()
    if passed:
        print("!! %s 回溯通过 —— 但这只是七个点上的模型选择,Tier-2 的门保持红色;" % passed)
        print("   解锁需要按 docs/07 的协议新采保留点,由人工核查后才动门。")
    else:
        print("**全族不过 —— 单参数全局重叠模型这条路走完了,负结果成立。**")
        print("下一步不是更多参数(1 个标定点只定得了 1 个参数,再多就是拿保留点")
        print("调参),而是新的测量:两臂各自的「双流重叠时间线」(docs/07 协议)。")


if __name__ == "__main__":
    from .calibrate import flat_supernode
    evaluate(flat_supernode())
