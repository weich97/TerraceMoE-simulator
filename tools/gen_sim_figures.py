# -*- coding: utf-8 -*-
"""生成仿真器结果图 F7-F10(SVG,文字走客户端字体,GitHub 直接渲染)。

与 tools/gen_figures.py(T-Route 消融,数字内嵌)不同,本脚本的每个数字都
**现场调用 sim/ 模块算出**(唯一例外:F12 内嵌实测记录,段首注明)——
图 = f(代码 + 标定常数),重跑 SVG 逐字节一致(蒙特卡洛固定种子 +
svg.hashsalt 固定 + 元数据去时间戳)。docs/05、docs/07、docs/08 表里的
数字以本脚本与 sim 模块输出为准。
跑法:python tools/gen_sim_figures.py   (需要 matplotlib)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import TwoSlopeNorm   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.calibrate import flat_supernode, synthetic          # noqa: E402
from sim.core import MoEGeometry, one_hop_call, two_hop_call  # noqa: E402
from sim.overlap import evaluate                              # noqa: E402
from sim.sweep import CHAIN_SCENARIOS                         # noqa: E402
from sim.uncertainty import breakeven_ratio, heatmap, mc_band  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none",
    "svg.hashsalt": "terracemoe-sim",   # 固定随机 id,连同 metadata Date=None
                                        # 保证同代码重跑 SVG 逐字节一致
    "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "PingFang SC",
                        "SimHei", "DejaVu Sans", "sans-serif"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

C = {"m0": "#8A8F98", "m1": "#B48EAD", "m2": "#5B8DB8", "m3": "#C4823B",
     "m4": "#3A7D5C", "m5": "#B35A5A", "band": "#5B8DB8"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------- F7 重叠模型族全灭
res = evaluate(flat_supernode(), verbose=False)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                              gridspec_kw={"width_ratios": [1.15, 1]})
order = ["M0", "M1", "M2", "M3", "M4", "M5"]
mk = {"M0": "x", "M1": "v", "M2": "o", "M3": "^", "M4": "s", "M5": "D"}
lo = hi = None
for k in order:
    r = res[k]
    xs = [row[1] for row in r["rows"]]     # 实测 G
    ys = [row[2] for row in r["rows"]]     # 预测 G
    ax.scatter(xs, ys, marker=mk[k], s=44, color=C[k.lower()], alpha=0.85,
               label="%s(MAE %.3f)" % (k, r["mae"]))
    pts = xs + ys
    lo = min(pts) if lo is None else min(lo, min(pts))
    hi = max(pts) if hi is None else max(hi, max(pts))
pad = 0.04
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333", lw=1)
ax.fill_between([lo - pad, hi + pad], [lo - pad - 0.035, hi + pad - 0.035],
                [lo - pad + 0.035, hi + pad + 0.035], color="#3A7D5C", alpha=0.12)
ax.text(hi + pad - 0.01, hi + pad - 0.032, "±0.035 容差带", color="#3A7D5C",
        ha="right", fontsize=8.5)
ax.set_xlabel("实测 G(6 个保留几何)")
ax.set_ylabel("预测 G")
ax.set_title("六个单参数重叠模型族:无一进带", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
_style(ax)

maes = [res[k]["mae"] for k in order]
ax2.bar(range(6), maes, width=0.55, color=[C[k.lower()] for k in order],
        edgecolor="white")
ax2.axhline(0.025, color="#B33", ls="--", lw=1.2)
ax2.text(5.3, 0.028, "门 MAE≤0.025", color="#B33", ha="right", fontsize=9)
for i, k in enumerate(order):
    ax2.text(i, maes[i] + 0.003, "%.3f" % maes[i], ha="center", fontsize=8)
ax2.set_xticks(range(6), order)
ax2.set_ylabel("保留点 MAE")
ax2.set_title("最好的族(M4)也差门 2 倍 —— 负结果成立", fontsize=11)
_style(ax2)
fig.suptitle("Tier-2 攻坚:单参数全局重叠模型此路不通(拟合只用 flag,保留点未参与)",
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f7-overlap-families.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F8 breakeven 热图
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
mats = [heatmap(chain) for _, chain in CHAIN_SCENARIOS[:2]]
_all = [v for mat, _, _ in mats for row in mat for v in row]
norm = TwoSlopeNorm(vmin=min(_all), vcenter=1.0, vmax=max(_all))   # 两 panel 共用,colorbar 对两边都成立
for ax, (name, chain), (mat, ratios, toks) in zip(axes, CHAIN_SCENARIOS[:2], mats):
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm,
                   origin="lower")
    for i in range(len(toks)):
        for j in range(len(ratios)):
            ax.text(j, i, "%.2f" % mat[i][j], ha="center", va="center",
                    fontsize=7.2,
                    color="#222" if 0.6 < mat[i][j] < 1.8 else "#eee")
    ax.set_xticks(range(len(ratios)), ["%g" % r for r in ratios], fontsize=8)
    ax.set_yticks(range(len(toks)), ["%d" % t for t in toks], fontsize=8)
    ax.set_xlabel("层级比(快边带宽 / 慢边带宽)")
    ax.set_ylabel("token / rank")
    be = breakeven_ratio(chain)
    ax.set_title("%s\nbreakeven 层级比 = %.2f(q=3)" % (name, be), fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.85, label="一跳耗时 / 两跳耗时(>1 两跳快)")
fig.suptitle("哪片区域值得两跳:绿区 = 两跳赢(通信级仿真,16 组 × 8)", y=1.04)
fig.savefig(os.path.join(OUT, "f8-breakeven-map.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F9 蒙特卡洛误差带
fig, ax = plt.subplots(figsize=(8.6, 4.6))
band_ratios = [1.03, 1.5, 2, 2.5, 3.2, 4, 4.5, 5.5, 6.5, 8, 11, 15.7]
tier_colors = ["#B35A5A", "#5B8DB8", "#3A7D5C"]
for (name, chain), col in zip(CHAIN_SCENARIOS, tier_colors):
    p5s, meds, p95s = [], [], []
    for r in band_ratios:
        p5, med, p95 = mc_band(r, chain)
        p5s.append(p5); meds.append(med); p95s.append(p95)
    ax.plot(band_ratios, meds, "o-", color=col, lw=1.6, ms=4, label=name)
    ax.fill_between(band_ratios, p5s, p95s, color=col, alpha=0.18)
    be = breakeven_ratio(chain)
    if 1.0 < be < 32.0:
        ax.axvline(be, color=col, ls=":", lw=1)
        ax.text(be, 0.32, "%.1f" % be, color=col, ha="center", fontsize=8.5)
ax.axhline(1.0, color="#333", lw=1)
ax.set_xscale("log")
ax.set_xticks(band_ratios)
ax.set_xticklabels(["%g" % r for r in band_ratios], fontsize=8)
ax.minorticks_off()
ax.set_xlabel("层级比(log 轴);虚线 = 各实现档的 breakeven")
ax.set_ylabel("一跳耗时 / 两跳耗时(>1 两跳快)")
_, _, flat_p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])
hier_p5, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])
ax.set_title("标定不确定度传播(400 次蒙特卡洛,带 = [p5, p95]):\n"
             "扁平列最有利情形 p95=%.2f %s;8x 列最不利情形 p5=%.2f %s"
             % (flat_p95, "≤1(判负鲁棒)" if flat_p95 <= 1.0 else "!! >1",
                hier_p5, ">1(方向鲁棒)" if hier_p5 > 1.0 else "!! <1"),
             fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f9-uncertainty.svg"), metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F10 规模效应与 α 出处
fig, ax = plt.subplots(figsize=(8.2, 4.2))
worlds = [32, 64, 128, 256, 512]
for like_meas, col, lab in ((True, "#3A7D5C", "α = 本机实测形状(>128 为借来的点)"),
                            (False, "#8A8F98", "α = 平坦 0.05 ms(反事实机器)")):
    ys = []
    for w in worlds:
        ng = w // 8
        c = synthetic(3.2, alpha_like_measured=like_meas,
                      chain_us_per_row=CHAIN_SCENARIOS[1][1])
        g = MoEGeometry(name="scale", n_groups=ng, R=8, k=6, M=2,
                        seq=4096, mbs=1, gbs=w * 4096)
        ys.append(one_hop_call(c, g) / two_hop_call(c, g))
    ax.plot(worlds, ys, "o-", color=col, lw=1.8, ms=5, label=lab)
ax.axvspan(128, 512, color="#C4823B", alpha=0.10)
ax.text(250, ax.get_ylim()[0] + 0.06, "α(256)/α(512) 是借来的曲线形状\n"
        "—— 阴影区结论随 α 出处摆动,换机必须重标", color="#9A6B2F", fontsize=8.5)
ax.axhline(1.0, color="#333", lw=1)
ax.set_xscale("log", base=2)
ax.set_xticks(worlds)
ax.set_xticklabels([str(w) for w in worlds])
ax.set_xlabel("总卡数(组数 × 8;层级比固定 3.2,融合 kernel 档)")
ax.set_ylabel("一跳耗时 / 两跳耗时")
ax.set_title("规模效应:大集群里两跳的额外赢面主要来自 α(world) 的增长\n"
             "—— 而 α 形状是机器性质,不是拓扑性质", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f10-scale-alpha.svg"), metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F11 判决床全景
# G 值直接取自 sim.validate.HOLDOUTS(步级实测真值,同一处出处);
# 每点的发数/散布是测量记录,在 HOLDOUTS 注释里,这里内嵌并同步维护。
from sim.validate import HOLDOUTS  # noqa: E402

GM = {h.geom.name: h.g_measured for h in HOLDOUTS}
NINFO = {"flag": "n=7\nt=3.27, p≈0.017", "tok2x": "n=3\n散布 0.8%",
         "tok4x": "n=3\n散布 0.6%", "k8m2": "n=1", "k8m4": "n=1",
         "n8": "n=1", "n4": "n=1"}

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
panels = [
    ("规模轴:节点数(带宽扁平机上\n节点越少,两跳可省越少)",
     [("4", "n4"), ("8", "n8"), ("16", "flag")], "节点数"),
    ("token 轴:micro-batch(token 越多\n越带宽主导,两跳越吃亏)",
     [("×1", "flag"), ("×2", "tok2x"), ("×4", "tok4x")], "token 倍率"),
    ("负载轴:k 与 M(跨组扇出越宽\n两跳越吃亏)",
     [("k6M2", "flag"), ("k8M2", "k8m2"), ("k8M4", "k8m4")], "路由负载"),
]
for ax, (title, pts, xlab) in zip(axes, panels):
    xs = list(range(len(pts)))
    ys = [GM[key] for _, key in pts]
    cols = ["#3A7D5C" if y > 1 else "#B35A5A" for y in ys]
    ax.bar(xs, [y - 1.0 for y in ys], width=0.5, bottom=1.0, color=cols,
           edgecolor="white")
    for x, y, (_, key) in zip(xs, ys, pts):
        ax.text(x, y + (0.006 if y >= 1 else -0.006), "%.4f" % y,
                ha="center", va="bottom" if y >= 1 else "top", fontsize=8.5)
        ax.text(x, 0.796, NINFO[key], ha="center", fontsize=7, color="#666")
    ax.axhline(1.0, color="#333", lw=1)
    ax.set_xticks(xs, [lab for lab, _ in pts])
    ax.set_xlabel(xlab)
    ax.set_ylim(0.78, 1.08)
    ax.set_title(title, fontsize=9.5)
    _style(ax)
axes[0].set_ylabel("G = t(一跳) / t(两跳)\n(>1 = 两跳快)")
fig.suptitle("判决床全景:7 个几何的端到端 G(带宽扁平超节点,16 组 × 8,同配置只换 a2a)"
             " —— 唯一的正值落在微观基准的两跳占优窗口(4096 tok 档),三条轴走向全部与机制一致",
             y=1.04, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f11-verdict-bed.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F12 单边 vs 集合 a2a
# 注意:与本脚本其余图不同,F12 的数字是**内嵌的实测记录**(带宽扁平超节点,
# 8 die 节点内,行宽对齐口径,中位数;量具与判据见 tools/onesided/ 与 docs/08),
# 不是 sim 计算 —— 重测请跑 tools/onesided/bench_onesided.py。
MB = [2.10, 4.19, 6.29, 8.39, 12.58, 16.78, 25.17]
A2A = [78.9, 92.2, 98.4, 100.8, 102.8, 102.9, 104.0]
OFF1 = [17.6, 23.3, 41.0, 46.5, 52.4, 53.6, 58.1]        # 官方形态 block_dim=1
BFY1 = [13.6, 14.5, 14.7, 14.9, 14.8, 13.5, 13.0]        # 蝶形 block_dim=1
BFY8 = [43.3, 50.1, 59.4, 63.3, 65.2, 64.2, 70.5]        # 蝶形 block_dim=8

# 各臂的作废档(耗时不足各自量具地板 3x,读数作废,画空心)——索引对照归档 JSON:
# 蝶形 x8: over_floor 1.3/2.3/2.9 -> {0,1,2};官方 x1: 2.1/-/2.7 -> {0,2};蝶形 x1: 2.7 -> {0}
VOIDS = {"BFY8": {0, 1, 2}, "OFF1": {0, 2}, "BFY1": {0}}


def _arm(ax, ys, void_idx, marker, color, label):
    ax.plot(MB, ys, marker + "-", color=color, lw=1.6, ms=5, label=label)
    vx = [MB[i] for i in sorted(void_idx)]
    vy = [ys[i] for i in sorted(void_idx)]
    ax.plot(vx, vy, marker, color=color, ms=7, markerfacecolor="white", zorder=3)


fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(MB, A2A, "o-", color="#333333", lw=2.0, ms=5, label="集合 a2a(HCCL,对照臂)")
_arm(ax, BFY8, VOIDS["BFY8"], "s", "#3A7D5C", "单边·蝶形 8 核(空心 = 不足地板 3×,作废)")
_arm(ax, OFF1, VOIDS["OFF1"], "^", "#5B8DB8", "单边·官方 alltoall 形态(每对端一流)")
_arm(ax, BFY1, VOIDS["BFY1"], "v", "#B35A5A", "单边·蝶形单核(原样 block_dim=1)")
best = max(b / a for i, (b, a) in enumerate(zip(BFY8, A2A)) if i not in VOIDS["BFY8"])
ax.annotate("单边最好 = %.2f× a2a\n(判据 ≥1.15× 于 ≥2 档,预注册)" % best,
            xy=(MB[-1], BFY8[-1]), xytext=(11, 44), fontsize=9, color="#3A7D5C",
            arrowprops=dict(arrowstyle="->", color="#3A7D5C", lw=1))
ax.set_xlabel("每对端载荷(MB,行宽对齐口径)")
ax.set_ylabel("有效带宽(GB/s,本 die 出口字节 / 墙钟)")
ax.set_ylim(0, 115)
ax.set_title("单边(aclshmem put)vs 集合 a2a,8 die 节点内:三种单边实现全部不过判据\n"
             "单核 MTE 天花板 ~14 → 8 核 5×提升 → 仍只有 a2a 的 0.68×", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="center right")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f12-onesided.svg"), metadata={"Date": None})
plt.close(fig)

print("6 figures ->", OUT)
