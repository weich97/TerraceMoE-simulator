# TerraceMoE Simulator:面向层级化集群的 MoE 通信方法与实测标定仿真

三件东西:**T-Route**(层级对齐的路由约束)与 **T-A2A**(两跳分层 all-to-all)的方法与参考实现;一条**适用判据**(你的集群该不该用);以及一个**用实测标定、用独立测量验证**的通信仿真器(`sim/`)——在一台机器上标定,回答许多台集群上的问题。

> **一句话定位:这套方法面向层级化特征明显的集群** —— 快侧(节点/机柜/超节点内)与慢侧(跨节点/跨柜)带宽差在 1.5 倍以上(q≥3 口径)的互联。我们在一台**带宽扁平**的超节点上把它完整建成并测到了底:**在扁平互联上两跳不划算**(见 [适用判据](#适用判据先跑这个))。开源它,是因为判据、推导和实现对层级化集群依然成立——同族方法已被多个公开工作在层级化互联上验证有效。

*T-Route (hierarchy-aligned routing constraints) and T-A2A (two-hop hierarchical all-to-all) for MoE expert parallelism, with a quantitative applicability criterion. Built and measured to ground truth on a bandwidth-flat supernode — where it does **not** pay. Released because the criterion, the derivation, and the implementation hold for hierarchical fabrics, where the same family of methods has public, positive evidence.*

---

## 方法

**T-Route** 在标准 top-k 路由上加两条约束:

1. **限组** —— 每个 token 的专家只落在 M 个组内(组边界 = 通信层级边界,M < N_g);
2. **组内等额** —— 每个入选组恰好 k/M 个专家。

前者让每 token 的跨组扇出有上界 M,后者让每条跨组消息的尺寸恒定。两条都是**架构性质,与数据无关**——通信量在编译期就有包络。与已有工作的关系:DeepSeek-V3 的 node-limited routing 只有限组;MoGE 的等额配额等价于 M = N_g。T-Route 是两者的合取且 M < N_g。

**T-A2A** 把 dispatch 的一跳 all-to-all 换成两跳:跨组只发 1 份到目标组的代表卡(Hop A,去重),组内再散给真正的专家卡(Hop B)。每 token 跨慢链路的载荷从 q = k/M 行降到 1 行,代价是快侧多搬 q(1−1/R) 行。**这笔交易只在慢链路显著更贵时划算** —— 这就是适用判据。

## 适用判据(先跑这个)

```
python tools/breakeven.py --ratio <你实测的 快侧带宽/慢侧带宽>
```

两跳在字节上净赚的充要条件(组内 R 卡、每组 q 个专家):

$$r_{be} = \frac{(1-1/R)\,q}{q-1} \quad<\quad \frac{\beta_{fast}}{\beta_{slow}}$$

R=8 时:q=2 → 1.75,q=3 → 1.31,q=6 → 1.05,q=8 → 1.00。

| 互联形态 | 快/慢带宽比(量级) | 判定 |
|---|---|---|
| NVLink 域内 vs 跨节点 IB/RoCE | ~3–18× | **通过**。同族方法有公开正面证据(DeepSeek-V3 的 node-limited 路由 + IB→NVLink 两跳转发;TeleChat3-MoE 的分层 A2A 报 EP=16 训练吞吐 +15%;Pangu Ultra MoE 同族) |
| 服务器内 HCCS vs 跨服务器 RoCE | ~8× | **通过**,同上 |
| **带宽扁平的超节点内部**(统一交换,跨节点≈节点内) | **~1.0** | **不通过。我们实测:同配置端到端,两跳在大 micro-batch 下慢 12–22%,小 micro-batch 下打平。别用。** |

两条务必注意:

- **用实测比值,不要用标称。** 集合通信的有效带宽与标称经常差 2 倍以上,而且要在**你的真实消息尺寸**上测——带宽对消息尺寸/对齐的依赖可能制造整段假结论(我们踩过)。
- **字节只是账的一半。** 两跳多付一次集合通信的固定开销;判据通过说明"值得做实验",不等于"必然更快"。完整清单见 [docs/03-applicability.md](docs/03-applicability.md)。

## T-Route 的质量代价(与通信无关,独立成立)

![质量轴](docs/assets/f1-loss-axis.svg)

四条轴(质量/下游/负载/步时)的**完整消融结果、逐种子图与撤回记录**见 [docs/06-troute-results.md](docs/06-troute-results.md)。头条表(13.14B 总参 / 1.33B 激活,4 模式 × 4 种子,每臂 62.9B token,留出集 val loss):

| 档 | 约束 | Δ vs 无约束 top-k | 90% CI |
|---|---|---|---|
| `group_limited` | 仅限组(8 选 4) | +0.00276 | [+0.00223, +0.00329] |
| `quota_only` | 仅等额(= MoGE) | +0.00895 | [+0.00726, +0.01064] |
| **`full`(T-Route)** | 限组 + 等额 | **+0.00339** | [+0.00224, +0.00455] |

- 代价 +0.0034 nats,占预注册无损阈值 0.1 nats 的 **3.4%**;逐种子 12/12 同号。这不是"测不出差异"——三档的 CI 都不含 0,是分辨出来之后确认小。
- **`full` 的代价约为 `quota_only` 的 38%(低约 62%)**:限组给等额留了泄压口("用哪 M 组"仍自由)。
- 下游(HellaSwag / LAMBADA,±1.0 pp 预注册 TOST):等价。
- 负载:约束档的专家级负载熵不低于对照——"隐性容量损失换效率"排除。

**边界**:组间均衡是统计性质不是架构保证(对抗输入下组级 CV 可达 1.0);已验证到上述规模,不能线性外推到更大宽度或更极端的 M/N_g。

## 仓库结构

| 路径 | 内容 | 状态 |
|---|---|---|
| `terrace/routing.py` | T-Route 参考实现,四种消融模式同一函数切换 | **已验证**(质量消融 + 性质测试) |
| `terrace/ta2a*.py` | T-A2A 两跳链:计划、派发、打包、可微接缝 | **已验证位级正确**(由全仓 274 条 CPU 测试把守;端到端只在扁平超节点测过,见判据) |
| `terrace/ops/` | 到达链融合算子(AscendC):passthrough / K1 / K2 + CPU 可执行规格 | passthrough 设备位级验证通过;**K1 算法已证正确、设备端翻译有一处未修的越界 bug;K2 未上机验证**(见 [docs/04-kernel-status.md](docs/04-kernel-status.md)) |
| `sim/` | 实测标定的通信仿真器:集群规格 → 一跳/两跳耗时与外推(见 [docs/05-simulator.md](docs/05-simulator.md)) | **Tier-1(通信微观)验证门通过**(误差中位 8.1%);Tier-2(端到端步级)如实标注不通过,步级外推封禁 |
| `tools/breakeven.py` | 适用判据(解析式;与 sim/ 是同一套账的两条独立实现,测试互证) | — |
| `tests/` | 274 条测试,纯 CPU 可跑(不需要 NPU) | 全绿 |
| `docs/` | 设计文档 ×5 + 消融结果全集(docs/06,图先行) | — |
| `tools/gen_figures.py` | 结果图生成(数字内嵌,图可复现) | — |

**不在本仓库**:训练框架接入(上游训练栈/Megatron 垫片)、集群与测量脚本、实验原始数据、内部记录。接缝契约见 [docs/02-ta2a-design.md](docs/02-ta2a-design.md),接入方按契约自写垫片。

## 跑测试

```
python -m pytest tests/ -q        # 只需 torch,CPU 即可,约 70 秒
python -m sim.validate_micro      # 仿真器 Tier-1 验证门
python -m sim.sweep               # 跨集群外推(入口处查门;输出全部标注「仿真」)
```

## 诚实声明

这不是一个带着胜利数字发布的仓库。我们在一台**扁平**超节点上把这套方法建成、测透,结论是**在那类机器上不要用它**;把方法、判据和"为什么不划算"的完整推导开源,是让层级化集群的使用者不必重走这段路——**先算判据,再决定接入**。所有出现的数字要么给出推导,要么标注实测口径;涉及具体机器与内部路径的信息一律移除。

## License

Apache-2.0
