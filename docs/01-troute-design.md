# T-Route 设计

## 定义

标准 top-k 路由:token 的隐向量对 E 个专家打亲和分,取前 k 个。T-Route 在其上加两条约束,组边界 = 通信层级边界(节点 / 机柜 / 超节点,由部署决定),E 个专家均分为 N_g 组:

1. **限组(group-limited)**:先按"组内最高亲和分"选出 M 个组(M < N_g),token 的专家只从这 M 个组里出。
2. **组内等额(equal quota)**:每个入选组恰好取 q = k/M 个专家(要求 M | k)。

## 五步伪码

```
scores = sigmoid(h @ E_experts + bias)          # 亲和分,bias 为 aux-loss-free 均衡偏置
group_score = max_pool(scores, per_group)       # 每组的代表分
groups = topM(group_score)                      # 选 M 个组
experts = 对每个入选组: topq(scores[该组])       # 每组恰 q 个
gates = normalize(scores[experts])              # 门控归一化
```

四种消融模式(`global_topk` / `group_limited` / `quota_only` / `full`)在同一函数内以 `mode` 切换——**同一条代码路径是消融可比的前提**。见 `terrace/routing.py`。

## 哪些性质无条件成立,哪些只是统计成立

**无条件(架构保证,任何输入)**:

- 每 token 恰好 k 个专家、互不相同;
- 专家张成的组数 ≤ M ⇒ **每 token 的跨组扇出有上界**;
- 每个入选组恰 q 个专家 ⇒ **每条跨组消息的行数定长**(q × 每行 H·dtype 字节)。

**只是统计(可被对抗输入打破)**:

- **组间负载均衡**。M < N_g 时"用哪 M 组"由数据决定;均衡靠 bias 驱动,是统计性质。真实语料下组级 CV 与无约束对照同区间,但对抗构造下组级 CV 可达 1.0(`tests/test_routing.py` 附可复现反例)。要真正数据无关的流量矩阵,需要批级容量约束或全局分配——那是另一种路由算法,精度消融得重做。

## 与已有工作的逐条差异

| | 限组 | 等额 | M < N_g |
|---|---|---|---|
| DeepSeek-V3 node-limited | ✔ | ✘ | ✔ |
| MoGE(等额配额) | ✘ | ✔ | ✘(M = N_g) |
| **T-Route** | ✔ | ✔ | ✔ |

合取的意义:限组给等额留了泄压口——`quota_only` 要求在**全部**组里各放专家,`full` 只在被选中的 M 组内各放 q 个,"用哪 M 组"仍自由。质量消融证实 `full` 的代价约为 `quota_only` 的 38%,低约 62%(README 表:+0.00339 vs +0.00895)。

## 为什么这两条约束对通信有用

- 扇出上界 M ⇒ 跨组消息**条数**编译期已知;
- 等额 q ⇒ 跨组消息**尺寸**编译期已知(定长,无需先交换 counts 再算 splits);
- 两条合起来,分层 all-to-all(T-A2A)的 Hop A 才能做成**定形**通信——这是把 dispatch 从"数据相关的变长 a2a"变成"静态编排"的前提。

注意:定形只覆盖 Hop A(跨组段)。组内(Hop B)"哪 q 个专家"仍随数据变,组内散布仍是变长的——见 02 篇。
