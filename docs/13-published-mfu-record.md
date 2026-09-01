# 13. Published MoE training runs: cluster size against model size and MFU

**Status: gathered, not adopted.** This is a literature sweep run to test whether the
published record constrains the model-size band a cluster runs well, which is the lower
edge `sim/envelope.py` computes from the cost model. It has been through source and
arithmetic verification passes but nothing here has been checked by hand against the
primary sources, and none of it is used by any code in this repository. Treat every
figure as a lead to verify, not as a datum.

Extracted 93 runs before verification. Dropped and corrected entries are listed in the
scope section below.

---

## Scope: what survived

Dropped per verifier tags — `[drop]`: Kimi k1.5, Step‑2, FasterMoE, Janus, ScheMoE, MixNet. `[unsourced]`: DBRX, Llama 4 Scout, Llama 4 Maverick, Llama 4 Behemoth, Yuan 2.0‑M32, Comet. Duplicated runs merged across streams (DeepSeek‑V3, Pangu Ultra MoE, Skywork‑MoE, MegaScale‑MoE, MegaScale 175B, Llama 3 405B, MoE Parallel Folding, Megatron‑Core, Hunyuan‑Large, Ling‑Plus).

Two verifier corrections are load‑bearing below and are applied: the **MoE Parallel Folding Table 4 scaling sweeps use global batch 1024**, not 256 (the 256 figures belong to the Table 1/3 optimal‑config comparison), and **MegaScale's 12,288‑GPU aggregate PFLOP/s cell is defective** — only its MFU and tokens/s are used.

**The single most important fact about the record before any table:** of the surviving runs, roughly half never state how many accelerators they used. Missing a cluster size entirely: DeepSeek‑V2, DeepSeek LLM 67B, Qwen2‑57B‑A14B, Qwen3‑235B‑A22B, Qwen3‑30B‑A3B, Qwen3‑Coder‑480B, Qwen3‑Coder‑Next, Mixtral 8x7B, Mixtral 8x22B, Grok‑1, Switch‑C, Switch‑XXL, **Kimi K2 (1.04 T)**, GLM‑4.5, GLM‑4.5‑Air, Hunyuan‑Large, Step‑3, TeleChat3‑MoE 105B, Ling‑Plus, Chinchilla 70B. Frontier‑scale MoE reports systematically omit the one number this question is about.

---

## 1. Surviving runs with a stated accelerator count

### A. MoE runs (sorted by accelerator count)

Kind: **P** = production pre‑training, **RL** = reinforcement‑learning run, **B** = systems benchmark. "P/acc" = total parameters ÷ accelerators; "Act/acc" = active parameters ÷ accelerators. Throughput column names the metric the paper actually reports.

| Model | Kind | Total | Active | Accelerators | P/acc | Act/acc | Reported throughput (metric named) |
|---|---|---|---|---|---|---|---|
| MPF Qwen2‑57B‑A14B | B | 57 B | 14 B | 64 H100 | 891 M | 219 M | **MFU** 39.9 % (BF16) |
| GRIN MoE 16x3.8B | B | 42 B | 6.6 B | 64 H100 | 656 M | 103 M | "Throughput Per GPU" 7,077 — **unit not printed in the paper**; throughput study only, pre‑training hardware unstated |
| DeepSpeed‑MoE 1.3B+MoE‑128 | P | 52 B | 1.3 B | 128 A100 | 406 M | 10.2 M | **training samples/sec** 372 (aggregate) |
| MCore Qwen3‑235B (131 K seq) | B | 235 B | 22 B | 128 GB300 | 1,836 M | 172 M | **TFLOPS/GPU** 1,150; **tokens/s/GPU** 1,556 |
| MPF Mixtral 8x22B | B | 141 B¹ | 39 B¹ | 128 H100 | 1,102 M | 305 M | **MFU** 52.2 % |
| MPF Qwen2‑57B‑A14B | B | 57 B | 14 B | 128 H100 | 445 M | 109 M | **MFU** 39.7 % |
| MPF Mixtral‑8x22B‑G8T8 | B | — (not stated) | — | 128 H100 | — | — | **MFU** 30.0 % |
| OLMoE‑1B‑7B (ablation) | B | 6.9 B | 1.3 B | 128 H100 | 53.9 M | 10.2 M | **tokens/s/GPU** 23,600 (dense counterpart 37,500) |
| MegaScale‑MoE Internal‑352B | B | 352 B | not stated | 240 H800 | 1,467 M | — | **MFU** 32.48 %; **tokens/s** 272.9 k |
| MCore DeepSeek‑V3‑685B | B | 685 B | 37 B | 256 GB300/GB200 | 2,676 M | 145 M | **TFLOPS/GPU** 1233 / 1048 / 857; **tokens/s/GPU** 4,730 / 4,020 / 3,298. **No MFU reported** |
| MCore Qwen3‑235B | B | 235 B | 22 B | 256 GB300/GB200/H100 | 918 M | 85.9 M | **TFLOPS/GPU** 974 / 919 / 750 / 320; **tokens/s/GPU** 6,583 / 6,212 / 5,100 / 2,132. **No MFU** |
| MPF Mixtral 8x22B | B | 141 B¹ | 39 B¹ | 256 H100 | 551 M | 152 M | **MFU** 50.7 % |
| MPF Qwen2‑57B‑A14B | B | 57 B | 14 B | 256 H100 | 223 M | 54.7 M | **MFU** 38.1 % |
| MPF Mixtral‑8x22B‑G8T8 | B | — | — | 256 H100 | — | — | **MFU** 29.3 % (printed "18,4%" for the MCore baseline row) |
| MPF Llama3‑8x70B | B | — (not stated) | — | 256 H100² | — | — | **MFU** 43.7 % |
| OLMoE‑1B‑7B (final run) | P | 6.9 B | 1.3 B | 256 H100 | 27.0 M | 5.08 M | **none reported** for this run |
| MegaScale‑MoE Internal‑352B | B | 352 B | — | 480 H800 | 733 M | — | **tokens/s** 498.6 k (MFU figure‑only) |
| DeepSeek‑R1‑Zero | RL | 671 B | 37 B | 512 H800 | 1,311 M | 72.3 M | **none** — only 198 h wall clock / 101 K GPU‑h |
| DeepSeek‑R1 | RL | 671 B | 37 B | 512 H800 | 1,311 M | 72.3 M | **none** — only ~80 h / 41 K GPU‑h |
| Phi‑3.5‑MoE 16x3.8B | P | 42 B | 6.6 B | 512 H100‑80G | 82.0 M | 12.9 M | **none reported**; *derived* tokens/s/GPU 4,816 = 4.9e12 ÷ (512 × 23 d × 86,400 s) |
| MPF Mixtral 8x22B | B | 141 B¹ | 39 B¹ | 512 H100 | 275 M | 76.2 M | **MFU** 48.9 % |
| MPF Qwen2‑57B‑A14B | B | 57 B | 14 B | 512 H100 | 111 M | 27.3 M | **MFU** 36.6 % |
| MPF Mixtral‑8x22B‑G8T8 | B | — | — | 512 H100 | — | — | **MFU** 26.7 % |
| MegaScale‑MoE Internal‑352B | B | 352 B | — | 720 H800 | 489 M | — | **tokens/s** 740.1 k |
| MegaScale‑MoE Internal‑352B | B | 352 B | — | 960 H800 | 367 M | — | **tokens/s** 963.8 k |
| GLaM 64B/64E | P | 1.2 T | 96.6 B | 1,024 TPU‑v4 | 1,172 M | 94.3 M | **none** — 574 h / 213 MWh for 280 B tokens; *derived* 132 tokens/s/chip = 280e9 ÷ (574×3600) ÷ 1024 |
| MCore DeepSeek‑V3‑685B | B | 685 B | 37 B | 1,024 H100 | 669 M | 36.1 M | **TFLOPS/GPU** 368; **tokens/s/GPU** 1,412. No MFU |
| MPF Mixtral 8x22B | B | 141 B¹ | 39 B¹ | 1,024 H100 | 138 M | 38.1 M | **MFU** 44.9 % |
| MPF Qwen2‑57B‑A14B | B | 57 B | 14 B | 1,024 H100 | 55.7 M | 13.7 M | **MFU** 33.4 % |
| MPF Mixtral‑8x22B‑G8T8 | B | — | — | 1,024 H100 | — | — | **MFU** 25.5 % |
| MPF Llama3‑8x70B | B | — | — | 1,024 H100 | — | — | **MFU** 41.5 % |
| MegaScale‑MoE Internal‑352B | B | 352 B | — | 1,440 H800 | 244 M | — | **MFU** 27.89 %; **tokens/s** 1,407.7 k |
| MiniMax‑Text‑01 | P | 456 B | 45.9 B | 1,500–2,500 H800³ | 182–304 M | 18.4–30.6 M | **none for training** (the >75 % MFU is *inference* on H20) |
| Skywork‑MoE | P | 146 B | 22 B | 1,536 A800‑80G | 95.1 M | 14.3 M | **MFU** 38 % **and tokens/GPU/s** 690 |
| DeepSeek‑V3 | P | 671 B | 37 B | 2,048 H800 | 328 M | 18.1 M | **MFU** 43.73 % non‑causal / 38.94 % causal (BF16 peak); **TFLOPS/GPU** 432 / 385; **tokens/day** 272.80 B |
| GShard MoE(2048E,36L) | P | 600 B | not stated | 2,048 TPU‑v3 | 293 M | — | **steps/sec** 0.72 at 4 M tokens/batch; *derived* 1,406 tokens/s/core. Per‑operator ">85 % peak FLOPS" (FFN) / ">30 %" (attention) — **not end‑to‑end MFU** |
| Pangu Pro MoE 72B‑A16B | P | 71.99 B | 16.50 B | 4,096 Ascend 800T A2 | 17.6 M | 4.03 M | **relative MFU only** ("35 % relative increase"); baseline printed as "–", so **no absolute MFU exists** |
| TeleChat3‑MoE 438B | P | 438 B | not stated | 4,096⁴ | 107 M | — | **no absolute throughput** — relative gains only |
| Pangu Ultra MoE 718B | P | 718 B | 39 B | 6,000 Ascend | 120 M | 6.50 M | **MFU** 30.0 % **and TPS** 1.46 M tokens/s (baseline 18.9 % / 0.61 M on 4 K NPUs) |
| TeleChat3‑MoE 1119B | P | 1,119 B | not stated | 8,192 Ascend | 137 M | — | **no absolute throughput** — only relative gains (15 % hierarchical EP, 25–30 % firmware, etc.) |

¹ Mixtral 8x22B's 141 B/39 B come from Mistral's own announcement and released `config.json`; the systems paper never restates them.
² Table 4 prints "128" for this model's Folding and FSDP+EP rows, but Table 3 configures Llama3‑8x70B only at 256 GPUs and the body text describes this exact series. Verified as a printed typo; listed at 256.
³ Cluster size varied *during* the run — no single count, so no per‑GPU derivation is possible.
⁴ The 4,096 figure comes from an automatic‑parallelisation *search example*, not a stated production cluster. Excluded from all statistics below.

### B. Dense contrast runs (all marked dense)

| Model | Total = active | Accelerators | P/acc | Reported throughput (metric named) |
|---|---|---|---|---|
| Megatron‑SP 22B | 22 B | 8 A100 | 2,750 M | **MFU** 41.5 % / **HFU** 43.7 % |
| Megatron‑LM PTD‑P 1.7B | 1.7 B | 32 A100 | 53.1 M | **TFLOPS/GPU** 137, 44 % of peak — **recomputation‑inclusive, HFU‑flavoured** |
| Megatron‑SP 175B | 175 B | 64 A100 | 2,734 M | **MFU** 51.4 % / **HFU** 52.8 % |
| DeepSpeed 6.7B dense | 6.7 B | 128 A100 | 52.3 M | **training samples/sec** 70 (quality‑matched pair to the 52 B MoE above, same cluster) |
| MegaScale 175B | 175 B | 256 Ampere | 684 M | **MFU** 65.3 % (batch 768) |
| Megatron‑SP 530B | 530 B | 280 A100 | 1,893 M | **MFU** 56.0 % / **HFU** 57.0 % |
| MegaScale 175B | 175 B | 512 | 342 M | **MFU** 63.5 % |
| Megatron‑SP 1T | 1,008 B | 512 A100 | 1,969 M | **MFU** 56.3 % / **HFU** 57.0 % |
| TeleChat2 115B | 115 B | 512 Ascend | 225 M | **MFU** 36.3 % (DP8/TP8/PP8, 1 M tokens/batch) |
| MegaScale 175B | 175 B | 768 / 1,024 | 228 / 171 M | **MFU** 61.3 % / 59.0 % |
| Megatron‑SP 530B, DP=8 | 530 B | 2,240 A100 | 237 M | **MFU** 54.2 % (down from 56.0 % at 280) |
| MegaScale 175B | 175 B | 3,072 / 6,144 / 8,192 / 12,288 | 57.0 / 28.5 / 21.4 / 14.2 M | **MFU** 59.1 / 57.3 / 54.9 / 55.2 % (batch 6144) |
| Megatron‑LM PTD‑P 1008B | 1,008 B | 3,072 A100 | 328 M | **TFLOPS/GPU** 163, 52 % of peak — recomputation‑inclusive |
| TeleChat2 115B | 115 B | 4,096 Ascend | 28.1 M | **MFU** 33.8 % (DP64/TP8/PP8, 4 M tokens/batch) |
| PaLM 540B | 540.35 B | 6,144 TPU‑v4 | 87.9 M | **MFU** 46.2 % **and HFU** 57.8 % **and tokens/s** 238.3 K |
| Llama 3.1 405B | 405 B | 8,192 H100 | 49.4 M | **MFU** 43 % (BF16) **and TFLOPs/GPU** 430 |
| Pangu Ultra 135B | 135 B | 8,192 Ascend | 16.5 M | **MFU** ~43 % baseline → **over 52 %** optimised |
| Llama 3.1 405B | 405 B | 16,384 H100 | 24.7 M | **MFU** 41 % **and TFLOPs/GPU** 400 |

---

## 2. The relation, as a band

**MoE, production pre‑training runs with a stated cluster (n = 10; 8 report active params):**

| Quantity | Min | Median | Max | Spread |
|---|---|---|---|---|
| Total params per accelerator | **17.6 M** (Pangu Pro MoE) | **128 M** | **1,172 M** (GLaM) | **67×** |
| Active params per accelerator | **4.03 M** (Pangu Pro MoE) | **11.5 M** | **94.3 M** (GLaM) | **23×** |

**All MoE rows including RL and benchmark sweeps (n = 32; 25 with active params):**

| Quantity | Min | Median | Max | Spread |
|---|---|---|---|---|
| Total params per accelerator | **17.6 M** | **387 M** | **2,676 M** (DeepSeek‑V3‑685B on 256 GB200) | **152×** |
| Active params per accelerator | **4.03 M** | **54.7 M** | **305 M** (Mixtral 8x22B on 128 H100) | **76×** |

**Dense contrast (n = 30):** 14.2 M – 2,750 M per accelerator, median 82 M, spread **193×**.

**How wide is the band? So wide that it constrains nothing, and I will not fit a line through it.**

The decisive evidence is not the aggregate spread — it is the spread *at a single, fixed cluster size*, which removes scale as a variable entirely:

- **At 256 accelerators:** OLMoE‑1B‑7B at 27.0 M params/GPU and Megatron‑Core DeepSeek‑V3‑685B at 2,676 M params/GPU. **99× apart, both running.**
- **At 1,024 accelerators:** MPF Qwen2‑57B at 55.7 M and GLaM 64B/64E at 1,172 M. **21× apart.**
- **At 512 accelerators:** Phi‑3.5‑MoE at 82 M and DeepSeek‑R1 at 1,311 M. **16× apart.**
- **At 4,096–8,192 accelerators:** Pangu Pro MoE at 17.6 M and TeleChat3‑MoE 1119B at 137 M — and dense Pangu Ultra 135B at 16.5 M on the *same vendor's 8,192‑NPU cluster* as the 1119 B MoE at 137 M.

A "range of model sizes a cluster runs well" that spans 16–99× at fixed cluster size is not a design constraint. Anyone sizing a cluster from this record would be told that almost any model fits.

What the record *does* pin, weakly, are the two edges — and only one of them by memory:

- **Upper edge (observed, not demonstrated):** ~2.7 B total parameters per accelerator on 192 GB devices (DeepSeek‑V3‑685B on GB200/GB300, PP4×VPP4×EP64, which Megatron‑Core states requires 199.5 GB/GPU in BF16); ~2.0 B on 80 GB devices (Megatron‑SP 1T on 512 A100‑80GB). This ceiling is a function of the parallelism strategy, not the parameter density: MoE Parallel Folding Table 3 shows Llama3‑8x70B OOM under FSDP and under TP+EP+DP at 256 GPUs while MCore w/ Folding runs at 41.6 % MFU on the identical hardware. MegaScale likewise "decrease[s] the batch size to 768 due to GPU memory limit" at 256–1024 GPUs. The ceiling moves with HBM and with software, and no paper reports the parameter density at which it actually broke.
- **Lower edge:** see §3. It is not in the record.

---

## 3. The lower edge specifically

### 3a. What is measured: a slope, not a cliff

Every published MFU‑versus‑scale sweep holds global batch fixed and adds accelerators — which *is* the "too small a model on too many accelerators" experiment. Every one of them produces a gentle, near‑log‑linear decline. Computing the relative loss per doubling of accelerator count:

| Sweep | Range | MFU | Per doubling |
|---|---|---|---|
| MegaScale‑MoE Internal‑352B (MoE) | 240 → 1,440 | 32.48 → 27.89 % | **−5.7 %** |
| MPF Mixtral‑8x22B‑G8T8 (MoE, fine‑grained) | 128 → 1,024 | 30.0 → 25.5 % | **−5.3 %** |
| MPF Mixtral 8x22B (MoE) | 128 → 1,024 | 52.2 → 44.9 % | **−4.9 %** |
| MPF Qwen2‑57B‑A14B (MoE) | 64 → 1,024 | 39.9 → 33.4 % | **−4.3 %** |
| MPF Llama3‑8x70B (MoE, largest) | 256 → 1,024 | 43.7 → 41.5 % | **−2.5 %** |
| MegaScale 175B, batch 768 (dense) | 256 → 1,024 | 65.3 → 59.0 % | −4.9 % |
| Llama 3.1 405B (dense) | 8,192 → 16,384 | 43 → 41 % | −4.7 % |
| MegaScale 175B, batch 6144 (dense) | 3,072 → 12,288 | 59.1 → 55.2 % | −3.4 % |
| TeleChat2 115B (dense, Ascend) | 512 → 4,096 | 36.3 → 33.8 % | −2.4 % |
| Megatron‑SP 530B → DP=8 (dense) | 280 → 2,240 | 56.0 → 54.2 % | −1.1 % |

Across two vendors, three silicon families, dense and MoE, the published slope is **−2 % to −6 % relative MFU per doubling of accelerator count at fixed global batch**, and *larger models decline more slowly* ("The results show the scalability of MoE parallel folding up to 16x nodes with little MFU drops, especially for large-scale models like Llama3-8x70B, where the MFU only drops from 43.7% to 41.5%").

**No published curve shows a knee.** Nobody has pushed a run into the collapse regime and reported it.

### 3b. The causal statements — and they disagree

Two papers attribute their decline, and to different mechanisms.

MegaScale‑MoE blames pipeline bubbles:

> "As the number of GPUs increases, the MFU (Model FLOPs Utilization) of MegaScale-MoE declines from 32.48% to 27.89%. This is expected, as the batch size is fixed and the number of micro-batches for each pipeline decreases with more GPUs, leading to more bubbles."

Llama 3 blames per‑DP‑group batch, and gives the only *quantified* per‑device‑batch elasticity in the record:

> "The slight drop in MFU to 41% on 16K GPUs with DP=128 compared to 43% on 8K GPUs with DP=64 is due to the lower batch size per DP group needed to keep the global tokens per batch constant during training."

Table 4 supplies the numbers behind it: batch/DP falls **32 → 16 sequences** (262,144 → 131,072 tokens per DP group) and MFU falls 43 % → 41 %. Halving the per‑DP‑group batch cost 2 MFU points. That is the closest thing to a measured lower‑edge gradient anywhere in the record, and it is dense.

Hagemann et al. complicate the naive reading further — smaller *micro*-batches were better, not worse:

> "For all model types, a micro-batch size of 1 achieves the highest MFU. Generally, smaller micro-batch sizes correlate with better MFU performance."

So per‑device work is not one quantity. Micro‑batch size, micro‑batches per pipeline, and per‑DP‑group batch pull in different directions, and no paper separates them.

### 3c. Is there a stated minimum per‑device batch? **No.**

I searched the surviving record. **No paper states a minimum per‑device batch size, in tokens or sequences, below which MFU falls off.** The nearest statements are mechanistic, not numeric:

- Ludziejewski et al. 2025 names the mechanism without measuring it: *"it may only be possible to run a large model with a small batch size due to limited GPU memory, leading to low hardware utilization."*
- Zhang et al. bound the *global* batch from the optimisation side — *"CBS scales primarily with data size rather than model size"* — which caps how many accelerators can be fed before per‑device batch must shrink, but attaches no MFU number.
- Megatron‑Core's *"For DeepSeek-V3 on 256 H100 GPUs, at around 50% MFU and sequence lengths of at least 16K…"* is a stated assumption inside an optimizer‑offloading trade‑off discussion, **not a measurement of any tabulated run** (verifier‑flagged). It should not be cited as a measured MFU.

What *can* be derived, from paper‑stated global batch ÷ paper‑stated accelerator count, is where the published record actually sits. This is an aggregate work‑per‑device figure, **not** a per‑DP‑group batch — TP and PP replicate tokens across devices, so it understates per‑device token load in TP/PP‑heavy configurations:

| Run | Arithmetic | tokens/accelerator/step | Reported |
|---|---|---|---|
| MCore DeepSeek‑V3 @256 | 8192×4096 ÷ 256 | 131,072 | 1,233 TFLOPS/GPU |
| MPF Qwen2‑57B @64 | 1024×4096 ÷ 64 | 65,536 | MFU 39.9 % |
| DeepSeek‑V3 (production) | 15,360×4096 ÷ 2,048 | 30,720 | **MFU 43.73 %** |
| MegaScale‑MoE @240 | 720×8192 ÷ 240 | 24,576 | MFU 32.48 % |
| **MegaScale‑MoE @1440** | 720×8192 ÷ 1,440 | **4,096** | **MFU 27.89 %** |
| **MPF Qwen2‑57B / Mixtral / G8T8 @1024** | 1024×4096 ÷ 1,024 | **4,096** | **MFU 33.4 % / 44.9 % / 25.5 %** |
| Pangu Pro MoE, reasoning phase | 16 M ÷ 4,096 | 3,906 | relative MFU only |
| GShard 600B | 4 M ÷ 2,048 | 1,953 | per‑operator peak only |
| Pangu Pro MoE, general phase | 4 M ÷ 4,096 | **977** | relative MFU only |
| — dense — | | | |
| Llama 3.1 405B @16,384 | 16 M ÷ 16,384 | 977 | MFU 41 % |
| MegaScale 175B @12,288 | 6144×2048 ÷ 12,288 | 1,024 | MFU 55.2 % |
| Megatron‑SP 22B @8 | 4×2048 ÷ 8 | 1,024 | MFU 41.5 % |
| TeleChat2 115B @4,096 | 4 M ÷ 4,096 | 977 | MFU 33.8 % |
| PaLM 540B | 2048×2048 ÷ 6,144 | **683** | **MFU 46.2 % / HFU 57.8 %** |

**The single sharpest finding in this section: no MoE run anywhere in the surviving record reports an absolute MFU below ~4,096 tokens per accelerator per step.** The dense record goes 4–6× lower — to 683 tokens/accelerator/step at PaLM's 46.2 % MFU — and still shows no collapse. The two MoE runs that go below 4,096 (Pangu Pro MoE at 977 and 3,906) publish only a *relative* MFU, so they contribute nothing.

The lowest absolute MoE MFU published under a competent configuration is **25.5 %** (Mixtral‑8x22B‑G8T8, 1,024 H100). That is a degradation, not a collapse.

### 3d. Minimum tokens per expert? **Not stated anywhere.**

No paper states a minimum tokens‑per‑expert. Deriving the *global* figure from stated global batch × top‑k ÷ routed experts (this is a global count; the per‑expert‑instance figure requires the DP degree, which most papers do not give):

| Run | Arithmetic | Global tokens/expert/step |
|---|---|---|
| MiniMax‑Text‑01 (final) | 128 M × 2 ÷ 32 | 8.0 M |
| GLM‑4.5 | 64 M × 8 ÷ 160 | 3.2 M |
| DeepSeek‑V3 | 62.91 M × 8 ÷ 256 | 1.97 M |
| Pangu Pro MoE (reasoning) | 16 M × 8 ÷ 64 | 2.0 M |
| DeepSeek‑V2 | 37.75 M × 6 ÷ 160 | 1.42 M |
| Kimi K2 | 67 M × 8 ÷ 384 | 1.40 M |
| MPF Mixtral 8x22B | 4.19 M × 2 ÷ 8 | 1.05 M |
| Pangu Pro MoE (general) / OLMoE | 4 M × 8 ÷ 64 | 0.50 M |
| MPF Qwen2‑57B‑A14B | 4.19 M × 8 ÷ 64 | 0.52 M |
| MegaScale‑MoE Internal‑352B | 5.90 M × 3 ÷ 32 | 0.55 M |
| **GShard 600B** | 4 M × 2 ÷ 2,048 | **3,906** |

The frontier clusters at 0.5 M – 8 M globally. GShard is three orders of magnitude below everyone else — 2,048 experts on 2,048 cores, one expert per device, ~3,900 tokens per expert per step — and it still reported *">85% peak FLOPS"* on the feed‑forward and projection operators. That single data point argues the expert GEMM itself is not the binding constraint until the token count per expert is very small indeed, which pushes the real lower edge further down than intuition suggests. But it is a per‑operator claim on a 2020 TPU‑v3 run, not an end‑to‑end MFU, and it is one point.

### 3e. Mechanism evidence, quantified but at the wrong scale

- **Lina** (16 A100): all‑to‑all is *"an average of 34.1%"* of step time and *"the average GPU SM efficiency during all-to-all is 3.7%"*. That is the collapse mechanism measured directly — a third of the step at 3.7 % utilisation — but at 16 GPUs, not 1,000+.
- **Tutel**: kernel optimisations give 3.52× at 16 GPUs but **1.04× at 2,048**, while 2DH All‑to‑All gives 4.25× at 2,048. What dominates flips completely with scale.
- **MegaScale‑MoE** on why MoE's ceiling is structurally lower and *falling*: *"the MFU value decreases as GPU compute capability increases. This is because, unlike dense models, MoE models involve many memory-intensive operations like routing, local scatter, and gather, which remain time-consuming since memory bandwidth does not scale as quickly as compute capabilities."*
- **Granularity is the same lower edge in a different variable.** Mixtral 8x22B reparameterised to 64 experts at 1/8 expert width (G8T8) drops from 52.2 % to 30.0 % MFU at identical capacity on 128 GPUs — *"the smaller hidden sizes decrease GEMM efficiency."* Too little work per GEMM produces the same failure whether it comes from too many devices or too many narrow experts.
- **The confounder in the dense record:** Megatron‑SP holds params/accelerator roughly constant (2,750 / 2,734 / 1,893 / 1,969 M) while model size grows 22 B → 1 T, and MFU *rises* 41.5 % → 56.3 %. Megatron‑LM PTD‑P shows the same, 44 % → 52 % (recomputation‑inclusive, so HFU‑flavoured — not comparable to the MFU numbers above without adjustment), and attributes it to *"larger matrix multiplications."* Absolute model size and per‑device work move together in every published sweep. Neither has been isolated.

### 3f. Honest verdict

**The published record does not locate the lower edge.** It shows a shallow, consistent −2 % to −6 % MFU per doubling of accelerators; it shows that MoE sits 10–20 MFU points below dense at comparable scale (best real MoE production run 43.7 %, best MoE benchmark 52.2 %, versus 56–70 % dense); and it shows that nobody has published a run at fewer than ~4,096 tokens per accelerator per step for MoE with an absolute MFU attached. The edge is below every published point, and the papers that come closest publish only relative numbers.

---

## 4. What would have to be measured to pin the lower edge

1. **MFU versus per‑device batch at fixed model and fixed cluster.** Take one MoE model on one fixed accelerator count and sweep global batch down by 8–16×, holding parallelism constant. Nobody has published this. Every existing sweep varies cluster size *and* per‑device work simultaneously, and Llama 3's 43 %→41 % at DP 64→128 is a single point on a line nobody has drawn.

2. **Push a sweep past the knee and report where it is.** Continue the MegaScale‑MoE and MoE Parallel Folding curves below 4,096 tokens/accelerator/step to 1,024, 512, 256. The record's shallow slope is only evidence that the collapse is not yet visible.

3. **Report DP degree with every MFU number.** MFU without the DP degree is unreconstructable: DeepSeek‑V3 gives PP=16 and EP=64 but not DP; Pangu Ultra MoE gives TP/PP/VPP/EP but not DP; Pangu Pro MoE's DP is not even derivable (4096 ÷ (TP8 × PP5) is not an integer). Skywork‑MoE is the exception (12 × 4 × 32 = 1,536, exactly its GPU count) and should be the template.

4. **Tokens per expert *instance* per step, not globally.** This requires DP degree and EP degree together. Only the global figure is currently derivable, and it differs from the per‑instance figure by the DP degree — which for a 2,048‑GPU DeepSeek‑V3‑class run is a factor of 2, and for a 16,384‑GPU run could be 100×.

5. **Separate the two candidate mechanisms.** MegaScale‑MoE attributes its entire decline to pipeline bubbles (fewer micro‑batches per pipeline); Llama 3 attributes its entire decline to per‑DP‑group batch. Both cannot be the general answer. Vary PP degree at fixed per‑DP batch, and per‑DP batch at fixed PP degree, and report both curves.

6. **Time breakdown at scale.** Lina's all‑to‑all fraction (34.1 %) and SM efficiency during all‑to‑all (3.7 %) are the right instruments; they exist only at 16 GPUs. Publish that breakdown at 256, 1,024, 4,096.

7. **Expert‑GEMM M‑dimension with its imbalance distribution.** Megatron‑Core's benchmarks *"use force-balanced routing"* and MoE Parallel Folding uses *"token drop training with a capacity factor equal to 1"* — both remove exactly the routing imbalance that bites hardest when per‑expert token counts are small. The lower edge must be measured with real routing.

8. **Dense/MoE matched pairs across a scale sweep, not at one point.** The only matched pairs are OLMoE (23,600 vs 37,500 tokens/s/GPU at 128 H100), DeepSpeed‑MoE (372 vs 70 samples/s at 128 A100), and GRIN (7,077 vs 8,176 per‑GPU at 64 H100) — all single‑cluster‑size. The MoE penalty as a function of scale is unmeasured.

9. **Report MFU and HFU together with the FLOPs formula.** Only PaLM (46.2 % / 57.8 %) and Korthikanti (41.5/43.7, 51.4/52.8, 56.0/57.0, 56.3/57.0) do both. Megatron‑LM PTD‑P's "% of peak" is recomputation‑inclusive and is silently incomparable to every MFU quoted elsewhere; without the formula, cross‑paper curves cannot be assembled.

10. **State the cluster size at all.** Twenty surviving runs — including Kimi K2 at 1.04 T, Qwen3‑235B, GLM‑4.5, Hunyuan‑Large, Mixtral, Grok‑1, and Switch‑C — do not. Until frontier MoE reports publish accelerator counts alongside MFU, no amount of re‑analysis will recover this relation from the literature.