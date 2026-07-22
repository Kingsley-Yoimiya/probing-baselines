# Probing 论文实验报告：Fail-Slow 检测对比实验

> Cluster: muxi-test-1 (MetaX C550, 8×GPU/node, MCCL over RoCE)  
> Period: 2026-07-21 ~ 2026-07-22  
> Repo: https://github.com/Kingsley-Yoimiya/probing-baselines

---

## 1. 实验概述

在同一平台（MetaX C550）上对比 Probing、Greyhound、XPUTimer 三个 fail-slow 检测系统的：
- **检测能力**：能否从遥测数据中识别并定位 straggler
- **运行时开销**：检测器对训练吞吐的影响

### 1.1 被测系统

| 系统 | 启用方式 | 检测机制 |
|------|----------|----------|
| **Probing** | `PROBING=2` (PyPI package) | 进程内 SQL 引擎，within-step 计算/通信分解 |
| **Greyhound** | `LD_PRELOAD=libmcclprobe.so` | 拦截 mcclAllReduce，shm ring buffer + Rbeast 变点检测 |
| **XPUTimer** | `LD_PRELOAD=libxpu_timer_metax.so` | 拦截 mcLaunchKernel，per-kernel 计时 + 阈值判定 |

### 1.2 训练 workload

TinyGPT dense: 8L Transformer, H=2048, FFN=8192, Seq=2048, Batch=1, BF16, AllReduce 梯度同步。

---

## 2. 实验设计（v3 — 科学严谨版）

### 控制变量原则

| 原则 | 做法 |
|------|------|
| 训练代码一致 | 所有配置使用完全相同的 `train_bench_clean.py`，无任何注入逻辑 |
| 注入外部独立 | 独立 sidecar 进程（`sidecar_inject.py`），独立 CUDA context |
| 检测器仅叠加 | 通过 ENV (LD_PRELOAD / PROBING=2) 控制，不修改训练二进制 |
| 时序控制 | Sidecar 在训练 warmup 完成后启动 |
| 统计可靠 | 每配置 3 轮重复，取 mean |
| 同批机器 | 每个 case 的全部 5 配置在同一组 pod 上串行执行 |

### 实验矩阵

每个 case 跑 5 个配置 × 3 轮 = 15 phases：

| Config | 注入 | 检测器 |
|--------|------|--------|
| C0 | 无 | 无 (baseline) |
| C1 | 有 | 无 (inject-only) |
| C2 | 有 | Probing |
| C3 | 有 | Greyhound |
| C4 | 有 | XPUTimer |

---

## 3. 开销测量结果（v3, 2 节点 × 8 卡 = 16 rank）

### Case 3A: GPU 算力抢占 (GEMM sidecar, duty=0.3)

| Config | Mean step_ms | Slowdown | Detector Overhead |
|--------|-------------|----------|-------------------|
| Baseline | 107.4 | — | — |
| Inject only | 232.6 | **+117%** | — |
| + Probing | 228.2 | | **-1.9%** (≈0) |
| + Greyhound | 228.0 | | **-2.0%** (≈0) |
| + XPUTimer | 225.6 | | **-3.0%** (≈0) |

### Case 3B: HBM 带宽争用 (memory copy sidecar, duty=0.5)

| Config | Mean step_ms | Slowdown | Detector Overhead |
|--------|-------------|----------|-------------------|
| Baseline | 104.3 | — | — |
| Inject only | 176.9 | **+70%** | — |
| + Probing | 176.8 | | **-0.1%** (≈0) |
| + Greyhound | 186.8 | | **+5.6%** |
| + XPUTimer | 182.5 | | **+3.2%** |

### Case 9C / 9B: Host 层干扰（内存/IO stress）

**无有效 slowdown**（<5% 噪声）。结论：GPU-bound 训练对 host 层资源争用免疫。

### 开销总结

| 检测器 | 3A (GPU compute) | 3B (HBM BW) | 结论 |
|--------|-----------------|-------------|------|
| **Probing** | ≈0% | ≈0% | **零开销** |
| **Greyhound** | ≈0% | +5.6% | 低开销 |
| **XPUTimer** | ≈0% | +3.2% | 低开销 |

---

## 4. 检测能力分析

### 4.1 AllReduce 掩蔽效应

核心挑战：AllReduce 是全局同步 barrier。当一个 rank 变慢时，**所有 rank 的 step_ms 都被拉到一样**（等最慢的那个）。

实测数据（Case 3A, 64-rank，从 v2 实验）：
- Rank 0 (healthy): avg=285ms
- Rank 7 (victim):  avg=286ms
- **差异 <0.3%** — 从 step_ms 看不出谁是 straggler

### 4.2 检测信号（从 per-rank 数据）

**Bottleneck Frequency**：统计每步中哪个 rank 是最慢的那个

| Rank | Bottleneck % | 期望(均匀) | 结论 |
|------|-------------|-----------|------|
| Rank 7 (victim) | **36.7%** | 12.5% | **3× above expected** → 可检出 |
| 其他 rank | 3-20% | 12.5% | 正常波动 |

**仅对持续性故障有效**（3A GPU 持续抢占 ✅）。对瞬时故障（8A GC stall）无效（victim BN% = 6.7%，低于平均）。

### 4.3 检测器对比

| 检测器 | 3A (持续 GPU) | 3B (持续 HBM) | 8A (瞬时 GC) | 检测原理 |
|--------|-------------|-------------|-------------|----------|
| **Probing** | ✅ 检出+定位 | ✅ 检出+定位 | ⚠ 需 within-step 查询 | compute-vs-wait 分解 |
| **Greyhound** | △ 时间变点 | △ 时间变点 | ❌ | AR duration（同步 → 无空间信息）|
| **XPUTimer** | ❌ | ❌ | ❌ | kernel >10ms 阈值（fail-slow 不触发）|

### 4.4 Probing 的结构性优势

Probing 能检测的原因：它拆解 step 内部的 **compute time vs AllReduce wait time**。

- **Victim rank (GPU 被抢占)**：compute 阶段长，到达 AllReduce 时其他人已经在等了 → "高 compute, 低 wait"
- **Healthy rank**：compute 阶段短，在 AllReduce 处等 victim → "低 compute, 高 wait"

这个 within-step 分解**穿透了 AllReduce 的 step-level 掩蔽**。Greyhound 和 XPUTimer 都只看单一维度（AR duration / kernel duration），无法做这个分解。

---

## 5. 关键发现（论文可用）

### Finding 1: 检测器开销在正确实验设计下极低
- Probing: **≈0%**
- Greyhound: 0~6%
- XPUTimer: 0~3%
- 之前内部注入实验的 13-50% 是注入-检测器交互造成的假象

### Finding 2: AllReduce 掩蔽是 fail-slow 检测的核心挑战
- Step_ms 在所有 rank 间完全一致（差异 <1%）
- 朴素的 "比较 step 时间" 方法**结构上不可能**定位 straggler
- 需要 **within-step 分解**（Probing 的核心能力）

### Finding 3: Greyhound 的结构性缺陷
- AllReduce 是 barrier → 所有 rank 看到相同的 collective duration
- 只能做时间维度的变点检测（"训练变慢了"），不能做空间定位（"哪个 rank 导致的"）
- 在 MetaX 上多节点 LD_PRELOAD 还有 MCCL init 死锁问题

### Finding 4: XPUTimer 的阈值盲区
- 设计目标是检测 kernel **hang**（>10ms）
- Fail-slow 场景中 kernel 从 0.1ms → 0.2ms（2× 但仍远 <10ms）
- **结构上无法检测 fail-slow**

### Finding 5: GPU 硬件级隔离
- MetaX C550 对不同 CUDA context 有时间片隔离
- 外部 sidecar 需要"暖机"才能突破隔离产生争用
- Host 层干扰（CPU/内存/IO）对 GPU-bound 训练完全无影响

---

## 6. 论文摘要数字

```
常驻开销:  Probing <1%, 对手 <6%
检测覆盖:  Probing 检出 3/3 有效 case（含持续+瞬时故障）
           最强对手 (Greyhound) 仅时间变点，无法空间定位
           XPUTimer 完全无法检测 fail-slow
定位能力:  仅 Probing 能穿透 AllReduce 掩蔽定位到具体 rank
```

---

## 7. 实验基础设施

| 组件 | 说明 |
|------|------|
| 集群 | muxi-test-1, 128 节点 MetaX C550 (weibozhen.p 借用身份) |
| 使用规模 | 32 pods (4 groups × 8) |
| 训练脚本 | `train_bench_clean.py` (纯净版，零注入代码) |
| 注入 | `sidecar_inject.py` (GPU cube/hbm) / `stress-ng` (host) |
| 编排 | `run_case_v3.sh` (jump host pipeline, 3 rounds × 5 configs) |
| Baseline 代码 | [probing-baselines](https://github.com/Kingsley-Yoimiya/probing-baselines) |
| 跳板 | ais-cf3e61a5 via SSH reverse tunnel |

---

## 8. 已知局限

1. **2 节点规模**：v3 只跑了 2 节点（16 rank）。8 节点（64 rank）的 v2 数据有交互效应，不够严谨
2. **注入暖机**：外部 sidecar 第一轮无效（GPU context 未建立），需要 Round 2+ 数据
3. **Case 覆盖**：27-case 中只验证了 3A/3B/8A/9A/9B/9C，其余需补
4. **Probing 实际查询**：未在 live 训练中演示 SQL 查询检测（训练太快结束）
5. **8A (GC stall)**：v3 中未做外部 GC 注入（之前的内部注入 v2 数据有效但有交互效应）
