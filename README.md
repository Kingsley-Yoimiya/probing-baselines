# probing-baselines

把**原版依赖 NVIDIA CUDA/NCCL、在沐曦（MetaX）上跑不起来**的 fail-slow 检测 baseline，适配到沐曦 C550 GPU 上跑起来。

## 背景

在沐曦集群评测 fail-slow 检测 baseline 时，5 个开源系统里有 2 个卡在 `ENV-BLOCKED`——它们 hook NVIDIA NCCL / 依赖 CUDA ABI，而沐曦只有 `libmccl.so` + `/dev/mxcd` + `mx-smi`，没有 CUDA/NCCL/nvidia-smi。本仓专门把这两个移植到沐曦：

| 目录 | Baseline | 上游 | 原版阻塞点 |
|---|---|---|---|
| `greyhound-metax/` | Greyhound (ATC'25) | [wutianyuan1/Greyhound](https://github.com/wutianyuan1/Greyhound) | LD_PRELOAD hook `libnccl.so.2`；`.cu` 微基准；Docker/Redis |
| `xputimer-metax/` | XPUTimer (dlrover) | [intelligent-machine-learning/dlrover](https://github.com/intelligent-machine-learning/dlrover) `xpu_timer` | Bazel 找 CUDA SDK；hook cudaLaunchKernel/cuBLAS/NCCL |

## 适配思路（关键前提）

MCCL 大概率直接导出 `nccl*` 同名符号（MACA 是 cuda-alike 层），所以 hook 层可能**只需改 dlopen 的库路径**，不必改符号名。上机第一步 `nm -D libmccl.so | grep nccl` 确认。

## 目录

- `greyhound-metax/` — Greyhound 源码 + 沐曦适配改动 + 跑通日志
- `xputimer-metax/` — XPUTimer 源码 + 沐曦适配改动 + 跑通日志
- `results/` — 各 baseline 在沐曦上的运行结果、踩坑记录、适配笔记

## 运行环境

- 集群 `vc-c550-h3c-test`，MetaX C550-PL，PyTorch `2.6.0+metax3.3.0.2`
- 借用身份 `weibozhen.p` 访问；落盘一律 `yinjinrun.p`（详见 myportal 集群身份规则）

## 状态

移植进行中。各 baseline 的适配进度与"能跑到哪一步"见各子目录与 `results/`。
