# XPUTimer → MetaX C550 检测后端（metax_probe）

把 dlrover 的 fail-slow 检测系统 **XPUTimer**（原版 hook CUDA/NCCL + Bazel 构建）
适配到**沐曦 MetaX C550 GPU**，让其**检测机制**（kernel/集合通信计时 + hang/slow 判定）
在沐曦上真实跑起来。

完整上机日志、符号确认、踩坑与修复见：`../../../results/xputimer-metax-log.md`。

## 关键结论（与任务初始假设的差异）

任务假设“MCCL 导出 `nccl*` 同名符号”**不成立**。实测（`nm -D`）：

- MACA 用 **`mc*` 前缀**：`mcLaunchKernel`（== cudaLaunchKernel）、`mcEvent*`（== cudaEvent*）、`mcGetDeviceProperties`。
- MCCL 用 **`mccl*` 前缀**：`mcclAllReduce/mcclAllGather/...`（== nccl*，但改名）。
- torch `2.6.0+metax3.3.0.2` **直接链接 `mc*`/`mccl*` 裸符号**（`nm -D libtorch_cuda.so` → `U mcLaunchKernel`、`U mcclAllReduce`）。

因此 LD_PRELOAD interposer 必须导出 **`mc*`/`mccl*`** 名（不是 nvidia 后端的 `cuda*`/`nccl*`）。

## 为什么不走原版 Bazel `-DXPU_METAX`

原版 `GpuTimerManager` 依赖 brpc/boost/perfetto/prometheus-cpp/protobuf，全靠 Bazel 6.2.0 从 GitHub 拉。
沐曦 pod 无 bazel/cmake、github 抓取不稳 → 那套依赖图不可构建。但 fail-slow **检测回路**
（event 计时 + poller + hang/slow 判定 + 指标）**不需要**这些。故本目录用 **plain g++** 直编一个
自包含 hook，忠实复刻检测机制并在真机跑通。

`config_subparsers/metax.py` + `common/platform.cc` 的 `XPU_METAX` 分支保留为**官方 in-tree 后端注册**
（等 Bazel 依赖可用时可接回原构建）。

## 文件

| 文件 | 作用 |
|---|---|
| `xpu_timer_metax_hook.cc` | LD_PRELOAD 检测 hook。导出 `mcLaunchKernel` + 17 个 `mccl*`（含 Ext/AllToAll）；event 计时 + 后台 poller + HANG/SLOW 判定 + Prometheus/JSONL 导出 |
| `build_metax_hook.sh` | `g++ -std=c++17 -O2 -fPIC -shared -ldl -lpthread`（pod 内跑） |
| `metax_selftest.py` | 单 GPU workload（matmul/elementwise）+ `--inject-hang` |
| `metax_dist_test.py` | 2+ rank `torch.distributed`（allreduce/allgather）+ `--desync-rank` 掉队注入 |

## 构建 + 运行（在沐曦 worker pod 内）

```bash
# 1. 构建（需 /opt/maca + g++；不需要 Bazel）
bash build_metax_hook.sh            # -> libxpu_timer_metax.so

# 2. 单 GPU：kernel 计时 + SLOW 检测
XPU_TIMER_DUMP_DIR=/tmp/xpu_out XPU_TIMER_SLOW_REPORT_US=500 \
LD_PRELOAD=./libxpu_timer_metax.so python metax_selftest.py --iters 100

# 3. 2-rank：集合通信拦截 + fail-slow（掉队 rank → 对端集合通信 HANG）
XPU_TIMER_HANG_TIMEOUT_MS=1500 XPU_TIMER_DUMP_DIR=/tmp/xpu_dist \
LD_PRELOAD=./libxpu_timer_metax.so \
torchrun --nproc_per_node=2 metax_dist_test.py --iters 60 --desync-rank 1 --desync-at 20

# 4. MoE 常用 AllToAll 覆盖
LD_PRELOAD=./libxpu_timer_metax.so \
torchrun --nproc_per_node=2 metax_dist_test.py --iters 20 \
  --exercise-alltoall --exercise-alltoallv
```

产物：`$XPU_TIMER_DUMP_DIR/metax_metrics.rank<R>.pid<P>.prom`（Prometheus 文本）+ `metax_trace.*.jsonl`。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `XPU_TIMER_ENABLE` | 1 | 总开关 |
| `XPU_TIMER_HOOK_LAUNCH` | 1 | 是否计时 `mcLaunchKernel` |
| `XPU_TIMER_LAUNCH_SAMPLE` | 1 | 每 N 次 launch 计 1 次（降开销） |
| `XPU_TIMER_HANG_TIMEOUT_MS` | 2000 | op 未就绪超此值 → 标 HANG |
| `XPU_TIMER_SLOW_REPORT_US` | 0(关) | op 延迟 ≥ 此值 → 标 SLOW |
| `XPU_TIMER_DUMP_DIR` | /tmp/xpu_timer_metax | 产物目录 |
| `XPU_TIMER_MCCL_LIB` / `XPU_TIMER_MACA_RUNTIME_LIB` | libmccl.so / libmcruntime.so | 原库路径覆盖 |

## 检测机制映射（上游 NVIDIA → MetaX）

| XPUTimer 组件 | NVIDIA | MetaX | 状态 |
|---|---|---|---|
| kernel launch 拦截 | `cudaLaunchKernel` | `mcLaunchKernel` (RTLD_NEXT) | ✅ |
| 集合通信拦截 | `nccl*` | `mccl*` base/Ext + AllToAll(v) | ✅ |
| event 计时 | `cudaEvent*` | `mcEvent*` (dlopen libmcruntime) | ✅ |
| getDeviceName | `cudaGetDeviceProperties` | `mcGetDeviceProperties` | ✅ |
| hang/slow 判定 | `GpuTimerManager::doHang` | 同逻辑复刻 | ✅ |
| cuBLAS GEMM shape | `cublas*`/`cublasLt*` | `mcblas*` | ⚪ 桩过（GEMM kernel 仍被计时，缺 matmul label） |
| comm 拓扑解析 | `parse_params` | MCCL comm 结构体 | ⚪ 桩过（集合通信按 kernel 计时 + API 元数据） |

集合通信内部的 `mcLaunchKernel` 会继承当前 `mccl*` API 名，指标以
`coll_kernel:mcclAllReduce` 等名称输出，因此 HANG/SLOW 不再只显示匿名
`kern_0x...`。若 MCCL 将 launch 转移到内部线程，无法继承线程局部标签的
kernel 仍按普通 kernel 记录。
