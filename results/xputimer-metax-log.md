# XPUTimer → MetaX C550 适配日志

上机环境：沐曦 vc-c550-h3c-test，借用 weibozhen.p，pod `muxi-1tb-moe-1024-worker-{111..118}`（8 台 / 64 卡）。
主用 worker-118（GPU 空闲，训练进程已 defunct）。落盘 `/afs-a3-weight-share/yinjinrun.p/`。

代码工作区：`/Users/yinjinrun/Codespace/probing-baselines/xputimer-metax-clone/`（上游 dlrover clone，focus `xpu_timer/`）。
本地参考副本：`/Users/yinjinrun/Codespace/fail-slow-opensource-study/vendors/dlrover/xpu_timer/`。

---

## 里程碑 1：符号确认（完成）

### 环境
- MACA Version 3.3.0.15，KMD 3.4.24，mx-smi 2.2.9，8x MetaX C550-PL（65536 MiB）。
- MACA root：`/opt/maca`（软链 `/opt/maca-3.3.0`）。lib 在 `/opt/maca/lib`。
- Torch：`2.6.0+metax3.3.0.2`，conda env `/opt/conda`（python3.12），`torch.cuda.is_available()=True`。
- 训练用 conda python：`/opt/conda/bin/python`。系统 `/usr/bin/python3` 无 torch。

### 关键发现：MACA 用 `mc*` 前缀，不是 `nccl*`/`cuda*`同名
任务假设“MCCL 导出 nccl* 同名符号”**不成立**。实测：

- **集合通信** `libmccl.so`：导出 `mccl*`（`mcclAllReduce/mcclAllReduceExt/mcclAllGather/mcclReduceScatter/mcclBroadcast/mcclBcast/mcclSend/mcclRecv/mcclReduce/mcclAllToAll{,v,d}/mcclGroupStart/mcclGroupEnd/mcclCommInitRank/mcclCommUserRank/mcclCommCount/mcclGetVersion`）。无 `nccl*` 导出。
- **runtime（cuda-alike）** `libmcruntime.so`：`mcLaunchKernel/mcLaunchKernelExC/mcModuleLaunchKernel/mcGetDeviceProperties/mcStreamSynchronize/mcEventCreate/mcEventRecord/mcEventElapsedTime/mcEventSynchronize/mcEventQuery/mcEventDestroy/mcStreamCreate...`。完整 CUDA event 计时 API 齐全（`mcEventElapsedTime` 存在 → XPUTimer 的 event 计时可原样映射）。
- **BLAS** `libmcblas.so`/`libmcblasLt.so`：`mcblasGemmEx/mcblasSgemm/.../mcblasLtMatmul`（`mcblas*` 前缀，cuBLAS-alike）。
- 还有 `libruntime_cu.so`（wrapper 层，含 `wcudaMallocHost` 等 `w`-前缀 cuda 兼容包装）。

### torch 实际链接的是 `mc*` 裸符号（决定 hook 策略）
`ldd libtorch_cuda.so` → 直接 NEEDED `libmccl.so libmcruntime.so libruntime_cu.so libmcblas.so libmcblasLt.so ...`。
`nm -D libtorch_cuda.so | grep U` → `U mcLaunchKernel`、`U mcclAllReduce`、`U mcclAllReduceExt`。
→ **torch 调用未定义符号是 `mcLaunchKernel` / `mcclAllReduce`**，因此 LD_PRELOAD 拦截必须导出 `mc*` / `mccl*` 名（不是 nvidia 后端的 `cudaLaunchKernel`/`ncclAllReduce`）。

### MetaX 的 CUDA 兼容层（编译期，不是运行期同名）
- `/opt/maca/tools/cu-bridge`：cuda→maca 源码级转换（类似 HIP hipify）。
  - `include/nccl.h`（shim）、`include/bridge/runtime/cuda_to_maca_mcr_adaptor.h`：`#define cudaLaunchKernel wcudaLaunchKernel`、`cudaLaunchKernelExC → mcLaunchKernelExC`、`cudaGetDeviceProperties → wcudaGetDeviceProperties`、`cudaEventElapsedTime → wcudaEventElapsedTime`、`cudaEventRecord → wcudaEventRecord` 等。
  - 即：CUDA 源码经 cu-bridge 编译，符号被 `#define` 改写成 `mc*`/`w*`。运行期真正的符号是 `mc*`。
- 结论：**适配策略 = LD_PRELOAD 一个导出 `mcLaunchKernel` + `mccl*` 的 .so，用 `dlsym(RTLD_NEXT,...)` 拿原实现，中间插 event 计时**。cuBLAS(`mcblas*`)与 comm 拓扑解析先桩过。

---

## 里程碑 2：构建策略决策 + g++ 编出 .so（完成）

### 为什么不走原版 Bazel `-DXPU_METAX`
- 原版 `GpuTimerManager` 依赖 brpc + boost + perfetto + prometheus-cpp + protobuf，全靠 Bazel 6.2.0 从 GitHub 拉取。
- pod 现状：无 bazel/bazelisk、无 cmake，只有 gcc/g++ 11.4 + make；pod 能连 pypi 但 **github 抓取不稳**（`urlopen(github)` 超时）。
- 即便硬装 bazelisk，这套依赖图在共享训练 pod 上编译要数小时，且大概率卡在网络/perfetto。→ **判定为硬阻塞路径**。
- 关键洞察：fail-slow **检测机制**（event 计时 + 后台 poller + hang/slow 判定 + 指标导出）本身**不需要** brpc/boost/perfetto。这些只是上报通道与花哨 trace。

### 采用策略：自包含 MetaX hook（g++ 直编），忠实复刻检测回路
新增 `xputimer-metax/xpu_timer/metax_probe/`：
- `xpu_timer_metax_hook.cc`：LD_PRELOAD interposer。导出 **`mc*`/`mccl*`** 符号（不是 nvidia 后端的 `cuda*`/`nccl*`——因为 torch-metax 链接的是 mc 裸符号），`dlsym(RTLD_NEXT,...)` 拿原实现（== 上游 `SETUP_DLSYM` 宏语义）。
  - 复刻 `NvidiaGpuTimer::startRecord/endRecord`：每个 op 用 `mcEventRecord(start/stop)` 在其 stream 上打点。
  - 复刻 `GpuTimerManager::doWork` 后台 poller：`mcEventQuery` 就绪→`mcEventElapsedTime` 折进直方图；超 `hang_timeout` 未就绪→标 HANG（fail-slow 信号）== 上游 `doHang`。
  - 复刻 `platform.cc::getDeviceName`：`mcGetDeviceProperties`。
  - 指标用 Prometheus text exposition 导出（沿用 `xpu_timer_common_kernel_*` 名）＋ JSONL trace。
  - hook 的符号：`mcLaunchKernel` + `mcclAllReduce/AllGather/ReduceScatter/Broadcast/Reduce/Send/Recv`。
  - **桩过**：mcblas GEMM shape 解析、mccl comm 拓扑（parse_params）——只影响 label 丰富度，不影响检测回路（符合任务“先桩过”）。
- `metax_selftest.py`：torch matmul/elementwise workload + `--inject-hang` 长 GPU 链路触发 HANG 检测。
- `build_metax_hook.sh`：`g++ -std=c++17 -O2 -fPIC -shared -ldl -lpthread`。**故意不 include maca 头**（自声明最小 ABI），避开 maca clang 专属的 `__device__` intrinsics。

### 编译结果（pod worker-118, /tmp/xputimer-metax-work）
```
g++ 11.4 -> libxpu_timer_metax.so (59KB)
nm -D 导出：mcLaunchKernel mcclAllReduce mcclAllGather mcclReduceScatter
            mcclBroadcast mcclReduce mcclSend mcclRecv  (T = 已导出)
```
**g++ 直编成功，无 Bazel。** 下一步：LD_PRELOAD 跑真实 workload 验证拦截+计时+HANG。

---

## 里程碑 3：LD_PRELOAD 上机跑通检测机制（完成，含硬 bug 修复）

### 单进程验证（worker-118, GPU0 空闲）
- baseline（无 preload）：torch matmul workload 正常，device=MetaX C550-PL。
- **带 hook**：`mcLaunchKernel` 成功拦截，device 检出 "MetaX C550-PL"，kernel 用 mcEvent 计时（elementwise 45us/33us、GEMM stub 未计）。202 条 JSONL trace，kernel 名成功 demangle（`vectorized_elementwise_kernel<CUDAFunctor_add<c10::Half>>` = `c+a`）。
- **SLOW 检测**：`XPU_TIMER_SLOW_REPORT_US=200` → 正确标记 >200us 的 kernel，`kernel_slow` 计数非零。

### 分布式 2-rank 踩坑 → 定位 → 修复（关键）
现象：`torchrun --nproc_per_node=2 ... init_process_group(backend=nccl)` + full hook → **rank SIGSEGV (exit -11)**；但 launch-only hook（不导出 mccl*）正常。

逐步定位：
1. launch-only（仅 `mcLaunchKernel`）分布式正常 → 罪魁是 `mccl*` interposer。
2. DEBUG 版打点：`[dbg] enter mcclAllReduce` 打印了，但 `[dbg] after orig` 没有 → **crash 在 `orig()` 调用里**。
3. 自检版打印指针：`RTLD_NEXT=(nil)`、`RTLD_DEFAULT=self`。
   **根因**：torch 是 **lazy `dlopen("libmccl.so")`**，我的 preload lib 加载时 libmccl 还不在全局搜索序里 → `dlsym(RTLD_NEXT,"mcclAllReduce")` 返回 **NULL** → `orig(...)` 调空指针 → SIGSEGV。
   （注：`mcLaunchKernel` 无此问题，因为 libmcruntime 被 torch NEEDED 直链，RTLD_NEXT 能找到。）

修复：mccl 原函数改用 **显式 `dlopen("libmccl.so", RTLD_GLOBAL) + dlsym(handle,...)`**（== 上游 `SETUP_DLSYM_WITH_DLOPEN`），env `XPU_TIMER_MCCL_LIB` 可覆盖。runtime helper 同理走 `dlopen("libmcruntime.so")`。
另一设计修正：**不在 mccl API hook 里打 mcEvent**（mccl bootstrap 会在私有 stream 上内部调用这些 API，打 event 会崩/死锁）。改为 mccl hook 只记 count+bytes 元数据；集合通信的**真实 kernel 延迟由 mcLaunchKernel 路径捕获**——这正是上游 nvidia 后端的做法（`ncclAllReduce` 不 bracket event，只 `interceptNcclInfo`，让 kernel 走 cudaLaunchKernel 计时）。

### 修复后结果（2x C550, GPU0+1）
- 正常 workload：`mcclAllReduce`×40 + `mcclAllGather`×40 拦截，集合通信正确完成，`[dist] done`，无 crash。指标含 `xpu_timer_common_coll_bytes_total` / `coll:mccl*` 计数。
- **FAIL-SLOW 注入（核心里程碑）**：rank1 在 iter20 straggle（模拟掉队/死 rank）→ rank0 的集合通信 kernel 卡住等待 → 超 1500ms 未就绪 → poller 标 **HANG**：
  ```
  [xpu_timer.metax][HANG] op=kern_0x...fe0 type=kernel outstanding=1500.0ms >= 1500ms (fail-slow candidate)
  [xpu_timer.metax][HANG] op=kern_0x...820 type=kernel outstanding=1500.1ms >= 1500ms (fail-slow candidate)
  xpu_timer_common_kernel_hang{name="kern_0x...820",type="kernel"} 1
  ```
  卡住的正是集合通信 kernel（地址与正常 run 一致）。**XPUTimer fail-slow 检测机制在沐曦 C550 上完整跑通。**

### 检测机制映射总结（上游 → MetaX）
| XPUTimer 组件 | 上游 (NVIDIA) | MetaX 适配 | 状态 |
|---|---|---|---|
| kernel launch 拦截 | `cudaLaunchKernel` | `mcLaunchKernel`（RTLD_NEXT） | ✅ 跑通 |
| 集合通信拦截 | `nccl*` | `mccl*`（显式 dlopen） | ✅ 跑通 |
| event 计时 | `cudaEvent*` | `mcEvent*`（dlopen libmcruntime） | ✅ 跑通 |
| getDeviceName | `cudaGetDeviceProperties` | `mcGetDeviceProperties` | ✅ "MetaX C550-PL" |
| poller/hang 检测 | `GpuTimerManager::doHang` | 同逻辑复刻 | ✅ HANG 触发 |
| SLOW 检测 | 延迟直方图 | 同逻辑 | ✅ 触发 |
| Prometheus 上报 | brpc+bvar | text exposition 直写 | ✅ 指标文件 |
| cuBLAS GEMM shape | `cublas*`/`cublasLt*` | `mcblas*`（**桩过**） | ⚪ 后置 |
| comm 拓扑 parse | `parse_params`+nccl 结构体 | `mccl*` 结构体（**桩过**） | ⚪ 后置 |

### 桩过项（不影响检测回路，符合任务“先桩过”）
- mcblas/mcblasLt GEMM 的 shape/dtype 解析：GEMM kernel 仍被 mcLaunchKernel 计时，只是没打 matmul 专属 label（TFLOPS 等）。
- mccl comm 拓扑（rank/nRanks/nNodes/commHash）：需重做 MCCL comm 结构体偏移解析，当前集合通信按 kernel 计时 + API 元数据，已够 fail-slow 检测。
