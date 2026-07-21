# Greyhound → 沐曦 (MetaX C550) 适配日志

> 目标：把 fail-slow 检测系统 Greyhound（原版依赖 NVIDIA CUDA/NCCL）适配到沐曦 C550 GPU，让其**检测机制**（hook 采集 + ACF/变点迭代时间估计 + 组对比）跑起来。缓解模块与 comm 拓扑逆向允许桩过。
>
> 上游：https://github.com/wutianyuan1/Greyhound ；适配副本：`probing-baselines/greyhound-metax/`
> 集群：`vc-c550-h3c-test`，借用身份 weibozhen.p，worker 119-126（8 台，64 卡）；落盘 `/afs-a3-weight-share/yinjinrun.p/`。

---

## 里程碑 1：libmccl 符号确认（关键前提验证）—— 结论与原假设相反

**原假设**：MCCL 直接导出 `nccl*` 同名符号，hook 层只需把 dlopen 路径从 `libnccl.so.2` 改成 `libmccl.so`。

**实测结论：假设不成立。必须做 `nccl*` → `mccl*` 符号改名（后缀相同、仅前缀不同）。**

### 证据

1. **libmccl.so 只导出 `mccl*`，不导出 `nccl*`**
   - 库位置：`/opt/maca-3.3.0/lib/libmccl.so`（`/opt/maca` 是软链），大小 ~369 MB，Dec 2025 构建。
   - `nm -D libmccl.so | grep -i nccl` → **0 个** `nccl*` 符号。
   - `readelf -sW` 里有 **8213** 个 `mccl` 相关符号；关键 collective 都在，命名规律是 `nccl`→`mccl`：
     `mcclCommInitRank / mcclCommInitRankConfig / mcclCommInitAll / mcclCommInitRankMulti / mcclSend / mcclRecv / mcclAllReduce / mcclAllGather / mcclReduceScatter / mcclBroadcast / mcclBcast / mcclGroupStart / mcclGroupEnd / mcclCommCount / mcclCommUserRank / mcclGetUniqueId / mcclCommDestroy / mcclReduce / mcclAllToAll(v) / ...` 另有 `*Ext` 变体。

2. **MetaX PyTorch (2.6.0+metax3.3.0.2) 真正调用的是 `mccl*`**（决定 hook 必须拦 `mccl*`）
   - torch 在 `/opt/conda/lib/python3.12/site-packages/torch`（训练用 `/opt/conda` 环境，非系统 python3）。
   - `libtorch_cuda.so`（真正的 ProcessGroupNCCL 后端）：**68 个 UND `mccl*`**、0 个 UND `nccl*`。去重后 33 个 mccl collective/comm 符号。→ **DDP/Megatron 的真实通信走 mccl***。
   - `libtorch_python.so`：24 个 UND `nccl*`，但都是 `torch::cuda::nccl::*` C++ 符号（`torch.cuda.nccl` 的 python 绑定），**不是** libnccl 的 C ABI，且不承载真实集合通信。

3. **cu-bridge 的 `nccl.h` 是「静态内联垫片」，不产生动态 `nccl*` 符号**（这是 Rosetta Stone）
   - `/opt/maca/tools/cu-bridge/include/nccl.h`：`#define ENABLE_CUDA_TO_MACA_ADAPTOR`，`#include "mccl.h"`，然后把每个 `ncclXXX` 定义成 **`static` inline 函数**直接调 `mcclXXX`（如 `static ncclResult_t ncclAllReduce(...) { return mcclAllReduce(...); }`），并 `typedef mcclComm_t ncclComm_t;` 等一整套类型/枚举别名。
   - 含义：任何用 cu-bridge `nccl.h` 编译的代码，`nccl*` 调用在编译期被内联成对 `mccl*` 的直接调用；链接期**没有** `nccl*` 动态符号可供 LD_PRELOAD 拦截。**所以 Greyhound 若仍导出 `nccl*`，永远拦不到训练流量**——必须导出并转发 `mccl*`。

### 适配定调（据此调整策略）

- **hook 层**：把 12+ 个拦截函数从 `ncclXXX` 改名为 `mcclXXX`，签名不变（`ncclResult_t`↔`mcclResult_t`、`ncclComm_t`↔`mcclComm_t`、`cudaStream_t`↔`mcStream_t` 全是同布局别名）。
- **dlopen 目标**：`/opt/maca/lib/libmccl.so`（经 `NCCL_PATH`/新增 `MCCL_PATH` 覆盖）。`dlsym` 取 `mccl*` 真符号转发。
- **类型/头文件**：用 cu-bridge 的 `nccl.h` 就能把 `ncclXXX` 类型别名映射到 `mcclXXX`；CUDA 运行时（cudaEvent/cudaMalloc/cudaMemcpy…）由 cu-bridge `cuda_runtime.h` → MACA `mc*` 实现。编译器用 `mxcc`/`cucc`。

---

## 环境事实（worker-126 实测）

| 项 | 结果 |
|---|---|
| 身份 | pod 内 `root`（uid 0），可自由装依赖 |
| GPU | 8× MetaX C550-PL，MX-SMI 2.2.9，MACA 3.3.0.15，KMD 3.4.24 |
| libmccl | `/opt/maca/lib/libmccl.so`（MCCL_VERSION_CODE 21605，即 2.16.5 对标） |
| 头文件 | `/opt/maca/include/mccl.h`、`mxccl.h`；cu-bridge `nccl.h`/`cuda*.h` |
| torch | `2.6.0+metax3.3.0.2` @ `/opt/conda`（**不可动**） |
| 编译器 | `g++/gcc 11.4`、`mxcc`（/opt/maca/mxgpu_llvm/bin）、`cucc`（/opt/maca/tools/cu-bridge/bin）；**cmake 缺失** |
| Python 检测核依赖 | 已有 numpy1.26 / pandas2.3 / scipy1.16 / matplotlib3.10；**缺 Rbeast、statsmodels、redis** |
| C++ 构建依赖 | **缺 boost（log/interprocess/process）、cpp_redis、tacopie、redis-server** |
| 网络出口 | pod 内可达 pypi.org:443 / mirrors.aliyun.com:443；有 apt-get、conda（清华 conda-forge 镜像） |
| /dev/shm | tmpfs 64G（够放 ring buffer） |
| AFS 落盘 | `/afs-a3-weight-share/yinjinrun.p/` 可写 ✔ |
| env | `MCCL_SOCKET_IFNAME=eth0`、`MCCL_IB_HCA=xscale_0..3`、`CUDA_PATH=/opt/maca/tools/cu-bridge`、`MACA_PATH=/opt/maca` |

**注意**：pod 内 `find /` 会因 AFS 挂载卡死超时；一律用精确路径探测。`timeout` 命令 macOS 侧没有，kubectl 调用靠工具自身超时。

---

## Greyhound 结构与「检测核」边界（读码结论）

数据面（C++ LD_PRELOAD，`detector/`）：
- `ncclprobe.cpp`：拦 `nccl{CommInitRank,CommInitRankConfig,CommSplit,Send,Recv,Bcast,Broadcast,AllGather,ReduceScatter,AllReduce,GroupStart,GroupEnd}`，每次调用把 Record（comm 地址、op 号、count、buff、datatype、pid、call_time、device、rank、aux、duration、numDevices、event_id）写入共享内存环形缓冲 `ncclRecord`。
- `global_status.cpp`：dlopen libnccl（**改 libmccl**）、dlsym 取真符号；还负责起 global/local controller 子进程 + 连 redis + 装 control_plane whl（**这些是缓解/控制面，检测核可先关**）。CUDA event 计时（`cudaEventRecord/Synchronize/ElapsedTime`）。
- `comm.cpp` `parse_communicator`：**手工逆向 NCCL 私有 `ncclComm` 内存布局**读 nRanks/rank/nChannels/ring/tree（MCCL 布局不同 → 最脆，按任务桩成用 `mcclCommCount`/`mcclCommUserRank` 返回 rank/nRanks）。
- `shm_storage.cpp`：boost::interprocess 共享内存环形缓冲（检测核必需，采集落点）。
- `event_handler.cpp`：控制信号 + p2p/gemm 微基准（缓解/验证阶段，检测核可先桩）。
- `matmul.cu`：GEMM 微基准（`.cu`，需 mxcc 重编；仅验证阶段用）。

控制面（Python，`detector/control_plane/`，硬件无关）：
- `slow_detection.py`：**检测核算法** —— `find_period`（statsmodels ACF 自相关找训练迭代周期）+ `find_performance_drop`（Rbeast 贝叶斯变点 + 退化幅度校验）。
- `local_analyzer.py`：`NcclRecord` 直接 mmap 读 `ncclRecord` 共享内存环形缓冲；`detect_failslow` 按 rank 分组跑 ACF+变点，估计每 rank 迭代时间。
- `global_analyzer.py`：把 comm 归到 TP/DP/PP clique，按组对比时延中位数找「慢 clique」（组对比）。

**检测核最小闭环（MVP）**：`mccl*` hook 采集 → 写 shm 环形缓冲 → Python `NcclRecord` 读 → `find_period`+`find_performance_drop`（迭代时间估计/变点）→ `global_analyzer` 组对比。可绕开 redis/controller/docker/Megatron 全栈。

---

## 里程碑 2：工具链与构建可行性（已验证可行）

### LD_PRELOAD 拦截可行性（决定性）
- `libtorch_cuda.so` 的 `DT_NEEDED` 里**含 `libmccl.so`**，且它引用的 `mccl*` 全部是 `GLOBAL DEFAULT UND`（未定义、默认全局绑定）。
- 结论：一个 **LD_PRELOAD 的 .so 只要导出同名 `mccl*`，就能抢先绑定、拦截 torch 的真实集合通信**。拦截路径成立 ✔。

### 编译器：cucc（不是 mxcc 裸用，也不是 g++）
- 系统 `g++ 11.4` 直接编不过：cu-bridge `cuda_runtime.h` → `mc_runtime_types.h` 找不到（缺 MACA include 根，且该头不在常规路径）。
- **`cucc`（`/opt/maca/tools/cu-bridge/bin/cucc`，bash 包装脚本）是正解**：自动带齐 MACA/cu-bridge 的 include 与 lib。实测综合用例（`#include <cuda_runtime.h> + <nccl.h> + boost/interprocess + dlopen/dlsym + cudaEvent* + 真调 mcclAllReduce`）**一次编过**，产出可用 `.so`；`.cu` 内核（`<<<>>>`）也用 cucc 编过。
- `<nccl.h>`/`<cuda_runtime.h>` 在 cucc 下自动解析到 cu-bridge 版本（`CUDA_PATH=/opt/maca/tools/cu-bridge`）。→ **源码里的 `ncclXxx` 类型/常量原样可用**（它们是 `mcclXxx` 的别名），只需改「导出函数名」和「dlsym 目标」。

### 依赖安装结果（都在 pod 内，未动 torch/系统运行时）
- Python（conda 环境）：`Rbeast`、`statsmodels`、`redis` 已 `pip install` 成功（numpy/pandas/scipy/matplotlib 本来就有）。
- C++：`apt-get install libboost-all-dev` 成功（boost 1.74，system g++ ABI 匹配）。
- **cpp_redis / tacopie / redis-server 仍缺**：这些只被「控制面/缓解」用（redis 拓扑缓存 + 控制信号）。**检测核 MVP 决定不依赖 redis**（见下解耦方案），故不装。

### mccl.h ABI 要点（写 hook 用）
- MCCL = NCCL 2.16.5 对标（`MCCL_VERSION_CODE 21605`，`MCCL_UNIQUE_ID_BYTES=128`）。
- 集合通信签名与 NCCL **逐字段一致**，仅前缀 `nccl`→`mccl`、stream 类型叫 `mcStream_t`（= `cudaStream_t` 别名）。
- **torch 同时调用「裸版」和「Ext 版」**（`mcclAllReduce` 与 `mcclAllReduceExt`、`mcclSend`/`mcclSendExt`、`mcclRecv`/`mcclRecvExt`、`mcclAllGather(Ext)`、`mcclReduceScatter(Ext)`）→ hook 必须**两版都拦**（签名相同）。
- `mcclCommSplit` 在 libmccl 中**不存在**（torch 也没引用）→ 原 `ncclCommSplit` hook 删掉。

---

## 里程碑 3：检测核解耦与移植方案（进行中）

**检测核最小闭环不依赖 redis**：hook 采集 `mccl*` → 写 `/dev/shm/ncclRecord` 环形缓冲 → Python `NcclRecord` mmap 读 → `find_period`(ACF) + `find_performance_drop`(Rbeast 变点) 估计每 rank 迭代时间 → `global_analyzer` 组对比。

关键洞察：`detect_failslow` 用的是 **`call_time`（每次调用的墙钟时间戳）的周期差**来估计迭代时间，**不依赖 `duration`**（cudaEvent 计时）。而 `duration` 只在 `STATE_PROFILE/VALIDATE`（缓解验证阶段，走 redis 控制信号）才填。→ **纯 MONITOR 模式（duration=0）就够跑检测核**，可彻底绕开 redis/controller/cudaEvent。

**编译期开关 `GREYHOUND_DETECTION_ONLY`**（Makefile.metax 默认开）：
- 编入：`ncclprobe.cpp`（改名 mccl* + Ext + 守卫）、`shm_storage.cpp`（shm 环形缓冲，核心数据面）、`global_status.cpp`（剥离 redis/controller）、`comm.cpp`（`parse_communicator` 桩）、`shm_topo.cpp`（只留 `Communicator`+`hash_nccl_id`，守卫掉 `NcclTopoConnection`）。
- 排除：`event_handler.cpp`（cpp_redis + 控制信号）、`matmul.cu`（GEMM 微基准，仅缓解验证用）。
- 守卫掉的 redis/cpp_redis/boost::process/EventHandler/NcclTopoConnection 全部 `#ifndef GREYHOUND_DETECTION_ONLY`，保留原代码，将来装了 cpp_redis 可一键开全功能。

## 待办 / 下一步

1. [进行中] 改造 `detector/`：hpp 守卫 + 重写 `ncclprobe.cpp`/`global_status.cpp`/`comm.cpp` + `Makefile.metax`（cucc）。
2. 在 pod 上编译 `libmcclprobe.so`（检测核版）。
3. 写最小 MetaX 通信 workload（torchrun 多进程重复 AllReduce 模拟训练迭代），LD_PRELOAD 采集。
4. 跑 Python 检测核：ACF 周期 + 变点迭代时间估计 + 组对比；注入 fail-slow 看能否检出。
5. 记录跑到哪步、卡在哪。

---

## 里程碑 4：编译通过 —— libmcclprobe.so 成功产出

`make -f Makefile.metax`（cucc）在 worker-126 上，**5 个 .o 全编过 + 链接成功**，产出 `libmcclprobe.so`（1.17 MB）。唯一告警无害：`--offload-arch=xcore1000 unused`（检测核纯 host 代码 + dlsym 转发，无 device kernel）。

### 导出符号验证（决定能否拦截 torch）
`nm -D libmcclprobe.so | grep " T mccl"` → **16 个拦截入口全部导出为已定义符号（T）**：
`mcclCommInitRank / mcclCommInitRankConfig / mcclSend(Ext) / mcclRecv(Ext) / mcclBcast / mcclBroadcast / mcclAllGather(Ext) / mcclReduceScatter(Ext) / mcclAllReduce(Ext) / mcclGroupStart / mcclGroupEnd`。
且 `nm -D | grep " U mccl"` **为空**——本 so 不静态依赖任何 mccl 符号，真实函数全部运行时 dlsym 转发（`get_function_ptr`→libmccl）。这正是 LD_PRELOAD interposer 应有形态。

### 改动清单（都在 `greyhound-metax/detector/`）
- **`ncclprobe.cpp`**（重写）：导出符号 `nccl*`→`mccl*`（`PROBE_FN`/`REAL_SYM` 宏在两套 ABI 间切换）；新增 `mccl*Ext` 变体拦截；删 `ncclCommSplit`（libmccl 无此符号）；`extern "C"` 保证导出名不 mangling。
- **`global_status.{hpp,cpp}`**：`#ifndef GREYHOUND_DETECTION_ONLY` 守卫掉 controller 进程 / whl 安装 / redis 等待；dlopen 默认路径改 `/opt/maca/lib/libmccl.so`；EventHandler 用 `mcclSend/mcclRecv` 解析。
- **`comm.cpp`**：`parse_communicator` 桩——用公开 API `mcclCommUserRank`+`mcclCommCount`（dlsym `RTLD_NOLOAD`）取 rank/nRanks，不逆向私有结构体；原 HackedComm 保留在 `#else` 分支。
- **`event_handler.hpp` / `shm_topo.{hpp,cpp}`**：detection-only 下把依赖 cpp_redis 的 `EventHandler`/`NcclTopoConnection` 换成 header-only no-op 桩。
- **`Makefile.metax`**（新增）：cucc 编译，`-DGREYHOUND_DETECTION_ONLY -DBOOST_LOG_DYN_LINK`，链 boost_log/thread/system/filesystem + pthread/rt/dl；排除 `event_handler.cpp` 和 `matmul.cu`。

---

## 里程碑 5：检测核在沐曦上端到端跑通 + fail-slow 检出成功

### LD_PRELOAD 拦截运行时验证（LD_DEBUG=bindings）
```
binding file libtorch_cuda.so to libmcclprobe.so: normal symbol `mcclAllReduce'
binding file libtorch_cuda.so to libmcclprobe.so: normal symbol `mcclAllReduceExt'
```
torch 的真实 `mcclAllReduce`/`Ext` 调用**确实绑定到我们的 probe**。probe `detect_sys_init` 触发、`get_function_ptr` 取到真 `mccl*`（如 `mcclAllReduce real=0x…cad10`），转发正常，torch 全程不崩、结果正确。

### 数据面：shm 环形缓冲工作正常
- boost::interprocess 在 `/dev/shm/ncclRecord`（+`recordLock`）创建缓冲（运行中实测 2.24 MB）。
- Python `multiprocessing.shared_memory.SharedMemory("ncclRecord")` 直接 mmap 读到（名字对齐）。
- **坑**：`NcclRecordStorage` 析构会 `remove("ncclRecord")`，进程退出即清；workload 加 `--hold-sec` 保活窗口，检测器在 hold 期间快照。

### 检测核：ACF 周期 + Rbeast 变点 + 组对比全通
标称 workload（4 GPU / 4 rank / 500 iter / 每 iter: AllReduce×2 + AllGather + ReduceScatter，AllGather/ReduceScatter 按原 TP 逻辑压缩），采集 **6120 条 Record**。

`detect_runner.py`（复用 `control_plane/slow_detection.py` 的 `find_period`+`find_performance_drop`，用 importlib 直接加载该文件，绕开 `control_plane/__init__.py` 拉起的 redis/controller/cvxpy 链）：
- `find_period`（statsmodels ACF）对每个 rank **恢复出周期 period=3**（压缩后每迭代 3 次记录：AllReduce + 压缩块 + AllReduce）。
- `find_performance_drop`（Rbeast 贝叶斯变点）估计每 rank 迭代时间 + 变点。

### fail-slow 注入检出（rank2 从 iter250 起每步 +15ms straggler）
```
rank 0: est_iter_time=0.0033s  change_points=[506641]
rank 1: est_iter_time=0.0033s  change_points=[509914]
rank 2: est_iter_time=0.0199s  change_points=[510370, 1417284]   <-- 多一个注入点变点
rank 3: est_iter_time=0.0034s  change_points=[509973]
median iter time = 0.0034s
  rank 2: 0.0199s  <-- SLOW (>=1.2x median)   ← 正确命中 straggler
```
- **迭代时间估计**：慢 rank 2 = 0.0199s，健康 rank ≈ 0.0033s（~6×）。
- **变点检测**：rank2 比健康 rank **多一个变点 t≈1417284us**，正是 iter250 注入时刻；健康 rank 只有收尾变点。
- **组对比**：`find_slow_clique` 同款逻辑（≥1.2× 中位数）**正确把 rank2 标为 SLOW**。

→ **Greyhound 的检测机制（hook 采集 → shm 环形缓冲 → ACF 周期 → 变点迭代时间估计 → 跨 rank 组对比）已在沐曦 C550 上完整跑通，并成功检出注入的 fail-slow straggler。**

### 产物落盘
- 本机：`results/greyhound-metax-run/`（snap.npy 标称、snap_slow.npy 注入、workload 日志）。
- AFS（yinjinrun.p）：`/afs-a3-weight-share/yinjinrun.p/greyhound-metax/20260721_124736-detcore/`（libmcclprobe.so + 两份 snapshot + 日志）。

### 复现命令（pod 内）
```bash
# 1) 编译（cucc）
cd /tmp/greyhound-metax/detector && make -f Makefile.metax    # -> libmcclprobe.so
# 2) 起带 probe 的 workload（注入 straggler）
cd ../metax_eval
NPROC=4 ITERS=500 SIZE=4194304 SLOW_AFTER=250 SLOW_RANK=2 SLOW_MS=15 HOLD_SEC=25 bash run_probe.sh
# 3) hold 窗口内跑检测核
python3 detect_runner.py --snapshot /tmp/greyhound-metax/run/snap.npy
```

## 已知限制 / 后续可选

- **`duration`（cudaEvent 集合通信精确耗时）在 MONITOR 模式为 0**：检测核只用 `call_time` 墙钟周期差估计迭代时间，已足够检出 fail-slow；若要 per-comm 耗时（`get_profile_results`），需开 PROFILE/VALIDATE 态（依赖 redis 控制信号）。
- **缓解模块（mitigation）与 comm 拓扑逆向已桩过**：`parse_communicator` 用公开 API 只取 rank/nRanks（ring/tree 空）；`event_handler.cpp`（p2p/gemm 微基准 + redis 控制面）、`matmul.cu` 未编入；`global_analyzer` 的 TP/DP/PP clique 划分依赖 Megatron 并行信息（需真实训练接线，本 MVP 用 rank 组对比替代）。
- **全功能路径**（redis 拓扑缓存 + 控制信号 + 缓解）：装 cpp_redis/tacopie/redis-server + cvxpy 后去掉 `-DGREYHOUND_DETECTION_ONLY` 即可编回原全栈（守卫代码都保留了）。
