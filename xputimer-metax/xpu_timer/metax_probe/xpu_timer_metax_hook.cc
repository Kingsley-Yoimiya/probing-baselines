// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
//
// xpu_timer_metax_hook.cc
// -----------------------
// MetaX (沐曦) C550 backend for the XPUTimer fail-slow detection mechanism.
//
// This is a *self-contained* re-implementation of XPUTimer's core detection
// loop (see xpu_timer/common/manager.h + xpu_timer/nvidia/{hook,nvidia_timer}.cc
// upstream) that compiles with plain g++ against the MACA SDK headers, without
// the heavy Bazel-only deps (brpc / boost / perfetto / protobuf / prometheus-cpp).
//
// Why a rewrite instead of -DXPU_METAX in the Bazel tree:
//   * MACA does NOT export cuda*/nccl* symbols. torch(2.6.0+metax) links the
//     mc*/mccl* symbols directly (verified: `nm -D libtorch_cuda.so` shows
//     `U mcLaunchKernel`, `U mcclAllReduce`). So the LD_PRELOAD interposer must
//     export mc*/mccl* names, not cuda*/nccl*.
//   * The pod has no Bazel 6.2.0, no cmake, flaky github -> the upstream deps
//     graph (brpc/boost/perfetto) is unbuildable here. The *detection mechanism*
//     (event timing + hang/slow poller + metric emission) needs none of that.
//
// Mechanism reproduced faithfully:
//   1. Interpose mcLaunchKernel (via dlsym(RTLD_NEXT)) + the mccl* collectives
//      (via explicit dlopen("libmccl.so")+dlsym, because torch dlopen()s libmccl
//      lazily so it is NOT in the RTLD_NEXT order when our interposer first runs).
//   2. Around each mcLaunchKernel, bracket a MACA event pair (mcEventRecord) on
//      the op's stream  ==  XPUTimer's NvidiaGpuTimer::startRecord/endRecord.
//      The mccl* collective kernels are ALSO timed via this same mcLaunchKernel
//      path (they enqueue kernels); the mccl* API hooks themselves only record
//      lightweight metadata (count+bytes), mirroring upstream interceptNcclInfo.
//   3. A background poller thread drains a work queue: when the stop event is
//      ready (mcEventQuery) it computes mcEventElapsedTime and folds it into a
//      per-kernel histogram; if an op has been outstanding longer than the hang
//      timeout it is flagged HANG  ==  XPUTimer's GpuTimerManager::doWork/doHang.
//   4. getDeviceName() via mcGetDeviceProperties  ==  common/platform.cc.
//   5. Metrics are dumped in Prometheus text-exposition format (the same metric
//      names XPUTimer exports: xpu_timer_common_kernel_*), plus a JSONL trace.
//
// cuBLAS(mcblas*) matmul shape parsing and mccl comm-topology (parse_params)
// are intentionally stubbed here -- they enrich labels but are NOT required for
// the fail-slow detection loop, per the task's "先桩过" guidance.

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---- MACA SDK types we need. We declare the minimal ABI ourselves so the hook
// ---- stays compilable even without the full maca headers on the include path.
// ---- IMPORTANT: we do NOT reference the mc* runtime helper symbols directly
// ---- (mcEvent*, mcGetDevice*) -- that would create undefined symbols that the
// ---- loader must resolve at LD_PRELOAD time, before torch has loaded
// ---- libmcruntime.so. Instead every runtime helper is resolved lazily via
// ---- dlopen("libmcruntime.so") + dlsym, matching XPUTimer's
// ---- SETUP_DLSYM_WITH_DLOPEN. Interposed originals still use dlsym(RTLD_NEXT).
extern "C" {
// dim3 (matches /opt/maca/include/common/__clang_maca_base_type.h)
struct dim3 {
  unsigned int x, y, z;
};
typedef struct MCstream_st* mcStream_t;
typedef struct MCevent_st* mcEvent_t;
typedef int mcError_t;                 // mcSuccess == 0
typedef struct mcclComm* mcclComm_t;   // opaque
typedef int mcclResult_t;              // mcclSuccess == 0
typedef int mcclDataType_t;
typedef int mcclRedOp_t;
}

namespace {
constexpr int kMcSuccess = 0;

// ----- config via env (mirrors XPU_TIMER_* knobs upstream) -----
struct Config {
  bool enable = true;
  bool time_launch_kernel = true;   // hook mcLaunchKernel
  int  launch_sample = 1;           // time 1/N launches (overhead knob)
  long hang_timeout_ms = 2000;      // op outstanding longer -> HANG
  long slow_report_us = 0;          // per-op latency >= this -> SLOW log (0=off)
  int  poller_sleep_us = 100;       // XPUTimer uses 100us
  std::string dump_dir = "/tmp/xpu_timer_metax";
  int  dump_interval_s = 5;
};

Config& cfg() {
  static Config c = [] {
    Config x;
    if (const char* e = getenv("XPU_TIMER_ENABLE")) x.enable = atoi(e) != 0;
    if (const char* e = getenv("XPU_TIMER_HOOK_LAUNCH")) x.time_launch_kernel = atoi(e) != 0;
    if (const char* e = getenv("XPU_TIMER_LAUNCH_SAMPLE")) x.launch_sample = std::max(1, atoi(e));
    if (const char* e = getenv("XPU_TIMER_HANG_TIMEOUT_MS")) x.hang_timeout_ms = atol(e);
    if (const char* e = getenv("XPU_TIMER_SLOW_REPORT_US")) x.slow_report_us = atol(e);
    if (const char* e = getenv("XPU_TIMER_DUMP_DIR")) x.dump_dir = e;
    if (const char* e = getenv("XPU_TIMER_DUMP_INTERVAL_S")) x.dump_interval_s = atoi(e);
    return x;
  }();
  return c;
}

inline uint64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ============================================================================
//  MACA runtime helper table -- resolved lazily via dlopen("libmcruntime.so").
//  This is XPUTimer's SETUP_DLSYM_WITH_DLOPEN pattern: we never create an
//  undefined mc* symbol that the loader would have to satisfy at preload time
//  (before torch loads MACA). Path overridable via XPU_TIMER_MACA_RUNTIME_LIB.
// ============================================================================
struct MacaRt {
  typedef mcError_t (*EventCreate_t)(mcEvent_t*);
  typedef mcError_t (*EventRecord_t)(mcEvent_t, mcStream_t);
  typedef mcError_t (*EventQuery_t)(mcEvent_t);
  typedef mcError_t (*EventElapsed_t)(float*, mcEvent_t, mcEvent_t);
  typedef mcError_t (*EventDestroy_t)(mcEvent_t);
  typedef mcError_t (*GetDeviceCount_t)(int*);
  typedef mcError_t (*GetDeviceProperties_t)(void*, int);

  EventCreate_t EventCreate = nullptr;
  EventRecord_t EventRecord = nullptr;
  EventQuery_t EventQuery = nullptr;
  EventElapsed_t EventElapsedTime = nullptr;
  EventDestroy_t EventDestroy = nullptr;
  GetDeviceCount_t GetDeviceCount = nullptr;
  GetDeviceProperties_t GetDeviceProperties = nullptr;
  bool ok = false;

  bool load() {
    if (ok) return true;
    const char* libname = getenv("XPU_TIMER_MACA_RUNTIME_LIB");
    // Prefer RTLD_NOLOAD-style: torch already loaded libmcruntime.so; a plain
    // dlopen returns the same handle without reloading.
    void* h = dlopen(libname ? libname : "libmcruntime.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("/opt/maca/lib/libmcruntime.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) {
      fprintf(stderr, "[xpu_timer.metax] cannot dlopen libmcruntime.so: %s\n",
              dlerror());
      return false;
    }
    EventCreate = (EventCreate_t)dlsym(h, "mcEventCreate");
    EventRecord = (EventRecord_t)dlsym(h, "mcEventRecord");
    EventQuery = (EventQuery_t)dlsym(h, "mcEventQuery");
    EventElapsedTime = (EventElapsed_t)dlsym(h, "mcEventElapsedTime");
    EventDestroy = (EventDestroy_t)dlsym(h, "mcEventDestroy");
    GetDeviceCount = (GetDeviceCount_t)dlsym(h, "mcGetDeviceCount");
    GetDeviceProperties = (GetDeviceProperties_t)dlsym(h, "mcGetDeviceProperties");
    ok = EventCreate && EventRecord && EventQuery && EventElapsedTime &&
         EventDestroy && GetDeviceCount && GetDeviceProperties;
    if (!ok)
      fprintf(stderr,
              "[xpu_timer.metax] some mc* runtime symbols missing "
              "(create=%p record=%p query=%p elapsed=%p count=%p prop=%p)\n",
              (void*)EventCreate, (void*)EventRecord, (void*)EventQuery,
              (void*)EventElapsedTime, (void*)GetDeviceCount,
              (void*)GetDeviceProperties);
    return ok;
  }
};

MacaRt& rt() {
  static MacaRt r;
  return r;
}

// Resolve an mccl symbol. torch dlopen()s libmccl.so lazily, so it is NOT in
// the RTLD_NEXT search order when our preloaded interposer first runs --
// dlsym(RTLD_NEXT, "mcclAllReduce") returns NULL and calling it segfaults.
// So we resolve the real symbol by explicitly dlopen("libmccl.so") + dlsym,
// exactly like XPUTimer's SETUP_DLSYM_WITH_DLOPEN (env XPU_TIMER_MCCL_LIB
// overrides the path). Cached per symbol.
void* resolve_mccl(const char* sym) {
  static void* handle = [] {
    const char* p = getenv("XPU_TIMER_MCCL_LIB");
    void* h = dlopen(p ? p : "libmccl.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!h) h = dlopen(p ? p : "libmccl.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("/opt/maca/lib/libmccl.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!h)
      fprintf(stderr, "[xpu_timer.metax] cannot dlopen libmccl.so: %s\n",
              dlerror());
    return h;
  }();
  if (!handle) return nullptr;
  void* fn = dlsym(handle, sym);
  // Fallback to RTLD_NEXT in case libmccl was already NEEDED-linked.
  if (!fn) fn = dlsym(RTLD_NEXT, sym);
  return fn;
}

// ---------- per-kernel aggregated metrics (histogram-lite) ----------
struct KernelStat {
  uint64_t count = 0;
  double sum_us = 0;
  double max_us = 0;
  uint64_t hang_count = 0;
  uint64_t slow_count = 0;
  std::string type;  // "kernel" | "coll"
};

// ---------- a single timed op: XPUTimer's NvidiaGpuTimer analogue ----------
struct TimedOp {
  mcEvent_t start = nullptr;
  mcEvent_t stop = nullptr;
  std::string name;
  std::string type;
  uint64_t problem_size = 0;   // bytes for coll, 0 otherwise
  uint64_t launch_wall_us = 0;
  bool hang_reported = false;
};

// ================= GpuTimerManager (singleton) =================
class Manager {
 public:
  static Manager& get() {
    static Manager m;
    return m;
  }

  bool enabled() const { return cfg().enable && running_.load(); }

  // acquire a reusable event pair (object pool == TimerPool<T> upstream)
  TimedOp* acquire() {
    std::lock_guard<std::mutex> lk(pool_mu_);
    TimedOp* op;
    if (!pool_.empty()) {
      op = pool_.back();
      pool_.pop_back();
    } else {
      op = new TimedOp();
      rt().EventCreate(&op->start);
      rt().EventCreate(&op->stop);
    }
    op->hang_reported = false;
    return op;
  }

  void record_start(TimedOp* op, mcStream_t s) {
    op->launch_wall_us = now_us();
    rt().EventRecord(op->start, s);
  }
  void record_stop_and_enqueue(TimedOp* op, mcStream_t s) {
    rt().EventRecord(op->stop, s);
    std::lock_guard<std::mutex> lk(q_mu_);
    work_.push_back(op);
  }

  // Lightweight collective-metadata recorder. Unlike kernel timing we do NOT
  // bracket mcEvents around the mccl call: mccl invokes these APIs internally
  // during comm bootstrap on fragile private streams, and recording events
  // there deadlocks/segfaults. The *collective kernel* is already timed via the
  // mcLaunchKernel hook; here we only fold invocation count + transferred bytes
  // into the per-op stats (this mirrors upstream's interceptNcclInfo, which
  // records metadata and lets cudaLaunchKernel time the actual work).
  void record_coll_meta(const char* name, uint64_t bytes) {
    std::lock_guard<std::mutex> lk(stat_mu_);
    KernelStat& s = stats_[std::string("coll:") + name];
    s.type = "coll";
    s.count++;
    s.sum_us += 0;  // latency comes from the kernel-timing path
    coll_bytes_[name] += bytes;
  }

  std::string device_name() {
    std::call_once(dev_once_, [this] { detect_device(); });
    return dev_name_;
  }

  void start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    if (!rt().load()) {
      fprintf(stderr,
              "[xpu_timer.metax] runtime not ready; detection disabled\n");
      started_.store(false);
      return;
    }
    detect_device();
    ensure_dump_dir();
    running_.store(true);
    poller_ = std::thread([this] { poll_loop(); });
    fprintf(stderr,
            "[xpu_timer.metax] started. device=%s pid=%d hook_launch=%d "
            "hang_timeout=%ldms dump=%s\n",
            dev_name_.c_str(), getpid(), cfg().time_launch_kernel,
            cfg().hang_timeout_ms, cfg().dump_dir.c_str());
  }

  // Lazy, thread-safe first-touch init: called from the first intercepted op,
  // by which point torch has already loaded libmcruntime/libmccl.
  void ensure_started() {
    std::call_once(start_once_, [this] { start(); });
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (poller_.joinable()) poller_.join();
    dump_metrics(/*final=*/true);
    fprintf(stderr,
            "[xpu_timer.metax] stopped. metrics -> %s (max_outstanding=%.1fms "
            "queue_peak=%zu)\n",
            dump_dir_prom_.c_str(), max_outstanding_us_ / 1000.0, queue_peak_);
  }

 private:
  Manager() = default;
  ~Manager() { stop(); }

  void detect_device() {
    dev_name_ = "UNKNOWN";
    int cnt = 0;
    if (!rt().ok || rt().GetDeviceCount(&cnt) != kMcSuccess || cnt == 0) return;
    // mcDeviceProp_t: .name is first field (char[256]); pass a big buffer.
    static char prop[8192];
    memset(prop, 0, sizeof(prop));
    if (rt().GetDeviceProperties(prop, 0) == kMcSuccess) {
      std::string full(prop);  // ".name" at offset 0, e.g. "MetaX C550-PL"
      dev_name_ = full.empty() ? "UNKNOWN" : full;
    }
  }

  void ensure_dump_dir() {
    std::string d = cfg().dump_dir;
    std::string cmd = "mkdir -p '" + d + "'";
    if (system(cmd.c_str()) != 0) { /* dir may already exist; ignore */ }
    // pid+rank suffix so multiple ranks on one node don't clobber each other.
    const char* rank = getenv("RANK");
    char suffix[64];
    snprintf(suffix, sizeof(suffix), "rank%s.pid%d",
             rank ? rank : "NA", getpid());
    dump_dir_prom_ = d + "/metax_metrics." + suffix + ".prom";
    dump_dir_trace_ = d + "/metax_trace." + suffix + ".jsonl";
    trace_.open(dump_dir_trace_, std::ios::out | std::ios::trunc);
  }

  void poll_loop() {
    uint64_t last_dump = now_us();
    while (running_.load()) {
      drain_once();
      uint64_t t = now_us();
      if (t - last_dump > (uint64_t)cfg().dump_interval_s * 1000000ULL) {
        dump_metrics(false);
        last_dump = t;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(cfg().poller_sleep_us));
    }
    drain_once();  // final flush
  }

  void drain_once() {
    std::deque<TimedOp*> local;
    {
      std::lock_guard<std::mutex> lk(q_mu_);
      local.swap(work_);
      if (local.size() > queue_peak_) queue_peak_ = local.size();
    }
    std::deque<TimedOp*> still_pending;
    for (TimedOp* op : local) {
      mcError_t st = rt().EventQuery(op->stop);
      if (st == kMcSuccess) {
        float ms = 0.f;
        rt().EventElapsedTime(&ms, op->start, op->stop);
        double us = ms * 1000.0;
        fold(op, us, /*hang=*/false);
        release(op);
      } else {
        // not ready: check for hang (fail-slow signal)
        uint64_t outstanding_us = now_us() - op->launch_wall_us;
        if (outstanding_us > max_outstanding_us_) max_outstanding_us_ = outstanding_us;
        if (!op->hang_reported &&
            outstanding_us > (uint64_t)cfg().hang_timeout_ms * 1000ULL) {
          op->hang_reported = true;
          fold(op, (double)outstanding_us, /*hang=*/true);
          fprintf(stderr,
                  "[xpu_timer.metax][HANG] op=%s type=%s outstanding=%.1fms "
                  ">= %ldms (fail-slow candidate)\n",
                  op->name.c_str(), op->type.c_str(), outstanding_us / 1000.0,
                  cfg().hang_timeout_ms);
        }
        still_pending.push_back(op);
      }
    }
    if (!still_pending.empty()) {
      std::lock_guard<std::mutex> lk(q_mu_);
      for (TimedOp* op : still_pending) work_.push_back(op);
    }
  }

  void fold(TimedOp* op, double us, bool hang) {
    std::lock_guard<std::mutex> lk(stat_mu_);
    KernelStat& s = stats_[op->name];
    s.type = op->type;
    if (hang) {
      s.hang_count++;
    } else {
      s.count++;
      s.sum_us += us;
      if (us > s.max_us) s.max_us = us;
      if (cfg().slow_report_us > 0 && us >= (double)cfg().slow_report_us) {
        s.slow_count++;
        fprintf(stderr,
                "[xpu_timer.metax][SLOW] op=%s type=%s latency=%.1fus >= %ldus\n",
                op->name.c_str(), op->type.c_str(), us, cfg().slow_report_us);
      }
      if (trace_.is_open()) {
        trace_ << "{\"ts_us\":" << op->launch_wall_us << ",\"name\":\""
               << op->name << "\",\"type\":\"" << op->type
               << "\",\"dur_us\":" << us << ",\"psize\":" << op->problem_size
               << "}\n";
      }
    }
  }

  void release(TimedOp* op) {
    std::lock_guard<std::mutex> lk(pool_mu_);
    pool_.push_back(op);
  }

  void dump_metrics(bool final) {
    std::lock_guard<std::mutex> lk(stat_mu_);
    std::ofstream f(dump_dir_prom_, std::ios::out | std::ios::trunc);
    if (!f.is_open()) return;
    f << "# XPUTimer MetaX backend metrics (Prometheus text exposition)\n";
    f << "# device=" << dev_name_ << " pid=" << getpid() << "\n";
    for (auto& kv : stats_) {
      const std::string& name = kv.first;
      const KernelStat& s = kv.second;
      double avg = s.count ? s.sum_us / s.count : 0.0;
      f << "xpu_timer_common_kernel_count{name=\"" << name << "\",type=\""
        << s.type << "\"} " << s.count << "\n";
      f << "xpu_timer_common_kernel_avg_us{name=\"" << name << "\",type=\""
        << s.type << "\"} " << avg << "\n";
      f << "xpu_timer_common_kernel_max_us{name=\"" << name << "\",type=\""
        << s.type << "\"} " << s.max_us << "\n";
      f << "xpu_timer_common_kernel_hang{name=\"" << name << "\",type=\""
        << s.type << "\"} " << s.hang_count << "\n";
      f << "xpu_timer_common_kernel_slow{name=\"" << name << "\",type=\""
        << s.type << "\"} " << s.slow_count << "\n";
    }
    for (auto& kv : coll_bytes_) {
      f << "xpu_timer_common_coll_bytes_total{name=\"" << kv.first << "\"} "
        << kv.second << "\n";
    }
    if (trace_.is_open()) trace_.flush();
    if (final && trace_.is_open()) trace_.close();
  }

  std::atomic<bool> started_{false};
  std::atomic<bool> running_{false};
  std::once_flag start_once_;
  std::thread poller_;
  std::once_flag dev_once_;
  std::string dev_name_;
  std::string dump_dir_prom_, dump_dir_trace_;
  std::ofstream trace_;

  std::mutex pool_mu_;
  std::vector<TimedOp*> pool_;
  std::mutex q_mu_;
  std::deque<TimedOp*> work_;
  std::mutex stat_mu_;
  std::unordered_map<std::string, KernelStat> stats_;
  std::unordered_map<std::string, uint64_t> coll_bytes_;
  uint64_t max_outstanding_us_ = 0;
  size_t queue_peak_ = 0;
};

// The .so may be LD_PRELOADed before torch loads libmcruntime.so, so we do NOT
// start the manager in a constructor. Instead the first intercepted op triggers
// ensure_started() (by then MACA is loaded). We still register an atexit-style
// destructor to flush metrics.
__attribute__((destructor)) void xpu_metax_dtor() { Manager::get().stop(); }

// ---- kernel-name resolution: dladdr on the device fn ptr (best-effort) ----
std::string resolve_kernel_name(const void* func) {
  Dl_info info;
  if (dladdr(func, &info) && info.dli_sname) return std::string(info.dli_sname);
  char buf[32];
  snprintf(buf, sizeof(buf), "kern_%p", func);
  return std::string(buf);
}

}  // namespace

// ============================================================================
//  Interposers.  Exported with default visibility so LD_PRELOAD wins over the
//  real symbols; original fetched with dlsym(RTLD_NEXT, ...) == XPUTimer's
//  SETUP_DLSYM macro.
// ============================================================================
#define EXPOSE_API __attribute__((visibility("default")))

extern "C" {

// ---- mcLaunchKernel : every kernel enqueue (the mc*-prefixed cudaLaunchKernel)
typedef mcError_t (*mcLaunchKernel_t)(const void*, dim3, dim3, void**, size_t,
                                      mcStream_t);
EXPOSE_API mcError_t mcLaunchKernel(const void* func, dim3 g, dim3 b,
                                    void** args, size_t shmem, mcStream_t s) {
  static mcLaunchKernel_t orig =
      (mcLaunchKernel_t)dlsym(RTLD_NEXT, "mcLaunchKernel");
  Manager& m = Manager::get();
  m.ensure_started();
  if (!m.enabled() || !cfg().time_launch_kernel)
    return orig(func, g, b, args, shmem, s);
  // sampling knob to bound overhead
  static thread_local uint64_t n = 0;
  if ((++n % cfg().launch_sample) != 0) return orig(func, g, b, args, shmem, s);

  TimedOp* op = m.acquire();
  op->name = resolve_kernel_name(func);
  op->type = "kernel";
  op->problem_size = 0;
  m.record_start(op, s);
  mcError_t rc = orig(func, g, b, args, shmem, s);
  m.record_stop_and_enqueue(op, s);
  return rc;
}

// ---- helper: bytes for a coll of `count` elements of dtype ----
#ifndef METAX_NO_COLL
static inline uint64_t dtype_bytes(mcclDataType_t dt) {
  // mccl.h: int8=0,uint8=1,int32=2,uint32=3,int64=4,uint64=5,float16=6,
  //         float32=7,float64=8,bfloat16=9
  switch (dt) {
    case 0: case 1: return 1;
    case 2: case 3: case 7: return 4;
    case 4: case 5: case 8: return 8;
    case 6: case 9: return 2;
    default: return 4;
  }
}

// Metadata-only collective hook. NO mcEvent bracketing on the mccl stream (that
// crashes during comm bootstrap). We record invocation count + bytes; the coll
// *kernel* latency is captured by the mcLaunchKernel hook. `stream` is unused
// but kept in the signature for clarity/future comm-topology parsing.
#ifdef METAX_COLL_DEBUG
#define COLL_DBG(NAME) fprintf(stderr, "[xpu_timer.metax][dbg] enter %s\n", NAME)
#define COLL_DBG2(NAME) fprintf(stderr, "[xpu_timer.metax][dbg] after orig %s\n", NAME)
#else
#define COLL_DBG(NAME)
#define COLL_DBG2(NAME)
#endif
#define COLL_META(NAME, EXPR)                         \
  do {                                                \
    COLL_DBG(NAME);                                   \
    Manager& _m = Manager::get();                     \
    _m.ensure_started();                              \
    if (_m.enabled()) _m.record_coll_meta(NAME, psize); \
    mcclResult_t _rc = (EXPR);                        \
    COLL_DBG2(NAME);                                  \
    return _rc;                                       \
  } while (0)

// ---- mccl collectives (the mccl*-prefixed nccl API) ----
typedef mcclResult_t (*mcclAllReduce_t)(const void*, void*, size_t,
                                        mcclDataType_t, mcclRedOp_t, mcclComm_t,
                                        mcStream_t);
EXPOSE_API mcclResult_t mcclAllReduce(const void* s, void* r, size_t count,
                                      mcclDataType_t dt, mcclRedOp_t op,
                                      mcclComm_t comm, mcStream_t stream) {
  static mcclAllReduce_t orig =
      (mcclAllReduce_t)resolve_mccl("mcclAllReduce");
  uint64_t psize = count * dtype_bytes(dt);
  COLL_META("mcclAllReduce", orig(s, r, count, dt, op, comm, stream));
}

typedef mcclResult_t (*mcclAllGather_t)(const void*, void*, size_t,
                                        mcclDataType_t, mcclComm_t, mcStream_t);
EXPOSE_API mcclResult_t mcclAllGather(const void* s, void* r, size_t sendcount,
                                      mcclDataType_t dt, mcclComm_t comm,
                                      mcStream_t stream) {
  static mcclAllGather_t orig =
      (mcclAllGather_t)resolve_mccl("mcclAllGather");
  uint64_t psize = sendcount * dtype_bytes(dt);
  COLL_META("mcclAllGather", orig(s, r, sendcount, dt, comm, stream));
}

typedef mcclResult_t (*mcclReduceScatter_t)(const void*, void*, size_t,
                                            mcclDataType_t, mcclRedOp_t,
                                            mcclComm_t, mcStream_t);
EXPOSE_API mcclResult_t mcclReduceScatter(const void* s, void* r,
                                          size_t recvcount, mcclDataType_t dt,
                                          mcclRedOp_t op, mcclComm_t comm,
                                          mcStream_t stream) {
  static mcclReduceScatter_t orig =
      (mcclReduceScatter_t)resolve_mccl("mcclReduceScatter");
  uint64_t psize = recvcount * dtype_bytes(dt);
  COLL_META("mcclReduceScatter", orig(s, r, recvcount, dt, op, comm, stream));
}

typedef mcclResult_t (*mcclBroadcast_t)(const void*, void*, size_t,
                                        mcclDataType_t, int, mcclComm_t,
                                        mcStream_t);
EXPOSE_API mcclResult_t mcclBroadcast(const void* s, void* r, size_t count,
                                      mcclDataType_t dt, int root,
                                      mcclComm_t comm, mcStream_t stream) {
  static mcclBroadcast_t orig =
      (mcclBroadcast_t)resolve_mccl("mcclBroadcast");
  uint64_t psize = count * dtype_bytes(dt);
  COLL_META("mcclBroadcast", orig(s, r, count, dt, root, comm, stream));
}

typedef mcclResult_t (*mcclReduce_t)(const void*, void*, size_t, mcclDataType_t,
                                     mcclRedOp_t, int, mcclComm_t, mcStream_t);
EXPOSE_API mcclResult_t mcclReduce(const void* s, void* r, size_t count,
                                   mcclDataType_t dt, mcclRedOp_t op, int root,
                                   mcclComm_t comm, mcStream_t stream) {
  static mcclReduce_t orig = (mcclReduce_t)resolve_mccl("mcclReduce");
  uint64_t psize = count * dtype_bytes(dt);
  COLL_META("mcclReduce", orig(s, r, count, dt, op, root, comm, stream));
}

typedef mcclResult_t (*mcclSend_t)(const void*, size_t, mcclDataType_t, int,
                                   mcclComm_t, mcStream_t);
EXPOSE_API mcclResult_t mcclSend(const void* s, size_t count, mcclDataType_t dt,
                                 int peer, mcclComm_t comm, mcStream_t stream) {
  static mcclSend_t orig = (mcclSend_t)resolve_mccl("mcclSend");
  uint64_t psize = count * dtype_bytes(dt);
  COLL_META("mcclSend", orig(s, count, dt, peer, comm, stream));
}

typedef mcclResult_t (*mcclRecv_t)(void*, size_t, mcclDataType_t, int,
                                   mcclComm_t, mcStream_t);
EXPOSE_API mcclResult_t mcclRecv(void* r, size_t count, mcclDataType_t dt,
                                 int peer, mcclComm_t comm, mcStream_t stream) {
  static mcclRecv_t orig = (mcclRecv_t)resolve_mccl("mcclRecv");
  uint64_t psize = count * dtype_bytes(dt);
  COLL_META("mcclRecv", orig(r, count, dt, peer, comm, stream));
}
#endif  // METAX_NO_COLL

}  // extern "C"
