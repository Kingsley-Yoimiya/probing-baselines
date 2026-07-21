#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <ctime>
#include <sys/time.h>
#include <memory>
#include <boost/log/trivial.hpp>

#include "shm_storage.hpp"
#include "config.hpp"
#include "global_status.hpp"
#include "comm.hpp"
#include "utils.hpp"

#define CHECK_PAUSE_MAGIC 503
#define RECORD_TOO_SMALL(count) ((count) < MIN_RECORD_OP_SIZE)

/* On MetaX, the collective library (libmccl.so) exports mccl* symbols with the
   same ABI as NCCL's nccl* symbols (nccl.h from cu-bridge is just a static
   alias shim over mccl.h).  The intercepted, *exported* entry points below are
   therefore named mccl* so LD_PRELOAD can interpose the real calls that
   libtorch_cuda.so makes.  The forwarded "real" function is looked up by the
   matching mccl* name.  On stock NVIDIA the same file exports nccl* names.
   RETRIEVE_REAL_FUNC(exported_type, real_symbol) resolves the real symbol. */
#ifdef GREYHOUND_DETECTION_ONLY
    #define REAL_SYM(nccl_name, mccl_name) mccl_name
#else
    #define REAL_SYM(nccl_name, mccl_name) nccl_name
#endif

// Resolve the real function pointer (typed like `sig_func`) from the loaded
// collective library by symbol name `real_name`.
#define RETRIEVE_REAL_FUNC(sig_func, real_name)\
    using func_t = typeof(sig_func);\
    auto real_func = reinterpret_cast<func_t*>(g_status.get_function_ptr(real_name));

// Backward-compatible helper: symbol name == function name.
#define RETRIEVE_NCCL_FUNC(func_name)\
    using func_t = typeof(func_name);\
    auto real_func = reinterpret_cast<func_t*>(g_status.get_function_ptr(#func_name));

static bool sys_inited = false;
static GlobalStatus g_status;

using namespace std::chrono;


static void detect_sys_init()
{
    if (sys_inited)
        return;
    const char* dbg = getenv("GREYHOUND_DEBUG");
    if (dbg) fprintf(stderr, "[greyhound] detect_sys_init: initializing probe (pid=%d rank=%s)\n",
                     getpid(), getenv("RANK") ? getenv("RANK") : "?");
    g_status.initialize(get_nccl_path("NCCL_PATH"));
    sys_inited = true;
    if (dbg) fprintf(stderr, "[greyhound] detect_sys_init: done\n");
}

static void get_linked_peers(ncclComm_t comm)
{
    std::shared_ptr<Communicator> parsed_comm = g_status.topo_buffer->find(reinterpret_cast<uint64_t>(comm));
    auto linked_comms = g_status.topo_buffer->find_linked_ranks(parsed_comm);
    std::stringstream ss;
    ss << "My global rank: " << parsed_comm->global_rank << ", group rank: " << parsed_comm->group_rank << "; linked comms: ";
    for (auto& i : linked_comms){
        ss << "(global: " << i->global_rank << ", group: " << i->group_rank << "), ";
    }
    BOOST_LOG_TRIVIAL(info) << ss.str();
}

static ncclResult_t log_communicator(std::shared_ptr<NcclTopoConnection> comm_cache,
                                     ncclComm_t hidden_comm, uint64_t comm_nccl_id_hash)
{
    Communicator parsed_comm;
    parse_communicator(hidden_comm, &parsed_comm);
    parsed_comm.id_hash = comm_nccl_id_hash;
    parsed_comm.debug_print();
    comm_cache->add_comm(parsed_comm);
    g_status.local_comms.push_back(parsed_comm);
    // For the first time we create a communicator with WORLD_SIZE, we save it for pausing.
    if (!(g_status.event_handler->has_world_comm())
        && (parsed_comm.num_devices == get_world_size(DistEngine::auto_find)) )
        g_status.event_handler->set_world_comm(hidden_comm);
    return ncclSuccess;
}


ncclResult_t log_event(const void* buff1, const void* buff2, size_t count,
                       ncclDataType_t datatype, ncclComm_t comm,
                       cudaStream_t stream, NcclNumber number, uint64_t aux)
{
    int dev_id = -1, numdevs = 0;
    char pcistr[PCI_STR_LEN] = {0};
    auto call_time = 0.0, call_duration = 0.0;

    if (datatype == ncclInt8 && count == CHECK_PAUSE_MAGIC)
        g_status.should_check = true;

    // skip operations with very small size (<1K)
    if (RECORD_TOO_SMALL(count))
        return ncclSuccess;

    /* Special Note! For tensor parallelism (TP), there are too many alternative
    ALLGATHER and REDUCE_SCATTER calls, with each of them has a small size, the
    interval between these calls are very short, and recording all of them will
    influence the performance significantly. The pattern is like (1 * AllReduce,
    n * (AllGather | ReduceScatter), 1 * AllReduce, ...). So we compress these
    AllGather | ReduceScatter to a single record. */
    auto can_compress = [=](NcclNumber call_id) {
        return (call_id == NcclNumber::ALL_GATHER || call_id == NcclNumber::REDUCE_SCATTER);
    };

    if (can_compress(number))
    {
        // If this call is AllGather or ReduceScatter (special operators in TP)
        g_status.update_accumulation(number, count, comm);
        return ncclSuccess;
    }

    call_time = g_status.time_since_initialize();
    cudaGetDevice(&dev_id);
    cudaDeviceGetPCIBusId(pcistr, PCI_STR_LEN, dev_id);

    if (can_compress(g_status.last_call_id) && (!can_compress(number)))
    {
        // the previous call is, but the current is not
        // we should first add this compressed record to the buffer
        Record compressed_record(
            (uint64_t)g_status.last_comm, g_status.repeated_call_num, g_status.accumulated_count,
            reinterpret_cast<uint64_t>(buff1), reinterpret_cast<uint64_t>(buff2),
            (uint64_t)(datatype), (uint64_t)(getpid()), (uint64_t)(call_time),
            (uint64_t)(dev_id), (uint64_t)(get_rank(DistEngine::auto_find)), aux,
            (uint64_t)g_status.accumulated_duration, (uint64_t)(numdevs)
        );

        if (g_status.in_group)  // If we are in a group, just push it into temp buffer, it will be logged after group end.
            g_status.tmp_record_buffer.push_back(compressed_record);
        else  // If not in group, write it down to buffer immediately.
            g_status.storage_buffer->addRecord(compressed_record.toVector());
        g_status.reset_accumulation(number);
    }

    // If this Op is not in group, we should record its execution time here.
    if (!g_status.in_group)
        call_duration = g_status.get_communication_time();

    Record record(
        (uint64_t)comm, (uint64_t)number, count, reinterpret_cast<uint64_t>(buff1),
        reinterpret_cast<uint64_t>(buff2), (uint64_t)(datatype),
        (uint64_t)(getpid()), (uint64_t)(call_time), (uint64_t)(dev_id),
        (uint64_t)(get_rank(DistEngine::auto_find)), aux, (uint64_t)(call_duration),
        (uint64_t)(numdevs)
    );
    if (g_status.in_group)  // If we are in a group, just push it into temp buffer, it will be logged after group end.
        g_status.tmp_record_buffer.push_back(record);
    else  // If not in group, write it down to buffer immediately.
        g_status.storage_buffer->addRecord(record.toVector());

    return ncclSuccess;
}


/* ==========================================================================
 * Intercepted collective / comm entry points.
 *
 * Exported symbol names are chosen by GREYHOUND_INTERCEPT (mccl* on MetaX,
 * nccl* on stock NVIDIA).  The type aliases (ncclComm_t, ncclResult_t,
 * cudaStream_t, ...) are identical between the two ABIs, so a single body
 * serves both; only the *name* under which we export and the real symbol we
 * dlsym differ.
 * ========================================================================== */

#ifdef GREYHOUND_DETECTION_ONLY
    #define PROBE_FN(nccl_name, mccl_name) mccl_name
#else
    #define PROBE_FN(nccl_name, mccl_name) nccl_name
#endif


extern "C" ncclResult_t PROBE_FN(ncclCommInitRank, mcclCommInitRank)(
    ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
{
    // initialize the detection system if it is not yet inited.
    detect_sys_init();

    // Logging & forward the function call
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclCommInitRank, mcclCommInitRank), REAL_SYM("ncclCommInitRank", "mcclCommInitRank"));

    auto ret = (*real_func)(comm, nranks, commId, rank);
    if (g_status.in_group)
    {
        g_status.comm_in_group = *comm;
        g_status.comm_nccl_id_hash = hash_nccl_id(commId.internal, NCCL_UNIQUE_ID_BYTES);
    } else {
        log_communicator(g_status.topo_buffer, *comm, hash_nccl_id(commId.internal, NCCL_UNIQUE_ID_BYTES));
    }

    BOOST_LOG_TRIVIAL(info) << "[ncclCommInitRank] nranks=" << nranks << ", rank=" << rank;
    return ret;
}


extern "C" ncclResult_t PROBE_FN(ncclCommInitRankConfig, mcclCommInitRankConfig)(
    ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config)
{
    // initialize the detection system if it is not yet inited.
    detect_sys_init();

    // Logging & forward the function call
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclCommInitRankConfig, mcclCommInitRankConfig), REAL_SYM("ncclCommInitRankConfig", "mcclCommInitRankConfig"));

    auto ret = (*real_func)(comm, nranks, commId, rank, config);
    if (g_status.in_group)
    {
        g_status.comm_in_group = *comm;
        g_status.comm_nccl_id_hash = hash_nccl_id(commId.internal, NCCL_UNIQUE_ID_BYTES);
    } else {
        log_communicator(g_status.topo_buffer, *comm, hash_nccl_id(commId.internal, NCCL_UNIQUE_ID_BYTES));
    }
    BOOST_LOG_TRIVIAL(info) << "[ncclCommInitRankConfig] nranks=" << nranks << ", rank=" << rank;
    return ret;
}

/* Note: NCCL's ncclCommSplit does not exist in libmccl (MetaX torch does not
   use it), so it is only intercepted on the stock-NVIDIA build. */
#ifndef GREYHOUND_DETECTION_ONLY
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config)
{
    RETRIEVE_NCCL_FUNC(ncclCommSplit);

    auto ret = (*real_func)(comm, color, key, newcomm, config);
    // The original `comm' was already added to the buffer, so we only need to log the new comm.
    char buf[128];
    if (g_status.in_group)
    {
        g_status.comm_in_group = *newcomm;
        g_status.comm_nccl_id_hash = hash_nccl_id(buf, NCCL_UNIQUE_ID_BYTES);
    } else {
        log_communicator(g_status.topo_buffer, *newcomm, hash_nccl_id(buf, NCCL_UNIQUE_ID_BYTES));
    }
    BOOST_LOG_TRIVIAL(warning) << "[ncclCommSplit] getting ncclUniqueID is not supported via split";
    BOOST_LOG_TRIVIAL(info) << "[ncclCommSplit] color=" << color << ", key=" << key;
    return ret;
}
#endif


/* --- Send / Recv (MetaX torch uses both plain and *Ext variants) --- */

static inline ncclResult_t probe_send(
    const char* real_name,
    const void* sendbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    using send_t = ncclResult_t(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    auto real_func = reinterpret_cast<send_t*>(g_status.get_function_ptr(real_name));
    g_status.add_timing_event(NcclNumber::SEND, count, stream);
    auto ret = (*real_func)(sendbuff, count, datatype, peer, comm, stream);
    log_event(sendbuff, nullptr, count, datatype, comm, stream, NcclNumber::SEND, (uint64_t)peer);
    return ret;
}

static inline ncclResult_t probe_recv(
    const char* real_name,
    void* recvbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    using recv_t = ncclResult_t(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    auto real_func = reinterpret_cast<recv_t*>(g_status.get_function_ptr(real_name));
    g_status.add_timing_event(NcclNumber::RECV, count, stream);
    auto ret = (*real_func)(recvbuff, count, datatype, peer, comm, stream);
    log_event(nullptr, recvbuff, count, datatype, comm, stream, NcclNumber::RECV, (uint64_t)peer);
    return ret;
}

extern "C" ncclResult_t PROBE_FN(ncclSend, mcclSend)(
    const void* sendbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    return probe_send(REAL_SYM("ncclSend", "mcclSend"), sendbuff, count, datatype, peer, comm, stream);
}

extern "C" ncclResult_t PROBE_FN(ncclRecv, mcclRecv)(
    void* recvbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    return probe_recv(REAL_SYM("ncclRecv", "mcclRecv"), recvbuff, count, datatype, peer, comm, stream);
}

#ifdef GREYHOUND_DETECTION_ONLY
extern "C" ncclResult_t mcclSendExt(
    const void* sendbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    return probe_send("mcclSendExt", sendbuff, count, datatype, peer, comm, stream);
}

extern "C" ncclResult_t mcclRecvExt(
    void* recvbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclComm_t comm, cudaStream_t stream)
{
    return probe_recv("mcclRecvExt", recvbuff, count, datatype, peer, comm, stream);
}
#endif


/* --- Bcast / Broadcast --- */

extern "C" ncclResult_t PROBE_FN(ncclBcast, mcclBcast)(
    void* buff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream)
{
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclBcast, mcclBcast), REAL_SYM("ncclBcast", "mcclBcast"));

    g_status.add_timing_event(NcclNumber::BCAST, count, stream);
    auto ret = (*real_func)(buff, count, datatype, root, comm, stream);
    log_event(buff, nullptr, count, datatype, comm, stream, NcclNumber::BCAST, (uint64_t)root);
    return ret;
}


extern "C" ncclResult_t PROBE_FN(ncclBroadcast, mcclBroadcast)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclBroadcast, mcclBroadcast), REAL_SYM("ncclBroadcast", "mcclBroadcast"));

    g_status.add_timing_event(NcclNumber::BROADCAST, count, stream);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, root, comm, stream);
    log_event(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::BROADCAST, (uint64_t)root);
    return ret;
}


/* --- AllGather (+ Ext) --- */

static inline ncclResult_t probe_allgather(
    const char* real_name,
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
    using ag_t = ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);
    auto real_func = reinterpret_cast<ag_t*>(g_status.get_function_ptr(real_name));
    g_status.add_timing_event(NcclNumber::ALL_GATHER, count, stream);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, comm, stream);
    log_event(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::ALL_GATHER, (uint64_t)0);
    return ret;
}

extern "C" ncclResult_t PROBE_FN(ncclAllGather, mcclAllGather)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
    return probe_allgather(REAL_SYM("ncclAllGather", "mcclAllGather"), sendbuff, recvbuff, count, datatype, comm, stream);
}

#ifdef GREYHOUND_DETECTION_ONLY
extern "C" ncclResult_t mcclAllGatherExt(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
    return probe_allgather("mcclAllGatherExt", sendbuff, recvbuff, count, datatype, comm, stream);
}
#endif


/* --- ReduceScatter (+ Ext) --- */

static inline ncclResult_t probe_reducescatter(
    const char* real_name,
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    using rs_t = ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
    auto real_func = reinterpret_cast<rs_t*>(g_status.get_function_ptr(real_name));
    g_status.add_timing_event(NcclNumber::REDUCE_SCATTER, count, stream);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, op, comm, stream);
    log_event(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::REDUCE_SCATTER, (uint64_t)op);
    return ret;
}

extern "C" ncclResult_t PROBE_FN(ncclReduceScatter, mcclReduceScatter)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return probe_reducescatter(REAL_SYM("ncclReduceScatter", "mcclReduceScatter"), sendbuff, recvbuff, count, datatype, op, comm, stream);
}

#ifdef GREYHOUND_DETECTION_ONLY
extern "C" ncclResult_t mcclReduceScatterExt(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return probe_reducescatter("mcclReduceScatterExt", sendbuff, recvbuff, count, datatype, op, comm, stream);
}
#endif


/* --- AllReduce (+ Ext) --- */

static inline ncclResult_t probe_allreduce(
    const char* real_name,
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    using ar_t = ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
    auto real_func = reinterpret_cast<ar_t*>(g_status.get_function_ptr(real_name));
    if (getenv("GREYHOUND_DEBUG")) {
        static int nfirst = 0;
        if (nfirst < 3) { fprintf(stderr, "[greyhound] %s count=%zu real=%p\n", real_name, count, (void*)real_func); nfirst++; }
    }
    g_status.add_timing_event(NcclNumber::ALL_REDUCE, count, stream);
    auto ret = (*real_func)(sendbuff, recvbuff, count, datatype, op, comm, stream);
    log_event(sendbuff, recvbuff, count, datatype, comm, stream, NcclNumber::ALL_REDUCE, (uint64_t)op);
    return ret;
}

extern "C" ncclResult_t PROBE_FN(ncclAllReduce, mcclAllReduce)(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return probe_allreduce(REAL_SYM("ncclAllReduce", "mcclAllReduce"), sendbuff, recvbuff, count, datatype, op, comm, stream);
}

#ifdef GREYHOUND_DETECTION_ONLY
extern "C" ncclResult_t mcclAllReduceExt(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
{
    return probe_allreduce("mcclAllReduceExt", sendbuff, recvbuff, count, datatype, op, comm, stream);
}
#endif


/* --- Group semantics --- */

extern "C" ncclResult_t PROBE_FN(ncclGroupStart, mcclGroupStart)()
{
    detect_sys_init();
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclGroupStart, mcclGroupStart), REAL_SYM("ncclGroupStart", "mcclGroupStart"));

    // When a new group starts, we reset its events to empty and mark we are in a group.
    g_status.group_start();
    return (*real_func)();
}


extern "C" ncclResult_t PROBE_FN(ncclGroupEnd, mcclGroupEnd)()
{
    RETRIEVE_REAL_FUNC(PROBE_FN(ncclGroupEnd, mcclGroupEnd), REAL_SYM("ncclGroupEnd", "mcclGroupEnd"));

    auto ret = (*real_func)();

    if (g_status.transparent)
        return ret;

    double t = g_status.get_communication_time();

    if (g_status.state == ControlState::STATE_MONITOR)
    {
        for (auto& rec: g_status.tmp_record_buffer)
            g_status.storage_buffer->addRecord(rec.toVector());
    }
    else if (!FLOAT_EQ(t, 0.0))
    {
        // If this Op can be compressed, just do add it to accumulation
        if (g_status.event_op == NcclNumber::ALL_GATHER || g_status.event_op == NcclNumber::REDUCE_SCATTER)
        {
            g_status.accumulated_duration += t;
            return ret;
        }
        else {
            for (auto& rec: g_status.tmp_record_buffer)
            {
                if (rec.duration == 0)
                    rec.duration = (uint64_t)t;
                g_status.storage_buffer->addRecord(rec.toVector());
            }
        }
        // BOOST_LOG_TRIVIAL(info) << "Op: " << ToString(g_status.event_op) << ", time: " <<  t << "us";
    }
    g_status.tmp_record_buffer.clear();


    // Check whether to change state / do validation
    if (g_status.should_check) {
        g_status.transparent = true;
        g_status.event_handler->handle_control_signal(g_status.curr_stream, &g_status.state);
        g_status.transparent = false;
        g_status.should_check = false;
    }

    if (g_status.comm_in_group != nullptr)
    {
        log_communicator(g_status.topo_buffer, g_status.comm_in_group, g_status.comm_nccl_id_hash);
        g_status.comm_in_group = nullptr;
        g_status.comm_nccl_id_hash = 0;
    }
    g_status.group_end();
    return ret;
}
