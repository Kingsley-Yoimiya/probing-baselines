/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MCCL_H_
#define MCCL_H_

#define ENABLE_HIP_TO_MACA_ADAPTOR 1
#include <mcr/mc_runtime.h>
// #include <mcr/mc_runtime_api.h>
// #include <common/maca_fp16.h>

#define MCCL_MAJOR 2
#define MCCL_MINOR 16
#define MCCL_PATCH 5
#define MCCL_SUFFIX ""

#define MCCL_VERSION_CODE 21605
#define MCCL_VERSION(X, Y, Z)                                                                      \
    (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#define MCCL_BFLOAT16        1
#define MCCL_GATHER_SCATTER  1
#define MCCL_ALLTOALLV       1
#define MCCL_MULTIRANKPERGPU 1

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t mcclTag;
/*! @brief Opaque handle to communicator */
typedef struct mcclComm *mcclComm_t;

#define MCCL_UNIQUE_ID_BYTES 128
typedef struct {
    char internal[MCCL_UNIQUE_ID_BYTES];
} mcclUniqueId;

/*! @brief Error type */
typedef enum {
    mcclSuccess            = 0,
    mcclUnhandledMacaError = 1,
    mcclSystemError        = 2,
    mcclInternalError      = 3,
    mcclInvalidArgument    = 4,
    mcclInvalidUsage       = 5,
    mcclRemoteError        = 6,
    mcclInProgress         = 7,
    mcclNumResults         = 8
} mcclResult_t;

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
typedef struct mcclConfig_v21400 {
    /* attributes that users should never touch. */
    size_t size;
    unsigned int magic;
    unsigned int version;
    /* attributes that users are able to customize. */
    int blocking;
} mcclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in MCCL error. */
#define MCCL_CONFIG_INITIALIZER                                                                    \
    {                                                                                              \
        sizeof(mcclConfig_t),                                 /* size */                           \
            0xcafebeef,                                       /* magic */                          \
            MCCL_VERSION(MCCL_MAJOR, MCCL_MINOR, MCCL_PATCH), /* version */                        \
            1                                                 /* blocking */                       \
    }

/*! @brief Return the MCCL_VERSION_CODE of the MCCL library in the supplied integer.
 *
 * @details This integer is coded with the MAJOR, MINOR and PATCH level of the
 * MCCL library
 */
mcclResult_t mcclGetVersion(int *version);
/// @cond include_hidden
mcclResult_t pmcclGetVersion(int *version);
/// @endcond

/*! @brief Generates an ID for mcclCommInitRank

    @details
    Generates an ID to be used in mcclCommInitRank. mcclGetUniqueId should be
    called once and the Id should be distributed to all ranks in the
    communicator before calling mcclCommInitRank.

    @param[in]
    uniqueId     mcclUniqueId*
                 pointer to uniqueId

*/
mcclResult_t mcclGetUniqueId(mcclUniqueId *uniqueId);
/// @cond include_hidden
mcclResult_t pmcclGetUniqueId(mcclUniqueId *uniqueId);
/// @endcond

/*! @brief Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
mcclResult_t mcclCommInitRankConfig(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank,
                                    mcclConfig_t *config);
/// @cond include_hidden
mcclResult_t pmcclCommInitRankConfig(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank,
                                     mcclConfig_t *config);
/// @endcond

/*! @brief Creates a new communicator (multi thread/process version).

    @details
    rank must be between 0 and nranks-1 and unique within a communicator clique.
    Each rank is associated to a device, which has to be set before calling
    mcclCommInitRank.
    mcclCommInitRank implicitly syncronizes with other ranks, so it must be
    called by different threads/processes or use mcclGroupStart/mcclGroupEnd.

    @param[in]
    comm        mcclComm_t*
                communicator struct pointer
    */
mcclResult_t mcclCommInitRank(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank);
/// @cond include_hidden
mcclResult_t pmcclCommInitRank(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank);
/// @endcond

/*! @brief Creates a new communicator (multi thread/process version) allowing multiple ranks per
   device.

    @details
    rank must be between 0 and nranks-1 and unique within a communicator clique.
    Each rank is associated to a MACA device, which has to be set before calling
    mcclCommInitRankMulti.
    Since this version of the function allows multiple ranks to utilize the same
    MACA device, a unique virtualId per device has to be provided by each calling
    rank.
    mcclCommInitRankMulti implicitly syncronizes with other ranks, so it must be
    called by different threads/processes or use mcclGroupStart/mcclGroupEnd.

    @param[in]
    comm        mcclComm_t*
                communicator struct pointer
    */
mcclResult_t mcclCommInitRankMulti(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank,
                                   int virtualId);
/// @cond include_hidden
mcclResult_t pmcclCommInitRankMulti(mcclComm_t *comm, int nranks, mcclUniqueId commId, int rank,
                                    int virtualId);
/// @endcond

/*! @brief Creates a clique of communicators (single process version).
 *
 * @details This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(mcclComm_t).
 * If devlist is NULL, the first ndev MACA devices are used.
 * Order of devlist defines user-order of processors within the communicator.
 * */
mcclResult_t mcclCommInitAll(mcclComm_t *comm, int ndev, const int *devlist);
/// @cond include_hidden
mcclResult_t pmcclCommInitAll(mcclComm_t *comm, int ndev, const int *devlist);
/// @endcond

/*! @brief Finalize a communicator.
 * @details mcclCommFinalize flushes all issued communications,
 * and marks communicator state as mcclInProgress. The state will change to mcclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling mcclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking. */
mcclResult_t mcclCommFinalize(mcclComm_t comm);
/// @cond include_hidden
mcclResult_t pmcclCommFinalize(mcclComm_t comm);
/// @endcond

/*! @brief Frees local resources associated with communicator object. */

mcclResult_t mcclCommDestroy(mcclComm_t comm);
/// @cond include_hidden
mcclResult_t pmcclCommDestroy(mcclComm_t comm);
/// @endcond

/*! @brief Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
mcclResult_t mcclCommAbort(mcclComm_t comm);
/// @cond include_hidden
mcclResult_t pmcclCommAbort(mcclComm_t comm);
/// @endcond

/*! @brief Returns a string for each error code. */
const char *mcclGetErrorString(mcclResult_t result);
/// @cond include_hidden
const char *pmcclGetErrorString(mcclResult_t result);
/// @endcond

/*! @brief Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
const char *mcclGetLastError(mcclComm_t comm);
/// @cond include_hidden
const char *pmcclGetLastError(mcclComm_t comm);
/// @endcond

/* Checks whether the comm has encountered any asynchronous errors */
mcclResult_t mcclCommGetAsyncError(mcclComm_t comm, mcclResult_t *asyncError);
/// @cond include_hidden
mcclResult_t pmcclCommGetAsyncError(mcclComm_t comm, mcclResult_t *asyncError);
/// @endcond

/*! @brief Gets the number of ranks in the communicator clique. */
mcclResult_t mcclCommCount(const mcclComm_t comm, int *count);
/// @cond include_hidden
mcclResult_t pmcclCommCount(const mcclComm_t comm, int *count);
/// @endcond

/*! @brief Returns the maca device number associated with the communicator. */
mcclResult_t mcclCommMcDevice(const mcclComm_t comm, int *device);
/// @cond include_hidden
mcclResult_t pmcclCommMcDevice(const mcclComm_t comm, int *device);
/// @endcond

/*! @brief Returns the user-ordered "rank" associated with the communicator. */
mcclResult_t mcclCommUserRank(const mcclComm_t comm, int *rank);
/// @cond include_hidden
mcclResult_t pmcclCommUserRank(const mcclComm_t comm, int *rank);
/// @endcond

/*! @brief Reduction operation selector */
/* Reduction operation selector */
typedef enum { mcclNumOps_dummy = 5 } mcclRedOp_dummy_t;
typedef enum {
    mcclSum  = 0,
    mcclProd = 1,
    mcclMax  = 2,
    mcclMin  = 3,
    mcclAvg  = 4,
    /* mcclNumOps: The number of built-in mcclRedOp_t values. Also
     * serves as the least possible value for dynamic mcclRedOp_t's
     * as constructed by mcclRedOpCreate*** functions. */
    mcclNumOps = 5,
    /* mcclMaxRedOp: The largest valid value for mcclRedOp_t.
     * It is defined to be the largest signed value (since compilers
     * are permitted to use signed enums) that won't grow
     * sizeof(mcclRedOp_t) when compared to previous MCCL versions to
     * maintain ABI compatibility. */
    mcclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(mcclRedOp_dummy_t))
} mcclRedOp_t;

/*! @brief Data types */
typedef enum {
    mcclInt8     = 0,
    mcclChar     = 0,
    mcclUint8    = 1,
    mcclInt32    = 2,
    mcclInt      = 2,
    mcclUint32   = 3,
    mcclInt64    = 4,
    mcclUint64   = 5,
    mcclFloat16  = 6,
    mcclHalf     = 6,
    mcclFloat32  = 7,
    mcclFloat    = 7,
    mcclFloat64  = 8,
    mcclDouble   = 8,
    mcclBfloat16 = 9,
    mcclNumTypes = 10
} mcclDataType_t;

/*! @brief mcclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
typedef enum {
    /* mcclScalarDevice: The scalar is in device-visible memory and will be
     * dereferenced while the collective is running. */
    mcclScalarDevice = 0,

    /* mcclScalarHostImmediate: The scalar is in host-visible memory and will be
     * dereferenced before the mcclRedOpCreate***() function returns. */
    mcclScalarHostImmediate = 1
} mcclScalarResidence_t;

/*! @brief mcclRedOpCreatePreMulSum
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
mcclResult_t mcclRedOpCreatePreMulSum(mcclRedOp_t *op, void *scalar, mcclDataType_t datatype,
                                      mcclScalarResidence_t residence, mcclComm_t comm);
/// @cond include_hidden
mcclResult_t pmcclRedOpCreatePreMulSum(mcclRedOp_t *op, void *scalar, mcclDataType_t datatype,
                                       mcclScalarResidence_t residence, mcclComm_t comm);
/// @endcond

/*! @brief mcclRedOpDestroy
 * @details Destroys the reduction operator *op*. The operator must have been created by
 * mcclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
 * destroyed as soon as the last MCCL function which is given that operator returns.
 */
mcclResult_t mcclRedOpDestroy(mcclRedOp_t op, mcclComm_t comm);
/// @cond include_hidden
mcclResult_t pmcclRedOpDestroy(mcclRedOp_t op, mcclComm_t comm);
/// @endcond

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*!
 * @brief Reduce
 *
 * @details Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclReduce(const void *sendbuff, void *recvbuff, size_t count, mcclDataType_t datatype,
                        mcclRedOp_t op, int root, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclReduce(const void *sendbuff, void *recvbuff, size_t count,
                         mcclDataType_t datatype, mcclRedOp_t op, int root, mcclComm_t comm,
                         mcStream_t stream);
/// @endcond

/*! @brief (deprecated) Broadcast (in-place)
 *
 * @details Copies count values from root to all other devices.
 * root is the rank (not the device) where data resides before the
 * operation is started.
 *
 * This operation is implicitely in place.
 */
mcclResult_t mcclBcast(void *buff, size_t count, mcclDataType_t datatype, int root, mcclComm_t comm,
                       mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclBcast(void *buff, size_t count, mcclDataType_t datatype, int root,
                        mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief Broadcast
 *
 * @details Copies count values from root to all other devices.
 * root is the rank (not the MACA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                            mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief All-Reduce
 *
 * @details Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
                           mcStream_t stream);
mcclResult_t mcclAllReduceExt(const void *sendbuff, void *recvbuff, size_t count,
                              mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
                              mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                            mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
                            mcStream_t stream);
/// @endcond

/*!
 * @brief Reduce-Scatter
 *
 * @details Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
mcclResult_t mcclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                               mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
                               mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                                mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
                                mcStream_t stream);
/// @endcond
mcclResult_t mcclReduceScatterExt(const void *sendbuff, void *recvbuff, size_t recvcount,
                                  mcclDataType_t datatype, mcclRedOp_t op, mcclComm *comm,
                                  mcStream_t stream);
/*! @brief All-Gather
 *
 * @details Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
mcclResult_t mcclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                            mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/// @endcond
mcclResult_t mcclAllGatherExt(const void *sendbuff, void *recvbuff, size_t sendcount,
                              mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/*! @brief Send
 *
 * @details Send data from sendbuff to rank peer.
 * Rank peer needs to call mcclRecv with the same datatype and the same count from this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mcclSend and mcclRecv operations
 * need to progress concurrently to complete, they must be fused within a mcclGroupStart/
 * mcclGroupEnd section.
 */
mcclResult_t mcclSend(const void *sendbuff, size_t count, mcclDataType_t datatype, int peer,
                      mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclSend(const void *sendbuff, size_t count, mcclDataType_t datatype, int peer,
                       mcclComm_t comm, mcStream_t stream);
/// @endcond

mcclResult_t mcclSendExt(const void *sendbuff, size_t count, mcclDataType_t datatype, int peer,
                         mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclSendExt(const void *sendbuff, size_t count, mcclDataType_t datatype, int peer,
                          mcclComm_t comm, mcStream_t stream);
/*! @brief Receive
 *
 * @details Receive data from rank peer into recvbuff.
 * Rank peer needs to call mcclSend with the same datatype and the same count to this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mcclSend and mcclRecv operations
 * need to progress concurrently to complete, they must be fused within a mcclGroupStart/
 * mcclGroupEnd section.
 */
mcclResult_t mcclRecv(void *recvbuff, size_t count, mcclDataType_t datatype, int peer,
                      mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclRecv(void *recvbuff, size_t count, mcclDataType_t datatype, int peer,
                       mcclComm_t comm, mcStream_t stream);
/// @endcond

mcclResult_t mcclRecvExt(void *recvbuff, size_t count, mcclDataType_t datatype, int peer,
                         mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclRecvExt(void *recvbuff, size_t count, mcclDataType_t datatype, int peer,
                          mcclComm_t comm, mcStream_t stream);

/*! @brief Gather
 *
 * @details Root device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 *
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
mcclResult_t mcclGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                        mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                         mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief Scatter
 *
 * @details Scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the data on root.
 *
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
mcclResult_t mcclScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                          mcclDataType_t datatype, int root, mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief All-To-All
 *
 * @details Device (i) send (j)th block of data to device (j) and be placed as (i)th
 * block. Each block for sending/receiving has count elements, which means
 * that recvbuff and sendbuff should have a size of nranks*count elements.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                          mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
mcclResult_t mcclAllToAllExt(const void *sendbuff, void *recvbuff, size_t count,
                             mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                           mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief All-To-Allv
 *
 * @details Device (i) sends sendcounts[j] of data from offset sdispls[j]
 * to device (j). In the same time, device (i) receives recvcounts[j] of data
 * from device (j) to be placed at rdispls[j].

 * sendcounts, sdispls, recvcounts and rdispls are all measured in the units
 * of datatype, not bytes.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
                           void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
                           mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
mcclResult_t mcclAllToAllvExt(const void *sendbuff, const size_t sendcounts[],
                              const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
                              const size_t rdispls[], mcclDataType_t datatype, mcclComm_t comm,
                              mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclAllToAllv(const void *sendbuff, const size_t sendcounts[], const size_t sdispls[],
                            void *recvbuff, const size_t recvcounts[], const size_t rdispls[],
                            mcclDataType_t datatype, mcclComm_t comm, mcStream_t stream);
/// @endcond

/*! @brief All-To-Alld
 *
 * @details Device (i) sends sendcounts[j] of data
 * to device (j). In the same time, device (i) receives recvcounts[j] of data
 * from device (j)

 * sendcounts, recvcounts are all measured in the units
 * of datatype, not bytes.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mcclResult_t mcclAllToAlld(const void *sendbuff[], const size_t sendcounts[], void *recvbuff[],
                           const size_t recvcounts[], mcclDataType_t datatype, mcclComm_t comm,
                           mcStream_t stream);
/// @cond include_hidden
mcclResult_t pmcclAllToAlld(const void *sendbuff[], const size_t sendcounts[], void *recvbuff[],
                            const size_t recvcounts[], mcclDataType_t datatype, mcclComm_t comm,
                            mcStream_t stream);
/// @endcond
mcclResult_t mcclAllToAlldExt(const void *sendbuff[], const size_t sendcounts[], void *recvbuff[],
                              const size_t recvcounts[], mcclDataType_t datatype, mcclComm_t comm,
                              mcStream_t stream);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since MCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping MCCL calls as being part of the same collective operation is done
 * using mcclGroupStart and mcclGroupEnd. mcclGroupStart will enqueue all
 * collective calls until the mcclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, mcclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and mcclCommInitRank can be used in conjunction
 * of mcclGroupStart/mcclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */

/*! @brief Group Start
 *
 * Start a group call. All calls to MCCL until mcclGroupEnd will be fused into
 * a single MCCL operation. Nothing will be started on the stream until
 * mcclGroupEnd.
 */
mcclResult_t mcclGroupStart();
/// @cond include_hidden
mcclResult_t pmcclGroupStart();
/// @endcond

/*! @brief Group End
 *
 * End a group call. Start a fused MCCL operation consisting of all calls since
 * mcclGroupStart. Operations on the stream depending on the MCCL operations
 * need to be called after mcclGroupEnd.
 */
mcclResult_t mcclGroupEnd();
/// @cond include_hidden
mcclResult_t pmcclGroupEnd();
/// @endcond

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
