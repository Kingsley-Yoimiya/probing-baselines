/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#define ENABLE_CUDA_TO_MACA_ADAPTOR

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// #if CUDART_VERSION >= 11000
// #include <cuda_bf16.h>
// #endif

#include "mccl.h"

#define NCCL_MAJOR MCCL_MAJOR
#define NCCL_MINOR MCCL_MINOR
#define NCCL_PATCH MCCL_PATCH
#define NCCL_SUFFIX MCCL_SUFFIX

#define NCCL_VERSION_CODE MCCL_VERSION_CODE
#define NCCL_VERSION MCCL_VERSION

typedef mcclComm_t ncclComm_t;
typedef mcclUniqueId ncclUniqueId;
#define NCCL_UNIQUE_ID_BYTES MCCL_UNIQUE_ID_BYTES

#define ncclSuccess             mcclSuccess
#define ncclUnhandledCudaError  mcclUnhandledMacaError
#define ncclSystemError         mcclSystemError
#define ncclInternalError       mcclInternalError
#define ncclInvalidArgument     mcclInvalidArgument
#define ncclInvalidUsage        mcclInvalidUsage
#define ncclRemoteError         mcclRemoteError
#define ncclInProgress          mcclInProgress
#define ncclNumResults          mcclNumResults

typedef mcclResult_t ncclResult_t;

#define ncclSum     mcclSum
#define ncclProd    mcclProd
#define ncclMax     mcclMax
#define ncclMin     mcclMin
#define ncclAvg     mcclAvg
#define ncclNumOps  mcclNumOps
typedef mcclRedOp_t ncclRedOp_t;

#define ncclInt8        mcclInt8
#define ncclChar        mcclChar
#define ncclUint8       mcclUint8
#define ncclInt32       mcclInt32
#define ncclInt         mcclInt
#define ncclUint32      mcclUint32
#define ncclInt64       mcclInt64
#define ncclUint64      mcclUint64
#define ncclFloat16     mcclFloat16
#define ncclHalf        mcclHalf
#define ncclFloat32     mcclFloat32
#define ncclFloat       mcclFloat
#define ncclFloat64     mcclFloat64
#define ncclDouble      mcclDouble
#define ncclBfloat16    mcclBfloat16
#define ncclNumTypes    mcclNumTypes

#define ncclScalarDevice        mcclScalarDevice
#define ncclScalarHostImmediate mcclScalarHostImmediate

#define NCCL_CONFIG_INITIALIZER MCCL_CONFIG_INITIALIZER

typedef mcclDataType_t ncclDataType_t;
typedef mcclScalarResidence_t ncclScalarResidence_t;
typedef mcclConfig_t ncclConfig_t;

static ncclResult_t  ncclGetVersion(int *version) {
    return mcclGetVersion(version);
}

static ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId) {
    return mcclGetUniqueId(uniqueId);
}

static ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
    return mcclCommInitRank(comm, nranks, commId, rank);
}

static ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
    return mcclCommInitAll(comm, ndev, devlist);
}

static ncclResult_t  ncclCommDestroy(ncclComm_t comm) {
    return mcclCommDestroy(comm);
}

static ncclResult_t  ncclCommAbort(ncclComm_t comm) {
    return mcclCommAbort(comm);
}

static const char*  ncclGetErrorString(ncclResult_t result) {
    return mcclGetErrorString(result);
}

static ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    return mcclCommGetAsyncError(comm, asyncError);
}

static ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count) {
    return mcclCommCount(comm, count);
}

static ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device) {
    return mcclCommMcDevice(comm, device);
}

static ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank) {
    return mcclCommUserRank(comm, rank);
}

static ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config) {
    return mcclCommInitRankConfig(comm, nranks, commId, rank, config);
}

static ncclResult_t  ncclCommInitRankMulti(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, int virtualId) {
    return mcclCommInitRankMulti(comm, nranks, commId, rank, virtualId);
}

static ncclResult_t  ncclCommFinalize(ncclComm_t comm) {
    return mcclCommFinalize(comm);
}

static ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, mcStream_t stream) {
    return mcclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

static ncclResult_t  ncclBcast(void* buff, size_t count, mcclDataType_t datatype, int root,
    ncclComm_t comm, mcStream_t stream) {
    return mcclBcast(buff, count, datatype, root, comm, stream);
}

static ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, mcStream_t stream) {
    return mcclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

static ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, mcStream_t stream) {
    return mcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

static ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    mcStream_t stream) {
    return mcclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

static ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, mcStream_t stream) {
    return mcclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

static ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, mcStream_t stream) {
    return mcclSend(sendbuff, count, datatype, peer, comm, stream);
}

static ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, mcStream_t stream) {
    return mcclRecv(recvbuff, count, datatype, peer, comm, stream);
}

static ncclResult_t ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, mcStream_t stream) {
    return mcclGather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream);
}

static ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, mcStream_t stream) {
    return mcclScatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream);
}

static ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, mcStream_t stream) {
    return mcclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

static ncclResult_t ncclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, mcStream_t stream) {
    return mcclAllToAllv(sendbuff, sendcounts, sdispls, recvbuff, recvcounts,
    rdispls, datatype, comm, stream);
}

static ncclResult_t ncclAllToAlld(const void *sendbuff[], const size_t sendcounts[],
    void *recvbuff[], const size_t recvcounts[],
    ncclDataType_t datatype, ncclComm_t comm, mcStream_t stream) {
    return mcclAllToAlld(sendbuff, sendcounts, recvbuff, recvcounts,
    datatype, comm, stream);
}

static ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    return mcclRedOpDestroy(op, comm);
}

static ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm) {
    return mcclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

static const char*  ncclGetLastError(ncclComm_t comm) {
    return mcclGetLastError(comm);
}

static ncclResult_t  ncclGroupStart() {
    return mcclGroupStart();
}

static ncclResult_t  ncclGroupEnd() {
    return mcclGroupEnd();
}

#endif // end include guard
