/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MCCL_H_
#define MCCL_H_

#if (defined(__MC_PLATFORM_MXCC__) || defined(__MCC__) || defined(__MXCC__))
#define ENABLE_HIP_TO_MACA_ADAPTOR 1
// #include <common/maca_fp16.h>
#endif

#include <mcr/mc_runtime.h>

#define NCCL_MAJOR 2
#define NCCL_MINOR 16
#define NCCL_PATCH 5
#define NCCL_SUFFIX ""

#define MCCL_VERSION_CODE 21605
#define MCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#define MCCL_BFLOAT16 1
// #define MCCL_FLOAT8 1
#define MCCL_GATHER_SCATTER 1
#define MCCL_ALLTOALLV 1

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/*! @brief      Opaque handle to communicator
    @details    A communicator contains information required to facilitate collective communications calls */
typedef struct mxcclComm* mxcclComm_t;
#define MCCL_COMM_NULL NULL

#define MCCL_UNIQUE_ID_BYTES 128
/*! @brief      Opaque unique id used to initialize communicators
    @details    The mxcclUniqueId must be passed to all participating ranks */
typedef struct { char internal[MCCL_UNIQUE_ID_BYTES]; /*!< Opaque array>*/} mxcclUniqueId;

/*! @defgroup   mxccl_result_code Result Codes
    @details    The various result codes that MCCL API calls may return
    @{ */

/*! @brief      Result type
    @details    Return codes aside from mxcclSuccess indicate that a call has failed */
  typedef enum {
    mxcclSuccess                 =  0, /*!< No error */
    mxcclUnhandledMacaError      =  1, /*!< Unhandled MACA error */
    mxcclSystemError             =  2, /*!< Unhandled system error */
    mxcclInternalError           =  3, /*!< Internal Error - Please report to MCCL developers */
    mxcclInvalidArgument         =  4, /*!< Invalid argument */
    mxcclInvalidUsage            =  5, /*!< Invalid usage */
    mxcclRemoteError             =  6, /*!< Remote process exited or there was a network error */
    mxcclInProgress              =  7, /*!< MCCL operation in progress */
    mxcclNumResults              =  8  /*!< Number of result types */
  } mxcclResult_t;
/*! @} */

#define MCCL_CONFIG_UNDEF_INT INT_MIN
#define MCCL_CONFIG_UNDEF_PTR NULL
#define MCCL_SPLIT_NOCOLOR -1

/*! @defgroup   mxccl_config_type Communicator Configuration
    @details    Structure that allows for customizing Communicator behavior via mxcclCommInitRankConfig
    @{ */

/*! @brief      Communicator configuration
    @details    Users can assign value to attributes to specify the behavior of a communicator */
typedef struct mxcclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;                 /*!< Should not be touched */
  unsigned int magic;          /*!< Should not be touched */
  unsigned int version;        /*!< Should not be touched */
  /* attributes that users are able to customize. */
  int blocking;                /*!< Whether or not calls should block or not */
  int cgaClusterSize;          /*!< Cooperative group array cluster size */
  int minCTAs;                 /*!< Minimum number of cooperative thread arrays (blocks) */
  int maxCTAs;                 /*!< Maximum number of cooperative thread arrays (blocks) */
  const char *netName;         /*!< Force NCCL to use a specfic network */
  int splitShare;              /*!< Allow communicators to share resources */
} mxcclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in an error. */
#define MCCL_CONFIG_INITIALIZER {                                        \
  sizeof(mxcclConfig_t),                            /* size */           \
  0xcafebeef,                                       /* magic */          \
  MCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */        \
  MCCL_CONFIG_UNDEF_INT,                            /* blocking */       \
  MCCL_CONFIG_UNDEF_INT,                            /* cgaClusterSize */ \
  MCCL_CONFIG_UNDEF_INT,                            /* minCTAs */        \
  MCCL_CONFIG_UNDEF_INT,                            /* maxCTAs */        \
  MCCL_CONFIG_UNDEF_PTR,                            /* netName */        \
  MCCL_CONFIG_UNDEF_INT                             /* splitShare */     \
}
/*! @} */

/* NCCL malloc and free function for all types of NCCL optimizations
 * (e.g. user buffer registration). The actual allocated size might
 * be larger than requested due to granularity requirement. */
mxcclResult_t  mxcclMemAlloc(void** ptr, size_t size);
mxcclResult_t pmxcclMemAlloc(void** ptr, size_t size);

mxcclResult_t  mxcclMemFree(void *ptr);
mxcclResult_t pmxcclMemFree(void *ptr);

/*! @defgroup   mxccl_api_version Version Information
    @details    API call that returns MCCL version
    @{ */

/*! @brief      Return the MCCL_VERSION_CODE of MCCL in the supplied integer.
    @details    This integer is coded with the MAJOR, MINOR and PATCH level of MCCL.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] version       Pointer to where version will be stored */

mxcclResult_t  mxcclGetVersion(int *version);
/*! @cond       include_hidden */
mxcclResult_t pmxcclGetVersion(int *version);
/*! @endcond */
/*! @} */

/*! @defgroup   mxccl_api_communicator Communicator Initialization/Destruction
    @details    API calls that operate on communicators.
                Communicators objects are used to launch collective communication
                operations.  Unique ranks between 0 and N-1 must be assigned to
                each HIP device participating in the same Communicator.
                Using the same HIP device for multiple ranks of the same Communicator
                is not supported at this time.
    @{ */

/*! @brief      Generates an ID for mxcclCommInitRank.
    @details    Generates an ID to be used in mxcclCommInitRank.
                mxcclGetUniqueId should be called once by a single rank and the
                ID should be distributed to all ranks in the communicator before
                using it as a parameter for mxcclCommInitRank.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] uniqueId      Pointer to where uniqueId will be stored */
mxcclResult_t  mxcclGetUniqueId(mxcclUniqueId* uniqueId);
/*! @cond       include_hidden */
mxcclResult_t pmxcclGetUniqueId(mxcclUniqueId* uniqueId);
/*! @endcond */

/*! @brief      Create a new communicator with config.
    @details    Create a new communicator (multi thread/process version) with a configuration
                set by users. See @ref mxccl_config_type for more details.
                Each rank is associated to a CUDA device, which has to be set before calling
                mxcclCommInitRank.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] comm          Pointer to created communicator
    @param[in]  nranks        Total number of ranks participating in this communicator
    @param[in]  commId        UniqueId required for initialization
    @param[in]  rank          Current rank to create communicator for. [0 to nranks-1]
    @param[in]  config        Pointer to communicator configuration */
mxcclResult_t  mxcclCommInitRankConfig(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank, mxcclConfig_t* config);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommInitRankConfig(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank, mxcclConfig_t* config);
/*! @endcond */

/*! @brief      Creates a new communicator (multi thread/process version).
    @details    Rank must be between 0 and nranks-1 and unique within a communicator clique.
                Each rank is associated to a CUDA device, which has to be set before calling
                mxcclCommInitRank.  mxcclCommInitRank implicitly syncronizes with other ranks,
                so it must be called by different threads/processes or use mxcclGroupStart/mxcclGroupEnd.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] comm          Pointer to created communicator
    @param[in]  nranks        Total number of ranks participating in this communicator
    @param[in]  commId        UniqueId required for initialization
    @param[in]  rank          Current rank to create communicator for */
mxcclResult_t  mxcclCommInitRank(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommInitRank(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank);
/*! @endcond */

/*! @brief Creates a new communicator (multi thread/process version) allowing multiple ranks per device.

    @details
    rank must be between 0 and nranks-1 and unique within a communicator clique.
    Each rank is associated to a MACA device, which has to be set before calling
    mxcclCommInitRankMulti.
    Since this version of the function allows multiple ranks to utilize the same
    MACA device, a unique virtualId per device has to be provided by each calling
    rank.
    mxcclCommInitRankMulti implicitly syncronizes with other ranks, so it must be
    called by different threads/processes or use mxcclGroupStart/mxcclGroupEnd.

    @param[in]
    comm        mxcclComm_t*
                communicator struct pointer
    */
  mxcclResult_t  mxcclCommInitRankMulti(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank, int virtualId);
/// @cond include_hidden
  mxcclResult_t pmxcclCommInitRankMulti(mxcclComm_t* comm, int nranks, mxcclUniqueId commId, int rank, int virtualId);
/// @endcond

/*! @brief      Creates a clique of communicators (single process version).
    @details    This is a convenience function to create a single-process communicator clique.
                Returns an array of ndev newly initialized communicators in comm.
                comm should be pre-allocated with size at least ndev*sizeof(mxcclComm_t).
                If devlist is NULL, the first ndev HIP devices are used.
                Order of devlist defines user-order of processors within the communicator.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] comm          Pointer to array of created communicators
    @param[in]  ndev          Total number of ranks participating in this communicator
    @param[in]  devlist       Array of GPU device indices to create for */
mxcclResult_t  mxcclCommInitAll(mxcclComm_t* comm, int ndev, const int* devlist);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommInitAll(mxcclComm_t* comm, int ndev, const int* devlist);
/*! @endcond */

/*! @brief      Finalize a communicator.
    @details    mxcclCommFinalize flushes all issued communications
                and marks communicator state as mxcclInProgress. The state will change to mxcclSuccess
                when the communicator is globally quiescent and related resources are freed; then,
                calling mxcclCommDestroy can locally free the rest of the resources (e.g. communicator
                itself) without blocking.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to finalize */
mxcclResult_t  mxcclCommFinalize(mxcclComm_t comm);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommFinalize(mxcclComm_t comm);
/*! @endcond */

/*! @brief      Frees local resources associated with communicator object.
    @details    Destroy all local resources associated with the passed in communicator object
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to destroy */
mxcclResult_t  mxcclCommDestroy(mxcclComm_t comm);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommDestroy(mxcclComm_t comm);
/*! @endcond */

/*! @brief      Abort any in-progress calls and destroy the communicator object.
    @details    Frees resources associated with communicator object and aborts any operations
                that might still be running on the device.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to abort and destroy */
mxcclResult_t  mxcclCommAbort(mxcclComm_t comm);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommAbort(mxcclComm_t comm);
/*! @endcond */

/*! @brief      Create one or more communicators from an existing one.
    @details    Creates one or more communicators from an existing one.
                Ranks with the same color will end up in the same communicator.
                Within the new communicator, key will be used to order ranks.
                MCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
                and will therefore return a NULL communicator.
                If config is NULL, the new communicator will inherit the original communicator's configuration
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Original communicator object for this rank
    @param[in]  color         Color to assign this rank
    @param[in]  key           Key used to order ranks within the same new communicator
    @param[out] newcomm       Pointer to new communicator
    @param[in]  config        Config file for new communicator. May be NULL to inherit from comm */
mxcclResult_t  mxcclCommSplit(mxcclComm_t comm, int color, int key, mxcclComm_t *newcomm, mxcclConfig_t* config);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommSplit(mxcclComm_t comm, int color, int key, mxcclComm_t *newcomm, mxcclConfig_t* config);
/*! @endcond */
/*! @} */

/*! @defgroup   mxccl_api_errcheck Error Checking Calls
    @details    API calls that check for errors
    @{ */

/*! @brief      Returns a string for each result code.
    @details    Returns a human-readable string describing the given result code.
    @return     String containing description of result code.

    @param[in]  result        Result code to get description for */
const char*  mxcclGetErrorString(mxcclResult_t result);
/*! @cond       include_hidden */
const char* pmxcclGetErrorString(mxcclResult_t result);
/*! @endcond */

/* Returns a human-readable message of the last error that occurred. */
const char*  mxcclGetLastError(mxcclComm_t comm);
/*! @cond       include_hidden */
const char* pmxcclGetLastError(mxcclComm_t comm);
/*! @endcond */

/*! @brief      Checks whether the comm has encountered any asynchronous errors
    @details    Query whether the provided communicator has encountered any asynchronous errors
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] asyncError    Pointer to where result code will be stored */
mxcclResult_t  mxcclCommGetAsyncError(mxcclComm_t comm, mxcclResult_t *asyncError);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommGetAsyncError(mxcclComm_t comm, mxcclResult_t *asyncError);
/*! @endcond */
/*! @} */

/*! @defgroup   mxccl_api_comminfo Communicator Information
    @details    API calls that query communicator information
    @{ */

/*! @brief      Gets the number of ranks in the communicator clique.
    @details    Returns the number of ranks in the communicator clique (as set during initialization)
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] count         Pointer to where number of ranks will be stored */
mxcclResult_t  mxcclCommCount(const mxcclComm_t comm, int* count);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommCount(const mxcclComm_t comm, int* count);
/*~ @endcond */

/*! @brief      Get the ROCm device index associated with a communicator
    @details    Returns the ROCm device number associated with the provided communicator.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] device        Pointer to where the associated ROCm device index will be stored */
mxcclResult_t  mxcclCommMcDevice(const mxcclComm_t comm, int* device);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommMcDevice(const mxcclComm_t comm, int* device);
/*! @endcond */

/*! @brief      Get the rank associated with a communicator
    @details    Returns the user-ordered "rank" associated with the provided communicator.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] rank          Pointer to where the associated rank will be stored */
mxcclResult_t  mxcclCommUserRank(const mxcclComm_t comm, int* rank);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommUserRank(const mxcclComm_t comm, int* rank);
/*! @endcond */
/*! @} */

/* Register CUDA buffer for zero-copy operation */
mxcclResult_t  mxcclCommRegister(const mxcclComm_t comm, void* buff, size_t size, void** handle);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommRegister(const mxcclComm_t comm, void* buff, size_t size, void** handle);
/*! @endcond */

/* Deregister CUDA buffer */
mxcclResult_t  mxcclCommDeregister(const mxcclComm_t comm, void* handle);
/*! @cond       include_hidden */
mxcclResult_t pmxcclCommDeregister(const mxcclComm_t comm, void* handle);
/*! @endcond */

/*! @defgroup   mxccl_api_enumerations API Enumerations
    @details    Enumerations used by collective communication calls
    @{ */

/*! @brief      Dummy reduction enumeration
    @details    Dummy reduction enumeration used to determine value for mxcclMaxRedOp */
typedef enum { mxcclNumOps_dummy = 5 } mxcclRedOp_dummy_t;

/*! @brief      Reduction operation selector
    @details    Enumeration used to specify the various reduction operations
                mxcclNumOps is the number of built-in mxcclRedOp_t values and serves as
                the least possible value for dynamic mxcclRedOp_t values constructed by
                mxcclRedOpCreate functions.

                mxcclMaxRedOp is the largest valid value for mxcclRedOp_t and is defined
                to be the largest signed value (since compilers are permitted to use
                signed enums) that won't grow sizeof(mxcclRedOp_t) when compared to previous
                MCCL versions to maintain ABI compatibility. */
typedef enum { mxcclSum        = 0, /*!< Sum */
               mxcclProd       = 1, /*!< Product */
               mxcclMax        = 2, /*!< Max */
               mxcclMin        = 3, /*!< Min */
               mxcclAvg        = 4, /*!< Average */
               mxcclNumOps     = 5, /*!< Number of built-in reduction ops */
               mxcclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(mxcclRedOp_dummy_t)) /*!< Largest value for mxcclRedOp_t */
             } mxcclRedOp_t;

/*! @brief      Data types
    @details    Enumeration of the various supported datatype */
typedef enum { mxcclInt8       = 0, mxcclChar       = 0,
               mxcclUint8      = 1,
               mxcclInt32      = 2, mxcclInt        = 2,
               mxcclUint32     = 3,
               mxcclInt64      = 4,
               mxcclUint64     = 5,
               mxcclFloat16    = 6, mxcclHalf       = 6,
               mxcclFloat32    = 7, mxcclFloat      = 7,
               mxcclFloat64    = 8, mxcclDouble     = 8,
               mxcclBfloat16   = 9,
#if defined(MCCL_FLOAT8)
               mxcclFp8E4M3    = 10,
               mxcclFp8E5M2    = 11,
               mxcclNumTypes   = 12 } mxcclDataType_t;
#else
               mxcclNumTypes   = 10 } mxcclDataType_t;
#endif
/*! @} */

/*! @defgroup   mxccl_api_custom_redop Custom Reduction Operator
    @details    API calls relating to creation/destroying custom reduction operator
                that pre-multiplies local source arrays prior to reduction
    @{ */

/*! @brief      Location and dereferencing logic for scalar arguments.
    @details    Enumeration specifying memory location of the scalar argument.
                Based on where the value is stored, the argument will be dereferenced either
                while the collective is running (if in device memory), or before the mxcclRedOpCreate()
                function returns (if in host memory). */
typedef enum {
  mxcclScalarDevice        = 0, /*!< Scalar is in device-visible memory */
  mxcclScalarHostImmediate = 1  /*!< Scalar is in host-visible memory */
} mxcclScalarResidence_t;

/*! @brief      Create a custom pre-multiplier reduction operator
    @details    Creates a new reduction operator which pre-multiplies input values by a given
                scalar locally before reducing them with peer values via summation. For use
                only with collectives launched against *comm* and *datatype*. The
                *residence* argument indicates how/when the memory pointed to by *scalar*
                will be dereferenced. Upon return, the newly created operator's handle
                is stored in *op*.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] op            Pointer to where newly created custom reduction operator is to be stored
    @param[in]  scalar        Pointer to scalar value.
    @param[in]  datatype      Scalar value datatype
    @param[in]  residence     Memory type of the scalar value
    @param[in]  comm          Communicator to associate with this custom reduction operator */
mxcclResult_t  mxcclRedOpCreatePreMulSum(mxcclRedOp_t *op, void *scalar, mxcclDataType_t datatype, mxcclScalarResidence_t residence, mxcclComm_t comm);
/*! @cond       include_hidden */
mxcclResult_t pmxcclRedOpCreatePreMulSum(mxcclRedOp_t *op, void *scalar, mxcclDataType_t datatype, mxcclScalarResidence_t residence, mxcclComm_t comm);
/*! @endcond */

/*! @brief      Destroy custom reduction operator
    @details    Destroys the reduction operator *op*. The operator must have been created by
                mxcclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
                destroyed as soon as the last MCCL function which is given that operator returns.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  op            Custom reduction operator is to be destroyed
    @param[in]  comm          Communicator associated with this reduction operator */
mxcclResult_t mxcclRedOpDestroy(mxcclRedOp_t op, mxcclComm_t comm);
/*! @cond       include_hidden */
mxcclResult_t pmxcclRedOpDestroy(mxcclRedOp_t op, mxcclComm_t comm);
/*! @endcond */
/*! @} */

/*! @defgroup   mxccl_collective_api Collective Communication Operations
    @details    Collective communication operations must be called separately for each
                communicator in a communicator clique.

                They return when operations have been enqueued on the HIP stream.
                Since they may perform inter-CPU synchronization, each call has to be done
                from a different thread or process, or need to use Group Semantics (see
                below).
    @{ */

/*! @brief      Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* into *recvbuff* using *op*
                operation.
                *recvbuff* may be NULL on all calls except for root device.
                *root* is the rank (not the HIP device) where data will reside after the
                 operation is complete.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Local device data buffer to be reduced
    @param[out] recvbuff      Data buffer where result is stored (only for *root* rank).  May be null for other ranks.
    @param[in]  count         Number of elements in every send buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator type
    @param[in]  root          Rank where result data array will be stored
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclReduce(const void* sendbuff, void* recvbuff, size_t count, mxcclDataType_t datatype,
    mxcclRedOp_t op, int root, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclReduce(const void* sendbuff, void* recvbuff, size_t count, mxcclDataType_t datatype,
    mxcclRedOp_t op, int root, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      (Deprecated) Broadcast (in-place)
    @details    Copies *count* values from *root* to all other devices.
                root is the rank (not the CUDA device) where data resides before the
                operation is started.
                This operation is implicitly in-place.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in,out]  buff      Input array on *root* to be copied to other ranks.  Output array for all ranks.
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank owning buffer to be copied to others
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclBcast(void* buff, size_t count, mxcclDataType_t datatype, int root,
    mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclBcast(void* buff, size_t count, mxcclDataType_t datatype, int root,
    mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Broadcast
    @details    Copies *count* values from *sendbuff* on *root* to *recvbuff* on all devices.
                *root* is the rank (not the HIP device) where data resides before the operation is started.
                *sendbuff* may be NULL on ranks other than *root*.
                In-place operation will happen if *sendbuff* == *recvbuff*.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to copy (if *root*).  May be NULL for other ranks
    @param[in]  recvbuff      Data array to store received array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank of broadcast root
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, mxcclDataType_t datatype, int root,
    mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, mxcclDataType_t datatype, int root,
    mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      All-Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* using *op* operation, and
                leaves identical copies of result on each *recvbuff*.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    mxcclDataType_t datatype, mxcclRedOp_t op, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    mxcclDataType_t datatype, mxcclRedOp_t op, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Reduce-Scatter
    @details    Reduces data in *sendbuff* using *op* operation and leaves reduced result
                scattered over the devices so that *recvbuff* on rank i will contain the i-th
                block of the result.
                Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
                should have a size of at least nranks*recvcount elements.
                In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result subarray
    @param[in]  recvcount     Number of elements each rank receives
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mxcclDataType_t datatype, mxcclRedOp_t op, mxcclComm_t comm,
    mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mxcclDataType_t datatype, mxcclRedOp_t op, mxcclComm_t comm,
    mcStream_t stream);
/*! @endcond */

/*! @brief      All-Gather
    @details    Each device gathers *sendcount* values from other GPUs into *recvbuff*,
                receiving data from rank i at offset i*sendcount.
                Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
                should have a size of at least nranks*sendcount elements.
                In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Input data array to send
    @param[out] recvbuff      Data array to store the gathered result
    @param[in]  sendcount     Number of elements each rank sends
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Send
    @details    Send data from *sendbuff* to rank *peer*.
                Rank *peer* needs to call mxcclRecv with the same *datatype* and the same *count*
                as this rank.
                This operation is blocking for the GPU. If multiple mxcclSend and mxcclRecv operations
                need to progress concurrently to complete, they must be fused within a mxcclGroupStart /
                mxcclGroupEnd section.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to send
    @param[in]  count         Number of elements to send
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclSend(const void* sendbuff, size_t count, mxcclDataType_t datatype, int peer,
    mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclSend(const void* sendbuff, size_t count, mxcclDataType_t datatype, int peer,
    mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Receive
    @details    Receive data from rank *peer* into *recvbuff*.
                Rank *peer* needs to call mxcclSend with the same datatype and the same count
                as this rank.
                This operation is blocking for the GPU. If multiple mxcclSend and mxcclRecv operations
                need to progress concurrently to complete, they must be fused within a mxcclGroupStart/
                mxcclGroupEnd section.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[out] recvbuff      Data array to receive
    @param[in]  count         Number of elements to receive
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclRecv(void* recvbuff, size_t count, mxcclDataType_t datatype, int peer,
    mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclRecv(void* recvbuff, size_t count, mxcclDataType_t datatype, int peer,
    mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Gather
    @details    Root device gathers *sendcount* values from other GPUs into *recvbuff*,
                receiving data from rank i at offset i*sendcount.
                Assumes recvcount is equal to nranks*sendcount, which means that *recvbuff*
                should have a size of at least nranks*sendcount elements.
                In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
                *recvbuff* may be NULL on ranks other than *root*.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to send
    @param[out] recvbuff      Data array to receive into on *root*.
    @param[in]  sendcount     Number of elements to send per rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank that receives data from all other ranks
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mxcclDataType_t datatype, int root, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    mxcclDataType_t datatype, int root, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      Scatter
    @details    Scattered over the devices so that recvbuff on rank i will contain the i-th
                block of the data on root.
                Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
                should have a size of at least nranks*recvcount elements.
                In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (on *root* rank).  May be NULL on other ranks.
    @param[out] recvbuff      Data array to receive partial subarray into
    @param[in]  recvcount     Number of elements to receive per rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank that scatters data to all other ranks
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mxcclDataType_t datatype, int root, mxcclComm_t comm,
    mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, mxcclDataType_t datatype, int root, mxcclComm_t comm,
    mcStream_t stream);
/*! @endcond */

/*! @brief      All-To-All
    @details    Device (i) send (j)th block of data to device (j) and be placed as (i)th
                block. Each block for sending/receiving has *count* elements, which means
                that *recvbuff* and *sendbuff* should have a size of nranks*count elements.
                In-place operation is NOT supported. It is the user's responsibility
                to ensure that sendbuff and recvbuff are distinct.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (contains blocks for each other rank)
    @param[out] recvbuff      Data array to receive (contains blocks from each other rank)
    @param[in]  count         Number of elements to send between each pair of ranks
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @brief      All-To-Allv
    @details    Device (i) sends sendcounts[j] of data from offset sdispls[j]
                to device (j). At the same time, device (i) receives recvcounts[j] of data
                from device (j) to be placed at rdispls[j].
                sendcounts, sdispls, recvcounts and rdispls are all measured in the units
                of datatype, not bytes.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref mxccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (contains blocks for each other rank)
    @param[in]  sendcounts    Array containing number of elements to send to each participating rank
    @param[in]  sdispls       Array of offsets into *sendbuff* for each participating rank
    @param[out] recvbuff      Data array to receive (contains blocks from each other rank)
    @param[in]  recvcounts    Array containing number of elements to receive from each participating rank
    @param[in]  rdispls       Array of offsets into *recvbuff* for each participating rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
mxcclResult_t  mxcclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @cond       include_hidden */
mxcclResult_t pmxcclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], mxcclDataType_t datatype, mxcclComm_t comm, mcStream_t stream);
/*! @endcond */

/*! @} */

/*! @defgroup   mxccl_group_api Group semantics
    @details    When managing multiple GPUs from a single thread, and since MCCL collective
                calls may perform inter-CPU synchronization, we need to "group" calls for
                different ranks/devices into a single call.

                Grouping MCCL calls as being part of the same collective operation is done
                using mxcclGroupStart and mxcclGroupEnd. mxcclGroupStart will enqueue all
                collective calls until the mxcclGroupEnd call, which will wait for all calls
                to be complete. Note that for collective communication, mxcclGroupEnd only
                guarantees that the operations are enqueued on the streams, not that
                the operation is effectively done.

                Both collective communication and mxcclCommInitRank can be used in conjunction
                of mxcclGroupStart/mxcclGroupEnd, but not together.

                Group semantics also allow to fuse multiple operations on the same device
                to improve performance (for aggregated collective calls), or to permit
                concurrent progress of multiple send/receive operations.
    @{ */

/*! @brief      Group Start
    @details    Start a group call. All calls to MCCL until mxcclGroupEnd will be fused into
                a single MCCL operation. Nothing will be started on the HIP stream until
                mxcclGroupEnd.
    @return     Result code. See @ref mxccl_result_code for more details. */
mxcclResult_t  mxcclGroupStart();
/*! @cond       include_hidden */
mxcclResult_t pmxcclGroupStart();
/*! @endcond */

/*! @brief      Group End
    @details    End a group call. Start a fused MCCL operation consisting of all calls since
                mxcclGroupStart. Operations on the HIP stream depending on the MCCL operations
                need to be called after mxcclGroupEnd.
    @return     Result code. See @ref mxccl_result_code for more details. */
mxcclResult_t  mxcclGroupEnd();
/*! @cond       include_hidden */
mxcclResult_t pmxcclGroupEnd();
/*! @endcond */
/*! @} */

/*! @brief is all rank use metax gpu
 * is all rank use metax gpu
 */
bool  mxcclAllRankMetax(mxcclComm_t comm);
/// @cond include_hidden
bool pmxcclAllRankMetax(mxcclComm_t comm);
/// @endcond

/*! @brief finish all current job
 * finish all current job
 */
mxcclResult_t  mxcclFinish(mxcclComm_t comm);
/// @cond include_hidden
mxcclResult_t pmxcclFinish(mxcclComm_t comm);
/// @endcond

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
