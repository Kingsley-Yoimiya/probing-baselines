// Copyright 2024 The DLRover Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xpu_timer/common/platform.h"

#include "xpu_timer/common/logging.h"
#include "xpu_timer/common/util.h"

namespace xpu_timer {
namespace platform {
#if defined(XPU_NVIDIA)

std::string getDeviceName() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    XLOG(FATAL) << "No CUDA devices found, abort";
    std::abort();
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::string full_device_name;
  std::string device_name;

  try {
    // Tesla P100-PCIE-16GB
    // NVIDIA A100-SXM4-80GB
    full_device_name = util::split(deviceProp.name, " ").at(1);
    device_name = util::split(full_device_name, "-").at(0);
  } catch (const std::out_of_range &e) {
    XLOG(ERROR) << "Device name parsing error origin name is "
                << deviceProp.name << " Fall back to A100";
    device_name = "A100";
  }

  XLOG(INFO) << "Device name " << device_name << " origin " << deviceProp.name;
  return device_name;
}

#endif

#if defined(XPU_METAX)
// MetaX (沐曦) C550. MACA is a cuda-alike layer that exports mc*-prefixed
// symbols (mcGetDeviceCount / mcGetDeviceProperties), NOT cuda*. The device
// property struct's first field is `char name[256]` (see
// /opt/maca/include/mcr/mc_runtime_types.h: mcDeviceProp_t), e.g. "MetaX C550-PL".
// Verified on-machine (see results/xputimer-metax-log.md, milestone 1).

std::string getDeviceName() {
  int deviceCount = 0;
  mcGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    XLOG(FATAL) << "No MACA devices found, abort";
    std::abort();
  }
  mcDeviceProp_t deviceProp;
  mcGetDeviceProperties(&deviceProp, 0);
  std::string full_device_name;
  std::string device_name;

  try {
    // "MetaX C550-PL" -> "C550-PL" -> "C550"
    full_device_name = util::split(deviceProp.name, " ").at(1);
    device_name = util::split(full_device_name, "-").at(0);
  } catch (const std::out_of_range &e) {
    XLOG(ERROR) << "Device name parsing error origin name is "
                << deviceProp.name << " Fall back to C550";
    device_name = "C550";
  }

  XLOG(INFO) << "Device name " << device_name << " origin " << deviceProp.name;
  return device_name;
}

#endif
}  // namespace platform
}  // namespace xpu_timer
