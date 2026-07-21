# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MetaX (沐曦) C550 build subparser for XPUTimer.
#
# Mirrors config_subparsers/nvidia.py but targets the MACA stack:
#   * -DXPU_METAX compile flag (instead of -DXPU_NVIDIA)
#   * SDK path defaults to /opt/maca (instead of /usr/local/cuda)
#   * links libmcruntime (the cuda-alike runtime) instead of libcudart
#   * version comes from MACART_VERSION in mcr/mc_runtime_api.h
#
# IMPORTANT (see results/xputimer-metax-log.md): MACA does NOT export cuda*/nccl*
# symbols. torch(2.6.0+metax) links the mc*/mccl* symbols directly. The runnable
# detection backend therefore lives in xpu_timer/metax_probe/ and interposes the
# mc*/mccl* names via LD_PRELOAD. This subparser wires the Bazel build for the
# in-tree metax backend once the heavy deps (brpc/boost/perfetto) are available;
# on a bare pod without Bazel, use xpu_timer/metax_probe/build_metax_hook.sh.

import textwrap

from . import BaseBuildRender


class BuildRender(BaseBuildRender):
    @classmethod
    def add_arguments(cls, parser):
        BaseBuildRender.add_arguments(parser)

    def __post_init__(self):
        if self.args.sdk_path is None:
            self.args.sdk_path = "/opt/maca"

        delete_packate = ",".join(
            f"//xpu_timer/{non_target}/..." for non_target in self.args.non_target
        )
        self.bazelrc_config.append(f"build --deleted_packages={delete_packate}")

    def rend_config_bzl(self):
        metax_config = textwrap.dedent(
            """
        XPU_TIMER_CONFIG = struct(
            linkopt = [
                "-Wl,--version-script=$(location //xpu_timer/metax:only_keep_mx.lds)",
                "-L{maca_path}/lib",
                "-lmcruntime",
            ],
            copt = [
                "-DXPU_METAX",
            ],
            deps = ["@maca//:maca_headers", "@mccl//:mccl_h", "//xpu_timer/metax:only_keep_mx.lds"],
            py_bin = [
                "//xpu_timer/metax:intercepted.sym.default",
            ],
            gen_symbol = ["//xpu_timer/metax:gen_metax_symbols.py"],
            timer_deps = ["//xpu_timer/metax:metax_timer"],
            hook_deps = ["//xpu_timer/metax:metax_hook"],
        )

        """
        )

        self.xpu_timer_config.append(metax_config.format(maca_path=self.sdk_path))
        return "\n".join(self.xpu_timer_config)

    def rend_bazelrc(self):
        self.bazelrc_config.append(f"build --repo_env=MACA_HOME={self.sdk_path}")
        return "\n".join(self.bazelrc_config)

    def setup_files(self):
        with open("WORKSPACE.template") as f:
            workspace = f.read()

        deps = textwrap.dedent(
            """
            load("//third_party/maca:maca_workspace.bzl", "maca_workspace")
            maca_workspace()

            load("//third_party/mccl:mccl_workspace.bzl", "mccl_workspace")
            mccl_workspace()
            """
        )
        return workspace + deps

    def setup_platform_version(self):
        # /opt/maca/include/mcr/mc_runtime_api.h:#define MACART_VERSION 3000
        # Also carry the MACA release (e.g. 3.3.0.x) if present in Version.txt.
        version = None
        path = f"{self.sdk_path}/include/mcr/mc_runtime_api.h"
        pattern = "#define MACART_VERSION"
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(pattern):
                        version = line.split(pattern)[-1].strip()
                        break
        except FileNotFoundError:
            pass
        if version is None:
            # Fallback: parse /opt/maca/Version.txt (e.g. "3.3.0.15")
            version = "0"
        return f"maca{version}", "METAX"
