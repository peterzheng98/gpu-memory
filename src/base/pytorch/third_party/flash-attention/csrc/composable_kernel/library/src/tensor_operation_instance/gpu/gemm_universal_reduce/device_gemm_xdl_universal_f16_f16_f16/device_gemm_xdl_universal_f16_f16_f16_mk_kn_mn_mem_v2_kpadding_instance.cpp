// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_f16_f16_f16_mk_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemmV2R1<Row,
                                               Row,
                                               DsLayout,
                                               Row,
                                               F16,
                                               F16,
                                               DsDataType,
                                               F16,
                                               PassThrough,
                                               PassThrough,
                                               PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_reduce_f16_f16_f16_mk_kn_mn_mem_instances<Interwave,
                                                                            GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
