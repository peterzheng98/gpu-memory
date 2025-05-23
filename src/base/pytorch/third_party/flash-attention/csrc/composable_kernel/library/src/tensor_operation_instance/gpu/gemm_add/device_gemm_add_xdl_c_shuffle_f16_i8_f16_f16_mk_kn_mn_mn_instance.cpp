// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/utility/sequence.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// e = elementwise((a * b), d0, d1)
// outout: e[m, n]
// input: a[m, k], b[k, n], d0[m, n], d1[m, n]
using device_gemm_add_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_generic_instances = std::tuple<
    // clang-format off
        // M/N/K padding
        //##############################|      A|      B|        Ds|      E| AData| BData| AccData| CShuffle|    DsData| EData|           A|           B|            CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################| Layout| Layout|    Layout| Layout|  Type|  Type|    Type| DataType|      Type|  Type| Elementwise| Elementwise|    Elementwise| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //##############################|       |       |          |       |      |      |        |         |          |      |   Operation|   Operation|      Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //##############################|       |       |          |       |      |      |        |         |          |      |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGemmMultipleD_Xdl_CShuffle<    Row,    Row, Row_Tuple,    Row,   F16,    I8,     F32,      F32, F16_Tuple,   F16, PassThrough, PassThrough,            Add, GemmMNKPadding,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>
    // clang-format on
    >;

using device_gemm_add_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances = std::tuple<
    // clang-format off
        // M/N/K padding
        //##############################|      A|      B|        Ds|      E| AData| BData| AccData| CShuffle|    DsData| EData|           A|           B|            CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################| Layout| Layout|    Layout| Layout|  Type|  Type|    Type| DataType|      Type|  Type| Elementwise| Elementwise|    Elementwise| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //##############################|       |       |          |       |      |      |        |         |          |      |   Operation|   Operation|      Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //##############################|       |       |          |       |      |      |        |         |          |      |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGemmMultipleD_Xdl_CShuffle<    Row,    Row, Row_Tuple,    Row,   F16,    I8,     F32,      F32, F16_Tuple,   F16, PassThrough, PassThrough,            Add, GemmMNKPadding,        1,   256,    16,   128,    32,   8,   8,   16,   16,    1,    2,     S<4, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1, LoopScheduler::Default,        PipelineVersion::v1>,
        DeviceGemmMultipleD_Xdl_CShuffle<    Row,    Row, Row_Tuple,    Row,   F16,    I8,     F32,      F32, F16_Tuple,   F16, PassThrough, PassThrough,            Add, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              1,         1,           1,           1,               S<1, 16, 1, 4>,               1, LoopScheduler::Default,        PipelineVersion::v1>,
        DeviceGemmMultipleD_Xdl_CShuffle<    Row,    Row, Row_Tuple,    Row,   F16,    I8,     F32,      F32, F16_Tuple,   F16, PassThrough, PassThrough,            Add, GemmMNKPadding,        1,    64,    16,    16,    64,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              1,         1,           1,           1,               S<1, 16, 1, 4>,               1, LoopScheduler::Default,        PipelineVersion::v1>
    // clang-format on
    >;

void add_device_gemm_add_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    I8,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    Add>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_add_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_generic_instances{});
    add_device_operation_instances(
        instances, device_gemm_add_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
