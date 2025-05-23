// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

#if !defined(CK_USE_AMD_MFMA_GFX950)
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmPadded  = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
#endif

// c[g, m, n] = a[g, m, k] * b[g, n, k]
template <bool Masking>
using device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances =
    std::tuple<
// clang-format off
        //#######################################| ALayout| B0Layout| B1Layout| CLayout| AData| B0Data| B1Data| CData| AccData| CShuffle|           A|          B0|        Acc0|          B1|           C|           GEMM| NumGemmK| Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|    MaskOut|
        //#######################################|        |         |         |        |  Type|   Type|   Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Elementwise| Elementwise| Specialization| Prefetch|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|      Upper|
        //#######################################|        |         |         |        |      |       |       |      |        |         |   Operation|   Operation|   Operation|   Operation|   Operation|               |    Stage|      |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|   Triangle|
        //#######################################|        |         |         |        |      |       |       |      |        |         |            |            |            |            |            |               |         |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                |           |
#if defined(CK_USE_AMD_MFMA_GFX950)
#else
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    256,   128,    32,    64,    32,   8,   8,    2,   32,   32,     2,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    256,   128,    32,   128,    32,   8,   8,    2,   32,   32,     2,     4,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   256,    32,    64,    32,   8,   8,    2,   32,   32,     1,     8,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   256,    32,   128,    32,   8,   8,    2,   32,   32,     1,     8,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   128,    64,    64,    32,   8,   8,    2,   32,   32,     1,     4,     2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   128,    32,    64,    32,   8,   8,    2,   32,   32,     1,     4,     2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,    128,   128,    32,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,     64,   256,    32,   128,    32,   8,   8,    2,   16,   16,     1,    16,     8,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           8,               S<1, 16, 1,16>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,     64,   256,    32,    64,    32,   8,   8,    2,   16,   16,     1,    16,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,     64,   256,    64,   128,    32,   8,   8,    2,   16,   16,     1,    16,     8,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           8,               S<1, 16, 1,16>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,    GemmDefault,        1,   256,     64,   256,    64,    64,    32,   8,   8,    2,   16,   16,     1,    16,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           4,               S<1, 32, 1, 8>,               8,   Masking>,
        // Padded fallback kernel
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,   128,    64,   128,    32,   8,   8,    2,   32,   32,     1,     4,     4,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,    64,    32,   128,    32,   8,   8,    2,   32,   32,     1,     2,     4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,      S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,       true,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>
#endif // defined(CK_USE_AMD_MFMA_GFX950)
       // clang-format on
        >;

template <bool Masking>
using device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_irregular_k_instances =
    std::tuple<
// clang-format off
        //#######################################| ALayout| B0Layout| B1Layout| CLayout| AData| B0Data| B1Data| CData| AccData| CShuffle|           A|          B0|        Acc0|          B1|           C|           GEMM| NumGemmK| Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|    MaskOut|
        //#######################################|        |         |         |        |  Type|   Type|   Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Elementwise| Elementwise| Specialization| Prefetch|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|      Upper|
        //#######################################|        |         |         |        |      |       |       |      |        |         |   Operation|   Operation|   Operation|   Operation|   Operation|               |    Stage|      |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|   Triangle|
        //#######################################|        |         |         |        |      |       |       |      |        |         |            |            |            |            |            |               |         |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                |           |
#if defined(CK_USE_AMD_MFMA_GFX950)
#else
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    256,   128,    40,    64,    32,   4,   4,    2,   32,   32,     2,     4,     2,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    256,   128,    40,   128,    32,   4,   4,    2,   32,   32,     2,     4,     4,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,   256,    40,    64,    32,   4,   4,    2,   32,   32,     1,     8,     2,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,   256,    40,   128,    32,   4,   4,    2,   32,   32,     1,     8,     4,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,   128,    40,    64,    32,   4,   4,    2,   32,   32,     1,     4,     2,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>,
        DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<     Row,      Col,      Row,     Row,   F16,    F16,    F16,   F16,     F32,      F16, PassThrough, PassThrough,       Scale, PassThrough, PassThrough,     GemmPadded,        1,   256,    128,   128,    40,   128,    32,   4,   4,    2,   32,   32,     1,     4,     4,     S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false,      S<2,128, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               4,               4,      false,     S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false,           1,           2,               S<1, 32, 1, 8>,               8,   Masking>
#endif // defined(CK_USE_AMD_MFMA_GFX950)
       // clang-format on
        >;

void add_device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemm<Row,
                                                             Col,
                                                             Row,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             Scale,
                                                             PassThrough,
                                                             PassThrough,
                                                             false>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances<
            false>{});
    add_device_operation_instances(
        instances,
        device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_irregular_k_instances<
            false>{});
}

void add_device_batched_gemm_masking_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemm<Row,
                                                             Col,
                                                             Row,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             Scale,
                                                             PassThrough,
                                                             PassThrough,
                                                             true>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances<
            true>{});
    add_device_operation_instances(
        instances,
        device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_irregular_k_instances<
            true>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
