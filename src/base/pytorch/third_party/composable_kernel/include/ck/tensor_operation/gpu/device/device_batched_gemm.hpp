// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB,
                        ck::index_t BatchStrideC,
                        ck::index_t Batch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename BScaleType,
          typename CDataType,
          index_t ScaleBlockN,
          index_t ScaleBlockK,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemmV2BScale : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        ck::index_t StrideScaleB,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB,
                        ck::index_t BatchStrideC,
                        ck::index_t BatchStrideScaleB,
                        const void* p_b_scale,
                        ck::index_t Batch,
                        ck::index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual bool GetPermuteB()         = 0;
    virtual ck::index_t GetKPerBlock() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceBatchedGemmPtr = std::unique_ptr<DeviceBatchedGemm<ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               ADataType,
                                                               BDataType,
                                                               CDataType,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
