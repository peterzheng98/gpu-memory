
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "moe_smoothquant_instance_common.hpp"

// clang-format off
//                                                  rm rn tm tn  vn  pd    2p
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 1, 4, 64, 8, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 2, 4, 64, 4, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 4, 4, 64, 2, true, false>>(const S&, A);
template float moe_smoothquant_<trait_<ck_tile::bf16_t, 1, 8, 4, 64, 1, true, false>>(const S&, A);
// clang-format on
