/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file sparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_flash_attention_template_tiling_key.h"

// A6 (ascend960dt/pr) shares the regbase arch35 kernel family even when the
// compiler does not advertise __CCE_AICORE__ == 310 for it (same rule the
// GDN FwdH kernel uses).
#if (__CCE_AICORE__ == 310) || (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#define SFA_ARCH35 1
#include "arch35/sparse_flash_attention_kernel_mla.h"
#else
#include "arch22/sparse_flash_attention_kernel_mla.h"
#endif

using namespace AscendC;

#if defined(SFA_ARCH35)
#if defined(__DAV_C310_CUBE__)
#define SFA_OP_IMPL(templateClass, tilingdataClass, NO_ROPE, ...)                                 \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SFAMatmulService<__VA_ARGS__, NO_ROPE>, BaseApi::SFAMatmulServiceDummy<__VA_ARGS__, NO_ROPE>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__, NO_ROPE>, BaseApi::SFAVectorService<__VA_ARGS__, NO_ROPE>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, user, nullptr, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#else
#define SFA_OP_IMPL(templateClass, tilingdataClass, NO_ROPE, ...)                                 \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SFAMatmulService<__VA_ARGS__, NO_ROPE>, BaseApi::SFAMatmulServiceDummy<__VA_ARGS__, NO_ROPE>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__, NO_ROPE>, BaseApi::SFAVectorService<__VA_ARGS__, NO_ROPE>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                       \
        const tilingdataClass *__restrict tilingData = &tiling_data_in;                                     \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, user, tilingData, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#endif
#else
// A2/A3 kernel has no NoPE variant; the token keeps the call sites identical.
#define SFA_OP_IMPL(templateClass, tilingdataClass, NO_ROPE, ...)                                 \
    do {                                                                                          \
        templateClass<SFAType<__VA_ARGS__>> op;                                                   \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
	    blocktable, queryRope, keyRope, attentionOut, softmaxMax, softmaxSum, user, tiling_data, tiling, &tPipe);         \
        op.Process();                                                                             \
    } while (0)
#endif

template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
 __global__ __aicore__ void
sparse_flash_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
                       __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
                       __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                       __gm__ uint8_t *attentionOut, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
                       __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    // A6 (ascend960dt/pr) exposes 64 AIC and 64 AIV. A 1:2 request makes every
    // AIC wait forever for a second AIV that is never scheduled.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

#if defined(SFA_ARCH35)
    // NoPE (rope_head_dim=0) keeps a 512-wide K dimension; RoPE keeps 576.
    // Select at runtime from the tiling so a single tiling key still serves both.
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingDataMla, dispatchTiling, tiling);
    if (dispatchTiling.baseParams.ropeHeadDim == 0) {
        if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 &&
                      ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
            SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, true, bfloat16_t,
                bfloat16_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION, static_cast<SFA_LAYOUT>(LAYOUT_T),
                static_cast<SFA_LAYOUT>(KV_LAYOUT_T), static_cast<SFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
        } else {
            SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, true, half, half,
                float, half, FLASH_DECODE, PAGE_ATTENTION, static_cast<SFA_LAYOUT>(LAYOUT_T),
                static_cast<SFA_LAYOUT>(KV_LAYOUT_T), static_cast<SFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
        }
    } else {
        if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 &&
                      ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
            SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, false, bfloat16_t,
                bfloat16_t, float, bfloat16_t, FLASH_DECODE, PAGE_ATTENTION, static_cast<SFA_LAYOUT>(LAYOUT_T),
                static_cast<SFA_LAYOUT>(KV_LAYOUT_T), static_cast<SFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
        } else {
            SFA_OP_IMPL(BaseApi::SparseFlashAttentionKernelMla, SparseFlashAttentionTilingDataMla, false, half, half,
                float, half, FLASH_DECODE, PAGE_ATTENTION, static_cast<SFA_LAYOUT>(LAYOUT_T),
                static_cast<SFA_LAYOUT>(KV_LAYOUT_T), static_cast<SFATemplateMode>(TEMPLATE_MODE), IS_SPLIT_G);
        }
    }
#else
    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_FLOAT16 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
        SFA_OP_IMPL(SparseFlashAttentionMla, SparseFlashAttentionTilingDataMla, false, half, half, half,
            FLASH_DECODE, static_cast<SFA_LAYOUT>(LAYOUT_T), static_cast<SFA_LAYOUT>(KV_LAYOUT_T), TEMPLATE_MODE);
    } else { // bf16
        SFA_OP_IMPL(SparseFlashAttentionMla, SparseFlashAttentionTilingDataMla, false, bfloat16_t, bfloat16_t,
            bfloat16_t, FLASH_DECODE, static_cast<SFA_LAYOUT>(LAYOUT_T), static_cast<SFA_LAYOUT>(KV_LAYOUT_T),
            TEMPLATE_MODE);
    }
#endif
}
