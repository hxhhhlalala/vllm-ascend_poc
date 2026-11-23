# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Dependencies may add op_host directly, bypassing the operator CMakeLists.
# A6 registers the in-tree SFA host implementation; the arch35 kernel selects
# its NO_ROPE variant from the tiling, so NoPE (GLM-5.3-Flash C16) is served.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(sparse_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for sparse_flash_attention")
    target_sources(op_host_aclnnInner PRIVATE
        sparse_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SparseFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_flash_attention ACLNNTYPE aclnn_inner)
endif()
