# A6 SparseFlashAttention / GLM-5.3 C16 — 双方案改动说明与验证交接

> 本文件为交接文档（未纳入 git）。两个方案分别在两个分支上，均基于
> `br_poc_a6_1106` 的基线提交 `5bbea77b7`。

## 0. 背景与结论

- **GLM-5.3-Flash C16 确认走 NoPE**：`mla_use_nope=true`、`qk_rope_head_dim=0`
  （`tests/e2e/pull_request/four_card/glm53flash_assets/config.json`）。
- A5/A6 上 `SFA_NOPE_PROLOG_V3` 均不可用，GLM NoPE 走 `PreprocessType.NATIVE`，
  最终进入 `vllm_ascend/attention/sfa_v1.py::sparse_mla()`。
- A6 现状：`SparseMLAMetadataState.use_smla=False` → `smla_metadata=None` →
  `_C_ascend.npu_sparse_flash_attention` 解析到 **CANN 9.3 arch92 SDK**，
  以 `query_rope=None/key_rope=None`（rope=0）执行。
- 已确认的两个问题：
  1. TND 图 bucket 的尾部 padding 行可能保留脏数据/NaN；
  2. 本仓库 in-tree arch35 SFA kernel 把 `dSizeRope=64` 写死，**无法**处理 rope=0。
- 两个方案：
  - **A（新分支）**：A6 复用 A5 的交付包 `SparseFlashMla` 路径（`use_smla` 纳入 A6）。
  - **B（当前分支）**：给 in-tree arch35 SFA 增加 `NO_ROPE` 变体，并把 SFA 编入 A6 自定义包。

## 1. 分支与提交

| 方案 | 分支 | 提交 | 说明 |
| --- | --- | --- | --- |
| B（in-tree `_ascend_C` + NO_ROPE） | `br_poc_a6_1106` | `f4eafb9c4` | 当前分支，已提交 |
| A（交付包 `SparseFlashMla`） | `br_poc_a6_1106_sfa_smla` | `206f15e07` | 从基线新建，已提交 |

两分支都包含 **共享的 TND 尾部修复**（`vllm_ascend/attention/sfa_v1.py` 的
`_zero_tnd_capacity_tail` / `masked_fill`）和对应 UT。

## 2. 方案 A：A6 走交付包 `SparseFlashMla`

### 设计
A6 不编 in-tree SFA；NoPE 的 C16 路径与 A5 完全一致：
`sparse_mla()` → `metadata.smla_metadata != None` → `sparse_flash_mla()`
（`torch.ops.cann_ops_transformer.sparse_flash_mla`）。

### 关键改动
- `vllm_ascend/attention/sfa_v1.py`
  - `SparseMLAMetadataState.use_smla`：由「仅 `FP8_OPTIMIZED`」改为
    「`FP8_OPTIMIZED` 或 `A6`」；C8/turboquant 仍然关闭 SMLA（保持 QSFA）。
  - 新增 `_zero_tnd_capacity_tail(output, cumulative_query_lengths)`：
    用 device 侧 `query_start_loc[-1]` 做 `masked_fill`，graph replay 安全。
  - SMLA 分支在 A6 上调用该函数清尾部；非 SMLA 分支保留同一函数。
- `tests/ut/attention/test_sfa_nope_metadata.py`
  - 新增 `test_a6_uses_delivered_smla_path`：A6 profile 下 `use_smla is True`。
- `tests/ut/attention/test_sfa_nope_forward.py`
  - 新增 `_zero_tnd_capacity_tail` 单元测试。
- `docs/source/developer_guide/a6_adaptation.md`、SFA `CMakeLists.txt` 注释：记录选择。
- in-tree SFA 的 A6 early-return **保留**（A6 仍不编该 OPP）。

### 优点 / 限制
- 优点：改动小；与 A5 同一算子契约，无 kernel 移植。
- 限制：依赖 A6 安装的 CANN 包里存在 `sparse_flash_mla` 与
  `sparse_flash_mla_metadata`；`generate_smla_plan` 的 core 数假设需要实测
  （已知 A5 metadata 有 64-core 全局假设）。

## 3. 方案 B：in-tree arch35 SFA + `NO_ROPE`

### 设计
arch35 SFA 新增 `NO_ROPE` 模板轴（与 QSFA 的做法一致），在 kernel 入口
**运行时**读取 `tiling.baseParams.ropeHeadDim` 选择变体，因此 tiling key 不变。

NoPE 下的布局：
- K 维 = 512（nope），不再拷贝 query_rope/key_rope；
- UB/GM 的 KV tile 按 512 行宽打包；
- **L1/GM 缓冲区容量仍保持 576**：BMM1 的 NZ 布局把 512 放在前面的 D-block，
  `ProcessVec1` 把概率 P 写到 `s2BaseSize*512` 偏移处，复用 rope 的 64 列空间。
- TND 尾部：`needInit==0` 时在 `CleanOutput` 调用 `ClearOutputTail()`，
  从 GM 读 `cuSeqlensQGm[bSize-1]`（replay 安全），按 32B 对齐分核清零。

### 关键改动
- `csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_common_arch35.h`
  - `CUBE_BLOCK_TRAITS_CONST_FIELDS` 增加 `X(NO_ROPE, bool, false)`。
- `.../sparse_flash_attention_service_cube_mla.h`
  - `dBaseSize = NO_ROPE ? 512 : 576`；跳过 queryRope 的 offset/copy；
  - `kvTileDim = NO_ROPE ? 512 : 576` 用于 v0Res→L1 拷贝。
- `.../sparse_flash_attention_service_vector_mla.h`
  - `dVTemplateTypeInput = NO_ROPE ? 512 : 576`；跳过 keyRope 拷贝；
  - `dstStride` 在 NoPE 下为 0（两行 512 紧凑）；
  - `stage0OutBuf` 按 `dVTemplateTypeInput*16` 申请；
  - `sharedParams.dSizeRope = NO_ROPE ? 0 : 64`；
  - 新增 `ClearOutputTail()` 并挂到 `CleanOutput()` 的 TND else 分支；
  - `keyRopeGm` 绑定加 `if constexpr (!NO_ROPE)`。
- `.../sparse_flash_attention_kernel_mla.h`
  - `constInfo.dSizeRope = NO_ROPE ? 0 : 64`（`NO_ROPE` 来自 `ARGS_TRAITS`）；
  - 注释说明 576 容量与 P 复用。
- `.../op_kernel/sparse_flash_attention.cpp`
  - `SFA_OP_IMPL` 增加 `NO_ROPE` 形参并传给 cube/vector；
  - 入口新增 `GET_TILING_DATA_WITH_STRUCT(...)` + `ropeHeadDim==0` 运行时分支；
  - arch22 路径不受影响（形参占位）。
- `csrc/attention/sparse_flash_attention/CMakeLists.txt` / `op_host/CMakeLists.txt`
  - 删除 A6 early-return，A6 正常注册 in-tree SFA。
- `csrc/build_aclnn.sh`
  - A6 `CUSTOM_OPS_ARRAY` 加回 `"sparse_flash_attention"`。
- `.../sparse_flash_attention_torch_adpt.h`、`README.md`、host tiling 注释：同步说明。
- 测试：
  - `tests/ut/_tools/test_a6_sfa_build.py`：A6 现在必须注册；A5/A6 混合打包允许。
  - `tests/e2e/.../test_sparse_flash_attention.py`：新增
    `test_sparse_flash_attention_nope_tnd_output_padding`（NoPE + BF16 + padding bucket）。
  - `tests/ut/attention/test_sfa_nope_forward.py`：`sparse_mla` 尾部清零 UT。

### 优点 / 限制
- 优点：A6 不依赖 SDK SFA；与 A5 共用同一套 in-tree kernel 源码。
- 限制：NO_ROPE 是手工移植，**必须实机编译 + 数值验证**；host tiling 在 A6 上走
  `isA5_==false` 的 A2/A3 分支（workspace/core 预算需要确认）；AIC:AIV=1:2 假设需要确认。

## 4. 共享修复（两分支都有）

`tilelang` 无关：`vllm_ascend/attention/sfa_v1.py`
- `sparse_mla()` 的两个分支在算子返回后清零 TND 容量尾部；
- 用 device tensor（`query_start_loc[1:][-1]`）而不是 host int，保证 ACLGraph replay 正确；
- 语义：valid 行不变，只把 `[actual, T)` 变成 0。

## 5. 已完成的自检 / 未完成

已做：
- `python -m py_compile`：`sfa_v1.py`、两个 UT 文件、e2e 测试文件。
- `ruff check`：以上文件全部通过。
- 静态审查：B 分支所有 rope 路径（queryRope/keyRope 拷贝、偏移计算）均有
  `NO_ROPE` 保护；kernel 入口确实从 `baseParams.ropeHeadDim` 分派；C8 路径未改。
- 分支清洁：两分支各自一个提交，工作区干净。

未做（本机无 CANN/torch_npu/NPU）：
- C++/AscendC 编译（需要服务器）。
- NPU 上的 UT / e2e / 模型。
- CMake 构建选择 UT（本机无 cmake，测试会自动 skip）。

## 6. 服务器验证步骤

### 6.1 环境记录（先做）
- 记录：CANN 版本、torch_npu、SOC（`ascend960dt/pr`）、AIC/AIV 实际核数、
  驱动、安装的 `cann_ops_transformer`/`npu_ops_transformer`。
- 确认 A6 编译时 `__CCE_AICORE__ == 310`（arch35 分支生效）：
  可在 `sparse_flash_attention.cpp` 临时加 `static_assert` 或编译日志确认。
- 基线对比：同一模型、同一输入、同一量化（GLM-5.3 w4a4c16）。

### 6.2 通用构建
```bash
# 以仓库实际安装方式为准；核心是让 build_aclnn 用 ascend960dt 编 SFA
bash csrc/build_aclnn.sh "$PWD" ascend960dt
# 或 python setup.py build_aclnn
```
B 分支应看到 `sparse_flash_attention` 出现在 A6 的 op 列表与编译产物中；
A 分支 A6 不应编译该 op（early-return 保留）。

### 6.3 方案 A 验证
1. 确认安装包内存在：
   ```python
   import torch, cann_ops_transformer  # noqa
   print(torch.ops.cann_ops_transformer.sparse_flash_mla)
   print(torch.ops.cann_ops_transformer.sparse_flash_mla_metadata)
   ```
2. 跑 UT：
   ```bash
   pytest tests/ut/attention/test_sfa_nope_metadata.py -k a6_uses_delivered_smla
   pytest tests/ut/attention/test_sfa_nope_forward.py -k zero_tnd
   ```
3. GLM-5.3-Flash C16：eager + FULL/PIECEWISE ACLGraph；构造 capacity shrink
   （如 128→64），断言 valid 行数值一致、尾部为 0。
4. 与 A5 同一 prompt 的 attention 输出/LSE 对照。
5. 重点观察 `generate_smla_plan` 是否因 A6 core 数与 A5 不同而失败或错位。

### 6.4 方案 B 验证
1. 编译：`sparse_flash_attention` 的 arch35 源码在 `ascend960dt` 下编译通过。
2. 注册优先级：确认 `ASCEND_CUSTOM_OPP_PATH` 中 `custom_transformer` 的
   `aclnnSparseFlashAttention` 生效（可临时 `A6`/`ASCEND_CUSTOM_OPP_PATH` 对照）。
3. UT/e2e：
   ```bash
   pytest tests/ut/_tools/test_a6_sfa_build.py        # 需要 cmake
   pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_attention.py \
          -k "nope_tnd_output_padding or tnd_output_padding"
   ```
4. 数值对照：
   - rope=64（Kimi K3 / 现有 MLA 用例）：in-tree vs 原 SDK 输出逐元素对比；
   - NoPE（GLM C16）：in-tree vs Python/CPU 参考；可选 vs 方案 A 的 SMLA 输出。
5. ACLGraph：先向输出填 NaN 再 replay，确认 valid 行不被污染、尾部为 0。
6. workspace/core：A6 走 `isA5_==false` 分支，确认不越界；记录 AIC/AIV 核数。
7. 性能：与 SDK arch92 / 方案 A 对比 prefill、decode、长序列。

### 6.5 A/B 对照方法
- 同一份 `hidden_states`、`topk_indices`、KV cache，分别跑两分支；
- dump `attention_out`（只比较 `[0, actual)` 行）+ 尾部；
- 若数值不同，优先分点：rope 拷贝、LSE、softmax 精度、tiling core 分档。

## 7. 待验证 / 风险清单

方案 A：
- [ ] CANN 9.3 包是否提供 `sparse_flash_mla` + metadata（名称/命名空间）。
- [ ] SMLA metadata 的 core 数与 A6 实际核数匹配（A5 是 64-core 假设）。
- [ ] A6 上 `use_smla` 打开后，所有 NoPE 层（含 MTP/draft）是否都走同一路径。
- [ ] 尾部：若交付算子本身已清零，host `masked_fill` 只是冗余（性能代价）。
- [ ] C8：`enable_sparse_sfa_c8` 仍关闭 SMLA，确认 QSFA 路径无回归。

方案 B：
- [ ] arch35 在 `ascend960dt/pr` 编译（`__CCE_AICORE__==310` 假设）。
- [ ] NO_ROPE 的 512 打包与 L1 P-tile 复用是否与预期一致（数值验证）。
- [ ] host tiling `isA5_==false` 的 workspace 公式是否覆盖 A6 实际核数。
- [ ] `s2BaseSize=128`、AIC:AIV=1:2 与 A6 是否一致。
- [ ] `ClearOutputTail` 的 `GetBlockNum/CV_RATIO` 分核是否覆盖全部 AIV。
- [ ] LSE（DCP）路径的尾部 softmax max/sum 目前只清 attention_out（与 QSFA 一致）。
- [ ] C8：QSFA 未改，确认 A6 上 QSFA + SFA 同时注册无冲突。

## 8. 命令速查

```bash
# 分支
git checkout br_poc_a6_1106              # B
git checkout br_poc_a6_1106_sfa_smla     # A

# Python 自检
python -m py_compile vllm_ascend/attention/sfa_v1.py
python -m ruff check vllm_ascend/attention/sfa_v1.py

# 构建（按实际环境）
bash csrc/build_aclnn.sh "$PWD" ascend960dt
# 或
python setup.py build_aclnn
```
