# A6 arch35 SFA：1:1 混核移植评审与本分支实现

> 分支：`br_poc_a6_1106_arch35_a6cv11`（基线 `br_poc_a6_1106` @ `a2844b50b`，含 NoPE 改动）
> 目的：评审 `HANDOFF.md` / `方案评审稿` 的思路与执行方案，并给出可上机验证的移植版本。

---

## 1. 结论先说

**方向判断正确，执行方案不成立。**

- 「先证明 arch35 在 A6 能算完一次，再谈尾部清零」——正确，且他们实验发现
  `T=236`（无 padding）也挂，推到「两层问题」——正确。
- 但他们的三处 A6 改动（`CV_RATIO=1`、`KERNEL_TYPE_MIX_AIC_1_1`、2D LoadData）
  **不足以让 arch35 跑起来**，因为 arch35 的 1:2 语义不只写在 `CV_RATIO` 里，
  而是分布在 7 个层面；其中一层（跨核 flag）会直接造成他们观测到的
  「64 个 AIC 全部停在同一个 PC 等 flag」。
- 他们的两个「已排除」结论**无效**，不能作为后续决策依据：
  - `SFA_SERIAL_MODE=1 仍挂`：串行化没有修掉 §2.1 的双 flag 等待，必挂，不能说明串行路线不可行。
  - `GetSubBlockIdx() 强制 0 仍挂`：同样没碰 §2.1，`subBlockIdx` 不是这个死锁的变量。
- 他们认为最可能的 H1（M 维语义）、H4（UB 越界）方向对，但**排序反了**：
  H4 在 1:1 下是**结构性**的（`CV_RATIO=1` 会让 UB 需求约 377 KB），不是「偶发 341」。

## 2. arch35 的 1:2 契约到底写在哪（逐层，带代码位置）

| # | 层面 | 位置 | 1:1 下必须怎么改 |
|---|---|---|---|
| L1 | **跨核 flag 协议** | `csrc/attention/common/op_kernel/buffer.h`：`AIV0_AIV1_OFFSET=16`；`WaitCrossCore`/`SetCrossCore` 中 AIC 侧对每个 id 做**两次** `CrossCore{Wait,Set}Flag(id)` 与 `(id+16)` | AIC 只做一次。**不改必死锁**：`id+16` 永远没人置位 ⇒ 64 个 AIC 全停在同一条 wait（正是 F5 的现象） |
| L2 | **核映射** | `kernel_mla.h` `Init()`：AIV 侧 `aicIdx = aivIdx >> 1` | `aicIdx = aivIdx`。否则 AIV 2p/2p+1 都算作 AIC p 的分区，AIC p≥32 找不到对端，同样表现成 F5 |
| L3 | **M 语义** | `kvcache.h` `ComputeSouterParam`：`halfMRealSize=(mRealSize+1)>>1`，`subBlockIdx==1` 取后半 | 单 AIV 要覆盖整个 M 块（或分趟处理，见 §3） |
| L4 | **Fixpipe 双目标** | `service_cube_mla.h` 两处 `dualDstCtl = 1`（`M/2` 写两个 AIV 的 UB） | 1:1 必须是单目标（`0`）；1:1 下不存在「另一个 AIV 的 UB」 |
| L5 | **s2 的 KV 搬运切分** | `service_vector_mla.h` `CalSparseCalSize`：`subBlockIdx==0 ? 前半 : 后半` | 单 AIV 搬全部 s2；否则 cube 的 bmm1 会读到未初始化的一半 K |
| L6 | **host 启动配置** | `op_host/...tiling.cpp` `CalcBlockDim`：`aivNum = 2 * usedCoreNum_` | `aivNum = usedCoreNum_`。请求 128 个 AIV 但只有 64 个 ⇒ 后一半 AIC 永久等待（这就是 `errcode 259` 的来源，而不是「1:2 模式非法」） |
| L7 | **host 分支** | `isA5_ = (npuArch_ == DAV_3510)`；A6 走 `else`（A2/A3 的 workspace 布局），但 kernel 是 arch35 | A6 必须走 arch35 的 workspace 分支；否则 v0Res 三缓冲区域对不上 |

另外两处他们没提到、但会影响 A6：

- `matmul.h` 的 2D/3D 选择其实已经是一个现成开关
  （`#if ((__CCE_AICORE__ == 310) || ... || (__NPU_ARCH__ == 5102))`），
  `#else` 才是 3D。他们的 F2 不需要拷 128 行，只要把 `9201/9202` 加进这个条件。
- `sparse_flash_attention.cpp` 用 `#if (__CCE_AICORE__ == 310)` 选 arch35/arch22。
  同仓 `chunk_gated_delta_rule_fwd_h_vllm.cpp` 的写法是
  `#elif (__CCE_AICORE__ == 310) || (__NPU_ARCH__ == 9201)`，说明 A6 未必满足
  `__CCE_AICORE__ == 310`。这一条不确认，可能一直在编 arch22。

## 3. 为什么 `CV_RATIO = 1` 不是答案：UB 预算是结构性的

arch35 的 vector 缓冲按 `s1BaseSize / CV_RATIO = 32` 行申请，正是为了 UB 装得下。
`T=float`（arch35 的中间累加是 fp32），`s1BaseSize=64`，`dSizeV=512`：

| buffer | `CV_RATIO=2`（现状） | `CV_RATIO=1`（对方方案） |
|---|---|---|
| `mm1ResultSize × 2` | 2 × 16 KB = 32 KB | 2 × 32 KB = 64 KB |
| `mm2ResultSize`（bmm2 结果） | 64 KB | 128 KB |
| `stage2OutBuf`（运行累加器） | 64 KB | 128 KB |
| `stage0OutBuf × 2` | ~37 KB | ~37 KB |
| `stage1OutQue × 2` | ~17 KB | ~33 KB（`vec1Srcstride` 也要 33→65） |
| **合计** | **≈ 214 KB** | **≈ 390 KB** |

`stage2OutBuf` 与 `bmm2Res` 在 `FlashUpdateNew` 里**同时存活**，不能复用。
所以只要 A6 单核 UB < 390 KB，`CV_RATIO=1 + halfM=mRealSize` 必然 `InitBuffer` 失败或
越界（他们偶发的 `errcode 341` 与之一致）。这不是「调参」，是重排流水。

## 4. 本分支的做法：保留 1:2 缓冲，单 AIV 串行跑两趟

核心思路：**不碰任何 buffer 尺寸，把「原先两个 AIV 各跑一半」改成「一个 AIV 依次跑两趟」**。
每一趟在数学/布局上与原来的 AIV0 / AIV1 **完全等价**，因此：

- cube 侧 M=32、L0C/UB 尺寸不变；
- P tile 的 NZ 布局按原 AIV0/AIV1 的槽位写入（偏移 `16 * passStart`，`dstStride` 用
  `Align16(fullM) - halfM`）——与 1:2 的每个 AIV 逐字节一致；
- softmax 的 `sum/max` 缓冲仍按 `multiCoreInnerIdx % 2` 乒乓，两趟天然交替；
- 尾部清零（`InitOutputSingleCore` / `ClearOutputTail`）按 `AIV_PER_AIC` 分核。

### 4.1 新增常量

`sparse_flash_attention_common_arch35.h`

```cpp
constexpr uint32_t CV_RATIO = 2;        // 缓冲/流水行数，保持 2（所有平台）
#if __NPU_ARCH__ == 9201 || 9202
constexpr uint32_t AIV_PER_AIC = 1;     // 一个 AIC 配一个 AIV
#define FA_AIV_PER_AIC 1                // 供 common/op_kernel/buffer.h 使用
#else
constexpr uint32_t AIV_PER_AIC = 2;
#define FA_AIV_PER_AIC 2
#endif
```

`ConstInfo::mPasses`（`util_regbase.h`）= A6 且 `mBase >= 2` 时为 2，否则 1。
`mBase = gSize`（`IS_SPLIT_G` 时 `gSize/2`）。

### 4.2 改动清单

| 文件 | 改动 |
|---|---|
| `common/op_kernel/buffer.h` | 用 `FA_AIV_PER_AIC` 门控所有 `+ AIV0_AIV1_OFFSET` 的第二个 flag |
| `common/op_kernel/matmul.h` | 2D LoadData 的 `#if` 加入 `__NPU_ARCH__ == 9201/9202`（替代对方拷入的 128 行） |
| `arch35/sparse_flash_attention_common_arch35.h` | 新增 `AIV_PER_AIC` / `FA_AIV_PER_AIC`；`CV_RATIO` 保持 2 并加注释 |
| `arch35/util_regbase.h` | `RunParamStr/RunInfo` 加 `fullMRealSize`；`ConstInfo` 加 `mPasses` |
| `arch35/sparse_flash_attention_kernel_mla.h` | `subBlockIdx=0`、`aicIdx=aivIdx`（1:1）；`ProcessMainLoop` 把 `(gS1, pass)` 展平成一个流水节拍，`PRELOAD_NUM` 排空按流水单位追加；`ComputeAxisIdxByBnAndGs1` 加 `mHalf` 并给 `goIdx` 加趟偏移；`maxS2LoopCnt *= mPasses` |
| `arch35/sparse_flash_attention_kvcache.h` | `ComputeSouterParam(…, mHalf)` 按趟切 M；`halfMRealSize = 本趟行数`，`firstHalfMRealSize = passStart`；LSE 偏移按趟加 `passStart` |
| `arch35/sparse_flash_attention_service_cube_mla.h` | `dualDstCtl = (AIV_PER_AIC==2)`；bmm2 的 P 左矩阵基址加 `16 * firstHalfMRealSize`（1:1） |
| `arch35/sparse_flash_attention_service_vector_mla.h` | `CalSparseCalSize` 1:1 覆盖全部 s2；P 写入位置/`dstStride` 用 `passStart`/`fullM`；清零分核用 `AIV_PER_AIC`；`InitCubeVecSharedParams` 1:1 必然发布 |
| `op_kernel/sparse_flash_attention.cpp` | A6 `KERNEL_TYPE_MIX_AIC_1_1`；arch35 选择条件加 `__NPU_ARCH__ == 9201/9202` |
| `op_host/sparse_flash_attention_tiling.{h,cpp}` | 新增 `isA6`（`DAV_9201/9202`，白名单补 9202）；A6 走 arch35 workspace 分支；`CalcBlockDim` 的 `aivNum = (isA6?1:2)*usedCoreNum_` |

### 4.3 与对方「三处改动」的对应关系

| 对方改动 | 本分支 | 说明 |
|---|---|---|
| `CV_RATIO = 1` | **不采纳** | 会撑爆 UB（§3）；用 `AIV_PER_AIC` + `mPasses` 表达 1:1 |
| `KERNEL_TYPE_MIX_AIC_1_1` | 采纳 | `sparse_flash_attention.cpp` |
| `matmul.h` 拷入 2D LoadData | 采纳（更小） | 直接把 9201/9202 加入已有 2D 条件 |

## 5. 待验证（按顺序，每步留证据）

1. **确认编译目标**：A6 下 `sparse_flash_attention.cpp` 走的是 `SFA_ARCH35`
   （`__CCE_AICORE__==310` 或 `__NPU_ARCH__==9201/9202`）。在 `Init()` 里临时
   `printf` 或 `static_assert` 打一次块号/`AIV_PER_AIC`。
2. **启动配置**：`SFA block dim` 日志应为 `aiv Num: 64 aic Num: 64`（T=236 时
   `usedCoreNum_=64`），不再是 128/64。
3. **T=236 能算完**：先跑 `pad_probe.py`（无 padding），确认不再
   `timeout or trap`。若仍挂，按 §2 的 L1→L7 逐条核对实际生成的二进制里
   flag 次数是否为 1 次（`plog` 的 `Error symbol` / PC 是否还停在同一处）。
4. **数值**：`attention(236)` 与 arch92 包逐元素对比；重点看
   - P tile 布局（§4.1 的偏移公式）——若有效行数值错而尾部对，优先查这里；
   - LSE（`return_softmax_lse=True`）的趟偏移。
5. **padding**：`attention(276)` 的 `[236:]` 全 0，且 `[0:236]` 与
   `attention(236)` 逐位一致（`atol=0`）。
6. **ACLGraph**：先填 NaN 再 replay，确认 valid 行不被污染、尾部为 0。
7. **rope=64 回归**：非 NoPE 用例（Kimi/MLA）与 1:2 平台输出一致。
8. **C8**：QSFA 未改（它有自己的 `arch35/common/buffer.h` 与 CV_RATIO=2），
   在 A6 上确认 QSFA 仍可启动；若它也 1:2 启动失败，说明 §2 L6 的结论
   对 QSFA 同样成立，需要同样的 host 侧 `aivNum` 修正（未在本分支改）。
9. **性能**：单 AIV 串行两趟 ⇒ K/V 搬运约 2×，prefill 性能预计下降。
   正确性通过后再决定是否做「bmm1/softmax 保持 64 行、仅 bmm2/vec2 分两趟」的优化版。

## 6. 风险与已知未覆盖

- **P tile 布局公式**（`16*passStart` + `Align16(fullM)`）是从 1:2 的两趟写入
  反推的等价形式，未在硬件上验证；若有效行数值错，这是第一嫌疑。
- `IS_SPLIT_G`（gSize=128）路径同样按 `mPasses=2` 展开，但该路径整体
  只做了静态推导，没有单独的 128-head 用例。
- `SyncAll<false>()` 在 1:1 的核数语义沿用原实现，未改。
- 性能回退（§5.9）。
- **CMake 打包**：本分支沿用方案的 CMake 结论（A6 编 in-tree SFA）。
  对方的 `binary_info_config.json` / `tilingKey 33926` 问题与本分支无关：
  本分支没有新增模板轴（NoPE 走运行时 `ropeHeadDim` 分派），tiling key 集合不变。
- **若 A6 上确实无法 1:1 启动 arch35**：Plan B 仍是「A6 走交付包 arch92 +
  共享尾部 mask」（即 `br_poc_a6_1106_sfa_smla` 分支），该路线已实测可用。

## 7. 命令速查

```bash
git checkout br_poc_a6_1106_arch35_a6cv11

# 编译（按服务端实际脚本）
SFA_SKIP_VOID_OP_CHECK=1 BASE=$PWD PKG=/path/to/pkg \
  BUILD_SH=csrc/build_aclnn.sh bash base_build.sh

# 上卡
REFONLY=1 python3 pad_probe.py     # T=236
PADONLY=1 python3 pad_probe.py     # T=276
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_attention.py \
  -k tnd_output_padding --noconftest
```
