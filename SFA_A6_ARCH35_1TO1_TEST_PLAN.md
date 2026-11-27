# A6 arch35 SFA（1:1 移植）编译 / 测试命令单

> 分支：`br_poc_a6_1106_arch35_a6cv11`（基线 `a2844b50b`）
> 设计说明见同目录 `SFA_A6_ARCH35_1TO1_PORT.md`；本文件只给可执行步骤。
> 文中 `/home/hyh_sfa/...` 全部是**上次交接里的实际路径**，换机请整体替换。

---

## 0. 一句话目标

让 A6 上的 in-tree arch35 SFA 以 **64 AIC : 64 AIV（1:1）** 启动，T=236 能算完且数值正确，
T=276 的尾部全 0。若 T=236 仍 `timeout or trap`，按 §9 的表定位。

---

## 1. 取分支（不要 `git clean`）

沿用 `poc_repo` 最省事，先把**只与构建有关**的两处本地补丁存下来，再切分支。

```bash
cd /home/hyh_sfa/poc_repo

# 1) 先把两个本地构建补丁单独存出来（aicpu 符号关卡 + 空 bin 目录检查）
git diff -- csrc/CMakeLists.txt csrc/cmake > /home/hyh_sfa/build_local.patch
wc -l /home/hyh_sfa/build_local.patch        # 应非 0

# 2) 丢掉被本分支取代的改动（matmul.h 的 128 行拷贝、CV_RATIO=1 等）
git checkout -- csrc

# 3) 切到本分支
git fetch origin
git checkout -B br_poc_a6_1106_arch35_a6cv11 origin/br_poc_a6_1106_arch35_a6cv11

# 4) 只恢复构建补丁
git apply /home/hyh_sfa/build_local.patch
git status --short
# 期望只剩： csrc/CMakeLists.txt  csrc/cmake/scripts/util/ascendc_ops_config.py
```

> `csrc/third_party/*`（catlass 等）是未跟踪文件，**别 git clean**。
> 若必须用新 worktree：`git worktree add /home/hyh_sfa/poc_a6 br_poc_a6_1106_arch35_a6cv11`，
> 再 `cp -a /home/hyh_sfa/poc_repo/csrc/third_party/* 新树/csrc/third_party/` 并 apply 上面的补丁。

---

## 2. 环境

```bash
export PATH=/home/miniconda3/envs/hyh_ld/bin:$PATH
source /home/zhangyang/Ascend/cann-9.3.0/set_env.sh
export PYTHONPATH=/home/hyh_sfa/poc_repo
cd /home/hyh_sfa/poc_repo
```

`ASCEND_CUSTOM_OPP_PATH` **只在跑算子时设**，且必须指到 `vendors/custom_transformer` 这一层（上次的坑）：

```bash
PKG=/home/hyh_sfa/sfa_pkg_a6cv11
V=$PKG/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$V/op_api/lib:$LD_LIBRARY_PATH
```

---

## 3. 编译（只编 SFA）

```bash
cd /home/hyh_sfa
SFA_SKIP_VOID_OP_CHECK=1 \
BASE=/home/hyh_sfa/poc_repo \
PKG=/home/hyh_sfa/sfa_pkg_a6cv11 \
BUILD_SH=csrc/build_sfa_only.sh \
bash /home/hyh_sfa/base_build.sh 2>&1 | tee /home/hyh_sfa/build_a6cv11.log
```

判定：

```bash
grep -nE "Built target sparse_flash_attention|BUILD_DONE|pass1_rc|install_rc|sfa_bins" \
  /home/hyh_sfa/build_a6cv11.log | tail -20
ls -l /home/hyh_sfa/poc_repo/csrc/build/binary/ascend960dt/bin/sparse_flash_attention/
```

- 期望：`[100%] Built target sparse_flash_attention`，产物 `.o` 数量与旧包一致（NoPE 走运行时
  `ropeHeadDim` 分派，**没有新增模板轴/tiling key**）。
- 若 `install_rc=0` 且有 `.run`：直接装，跳到 §4。
- 若打包仍被无关算子拖停：走 §4 的手工覆盖。

---

## 4. 安装

### 4.1 优先：跑完打包

```bash
grep -n "install_rc" /home/hyh_sfa/build_a6cv11.log | tail -3
# install_rc=0 即可；PKG 目录下应有 vendors/custom_transformer
```

### 4.2 备选：手工覆盖（在旧包副本上）

```bash
PKG=/home/hyh_sfa/sfa_pkg_a6cv11
V=$PKG/vendors/custom_transformer
K=$V/op_impl/ai_core/tbe/kernel/ascend960dt/sparse_flash_attention

cp -a /home/hyh_sfa/sfa_pkg_repo2 $PKG                      # 以旧包为底
cp -a /home/hyh_sfa/poc_repo/csrc/build/binary/ascend960dt/bin/sparse_flash_attention/SparseFlashAttention_* \
      $K/
# tiling 库必须取 custom 目录下那份（两份 md5 不同）
cp /home/hyh_sfa/poc_repo/csrc/build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so \
   $V/op_impl/ai_core/tbe/op_tiling/liboptiling.so
md5sum $V/op_impl/ai_core/tbe/op_tiling/liboptiling.so \
       /home/hyh_sfa/poc_repo/csrc/build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so
```

`Cannot find tilingKey[33926]` 仍出现时，说明 `binary_info_config.json` 与本次产物的 key 集合不一致：

```bash
# 找到本次生成的配置并合并（cmake 目标名：gen_binary_info_config_json）
find /home/hyh_sfa/poc_repo/csrc/build -name binary_info_config.json -newer /home/hyh_sfa/build_local.patch
# 若仓库里已有 SFA 的 tiling key 列表，可只把 SFA 段合并进
# $V/op_impl/ai_core/tbe/kernel/config/ascend960dt/binary_info_config.json
```

---

## 5. 上卡闸门

```bash
npu-smi info -t proc-mem -i 5
# 必须看到 "No process in device." 才继续
export ASCEND_RT_VISIBLE_DEVICES=5
```

---

## 6. 三个必看信号（本分支专属）

### 6.1 host 侧：启动配置必须是 64/64

```bash
grep -rh "SFA block dim" $HOME/ascend/log/debug/plog/ 2>/dev/null | tail -3
# 期望： ... aiv Num: 64 aic Num: 64.
# 若仍是 aiv Num: 128 ⇒ host 的 isA6 没生效（DAV_9201/9202），回去查 tiling 库是否覆盖成功
```

### 6.2 确认编进去的是 arch35（可选，临时插桩）

`op_kernel/arch35/sparse_flash_attention_kernel_mla.h`，`Init()` 里 `ComputeConstexpr();` 之后加：

```cpp
#if defined(SFA_ARCH35)
    if (ASCEND_IS_AIV) {
        if (GetBlockIdx() == 0) {
            __asc_aicore::printf("SFA_ARCH35 aiv=%d mPasses=%d sub=%d\n",
                (int)constInfo.aivIdx, (int)constInfo.mPasses, (int)constInfo.subBlockIdx);
        }
    }
#endif
```

文件顶部 `#include "utils/debug/asc_printf.h"`。只打 block 0，避免环缓冲被冲掉。
期望：`mPasses=2 sub=0`（heads=64 的用例）。

### 6.3 若仍挂：AIC 是否还等两个 flag

```bash
grep -h "Error symbol\|errcode" $HOME/ascend/log/debug/plog/*.log | tail -40
```
- 1:1 正确时，AIC 不应再出现「两个不同 PC 的同一 wait 各 64 次」。
- 仍是两个 PC ⇒ `FA_AIV_PER_AIC` 没生效（buffer.h 的宏没吃到，检查 include 顺序）。

---

## 7. 探针（不依赖模型，最快）

把下面脚本存成 `/home/hyh_sfa/probe_a6_sfa.py`。

```python
import os, sys, torch, torch_npu  # noqa
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.manual_seed(53)
dev = "npu"
heads = int(os.getenv("HEADS", "64"))
page = 128
valid = int(os.getenv("VALID", "236"))
pad = int(os.getenv("PAD", "40"))
mode = int(os.getenv("MODE", "3"))          # 3: causal-ish (rope 用例)；0: dense (NoPE 用例)
nope = os.getenv("NOPE", "0") == "1"
dt = torch.bfloat16
scale = 1 / 24

pages = torch.randn(2, page, 1, 512, dtype=dt, device=dev)
pages[1].fill_(2)
block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=dev)
idx = torch.full((valid + pad, 1, 2048), -1, dtype=torch.int32, device=dev)
idx[:valid, 0, :page] = torch.arange(page, dtype=torch.int32, device=dev)
q = torch.randn(valid + pad, heads, 512, dtype=dt, device=dev)
qr = None if nope else q.new_zeros((valid + pad, heads, 64))
kr = None if nope else pages.new_zeros((2, page, 1, 64))
first = int(os.getenv("FIRST_END", "120"))
ql = torch.tensor([first, valid], dtype=torch.int32, device=dev)
kl = torch.tensor([page, page], dtype=torch.int32, device=dev)

def run(rows):
    out, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q[:rows], key=pages, value=pages, sparse_indices=idx[:rows],
        scale_value=scale, sparse_block_size=1, block_table=block_table,
        actual_seq_lengths_query=ql, actual_seq_lengths_kv=kl,
        query_rope=None if nope else qr[:rows], key_rope=kr,
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=mode,
        attention_mode=2, return_softmax_lse=False)
    return out

ref = run(valid)
print("REF_NAN", int(torch.isnan(ref).sum()), "REF_NZ", int(torch.count_nonzero(ref[:first])))
padded = run(valid + pad)
same = torch.equal(padded[:valid], ref)
tail_nz = int(torch.count_nonzero(padded[valid:]))
print(f"VALID_EQ={same} TAIL_NZ={tail_nz} SHAPE={tuple(padded.shape)}")
if os.getenv("SAVE"):
    torch.save(padded.cpu(), os.getenv("SAVE"))
sys.exit(0 if (same and tail_nz == 0) else 2)
```

运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
export ASCEND_CUSTOM_OPP_PATH=$V LD_LIBRARY_PATH=$V/op_api/lib:$LD_LIBRARY_PATH

# A) NoPE（GLM C16 形态，dense mask，可与 CPU 对拍）
NOPE=1 MODE=0 HEADS=8 VALID=40 PAD=8 FIRST_END=40 python3 probe_a6_sfa.py
# B) rope=64（MLA 常规形态）
NOPE=0 MODE=3 HEADS=64 VALID=236 PAD=40 python3 probe_a6_sfa.py
```

判据：`VALID_EQ=True` 且 `TAIL_NZ=0`，退出码 0。
只看 traceback 出现源码行会**误判成功**（上次的坑），必须 grep 到 `VALID_EQ=`/`TAIL_NZ=`。

---

## 8. 官方 UT

```bash
cd /home/hyh_sfa/poc_repo
export ASCEND_RT_VISIBLE_DEVICES=5
export ASCEND_CUSTOM_OPP_PATH=$V LD_LIBRARY_PATH=$V/op_api/lib:$LD_LIBRARY_PATH

# 本分支的两条 A6 用例（均 skipif != A6）
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_attention.py \
  -k tnd_output_padding --noconftest
# 期望：2 passed
```

对照（A6 上会 skip 的用例，别误判成失败）：
- `test_sparse_flash_attention_padded_indices_lse` —— 仅 A5；
- QSFA 的 `test_rope0_compact_matches_padded_bitwise` 等 —— 仅 A2/A3/A5。

C8 冒烟（A6 上无 skip）：

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_vllm.py \
  -k "test_kv_quant_sparse_flash_attention_vllm" --noconftest
```
> 注意：QSFA 仍是 1:2 启动。若它也报 259/超时，属预期缺口（host 的 `aivNum` 修正本分支只落在 SFA），
> 记录后跳过，不要据此判断 SFA 失败。

---

## 9. A/B 数值对照（in-tree vs SDK）

同一个探针跑两遍，只切 `ASCEND_CUSTOM_OPP_PATH`：

```bash
# in-tree（本分支）
ASCEND_CUSTOM_OPP_PATH=$V SAVE=/tmp/intree.pt NOPE=0 MODE=3 HEADS=64 VALID=236 PAD=40 \
  python3 probe_a6_sfa.py
# SDK（不设 custom 路径 ⇒ 解析到 CANN 的 arch92 实现）
env -u ASCEND_CUSTOM_OPP_PATH SAVE=/tmp/sdk.pt NOPE=0 MODE=3 HEADS=64 VALID=236 PAD=40 \
  python3 probe_a6_sfa.py
python3 - <<'PY'
import torch
a=torch.load("/tmp/intree.pt"); b=torch.load("/tmp/sdk.pt")
print("max_abs", (a.float()-b.float()).abs().max().item())
print("tail_intree", int(torch.count_nonzero(a[236:])), "tail_sdk", int(torch.count_nonzero(b[236:])))
PY
```

- `max_abs == 0` 是最理想（期望能达到，两边算法路径一致）；
- 若只在尾部不同 ⇒ 尾部清零位置问题（我们的 `needInit`/`ClearOutputTail` 路径）；
- 若有效行有差 ⇒ 回到 §6.2 的 P tile 趟偏移。

---

## 10. ACLGraph 重放

```python
import torch
out = run(valid + pad)
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    out = run(valid + pad)
out.fill_(float("nan"))
g.replay()
assert torch.equal(out[:valid], ref), "valid rows polluted"
assert int(torch.count_nonzero(out[valid:])) == 0, "tail not cleared"
print("GRAPH_OK")
```

---

## 11. 失败归因表

| 现象 | 最可能层 | 先查 |
|---|---|---|
| `SFA block dim` 仍是 `aiv Num: 128` | L6 host | tiling 库是否为 `custom/.../liboptiling.so`；`isA6` 是否命中 `DAV_9201/9202` |
| 64 AIC 停在两个不同 PC 的同一条 wait | L1 flag | `FA_AIV_PER_AIC` 是否在 `buffer.h` 之前定义为 1（编译单元 include 顺序） |
| AIC 停在一个 PC、AIV 停在一个 PC | L2 核映射 | `aicIdx == aivIdx` 是否生效 |
| `errcode 180` / MTE 不对齐 | 2D LoadData | `matmul.h` 第 86 行条件是否包含 9201/9202 |
| `errcode 341` / UB 越界 | 缓冲 | 本分支不再改 `CV_RATIO`；检查是否误用了旧包/旧 `.o` |
| T=236 能算完但有效行数值错 | P tile 趟偏移 | §4.1 的 `16*passStart` / `Align16(fullM)-halfM` |
| 有效行对、尾部非 0 | 尾部清零 | `needInit` 分支与 `ClearOutputTail`；`cuSeqlensQ` 是否为 nullptr |
| LSE 错位（开了 `return_softmax_lse`） | LSE 趟偏移 | `LoopSOuterOffsetInit` 里 A6 的 `+= firstHalfMRealSize` |
| 只有 gSize=128 的模型错 | IS_SPLIT_G | `mPasses` 与 `goIdx%2` 的组合；单独造 128-head 用例 |

---

## 12. 记录清单（每轮）

- `.o` md5、`liboptiling.so` md5、包路径；
- `npu-smi` 设备占用、plog 的 `errcode` 与 `Error symbol`；
- `SFA block dim` 日志行；
- 探针的 `VALID_EQ` / `TAIL_NZ` / `REF_NAN`；
- UT 的 passed/skipped 分类（A5-only、A2/A3/A5-only 的 skip 不算失败）。
