# FAG Arch35 Swizzle AI 交接手册

> 快照日期：2026-08-26  
> 工作区：`M:\Users\l00611801\Desktop\workpsace`  
> 目标：让后续 AI 能快速判断分支、修改 Host/Kernel/Python，并完成轮次、坐标、精度、确定性和性能校验。

## 1. 接手时先记住的结论

1. **当前代码永远是最终真值。** PDF、Python 和历史提交是设计依据与验证 oracle，但都可能落后于最新 Host 条件或性能门槛。开始修改前必须重新读取目标分支。
2. **先判断实际命中的模板和内部 layout，再分析 Swizzle。** 外部输入是 TND，不代表 Host 最终仍按 TND 处理；`SupportTrans2BS2N2GD` 可能将等长 TND（尤其 `B=1`）转换为非 TND 内部 layout。
3. **`isSplitByBlockIdx` 只表示是否启用 block-index 分核。** 不要把它复用成枚举 schedule mode。非 TND 确定性 BAND 的具体算法由独立字段 `deterBandScheduleMode` 表示。
4. **Host 和 Kernel 必须同时一致：** 分支条件、参数归一化、最大轮次、坐标、无效槽处理、索引基准、tiling split mode 转换，缺一不可。
5. **确定性不能只验证坐标覆盖。** 还要验证同轮 dQ 冲突、相邻轮流水冲突、dK/dV 列的跨核归属，以及 BN2S2 的同步区间。
6. **性能不能只看各核有效块数量。** 还要看总遍历轮次、槽位利用率、相对旧方案的轮次膨胀、Batch/Head 完成跨度和 KV 列跨核切分。
7. **非确定性方案不是确定性方案的一比一复制。** 非确定性允许同一 local step 出现同一行的原子累加，核心约束是覆盖和列归属，不需要套用确定性的同轮行唯一条件。

## 2. 当前代码快照与方案状态

以下信息只描述 2026-08-26 的工作区；分支可能继续变化，后续 AI 必须重新执行 `git status --short --branch` 和 `git log -1 --oneline`。

| 内容 | 仓库/版本 | 状态 |
| --- | --- | --- |
| 最新确定性非 TND BAND、GQA Dense、TND Dense Mode2 | `ops-transformer-zyj`，分支 `david_0820_tnd_dense_swi`，提交 `246a77cdb` | 当前确定性实现真值，快照时工作树干净 |
| 非确定性 Dense/Causal 严格按列分核 | `ops-transformer-xkbug` 的提交 `a7b1f837b`，分支 `david_0813` | 不在 `ops-transformer-xkbug` 当前 HEAD；接手时用 `git show` 或切到对应分支查看 |
| `ops-transformer-xkbug` 当前分支 | `david_0820_dpse`，提交 `9a33103eb` | 快照时有用户目录 `scripts/local/`，不要删除；该 HEAD 不是上述非确定性 Swizzle 真值 |
| 另一个历史仓 | `ops-transformer`，分支 `david_0822_oom`，提交 `a1523220e` | 有未跟踪文件 `arch35_tiling_code_review.pdf`，不要删除；不作为本文当前实现真值 |

重要提交：

| 提交 | 含义 |
| --- | --- |
| `246a77cdb` | 确定性 TND Dense Swizzle |
| `6c9f40483` | 确定性 GQA Dense Swizzle |
| `249fe87d6` | 确定性 BAND Swizzle 正式实现 |
| `a7b1f837b` | 非确定性大 Shape Dense/Causal 严格按列 Swizzle |

查看非确定性历史实现：

```powershell
cd M:\Users\l00611801\Desktop\workpsace\ops-transformer-xkbug
git show --stat a7b1f837b
git show a7b1f837b:attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_base.h
```

不要为了查看历史实现而覆盖用户当前工作树；优先用 `git show`、独立 worktree 或明确的新分支。

## 3. 目录与代码入口

### 3.1 确定性当前实现

仓库根目录：`M:\Users\l00611801\Desktop\workpsace\ops-transformer-zyj`

| 层次 | 文件 | 作用 |
| --- | --- | --- |
| Host 常量/声明 | `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h` | Swizzle 阈值、公共结构和声明 |
| Host 分支与选择器 | `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_normal_regbase.cpp` | `SelectBlockSchedule`、`SelectDeterBandSchedule`、`SelectGQADenseSchedule`、TND 安全/性能门槛 |
| Host 确定性轮次 | `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp` | 旧确定性轮次、prefix、同步相关计算 |
| TND Host | `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_varlen_regbase.cpp` | TND shape、prefix 和模板相关处理 |
| Tiling data | `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h` | `DeterBandScheduleMode`、`isSplitByBlockIdx`、`deterMaxRound`、同步数组等 ABI |
| Kernel 坐标算法 | `attention/flash_attention_score_grad/op_kernel/arch35/deter.h` | Dense/Causal/BAND/GQA/TND 的坐标函数 |
| Kernel 分发/轮次/同步 | `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_deter.h` | `CalBandDeterIndex`、`CalDeterMaxLoopNum`、`DeterSync` |
| Tiling key | `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_template_tiling_key.h` | 模板与编译键选择 |

### 3.2 Python 与方案文档

根目录：`M:\Users\l00611801\Desktop\workpsace\swizzle`

| 场景 | Python 真值/参考 | 文档 |
| --- | --- | --- |
| 非 TND、NoGQA、BAND 混合选择 | `deter/no_tnd/NoGQA/band_hybrid.py` | `deter/no_tnd/NoGQA/band_hybrid.pdf` |
| 非 TND、NoGQA、非等长 Causal | `deter/no_tnd/NoGQA/casual.py` | `deter/no_tnd/NoGQA/FAG确定性计算_LEFT_UP_CAUSAL非等长Swizzle方案.pdf` |
| 非 TND、GQA Dense | `deter/no_tnd/GQA/FAG_Dense_GQA_0813_v1.py` | `deter/no_tnd/GQA/batch-GQA-dense.pdf` 等 |
| 非 TND、GQA Causal/BAND | `deter/no_tnd/GQA/FAG_Causal_GQA_0915.py`、`FAG_Band_GQA_0906.py` | 同目录 PDF |
| TND Dense | `deter/tnd/FAG_Dense_TND_0105_deter_line.py` | 脚本图与历史方案 |
| 非确定性 Dense/Causal 严格按列 | `no_deter/non_deter_swizzle.py` | `no_deter/FAG非确定性Swizzle分核方案_Dense_Causal.md`、PDF、`VALIDATION.md` |
| SparseMode4 实验 | `deter/no_tnd/gpt/` | 多为实验方案，不能直接视为生产真值 |

注意：

- 文件名和已有变量中使用了历史拼写 `casual` / `enableCasualSwizzle`，语义实际是 causal。除非任务明确要求接口重构，否则保留已有标识符，避免扩大改动。
- `deter/no_tnd/NoGQA/README.md` 当前为空，不应作为依据。
- 当前 `band_hybrid.py` 能验证核心算法，但**尚未完整体现最新 C++ 的 90% 槽位门槛和 20% 轮次膨胀门槛**。
- TND Python Mode2 与 Kernel 坐标一致，但脚本本身没有完整模拟最新 Host 的 60% 槽位门槛和 20% 轮次膨胀门槛。

## 4. 统一术语和变量含义

不同脚本的字母含义曾经混用，开始分析前必须建立一张实际映射表。

| 符号 | 常见含义 |
| --- | --- |
| `k` | 使用的 AIC 核数；Kernel 中经常由 `coreNum / 2` 得到 |
| `m` | S1 基本块数，通常为 `s1Outer` |
| `n` | S2 基本块数，通常为 `s2Outer` |
| `p` | BAND 中每列向下/左侧有效宽度参数，来自 `s1Token` 归一化 |
| `q` | BAND 中每列向上/右侧有效宽度参数，来自 `s2Token` 归一化 |
| `b` | 算法中的扁平 batch 数；NoGQA 非 TND 常为 `(B-tailZeroCount)*N2`，不要误当原始 B |
| `g` / `G` / 脚本 `N1` | 每个 KV head 对应的 query head 分组数；具体脚本先看函数签名 |
| `n1` | Kernel TND 常为 `n2*g` |
| `j` | 核编号；多数确定性数学函数使用 1-based，非确定性脚本常用 0-based |
| `r` | 轮次；多数确定性数学函数使用 1-based，外层 `roundId` 和非确定性脚本常用 0-based |
| `batchId,s1Idx,s2Idx` | 基本块坐标；进入 Kernel 通用路径后通常会转换成 0-based |

必须同时记录：

- `cubeBaseM = s1Inner * s1CvRatio`
- `cubeBaseN = s2Inner * s2CvRatio`
- `m = ceil(actualS1/cubeBaseM)`
- `n = ceil(actualS2/cubeBaseN)`
- `splitAxis` 是 `BN2GS1S2` 还是 `BN2S2`
- `layoutType` 是外部输入值还是 Host 转换后的内部值
- `sparseMode`、`sparseType`、`deterSparseType` 三者不是同一个概念

## 5. 一次 Swizzle 的完整数据流

```text
输入 shape/layout/sparseMode/token
        ↓
Host shape 与 sparse 归一化（可能发生 TND → BS2N2GD）
        ↓
选择 splitAxis、cube base、blockOuter、enableSwizzle
        ↓
选择 isSplitByBlockIdx / deterBandScheduleMode / TND isTndSwizzle
        ↓
Host 计算 prefix、deterMaxRound、同步区间并写入 tiling data
        ↓
tiling key 选择 Kernel 模板
        ↓
Kernel 计算 maxLoopNum
        ↓
(core, round) → (batch/head, s1, s2)
        ↓
mask/边界过滤、确定性同步、实际计算
```

任何性能问题都先沿这条链确认真实分支。只根据用户输入参数或某个 Python 图推断，容易把非 Swizzle、另一模板或内部转 layout 的结果误认为目标算法性能。

## 6. 当前 Host 分支选择规则

### 6.1 非 TND 公共前提

`SelectBlockSchedule` 中的 `canSplitByBlockIdx` 当前等价于：

```text
enableSwizzle
&& layoutType != TND
&& splitAxis == BN2GS1S2
&& s1Inner*s1CvRatio == s2Inner*s2CvRatio
&& sparseType != UNSUPPORTED
```

非确定性场景直接把这个结果写入 `isSplitByBlockIdx`；实际 Dense/Causal 的 Kernel 入口还会检查 `enableDenseSwizzle` 或 `enableCasualSwizzle`。

### 6.2 非 TND 确定性旧 Dense/Causal/RIGHT_DOWN 路径

旧路径还要求：

```text
(B*N2) 为偶数
&& g == 1
&& s1 >= aicNum*128
&& deterSparseType 属于 DETER_DENSE / DETER_CAUSAL / 特定 RIGHT_DOWN BAND
```

当前 `causalCond` 直接依据 `deterSparseType == DETER_CAUSAL`。因此不要再次用原始 `sparseMode` 人为收窄；`sparseMode=0/2` 在 token 归一化后可能成为可支持的 DETER_CAUSAL 或 DETER_BAND。

RIGHT_DOWN_CAUSAL 有一个必须保留的早返回：如果它已满足旧 `isSplitByBlockIdx` 条件，则保持 `deterBandScheduleMode=DISABLED`，Kernel 继续走旧 `CalCausalSwizzleIndex`。历史上把该场景重新选择成 Hybrid DENSE 会增加轮次并造成性能回退。

### 6.3 非 TND 确定性 Hybrid BAND

在旧路径未早返回后，Hybrid BAND 入口要求：

```text
canSplitByBlockIdx
&& deterSparseType == DETER_BAND
&& g == 1
&& coreNum == 2*aicNum
&& actualBatch=(B-tailZeroCount)*N2 > 0
```

然后执行：

1. token 转块：`p = ceil(s1Token/cubeBase)+1`，`q = ceil(s2Token/cubeBase)+1`；负 token 的除法语义必须和 C++ `CeilDivideBy` 一致。
2. `NormalizeDeterBandScheduleParams` 做 shifted BAND 转换。
3. `SelectDeterBandSchedule` 在 BAND、满足安全条件的 DENSE、满足浪费条件的 lower CAUSAL 中选最少轮次方案。
4. 通过槽位利用率和相对旧实现轮次膨胀门槛后，设置 `isSplitByBlockIdx=true` 及独立的 `deterBandScheduleMode`。
5. 任一门槛失败时返回 `DISABLED`，继续旧 `GenBandInfo + CalBandIndex`。

`DeterBandScheduleMode` 的 ABI 值为：

```text
DISABLED=0, CAUSAL=1, DENSE=2, BAND=3
```

### 6.4 TND 确定性 Dense Mode2

入口核心条件：

```text
enableSwizzle
&& 内部 layoutType == TND
&& deterministic
&& splitAxis 属于 BN2GS1S2 / BN2S2
&& deterSparseType == DETER_DENSE
&& g == 1
&& IsTndDeterSwizzleSupported
&& B < TND_SWIZZLE_PREFIX_NUM
&& 不存在零长度 seq
&& tailZeroCount == 0
```

其中：

```text
enableSwizzle = (CheckExceedL2Cache() || CheckIsLargeInvalidBlk())
                && blockOuter == aicNum
```

`IsTndDeterSwizzleSupported` 同时检查精度安全和性能门槛：

- 每个 batch 必须满足 `m >= min(k,n)`；这是 Mode2 同轮 dQ 行唯一的必要条件。Python 注释中的 `m>=n` 只是 `k>=n` 时的特例。
- 聚合槽位利用率至少 60%。
- 候选最大轮次相对旧方案最多增长 20%。

### 6.5 GQA Dense 的 R 调整

`SelectGQADenseSchedule` 只在 `g>1` 时尝试调整：

```text
usedK = min(k, b*g*m, b*n)
baseR = max(ceil(b*n*g/usedK), ceil(n/m), g)
candidateR = ceil(baseR/g)*g
```

只有同时满足以下条件才从 `baseR` 调整到 `candidateR`：

- 轮次增量不超过 3%。
- 无效 ID 比例不超过 4%。
- `batchHeadNum/gcd(batchHeadNum,R)` 形成的遍历周期确实缩短，且小于 `k`。
- `ceil(k/candidatePeriod) <= m`，S1 行偏移足够。

R 的作用不是单纯减少总轮次，而是改善 Batch/Head 在轮次上的聚集程度，减少 KV 分组跨核切分并改善 L2 局部性。

## 7. 坐标与轮次公式

### 7.1 BAND 参数归一化必须 Host/Kernel 相同

先执行：

```text
p = min(p,m)
q = min(q,n)

if p < 0: (m,n,p,q) = (m,n+p,1,p+q)
if q < 0: (m,n,p,q) = (m+q,n,p+q,1)
```

Hybrid 内部还执行：

```text
m = min(m,n+p-1)
n = min(n,m+q-1)
```

这几步的顺序不可改变。Host 选择器、Kernel `CalDeterMaxLoopNum`、`GenBandHybridInfo` 和 Python 必须使用相同的有效 `m,n,p,q`。

### 7.2 Hybrid BAND

当 `p+q<=m`：

```text
L1 = q-1
L2 = min(n-q+1, m+2-p-q)
L3 = max(0, min(p+n-m-1, p+q-2))
bandBlocks = (2p-2+q)*L1/2
             + (p+q-1)*L2
             + (p+q-2)*L3
             - L3*(L3-1)/2
```

当 `p+q>m`：

```text
L1 = m-p
L2 = p+q-m
L3 = min(n-q,m-1)
bandBlocks = (p+m-1)*L1/2
             + m*L2
             + (2m-1-L3)*L3/2
```

列压缩与轮次：

```text
pairCount = max(0,min(L1,L3-p+1))
colsPerBatch = L1+L2+L3-pairCount
slot = p+q-1
bandMaxRound = ceil(b*colsPerBatch/k)*slot
```

Kernel `CalBandHybridIndex` 采用 1-based `j,r`：

```text
layer = (r-1)/slot
localRound = (r-1)%slot+1
globalColumn = layer*k+j
w = (globalColumn-1)/colsPerBatch+1
localColumn = (globalColumn-1)%colsPerBatch+1
x = y+localRound-q
有效区间：max(1,y-q+1) <= x <= min(m,y+p-1)
```

L1/L3 的部分短列会配对进同一个固定长度 slot。修改配对规则时必须同时验证每个原始 S2 列的唯一核归属。

### 7.3 Hybrid 中的 DENSE 候选

```text
denseK = min(k,m*b)
安全前提：denseK == k && min(denseK,n) <= m
denseMaxRound = ceil(n*b/denseK)*m
```

安全前提的本质是同轮活跃列映射到足够多的 S1 行，避免确定性 dQ 冲突；它比简单写成 `m>=k` 更宽松。

### 7.4 Hybrid 中的 lower CAUSAL 候选

```text
causalSize = m+q-1
lowerWaste = causalSize*(causalSize+1)/2-bandBlocks
upperWaste = (n+p-1)*(n+p)/2-bandBlocks
```

只使用 lower embedding，且要求（浪费门槛是源码采用的严格整数形式，不能随意改成浮点 10% 比较）：

```text
lowerWaste >= 0
lowerWaste <= upperWaste
lowerWaste <= (bandBlocks-1)/10
存在 batch pair
causalK == k
causalK <= causalSize
```

偶数 batch pair 使用 `CalCausalSwizzleIndex`，奇数尾 batch 使用 `CalCausalSingleBatchDeterIndex`。之后把 `s1` 减去 `q-1`，再按原 BAND 边界过滤。

### 7.5 旧 BAND 最大轮次

性能比较中的旧方案轮次必须与 Kernel `GenBandInfo` 完全一致：

```text
rm2 = p+q>m
    ? m*ceil(n*b/min(k,b*m))
    : ceil(n*b/k)*(p+q-1)

legacyMaxRound = max(hostDeterMaxRound,rm2)
```

不能只拿 Host 原始 `deterMaxRound` 比较，否则可能低估 Kernel 实际执行轮次。

### 7.6 TND Dense Mode2

每个 batch：

```text
columnNum = n*N2*g
batchRound = ceil(columnNum/k)*m
prefix[b+1] = prefix[b]+batchRound
```

将 Kernel 传入的 1-based `j,r` 减一后：

```text
delta = r-prefix[batch]
linear = (delta/m)*k+j
if linear >= n*(N2*g): invalid

head = linear/n
s2 = linear%n
s1 = (s2+delta)%m
```

这与 `FAG_Dense_TND_0105_deter_line.py` 的 `get_dense_batch_position_v2` 对应，也就是脚本 `mode_chosen>1` 的 Mode2，而不是 Mode1。

Mode2 的特点：

- 一个 `(batch,head,s2)` 完整列由同一核在连续 `m` 个 round phase 中完成。
- S1 通过 `(s2+delta)%m` 旋转，满足条件时同轮不重复写同一 dQ 行。
- 同一 batch/head 的 S1S2 更集中，有利于 L2；Mode1 的 head/S2 排列会破坏这一局部性。

### 7.7 非确定性严格按列 Dense

实现快照：`a7b1f837b`。

```text
flatBatchCount = B*N2*g
totalColumns = flatBatchCount*n
maxOwnedColumns = ceil(totalColumns/k)
swizzleMaxRound = maxOwnedColumns*m

localColumn = loopIdx/m
s1 = loopIdx%m
flatColumn = localColumn*k+coreId
flatBatch = flatColumn/n
s2 = flatColumn%n
```

每个 `(flatBatch,s2)` 完整列只有一个 owner core。这里**不增加 `g==1` 限制**，因为 flat batch 已包含 `g`，且非确定性允许多个 query group 通过原子操作更新同一物理 KV 梯度。

### 7.8 非确定性严格按列 Causal

当前范围是方形 lower triangle，`m==n`：

- 两个 flat batch 配成一个 `m*(m+1)` 无 padding 矩形，再将完整虚拟列 round-robin 分给核。
- 奇数尾 batch 从 `pairColumnCount%k` 开始 round-robin 分配真实三角列。
- 每个核的任务数按自己拥有的完整列长度累计，`swizzleMaxRound` 取所有核最大值。
- 理论负载差不超过最长列 `m`。

这一方案的约束是完整列唯一 owner，不要求同一个 local step 的 S1 行唯一。

## 8. BN2S2 为什么需要 DeterSync

TND Mode2 扩展到 `SPLIT_AXIS=BN2S2` 时，下标算法与 `BN2GS1S2` 一致，但同步需求不同。

Host `ConfigureTndDeterBn2S2Swizzle` 做两件事：

1. 将 `deterPrefix2` 填为 `-1`。Mode2 已保证每个 `(B,N2,S2)` 列固定在一个核，因此 dK/dV 不需要跨核尾部 merge。
2. 设置 `startNeedSyncRound[0]=1`，`endNeedSyncRound[0]=TND prefix end`。不同列仍可能原子累加到同一 dQ，必须约束连续轮次的累加顺序。

Kernel 仅在 `SPLIT_AXIS==BN2S2` 的编译分支调用 `DeterSync(loopIdx)`。原始 BN2S2 非 Swizzle 并不是无条件每轮同步，而是按 Host 计算的冲突区间同步；Mode2 重排后当前配置覆盖整个 Swizzle 轮次区间。

因此 BN2S2 Swizzle 对空槽特别敏感：无效槽不做有效计算，但仍会遍历轮次并承担全轮次同步开销。历史板测中 3.125% 槽位利用率用例回退约 21 倍；这也是保留硬槽位门槛的直接原因。

## 9. 精度和确定性必须验证的约束

### 9.1 坐标集合

对所有 `(core,round)` 穷举，构造实际有效坐标集合，与理论有效集合比较：

```text
missing == 0
duplicate == 0
extra == 0
outOfRange == 0
```

注意 duplicate 是“基本块是否被计算两次”，不等价于原子目标冲突。BAND 的 padding 坐标必须在进入实际计算前正确过滤。

### 9.2 dQ 确定性

至少验证：

- 同一个确定性 round 内，不同核不能命中同一物理 `(B,N1,S1)` dQ 行。
- 不能只看单轮；流水可能让 `round r` 与 `round r+1` 同时在途。必须按 Kernel 的 `JudgeIsNeedDeter` / 同步协议验证相邻轮冲突。
- TND Mode2 的安全条件 `m>=min(k,n)` 不能因为“随机测试没撞上”而删除。

### 9.3 dK/dV 确定性

- 理想情况：一个物理 `(B,N2,S2)` 列只归一个核。
- GQA 中多个 query group 对应同一个 KV head，脚本坐标的 `head` 不能直接当作物理 KV owner；必须折算到 `N2`。
- 如果列跨核，必须证明已有确定性 workspace、尾部 merge 或同步协议覆盖该冲突。
- BN2S2 中“dK/dV 不需 merge”和“dQ 需 DeterSync”是两件独立的事，不能互相替代。

### 9.4 轮次边界

逐项核对：

- Host 序列化轮次与 Kernel `CalDeterMaxLoopNum` 返回值一致。
- `round=0/1`、`core=0/1` 的转换一致。
- 第一轮、最后一轮有效；`maxRound+1` 必须无效。
- `DETER_TILING_SPLIT_MODE` 对 round 的拆分/合并及 `TransDeterRound`、`TransTilingSplitModeBack` 一致。
- 奇偶 batch、尾 batch、尾核和无效槽都覆盖。

## 10. 性能保护指标

### 10.1 槽位利用率

```text
slotUtilization = validBlockCount/(k*candidateMaxRound)
```

它衡量候选方案自身有多少 `(core,round)` 槽位真正计算有效块。低利用率意味着大量 round 只做索引、判断或同步。

当前阈值：

| 场景 | 最低利用率 |
| --- | --- |
| TND 确定性 Dense Swizzle | 60% |
| 非 TND 确定性 Hybrid BAND | 90% |

### 10.2 轮次膨胀率

```text
roundGrowth = candidateMaxRound/legacyMaxRound-1
```

它回答“新方案相对原方案是否回退”。当前 TND 和 Hybrid BAND 都限制在最多增长 20%。

两者区别：

- 槽位利用率是候选方案的绝对效率。
- 轮次膨胀率是相对旧方案的回退程度。
- 候选槽位较低但仍可能比旧方案轮次少；反之槽位看似不差，也可能因 batch 串行化而比旧方案慢。

当前 Hybrid BAND 的 90% 利用率本身已很强。在总有效工作相同的理想模型里，它大致把候选轮次限制在理论下界的 1.111 倍以内，因此 20% 膨胀门槛经常成为冗余保险。若后续要扩大覆盖，优先考虑降低槽位阈值并保留相对轮次门槛；但 BN2S2 全轮同步场景仍建议保留较硬的利用率下限。

### 10.3 还应记录的指标

单纯 `max(load)-min(load)` 不足以描述性能，建议 Python 每个 case 输出：

- `candidateRound`、`legacyRound`、round growth。
- valid / invalid / total slots 和 utilization。
- 每核 valid block 的 min/max/avg/P95。
- 每个 batch/head 的首轮、末轮和 completion span。
- 同一时刻活跃 batch 数。
- 每个物理 KV 列的 owner core 数、跨核列数量。
- 相邻轮命中同一 batch/head/列的比例，用于近似 L2 局部性。
- 每 batch 的最差槽位利用率，防止大 batch 掩盖小 batch 的病态尾组。
- BN2S2 的同步轮数/有效块数。
- D、Dv 尾块、mask、dropout 等导致的块成本差异；基本块数量相同不代表耗时完全相同。

## 11. 已知典型案例与结论

### 11.1 Hybrid BAND 严重回退案例

外部输入：

```text
B=1, N1=N2=32
seqQ=1024, seqKv=8192
D=192, Dv=128
sparseMode=3, layout=TND, deterministic=true
```

关键事实：B=1 等长 TND 被 Host 转成非 TND 内部 layout，最终进入 Hybrid BAND。归一化后约为：

```text
k=32,m=8,n=64,p=8,q=57,b=32
```

旧结果与新结果：

```text
Hybrid BAND maxRound = 4096
valid coordinates     = 15488
total slots           = 32*4096 = 131072
slot utilization      = 11.816%
invalid slots         = 115584

legacy Host round     = 484
legacy Kernel rm2     = 512
legacy actual round   = 512
round growth          = 8x
```

坐标覆盖是正确的：15488 个有效坐标无缺失、无重复；问题是性能而不是精度。当前 90% 利用率和 20% round growth 门槛都会将它退回旧实现。

### 11.2 Hybrid 高利用率参考案例

| `(k,m,n,p,q,b)` | 候选模式 | 候选/旧轮次 | 利用率 |
| --- | --- | --- | --- |
| `(28,55,55,55,1,13)` | CAUSAL | `715/1430` | 100% |
| `(28,55,55,55,55,2)` | DENSE | `220/220` | 约 98.214% |
| `(28,55,55,3,3,2)` | BAND | `20/55` | 约 96.071% |

这些用例适合做“门槛不能误杀正常收益 case”的回归样本。

### 11.3 TND Mode2 参考案例

```text
k=32,m=[63,2],n=[64,2],N1=16
```

每个 batch 满足 `m>=min(k,n)`，且总槽位满载，可进入 Mode2（仍需满足其余 Host/模板条件）。

```text
k=32,m=[63,2],n=[64,3],N1=16
```

短 batch 不满足 `2>=min(32,3)`，不能直接使用当前确定性 Mode2，否则存在 dQ 同轮冲突风险。若要优化这类 mixed case，需要设计按 batch 混合 schedule 或拆分短 batch，不能只放宽条件。

### 11.4 BN2S2 槽位实测

- 3.125% 槽位利用率 P0：约 21 倍回退，全轮次同步开销非常严重。
- 满槽 P1/P2：有明显收益，P2 加速约 39%～42%。
- 历史 mixed 测试实际走的是非 Swizzle tiling key，约 0～1.4% 差异不能作为 Swizzle 性能结论。

### 11.5 GQA Dense 的 28 核与 32 核

`k=32` 往往更容易让同一个 batch 内不同 head 在相近轮次完成，符合最初的 L2 局部性设计；`k=28` 在某些 `b*g` 与 R 的组合下遍历周期变长，同一 batch 的 head 被分散到较远轮次，且物理 KV 分组可能跨更多核。

这不是简单的“少 4 个核所以慢一点”。核数与 `b*g`、R 的 gcd/周期关系可能引起离散跃迁。R 调整只在局部性收益明显、轮次和空槽成本受控时启用。

## 12. 常见误区和已经踩过的坑

1. **只看外部 layout。** TND 可能已被 Host 转为非 TND。
2. **只看 `sparseMode`。** 真正的确定性分支经常依据归一化后的 `deterSparseType`。
3. **把 `isSplitByBlockIdx` 当枚举。** 会破坏可读性和既有语义；使用独立 mode 字段。
4. **在 `SaveToTilingData` 才决定 schedule。** 分支选择应在清晰的 selector/configure 函数完成，Save 阶段只序列化结果。
5. **Host 只算一个轮次，Kernel 又按另一公式重算。** 任何一侧修改都要搜索对应公式并双向核对。
6. **只验证总坐标 coverage。** 覆盖正确仍可能不确定；必须验证 dQ/dKV 冲突和相邻轮。
7. **只验证负载均衡。** Hybrid 回退 case 的每核有效负载并不离谱，但所有核仍遍历 4096 轮。
8. **用平均槽位掩盖病态 batch。** TND mixed length 应额外看 worst-batch utilization。
9. **认为 28 核与 32 核只差 12.5% 算力。** round-robin 周期、整除性和 L2 局部性可能完全变化。
10. **随意增加或删除 `g==1`。** 确定性方案的 `g==1` 常与物理 dKV 冲突有关；非确定性严格按列方案明确不需要该限制。
11. **把确定性公式直接搬到非确定性。** 非确定性允许同 local step 重复行，约束集合不同。
12. **用脚本的一条注释代替数学条件。** 例如 TND `m>=n` 只是 `m>=min(k,n)` 的特例。
13. **性能数据未确认 tiling key。** 必须先证明板测确实进入目标模板和 Swizzle 分支。
14. **删掉 RIGHT_DOWN 旧路径早返回。** 可能让本来走高性能 `CalCausalSwizzleIndex` 的 case 被重选成较慢 Hybrid DENSE。
15. **自动格式化或回退覆盖用户改动。** 工作树可能有未提交文件；只修改任务相关文件，禁止使用 `git reset --hard`。

## 13. 标准开发流程

### 阶段 A：确认分支和真实入口

1. 记录目标仓库、分支、HEAD 和工作树状态。
2. 从 Host 的 shape/sparse 归一化开始，追踪内部 layout、splitAxis、cube base。
3. 确认 `enableSwizzle`、`isSplitByBlockIdx`、`deterBandScheduleMode`、`isTndSwizzle` 和 tiling key。
4. 找到 Kernel 实际调用的坐标函数和 max-loop 函数。

建议诊断时记录：

```text
externalLayout, internalLayoutType, splitAxis
isDeterministic, sparseMode, sparseType, deterSparseType
B, N2, g, s1, s2, m, n, p, q
aicNum, coreNum, blockOuter
enableSwizzle, isSplitByBlockIdx, deterBandScheduleMode
deterMaxRound, tndPrefixEnd, tilingKey
```

临时日志仅用于诊断；无关的 valid/total slot 明细不应长期污染算子日志。

### 阶段 B：先做 Python oracle

1. 精确复刻 Host 的参数归一化和分支条件。
2. 精确复刻 Kernel 的 `(core,round)→coordinate`，明确 0/1-based。
3. 生成理论有效坐标集合。
4. 穷举和随机验证覆盖、唯一性、边界、确定性冲突。
5. 同时输出候选/旧轮次与性能指标。
6. 用图检查 Batch/Head 完成跨度和 L2 局部性，但图不是正确性证明。

不要只写能画图的脚本；脚本必须能无交互 self-test，并在失败时给出首个反例。

### 阶段 C：落 Host

1. 把选择逻辑放进具名 selector/configure helper，而不是塞进 `SaveToTilingData`。
2. 魔鬼数字放公共头文件 `flash_attention_score_grad_tiling_common_regbase.h`，命名体现具体场景和单位。
3. `isSplitByBlockIdx` 保持 bool 语义；新增算法用明确 enum/field。
4. 先算 candidate，再过精度安全条件和性能门槛，最后写入参数结构。
5. 选择失败必须自然回退旧路径，不能留下半初始化 schedule mode/round。

### 阶段 D：落 Kernel

1. 入口条件应直接对应 Host 字段或现有模板 flag。
2. Round 公式尽量只从 Host 读取；如果 Kernel 必须重算，要写注释说明必须与哪个 Host 函数一致。
3. 坐标函数保持纯粹：输入参数、输出 coordinate、无效返回。
4. 明确 schedule mode 的 DISABLED fallback。
5. 检查 split mode 的 round 转换、TND/非 TND 的 batch/head 解码和最后的 0-based 转换。

### 阶段 E：一致性验证

对代表性与边界 case 自动比较 Python 和 C++ 公式的：

- 分支选择结果。
- schedule mode。
- max round。
- 每个 `(core,round)` 的有效/无效状态。
- 有效坐标 `(batch,n2,g,s1,s2)`。
- 第一轮、最后一轮和尾 batch。

如果无法直接运行 Kernel，可写一个仅复刻 C++ 整数公式的独立 checker，并逐字段对照源码；不能用同一个 Python 函数同时充当“预期”和“实际”。

### 阶段 F：板测

至少准备：

- 高利用率、应使能并有收益的正例。
- 低利用率、应回退的负例。
- 阈值两侧边界。
- `k=28/32`。
- 奇偶 batch 和奇数尾 batch。
- `g=1` 与 `g>1`。
- `m<n`、`m=n`、`m>n`。
- sparseMode `0/2/3/4` 及 token 边界。
- TND mixed length。
- `BN2GS1S2` 与 `BN2S2`。

每条性能结论必须附实际 tiling key/分支证据。

## 14. 当前可执行验证命令

```powershell
cd M:\Users\l00611801\Desktop\workpsace\swizzle

python .\deter\no_tnd\NoGQA\band_hybrid.py --full-test
python .\deter\no_tnd\NoGQA\casual.py --full-test
python .\no_deter\non_deter_swizzle.py --full-test --demo
python .\deter\no_tnd\GQA\FAG_Dense_GQA_0813_v1.py --k 28 --m 58 --n 38 --b 10 --g 3
```

2026-08-26 本机复验结果：

```text
band_hybrid.py: PASS full-test cases=292 coordinates=203968
casual.py:      PASS full-test cases=16384 coordinates=3198720
non_deter:      Dense 4968 cases / 6574396 coordinates
                Causal 920 cases / 4065476 coordinates
                全部满足严格列唯一 owner
GQA v1 case:   baseR=41 → selectedR=42
                duplicate/missing/dq_conflict/dkv_conflict 均为 0
                split_kv 由 18 降为 0
```

`FAG_Dense_TND_0105_deter_line.py` 当前在模块尾部直接生成示例图，不是完整 CLI self-test；接手时建议先将可视化入口放入 `if __name__ == "__main__"`，再补无图形依赖的验证入口。

本 Windows 环境过去存在完整构建链不可用、全量 pre-commit 依赖 WSL/bash 的情况。应至少执行目标文件的 clang-format/pre-commit；如果完整构建因环境失败，要明确区分“环境未具备”与“代码验证失败”，不能宣称已完成编译验证。

## 15. Python、Host、Kernel 一致性检查表

提交前逐项打勾：

- [ ] Python 使用与 Host 相同的输入字段和整数类型语义。
- [ ] 负数 token 的 ceil-div 与 C++ 一致。
- [ ] p/q clamp、shift 和 m/n crop 顺序一致。
- [ ] Host 分支条件没有在 Python 中遗漏 layout、splitAxis、g、coreNum 等限制。
- [ ] Host schedule mode 与 Kernel enum 数值一致。
- [ ] `isSplitByBlockIdx` 未被用作多值 mode。
- [ ] Host candidate round 与 Kernel max-loop 完全一致。
- [ ] 旧方案比较轮次包含 Kernel 的 `rm2` 等二次上界。
- [ ] 0/1-based core、round、batch、s1、s2 转换一致。
- [ ] Tiling split mode 的 round 变换一致。
- [ ] 所有理论有效基本块恰好出现一次。
- [ ] 所有 padding/越界坐标最终无效。
- [ ] 确定性 dQ 同轮与相邻轮无未同步冲突。
- [ ] 确定性 dKV 列归属或 merge 协议正确。
- [ ] BN2S2 同步范围覆盖新 schedule，且空槽成本受门槛保护。
- [ ] 槽位、round growth、负载、局部性指标均已输出。
- [ ] 板测确认实际 tiling key 和目标分支。
- [ ] 未覆盖用户工作树中的无关修改和未跟踪文件。

## 16. 后续可改进项

1. 将最新的 90% Hybrid BAND 槽位门槛和 20% round growth 门槛同步回 `band_hybrid.py` 与 PDF，做到分支选择一一对应。
2. 给 TND Mode2 脚本增加 Host 入口模拟、60% 槽位门槛、20% round growth、BN2S2 同步成本和无图 self-test。
3. 建立统一 `schedule_oracle`：输入完整算子参数，先模拟 Host 内部 layout/sparse/template，再模拟 Kernel 坐标，而不是每个脚本只接收已经归一化的 `k,m,n...`。
4. 增加 worst-batch utilization、Batch/Head completion span 和 KV owner 数，避免只用平均槽位与 max/min load。
5. 将典型正例、负例和历史性能回退 case 固化成回归 JSON，供 Python、Host UT 和板测共用。
6. 为 `FAG_Dense_TND_0105_deter_line.py` 去掉 import 即执行的绘图副作用，并修复历史编码乱码注释。
7. GQA Causal/BAND 的 Python 与当前生产代码需要重新做一次逐函数审计，不能仅凭文件存在就假设仍一一对应。

## 17. 给下一个 AI 的推荐任务开场模板

可以把下面内容与具体需求一起发给接手 AI：

```text
请先完整阅读：
M:\Users\l00611801\Desktop\workpsace\swizzle\FAG_Swizzle_AI交接手册.md

目标仓库：<填写仓库>
目标分支/提交：<填写分支和HEAD>
目标场景：<确定性/非确定性，TND/非TND，Dense/Causal/BAND，GQA/NoGQA>
输入或问题：<填写参数与现象>

要求：
1. 先确认内部 layout、splitAxis、tiling key 和实际命中的分支，不根据外部输入直接推断。
2. 先更新/新增 Python oracle，自测覆盖、唯一性、边界、轮次和坐标。
3. 确定性额外验证 dQ 同轮/相邻轮冲突、dKV 列归属和同步协议。
4. 同时报告槽位利用率、候选/旧轮次、轮次膨胀、各核负载、Batch/Head 完成跨度和 KV 跨核切分。
5. Host 与 Kernel 同时修改并逐公式校验；isSplitByBlockIdx 保持 bool 语义。
6. 保留用户未提交改动，不使用破坏性 git 命令。
7. 完成后给出改动文件、分支条件、验证命令、验证结果和未完成风险。
```

## 18. 最后原则

Swizzle 的目标不只是“把所有基本块平均分给核”，而是同时满足四件事：

```text
正确覆盖
+ 确定性约束（若开启）
+ 可控的轮次/同步开销
+ Batch、Head、S2 列的局部性
```

任何新方案如果只优化其中一个指标，都容易在另一类 shape 上形成严重回退。最可靠的工作方式始终是：**先还原真实分支，再建立独立 Python oracle，最后让 Host 条件、Kernel 轮次和坐标逐项对齐。**
