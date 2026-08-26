# SparseMode4 按列分核迁移对话记录

- 导出日期：2026-08-06
- 工作目录：`M:\Users\l00611801\Desktop\workpsace`
- 相关目录：
  - `swizzle\deter\no_tnd`
  - `ops-transformer\attention\flash_attention_score_grad`

> 说明：较早轮次已经被系统压缩，因此本文不能作为平台的逐字原始聊天导出。
> “用户请求”尽量保留当前上下文中可见的原文；“处理结果”根据可访问的工作摘要整理。
> 最近几轮内容来自当前可见对话，但仍以技术记录为目的进行了排版。

## 一、需求演进

### 1. 将 Batch-First 方案迁移到 arch35

用户请求：

> 按照M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\SparseMode4_通用斜带区域确定性分核需求及代码_v2_batch_first.md中的方案，迁移到M:\Users\l00611801\Desktop\workpsace\ops-transformer\attention\flash_attention_score_grad中的arch35实现，具体参考isSplitByBlockIdx分支实现

随后用户切换分支并补充：

> 同样按照M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\SparseMode4_通用斜带区域确定性分核需求及代码_v1.1.md中的方案，迁移到M:\Users\l00611801\Desktop\workpsace\ops-transformer\attention\flash_attention_score_grad中的arch35实现，具体参考isSplitByBlockIdx分支实现，代码已为你切换到新分支

之后代码被回退：

> 代码被我回退了 重新应用下

处理结果摘要：

- 在 `flash_attention_score_grad` 的 arch35 确定性 BAND 分支中迁移 Batch-First 排布。
- 参考 `isSplitByBlockIdx` 分支接入。
- 保留按 batch 串行和按列分核的基本约束。

### 2. 分析第一版方案性能问题

用户请求：

> 上述第一个方案性能很差，已排除负载均衡原因，帮我看一下还有哪些因素导致，算子代码已经恢复到第一个方案

处理结果摘要：

- 排除宏观核间负载不均后，重点分析了：
  - 有效任务在 round 维度过于碎片化；
  - 同步轮数过多；
  - 部分 round 活跃核数不足；
  - 坐标访问和工作集跳变；
  - 调度公式的整数运算、分支及索引开销。
- 结论是仅看总任务量负载均衡不足以解释性能，round 数和每轮有效核利用率同样关键。

### 3. 修改 Batch-First Python 方案

用户请求：

> 根据这个结论，对M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\SparseMode4_通用斜带区域确定性分核需求及代码_v2_batch_first.md中的python脚本需要做修改吗？

> 可以帮忙修改一下python脚本，原始脚本在M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first.py，可以新建一个脚本

第一次实现后用户指出：

> 实现不对，没有遵循按列分核

处理结果摘要：

- 新建 `band_batch_first_compact.py`。
- 修正后严格保持同一 batch 内同一物理列始终由同一个 core 负责。
- 继续沿用原 Batch-First 的：
  - 每 `k` 列分组；
  - 相邻列组 snake；
  - 每两组旋转；
  - batch 旋转。
- 固定列 owner 后，将有效块构造成 row/core 二分多重图，通过完美匹配分解得到全局最少轮数：

```text
rounds_per_batch = max(最大行任务数, 最大 core 任务数)
```

- 该 Python 版本是全局 matching 参考实现，需要保存 `round × core` 调度信息。

验证摘要：

- 关键场景 `k=m=n=32, p=q=31`：每 batch 轮数从 61 降到 32。
- 9,072 组小规模组合验证通过。
- 验证了覆盖、无重复、同行互斥、同核互斥、固定列 owner 和确定性。

### 4. 将方案迁移到 Kernel

用户请求：

> 参考上面实现迁移到M:\Users\l00611801\Desktop\workpsace\ops-transformer\attention\flash_attention_score_grad中的arch35实现

初始 Kernel 实现摘要：

- 当 `effectiveN <= k` 时，不使用 matching 表，而使用循环边着色解析公式：

```text
round = (x - y) mod compactRoundCount
compactRoundCount = min(p + q - 1, max(m, effectiveN))
```

- 当 `effectiveN > k` 时先回退旧的列组隔离方案。
- 保持按列 owner、snake 和 batch 旋转。

验证摘要：

- 17,317 个解析分支场景通过验证。
- 9,072 个解析/回退组合场景通过验证。
- 未发现可用的 Ascend/CANN 编译工具链，因此只完成算法镜像验证和 `git diff --check`。

### 5. matching 表大小与替代方案

用户提问：

> 如果没有按照python通用的方法，有什么弊端

处理结果摘要：

- 解析公式不一定复现 Python 全局完美匹配的 round 编号。
- 对 `effectiveN > k` 的一般情况，如果坚持全局最优 matching，需要保存或运行时生成全局调度。
- 不使用通用 matching 可能无法跨列组填补空槽，因此某些边界场景轮数高于全局最优。

用户提问：

> 一般什么样的场景会走到effectiveN <= k

处理结果摘要：

```text
effectiveN = min(n, m + q - 1)
```

当有效列数不超过参与分核的 core 数 `k` 时进入该解析分支，常见于较小 `n`、较窄有效右边界或较多 core 的场景。

用户提问：

> 如果按照通用方案需要增加多大的matching 表

处理结果摘要：

- 若每个表项用一个 `uint32` block index：

```text
matching_table_bytes = k × rounds_per_batch × 4
```

- 基础表可在不同 batch 间复用，不需要乘 `b`。
- 满方阵示例：

| k / m | 估算大小 |
|---:|---:|
| 32 | 4 KiB |
| 64 | 16 KiB |
| 128 | 64 KiB |
| 256 | 256 KiB |

用户进一步指出：

> m有可能扩展到很大，这个占用大小接受不了，有没有其他方案

提出的替代方案：按列组解析 matching。

```text
每组最多 k 列
delta_g = max(组内最大行度, 组内最大列/owner-core 度)
组内 round = (x - y) mod delta_g
总轮数 = b × sum(delta_g)
```

特点：

- 不保存随 `m` 增长的二维 matching 表。
- 列组之间使用不重叠的 round 区间。
- 保持固定 column owner。
- 当前 `effectiveN <= k` 公式是单列组特例。
- 代价是不能跨列组填补空槽，部分不规则边界场景不如全局 matching。

### 6. 实现多列组解析方案

用户请求：

> 实现这个方案

实现结果摘要：

- 修改文件：
  - `ops-transformer\attention\flash_attention_score_grad\op_kernel\arch35\deter.h`
  - `ops-transformer\attention\flash_attention_score_grad\op_kernel\arch35\flash_attention_score_grad_kernel_deter.h`
- 新增解析计算：
  - 区间最大重叠；
  - 每列组 `delta_g`；
  - 每 batch 总轮数；
  - 组内模着色。
- 所有 `effectiveN` 均可使用分列组解析方案，不再只有 `effectiveN <= k` 才压缩。
- 保留列组 snake、group rotation 和 batch rotation。
- 为避免每个 round 从第 0 组开始扫描，增加 4 个 `int64_t` 的单调列组缓存。
- `BandInfo` 中增加一个 `roundsPerBatch` 标量缓存。
- 新增状态约 40 字节，不随 `m` 增长。

验证摘要：

- 18,150 组参数穷举通过。
- 每个有效块恰好出现一次。
- 同一 round 无同行冲突和同核多任务。
- 列始终归属原 owner core。
- 每列组轮数等于该组理论下界。
- 500 组随机缓存状态切换通过。
- `git diff --check` 通过。
- 当前环境没有 Ascend/CANN 工具链，未执行设备侧编译。

## 二、当前方案适用性讨论

用户提问：

> 当前这个方案对哪种输入最有利，哪种最劣势

结论摘要：

```text
D = p + q - 1
R_old = 列组数 × D
R_new = sum(delta_g)
```

最有利：

- 宽斜带或接近 Dense；
- `D` 明显大于矩阵实际可容纳的行/列度；
- `effectiveN` 接近 `k` 的整数倍；
- 各列组任务饱满；
- batch 较多，可通过旋转均摊物理核负载。

最劣势：

- 大矩阵窄带，内部列的有效任务数就是 `D`，导致 `delta_g = D`；
- 尾部列组很小；
- 边界高度不规则但又不能跨组填槽；
- `b=1`，batch rotation 无法均摊边界负载；
- 列组极多但每组任务很少。

相对旧 Batch-First，轮数不会增加，因为每组始终有 `delta_g <= D`；最差是轮数持平并增加少量解析开销。

## 三、具体输入分析

用户提问：

> m=32,n=64,k=32,p=31,q=64这种场景会有空跑的轮次吗

分析结果：

```text
effectiveN = 64
列组数 = 2
D = 94
delta_0 = 32
delta_1 = 32
每 batch 总轮数 = 64
```

有效块总数：

```text
32 × 64 - 1 = 2047
```

唯一无效块为 `(x=32, y=1)`，因为 `x-y=31 > p-1=30`。

结论：

- 没有整轮完全空跑。
- 只有一轮为 31 个核活跃、1 个核空闲。
- 其他 63 轮满 32 核。
- 槽位利用率约为 `99.951%`。
- 旧方案需要 `2 × 94 = 188` 轮，当前方案为 64 轮。

## 四、Python 全局 matching 与 Kernel 排布差异

用户提问：

> 你改完的算子的版本和M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_compact.py中排布不一样是什么原因，比如我本地尝试了输入k, m, n, b = 32, 32, 64, 8 ;p, q = 32, 63的时候python脚本的轮次编号和算子内实现的轮次编号完全不一致，算子内实现的轮次存在很大跳变

分析结果：

- `band_batch_first_compact.py` 使用所有列组共同参与的全局二分多重图 perfect matching。
- Kernel 使用列组隔离的解析 matching。
- 二者可以有相同总轮数，但 round 的着色编号和边分解顺序不同。

对于：

```text
k=32, m=32, n=64, b=8, p=32, q=63
```

两种方案均为 64 轮每 batch，但：

Python 全局 matching 大致交替列组：

```text
round 1：y=1..32
round 2：y=64..33
round 3：y=1..32
round 4：y=64..33
...
```

Kernel 分组解析：

```text
round 1..32：y=1..32
round 33..64：y=64..33
```

core 1 的 Kernel 排布：

```text
r1  -> (x=1,  y=1)
...
r32 -> (x=32, y=1)
r33 -> (x=32, y=64)
r34 -> 空闲
r35 -> (x=2,  y=64)
...
```

跳变来源：

- `(x-y) mod delta_g` 的模回绕；
- 第 32/33 轮的列组切换；
- 奇数列组的 snake 反向映射。

缓存只优化组查找，不改变排布。

如果要求与 Python 全局 matching 的 round 编号逐项一致，就必须复现全局 perfect matching 的分解顺序，通常需要 matching 表或运行时匹配算法，这与“不接受随规模增长的 matching 表”相冲突。

## 五、新建与 Kernel 一致的 Python 脚本

用户请求：

> 新建一个脚本和kernel实现保持一致

实现结果：

- 新建：`band_batch_first_group_analytic.py`
- 严格镜像 Kernel：
  - `effectiveN`；
  - 每 `k` 列分组；
  - `delta_g`；
  - 分组 round 前缀；
  - snake/rotation；
  - `(x-y) mod delta_g`；
  - 1-based batch/core/round/坐标。
- 提供：
  - 单点 `(core, round) -> (batch, x, y)` 查询；
  - round/core 三维矩阵构建；
  - 覆盖、冲突、owner 和利用率校验；
  - 指定 core 的逐轮打印；
  - 命令行参数；
  - 可视化输出。

验证摘要：

- 7,776 组参数与独立 Kernel 公式逐项对拍通过。
- `py_compile` 通过。
- 默认输入结果：

```text
k=32, m=32, n=64, b=8, p=32, q=63
列组轮数：[32, 32]
每 batch：64 轮
总轮数：512
有效块：2047 / batch
满核轮：504
部分核轮：8
空轮：0
槽位利用率：99.951172%
```

## 六、可视化风格对齐

用户请求：

> 画图部分按照我的之前的脚本风格

实现结果：

- 将 `band_batch_first_group_analytic.py` 的画图风格对齐 `band_batch_first_compact.py`：
  - 相同 core 配色；
  - 每页多 batch；
  - 支持多行多列；
  - 相同网格、坐标字号；
  - 白色圆角框加粗 round 标注；
  - 相同分页标题和画布尺寸逻辑。
- 输出命名：

```text
FAG_band_group_analytic_batch_first_p{p}_q{q}_page_{page}.png
```

- 新增命令行参数：

```text
--batches-per-figure
--ncols
--dpi
--no-annotate-round
--show
```

- 完成 Python 语法检查和多页可视化冒烟测试。

## 七、当前相关文件

### 方案与参考脚本

- `M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\SparseMode4_通用斜带区域确定性分核需求及代码_v1.1.md`
- `M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\SparseMode4_通用斜带区域确定性分核需求及代码_v2_batch_first.md`
- `M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first.py`
- `M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_compact.py`
- `M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_group_analytic.py`

### Kernel 实现

- `M:\Users\l00611801\Desktop\workpsace\ops-transformer\attention\flash_attention_score_grad\op_kernel\arch35\deter.h`
- `M:\Users\l00611801\Desktop\workpsace\ops-transformer\attention\flash_attention_score_grad\op_kernel\arch35\flash_attention_score_grad_kernel_deter.h`

## 八、常用运行命令

仅校验并打印统计：

```powershell
python M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_group_analytic.py `
  --k 32 --m 32 --n 64 --b 8 --p 32 --q 63 `
  --no-plot
```

打印 core 1 的逐轮排布：

```powershell
python M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_group_analytic.py `
  --k 32 --m 32 --n 64 --b 8 --p 32 --q 63 `
  --no-plot --core 1
```

生成可视化：

```powershell
python M:\Users\l00611801\Desktop\workpsace\swizzle\deter\no_tnd\band_batch_first_group_analytic.py `
  --k 32 --m 32 --n 64 --b 8 --p 32 --q 63 `
  --batches-per-figure 2 --ncols 2 --dpi 260
```
