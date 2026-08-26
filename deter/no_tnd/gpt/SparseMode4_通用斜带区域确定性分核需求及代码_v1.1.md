# SparseMode4 通用斜带区域确定性分核需求

> **文档说明**
>
> 本文档将需求说明和对应的完整实现代码放在同一个 Markdown 文件中。
> 当前版本为 `v1.1`，不包含测试代码。
>
> 后续每次代码修改均应同步更新本文档中的需求、变更说明和完整代码。
>
> 数学公式统一使用 `$...$`（行内）和 `$$...$$`（独立公式），提高 GitHub、VS Code、Typora 等 Markdown 预览器的兼容性。

---

## 1. 需求背景

当前脚本实现了 FlashAttentionScoreGrad 算子在非 TND、causal 下三角场景下的确定性分核。

每个 batch 包含一个形状为 `[m, n]` 的任务块矩阵，其中矩阵行坐标记为 `x`，列坐标记为 `y`。当前实现只处理 causal 下三角区域，其有效块满足：

$$
y \le x
$$

现需要在当前 causal 分核逻辑的基础上，将固定的下三角有效区域扩展为由参数 `p` 和 `q` 控制的通用斜带有效区域。

当前 causal 下三角场景应当作为新需求的一个特例：

$$
p=m,\qquad q=1
$$

---

## 2. 新增输入参数

在现有参数：

```python
m, n, b, core_id, round_id, k
```

基础上新增：

```python
p, q
```

参数含义如下。

### 2.1 参数 `p`

`p` 表示从 `[m, n]` 矩阵左上角开始，沿第一列向下连续计算，有多少个块属于有效区域。

即第一列中：

```text
(1, 1)
(2, 1)
...
(p, 1)
```

均为有效块。

通过位置 `(p, 1)` 绘制一条向右下方向延伸的 45 度斜线，该斜线作为有效区域的下边界，斜线上的块属于有效块。

参数范围：

$$
1\le p\le m
$$

### 2.2 参数 `q`

`q` 表示从 `[m, n]` 矩阵左上角开始，沿第一行向右连续计算，有多少个块属于有效区域。

即第一行中：

```text
(1, 1)
(1, 2)
...
(1, q)
```

均为有效块。

通过位置 `(1, q)` 绘制一条向右下方向延伸的 45 度斜线，该斜线作为有效区域的上边界，斜线上的块属于有效块。

参数范围：

$$
1\le q\le n
$$

---

## 3. 有效区域定义

矩阵坐标采用从 1 开始的编号：

$$
1\le x\le m
$$

$$
1\le y\le n
$$

其中：

- `x` 表示行编号，从上向下递增；
- `y` 表示列编号，从左向右递增。

由 `p` 确定的下边界满足：

$$
x-y=p-1
$$

由 `q` 确定的上边界满足：

$$
x-y=1-q
$$

夹在两条斜线之间、并包含两条边界线上的块，均属于有效块。

因此，一个块 `(x, y)` 有效，当且仅当：

$$
1-q\le x-y\le p-1
$$

等价地，对矩阵中的第 `x` 行，其有效列范围为：

$$
y_{\min}(x)=\max(1,\ x-p+1)
$$

$$
y_{\max}(x)=\min(n,\ x+q-1)
$$

当：

$$
y_{\min}(x)\le y\le y_{\max}(x)
$$

时，块 `(x, y)` 有效。

可定义统一判断函数：

```python
def is_valid_block(
        x: int,
        y: int,
        m: int,
        n: int,
        p: int,
        q: int
) -> bool:
    return (
        1 <= x <= m
        and 1 <= y <= n
        and 1 - q <= x - y <= p - 1
    )
```

---

## 4. 与当前 causal 场景的关系

当：

$$
p=m,\qquad q=1
$$

有效条件变为：

$$
0\le x-y\le m-1
$$

在矩阵坐标范围内，等价于：

$$
y\le x
$$

即当前脚本中的 causal 下三角区域。

逐行有效列范围为：

$$
y_{\min}(x)=1
$$

$$
y_{\max}(x)=\min(n,x)
$$

因此，新增的 `p`、`q` 方案必须能够在 `p=m、q=1` 时，退化为现有 `get_causal_batch_position()` 的分核结果或满足与其等价的确定性调度约束。

---

## 5. 示例说明

假设：

```text
m = 6
n = 8
p = 4
q = 3
```

有效条件为：

$$
-2\le x-y\le3
$$

每一行的有效列范围如下：

| 行`x` | 有效列范围 |
| ------: | ---------- |
|       1 | 1～3       |
|       2 | 1～4       |
|       3 | 1～5       |
|       4 | 1～6       |
|       5 | 2～7       |
|       6 | 3～8       |

对应的有效块示意如下，其中 `1` 表示有效块，`.` 表示无效块：

```text
      y →
      1 2 3 4 5 6 7 8
x=1  1 1 1 . . . . .
x=2  1 1 1 1 . . . .
x=3  1 1 1 1 1 . . .
x=4  1 1 1 1 1 1 . .
x=5  . 1 1 1 1 1 1 .
x=6  . . 1 1 1 1 1 1
```

该有效区域是一条由两条平行 45 度斜线夹出的斜带。

---

## 6. 分核目标

需要将 `b` 个 `[m, n]` 矩阵中的全部有效块分配给 `k` 个核执行。

分核需要满足以下约束。

### 6.1 完整覆盖

每个满足以下条件的有效块：

$$
1-q\le x-y\le p-1
$$

必须被分配一次。

对于任意：

```text
batch_id ∈ [1, b]
x ∈ [1, m]
y ∈ [1, n]
```

如果 `(x, y)` 为有效块，则必须存在唯一的：

```text
core_id
round_id
```

使该任务被执行。

### 6.2 不允许重复分配

同一个：

```text
(batch_id, x, y)
```

不能被两个核或两个轮次重复处理。

即有效任务与调度位置之间必须是一一映射。

### 6.3 无效块不得参与计算

对于不满足：

$$
1-q\le x-y\le p-1
$$

的块，位置计算函数必须返回 `None`，或者在调度过程中直接跳过。

`rounds_cube` 和 `core_cube` 中对应位置应继续保持为 `-1`。

### 6.4 同一行轮次唯一

确定性计算的核心约束为：

> 对于同一个 batch 的同一行 `x`，所有有效列对应的 `round_id` 必须互不相同。

形式化表示为：

对于固定的：

```text
batch_id
x
```

以及任意两个不同的有效列：

$$
y_1\ne y_2
$$

必须满足：

$$
round(batch_id,x,y_1)
\ne
round(batch_id,x,y_2)
$$

也就是说，在每一个 batch 的 `[m, n]` 矩阵中，同一行不能出现相同的轮次编号。

该约束用于保证同一输出行的不同列贡献不会在同一轮被多个核同时处理。

### 6.5 同一轮同行无冲突

等价地，在固定的：

```text
batch_id
round_id
```

下，每一行最多只能有一个有效块被调度：

$$
\#\{y\mid round(batch,x,y)=r\}\le1
$$

### 6.6 确定性映射

调度结果必须只由以下参数决定：

```text
m, n, b, p, q, k, core_id, round_id
```

不能依赖：

- 动态任务队列；
- 原子抢占；
- 核的实际执行速度；
- 工作窃取；
- 运行时非确定性顺序。

在输入参数相同的情况下，每次运行必须得到完全相同的：

```text
batch_id
x
y
core_id
round_id
```

映射关系。

### 6.7 负载尽量均衡

在满足行轮次唯一和确定性要求的前提下，应尽量使各核分配到的有效块数量均衡。

建议优先沿用当前脚本的按列粒度分核方式：

- 一个核在一组轮次中处理一个逻辑列或虚拟列；
- 不同逻辑列按照轮询方式分配给不同核；
- 各核获得的逻辑列数量最多相差 1；
- 对于边界位置，由于有效块数量不同，允许少量负载差异。

---

## 7. 新增位置计算接口

建议新增通用位置计算函数：

```python
def get_band_batch_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int
) -> Optional[Tuple[int, int, int]]:
    """
    根据核编号和轮次编号，计算当前核本轮需要处理的任务。

    返回：
        (batch_id, x, y)

    如果当前核在当前轮次没有有效任务，则返回：
        None
    """
```

也可以对现有函数进行扩展：

```python
def get_causal_batch_position(
        m: int,
        n: int,
        b: int,
        core_id: int,
        round_id: int,
        k: int,
        p: Optional[int] = None,
        q: Optional[int] = None
) -> Optional[Tuple[int, int, int]]:
```

默认值为：

```python
p = m
q = 1
```

从而保证现有调用方式不受影响。

但从代码语义清晰性考虑，推荐新增 `get_band_batch_position()`，并让原来的 causal 接口调用通用接口：

```python
def get_causal_batch_position(...):
    return get_band_batch_position(
        m=m,
        n=n,
        b=b,
        p=m,
        q=1,
        core_id=core_id,
        round_id=round_id,
        k=k,
    )
```

---

## 8. 调度实现要求

新逻辑应参考当前 `get_causal_batch_position()` 的实现方式，优先通过数学公式直接计算：

$$
(core\_id,round\_id)
\rightarrow
(batch\_id,x,y)
$$

不建议在设备侧预先生成全部有效块列表后再分核。

实现应包含以下步骤。

### 8.1 参数校验

需要检查：

```python
m > 0
n > 0
b > 0
k > 0
1 <= p <= m
1 <= q <= n
1 <= core_id <= k
round_id >= 1
```

参数非法时，应抛出异常或明确返回错误。

### 8.2 生成候选位置

根据：

```text
core_id
round_id
m
n
b
p
q
k
```

计算候选的：

```text
batch_id
x
y
```

候选位置生成方式应尽量延续现有按列分核、循环错位行坐标的设计。

### 8.3 有效性过滤

候选位置必须通过以下判断：

```python
1 <= batch_id <= b
1 <= x <= m
1 <= y <= n
1 - q <= x - y <= p - 1
```

否则返回 `None`。

### 8.4 边界映射

如果继续采用虚拟稠密矩形或 batch 拼接方案，则必须保证：

- 虚拟位置到真实位置的映射是一一映射；
- 映射后仍满足有效区域公式；
- 映射后同一行的轮次不重复；
- 不产生遗漏块；
- 不产生重复块；
- 斜线边界上的块必须被保留。

### 8.5 总轮次数计算

需要根据有效区域尺寸、batch 数量和核数计算一个确定的最大轮次数 `R`。

`R` 必须足够覆盖全部有效块，但不要求达到理论最小值。

建议新增：

```python
def get_band_max_rounds(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int
) -> int:
    ...
```

总有效块数可以按行计算：

$$
V_{\text{single}}
=
\sum_{x=1}^{m}
\max
\left(
0,
y_{\max}(x)-y_{\min}(x)+1
\right)
$$

其中：

$$
y_{\min}(x)=\max(1,x-p+1)
$$

$$
y_{\max}(x)=\min(n,x+q-1)
$$

全部 batch 的有效块数为：

$$
V_{\text{total}}=b\cdot V_{\text{single}}
$$

`R` 至少应满足：

$$
k\cdot R\ge V_{\text{total}}
$$

但由于还需要满足同行轮次唯一约束，实际 `R` 可以大于该理论下界。

---

## 9. 矩阵构建接口修改

建议新增：

```python
def _build_band_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int
):
    """
    构建全部 batch 的 round/core 矩阵。
    """
```

初始化方式保持不变：

```python
rounds_cube = np.full((b, m, n), -1, dtype=int)
core_cube = np.full((b, m, n), -1, dtype=int)
```

遍历所有核和轮次：

```python
for core_id in range(1, k + 1):
    for round_id in range(1, R + 1):
        pos = get_band_batch_position(
            m=m,
            n=n,
            b=b,
            p=p,
            q=q,
            core_id=core_id,
            round_id=round_id,
            k=k,
        )

        if pos is None:
            continue

        batch_id, x, y = pos
```

写入前必须进行重复检查：

```python
assert rounds_cube[batch_id - 1, x - 1, y - 1] == -1
assert core_cube[batch_id - 1, x - 1, y - 1] == -1
```

然后写入：

```python
rounds_cube[batch_id - 1, x - 1, y - 1] = round_id
core_cube[batch_id - 1, x - 1, y - 1] = core_id
```

---

## 10. 可视化要求

现有可视化接口需要增加 `p` 和 `q` 参数：

```python
def visualize_band_schedule(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        ...
):
```

图中需要展示：

- 有效块对应的核编号颜色；
- 有效块内部标注 `round_id`；
- 无效块显示为空白或统一灰色；
- 标题中显示 `p` 和 `q`；
- 可选绘制两条有效区域边界斜线。

标题格式建议为：

```text
Band Schedule | k=32, m=32, n=32, b=8, p=32, q=1
```

对于无效块：

```python
rounds_cube == -1
core_cube == -1
```

不得显示有效轮次或核编号。

---

## 11. 自动校验要求

需要新增调度结果校验函数：

```python
def validate_band_schedule(
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> None:
    ...
```

至少完成以下检查。

### 11.1 有效块完整覆盖检查

对于所有有效块：

```python
if 1 - q <= x - y <= p - 1:
```

必须满足：

```python
rounds_cube[batch_id, x, y] >= 1
1 <= core_cube[batch_id, x, y] <= k
```

### 11.2 无效块未分配检查

对于所有无效块，必须满足：

```python
rounds_cube[batch_id, x, y] == -1
core_cube[batch_id, x, y] == -1
```

### 11.3 同行轮次唯一检查

对于每个 batch 和每一行：

```python
round_ids = rounds_cube[batch_id, x, :]
round_ids = round_ids[round_ids >= 1]
assert len(round_ids) == len(set(round_ids))
```

### 11.4 核编号合法检查

所有已分配块的核编号必须满足：

```python
1 <= core_id <= k
```

### 11.5 总块数检查

实际分配块数必须等于：

$$
b\cdot
\sum_{x=1}^{m}
\max
\left(
0,
\min(n,x+q-1)
-
\max(1,x-p+1)
+1
\right)
$$

### 11.6 确定性检查

使用完全相同的参数连续构建两次调度矩阵，必须满足：

```python
np.array_equal(rounds_cube_1, rounds_cube_2)
np.array_equal(core_cube_1, core_cube_2)
```

---

## 12. 验收用例

### 用例一：兼容当前 causal 下三角

```text
m = 32
n = 32
b = 8
k = 32
p = 32
q = 1
```

预期：

- 有效条件为 `y <= x`；
- 每个 batch 有效块数为：

$$
\frac{32\times33}{2}=528
$$

- 总有效块数为：

$$
8\times528=4224
$$

- 每个 batch 的每一行不存在重复轮次；
- 结果满足当前 causal 场景的调度约束。

### 用例二：完整矩阵

```text
p = m
q = n
```

此时：

$$
1-n\le x-y\le m-1
$$

矩阵内全部块均有效。

预期：

- 每个 batch 有效块数为 `m * n`；
- 所有位置均被调度；
- 每一行的 `n` 个块具有不同轮次。

### 用例三：主对角线

```text
p = 1
q = 1
```

有效条件为：

$$
x-y=0
$$

即：

$$
x=y
$$

预期：

- 仅主对角线位置有效；
- 对于非方阵，仅有 `min(m, n)` 个有效块；
- 不允许调度任何非对角线块。

### 用例四：上三角

```text
p = 1
q = n
```

有效条件为：

$$
1-n\le x-y\le0
$$

等价于：

$$
x\le y
$$

预期有效区域为包含主对角线的上三角。

### 用例五：普通斜带

```text
m = 6
n = 8
b = 4
k = 8
p = 4
q = 3
```

预期单个 batch 的行有效块数分别为：

```text
3, 4, 5, 6, 6, 6
```

单个 batch 总有效块数为：

$$
3+4+5+6+6+6=30
$$

全部 batch 总有效块数为：

$$
4\times30=120
$$

### 用例六：长方形矩阵

```text
m = 64
n = 32
p = 20
q = 8
```

以及：

```text
m = 32
n = 64
p = 20
q = 8
```

预期：

- 有效区域严格按照统一公式判断；
- 不能因为 `m != n` 出现重复或遗漏；
- 同一 batch、同一行中轮次仍然唯一。

---

## 13. 兼容性要求

1. 原有 causal 调用场景必须继续支持。
2. `p=m、q=1` 时必须能够正常运行。
3. 除新增 `p`、`q` 参数外，其他参数含义不变。
4. 可视化输出格式尽量兼容现有脚本。
5. 首版可以沿用当前脚本以偶数 `b` 进行 batch 拼接的约束。
6. 如果 `b` 为奇数且暂不支持，应在参数检查阶段明确报错，不能静默遗漏最后一个 batch。
7. 后续可以扩展为奇数 `b` 场景下，对最后一个 batch 单独调度。

---

## 14. 最终交付形式

每次需求或代码发生修改时，交付一个完整的 Markdown 文档。该文档必须同时包含：

1. 本次需求说明；
2. 有效区域和确定性调度公式；
3. 接口及兼容性要求；
4. 验收条件；
5. 与本次需求对应的完整 Python 实现代码；
6. 本次代码变更说明和验证结果。

测试代码不需要附在 Markdown 文档中，也不单独作为交付文件提供。

---

## 15. 需求总结

新增参数 `p` 和 `q`，将原有 causal 下三角有效区域扩展为通用斜带有效区域。

有效块统一满足：

$$
\boxed{1-q\le x-y\le p-1}
$$

其中：

- `p` 决定第一列从上向下的有效块数量；
- `q` 决定第一行从左向右的有效块数量；
- 两条边界均为 45 度斜线；
- 边界线上的块属于有效块。

需要对所有有效块进行确定性分核，并保证：

- 每个有效块恰好执行一次；
- 无效块不执行；
- 同一个 batch 的同一行中不能出现相同轮次；
- 相同输入参数始终产生相同分核结果；
- 各核负载尽量均衡；
- `p=m、q=1` 时退化为当前 causal 下三角场景。

---

## 16. 当前完整实现代码

### 16.1 文件信息

- 实现文件：`band_balance_first.py`
- 文档版本：`v1.1`
- 有效区域：

$$
1-q\le x-y\le p-1
$$

- causal 兼容参数：`p=m, q=1`
- 本节包含与上述需求对应的完整实现代码。
- 本文档不附测试代码。

### 16.2 完整 Python 代码

```python
"""
FlashAttentionScoreGrad SparseMode4 通用斜带区域确定性分核。

有效区域由 p、q 控制。矩阵坐标采用 1-based：
    1 <= x <= m, 1 <= y <= n
有效块满足：
    1 - q <= x - y <= p - 1

调度思想：
1. 按逻辑列 (batch_id, y) 分核；一个逻辑列在一组连续轮次中由同一个核负责。
2. 一组轮次包含 p + q - 1 个“斜对角阶段”，阶段 d 对应 x - y = d。
3. 对固定阶段 d，有 x = y + d。不同列 y 会得到不同的行 x，因此同一个 batch、
   同一个 round 内不会有两个任务落到同一行。
4. 逻辑列按固定顺序切成大小不超过 k 的组，并对每组做确定性核旋转，以改善核间负载。

当 p=m、q=1 时，有效条件退化为 y <= x，即原 causal 下三角场景。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

Position = Tuple[int, int, int]


@dataclass(frozen=True)
class BandScheduleInfo:
    """通用斜带调度的静态信息。"""

    k: int
    m: int
    n: int
    b: int
    p: int
    q: int
    effective_n: int
    diagonal_count: int
    logical_column_count: int
    column_group_count: int
    total_rounds: int
    rotation_stride: int


@dataclass(frozen=True)
class BandScheduleStats:
    """调度校验与负载统计结果。"""

    valid_blocks_per_batch: int
    total_valid_blocks: int
    min_core_load: int
    max_core_load: int
    core_loads: Tuple[int, ...]


# ---------------------------------------------------------------------------
# 参数与几何定义
# ---------------------------------------------------------------------------


def _validate_common_parameters(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> None:
    """校验与具体 core/round 无关的输入参数。"""

    for name, value in (("m", m), ("n", n), ("b", b), ("p", p), ("q", q), ("k", k)):
        if not isinstance(value, int):
            raise TypeError(f"{name} 必须是 int，实际为 {type(value).__name__}")

    if m <= 0 or n <= 0 or b <= 0 or k <= 0:
        raise ValueError("m、n、b、k 必须大于 0")
    if not 1 <= p <= m:
        raise ValueError(f"p 必须满足 1 <= p <= m，当前 p={p}, m={m}")
    if not 1 <= q <= n:
        raise ValueError(f"q 必须满足 1 <= q <= n，当前 q={q}, n={n}")


def is_valid_block(
        x: int,
        y: int,
        m: int,
        n: int,
        p: int,
        q: int,
) -> bool:
    """判断 1-based 坐标 (x, y) 是否位于 p/q 指定的有效斜带内。"""

    return (
        1 <= x <= m
        and 1 <= y <= n
        and 1 - q <= x - y <= p - 1
    )


def get_row_valid_range(
        x: int,
        m: int,
        n: int,
        p: int,
        q: int,
) -> Optional[Tuple[int, int]]:
    """返回第 x 行的有效列闭区间 [y_min, y_max]，无有效块时返回 None。"""

    _validate_common_parameters(m, n, 1, p, q, 1)
    if not 1 <= x <= m:
        raise ValueError(f"x 必须满足 1 <= x <= m，当前 x={x}, m={m}")

    y_min = max(1, x - p + 1)
    y_max = min(n, x + q - 1)
    if y_min > y_max:
        return None
    return y_min, y_max


def get_column_valid_range(
        y: int,
        m: int,
        n: int,
        p: int,
        q: int,
) -> Optional[Tuple[int, int]]:
    """返回第 y 列的有效行闭区间 [x_min, x_max]，无有效块时返回 None。"""

    _validate_common_parameters(m, n, 1, p, q, 1)
    if not 1 <= y <= n:
        raise ValueError(f"y 必须满足 1 <= y <= n，当前 y={y}, n={n}")

    x_min = max(1, y - q + 1)
    x_max = min(m, y + p - 1)
    if x_min > x_max:
        return None
    return x_min, x_max


def count_valid_blocks_per_batch(m: int, n: int, p: int, q: int) -> int:
    """计算单个 batch 内有效块总数。"""

    _validate_common_parameters(m, n, 1, p, q, 1)
    total = 0
    for x in range(1, m + 1):
        y_min = max(1, x - p + 1)
        y_max = min(n, x + q - 1)
        if y_min <= y_max:
            total += y_max - y_min + 1
    return total


def _find_coprime_rotation_stride(k: int) -> int:
    """
    选择与 k 互质的核旋转步长。

    每个逻辑列组都对“局部槽位 -> core_id”的映射做一次旋转，使边界列和最后一个
    不满组不会长期落在固定核上。k=1 时返回 0。
    """

    if k == 1:
        return 0

    stride = k // 2 + 1
    while gcd(stride, k) != 1:
        stride += 1
    return stride % k


def get_band_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> BandScheduleInfo:
    """计算通用斜带调度所需的静态信息。"""

    _validate_common_parameters(m, n, b, p, q, k)

    # 对任意有效块，y <= m + q - 1。因此右侧超过该范围的列必然全无效。
    effective_n = min(n, m + q - 1)
    diagonal_count = p + q - 1
    logical_column_count = b * effective_n
    column_group_count = ceil(logical_column_count / k)
    total_rounds = column_group_count * diagonal_count

    return BandScheduleInfo(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        effective_n=effective_n,
        diagonal_count=diagonal_count,
        logical_column_count=logical_column_count,
        column_group_count=column_group_count,
        total_rounds=total_rounds,
        rotation_stride=_find_coprime_rotation_stride(k),
    )


def get_band_max_rounds(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> int:
    """返回覆盖全部有效块所需的确定性总轮次数。"""

    return get_band_schedule_info(m, n, b, p, q, k).total_rounds


# ---------------------------------------------------------------------------
# 核心确定性映射
# ---------------------------------------------------------------------------


def _get_band_batch_position_from_info(
        info: BandScheduleInfo,
        core_id: int,
        round_id: int,
) -> Optional[Position]:
    """使用预计算的 info 执行核心映射，供矩阵构建的热循环调用。"""

    if round_id > info.total_rounds:
        return None

    round0 = round_id - 1
    group_id = round0 // info.diagonal_count
    diagonal_id = round0 % info.diagonal_count

    group_start = group_id * info.k
    group_size = min(info.k, info.logical_column_count - group_start)
    if group_size <= 0:
        return None

    # 相邻两个列组使用“正序/逆序”蛇形分配，并共享同一个旋转量：
    # - 对列有效块数量单调变化的 causal/斜带边界，重列与轻列会在两个组内互补；
    # - 每两个组再旋转一次，令不满组和边界列不会长期固定在同一批核上。
    pair_rotation = ((group_id // 2) * info.rotation_stride) % info.k
    permuted_core = ((core_id - 1) - pair_rotation) % info.k
    if group_id % 2 == 0:
        local_slot = permuted_core
    else:
        local_slot = info.k - 1 - permuted_core

    if local_slot >= group_size:
        return None

    logical_column_id = group_start + local_slot

    # 逻辑列按 y 优先、batch 次优先排列：
    #   (y=1,batch=1..b), (y=2,batch=1..b), ...
    # 同一列权重会先分散到多个核，相比 batch-major 顺序可显著改善边界斜带的负载。
    y0, batch0 = divmod(logical_column_id, info.b)
    batch_id = batch0 + 1
    y = y0 + 1

    diagonal = 1 - info.q + diagonal_id
    x = y + diagonal

    if not is_valid_block(x, y, info.m, info.n, info.p, info.q):
        return None

    return batch_id, x, y


def get_band_batch_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    """
    根据 (core_id, round_id) 直接计算当前任务位置。

    返回：
        (batch_id, x, y)，均为 1-based。

    当前核在该轮没有有效任务时返回 None。

    映射公式：
        group       = (round_id - 1) // (p + q - 1)
        diagonal_id = (round_id - 1) %  (p + q - 1)
        d            = 1 - q + diagonal_id
        x            = y + d

    对固定 round，d 固定；在同一个 batch 内，不同 y 映射到不同 x，因而同行无冲突。
    """

    info = get_band_schedule_info(m, n, b, p, q, k)

    if not isinstance(core_id, int) or not isinstance(round_id, int):
        raise TypeError("core_id 和 round_id 必须是 int")
    if not 1 <= core_id <= k:
        raise ValueError(f"core_id 必须满足 1 <= core_id <= k，当前 core_id={core_id}, k={k}")
    if round_id < 1:
        raise ValueError(f"round_id 必须大于等于 1，当前 round_id={round_id}")

    return _get_band_batch_position_from_info(info, core_id, round_id)


def get_causal_batch_position(
        m: int,
        n: int,
        b: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    """
    causal 下三角兼容接口。

    causal 场景等价于 p=m、q=1，因此有效条件为 0 <= x-y <= m-1，即 y <= x。
    """

    return get_band_batch_position(
        m=m,
        n=n,
        b=b,
        p=m,
        q=1,
        core_id=core_id,
        round_id=round_id,
        k=k,
    )


# ---------------------------------------------------------------------------
# 矩阵构建、统计与校验
# ---------------------------------------------------------------------------


def _build_band_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        validate: bool = True,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """一次性构建全部 batch 的 round/core 矩阵。"""

    info = get_band_schedule_info(m, n, b, p, q, k)
    rounds_cube = np.full((b, m, n), -1, dtype=np.int64)
    core_cube = np.full((b, m, n), -1, dtype=np.int64)

    for round_id in range(1, info.total_rounds + 1):
        for core_id in range(1, k + 1):
            pos = _get_band_batch_position_from_info(
                info=info,
                core_id=core_id,
                round_id=round_id,
            )
            if pos is None:
                continue

            batch_id, x, y = pos
            index = (batch_id - 1, x - 1, y - 1)
            if rounds_cube[index] != -1 or core_cube[index] != -1:
                raise AssertionError(
                    "检测到重复任务分配："
                    f"position={(batch_id, x, y)}, "
                    f"old=(round={rounds_cube[index]}, core={core_cube[index]}), "
                    f"new=(round={round_id}, core={core_id})"
                )

            rounds_cube[index] = round_id
            core_cube[index] = core_id

    if validate:
        validate_band_schedule(
            rounds_cube=rounds_cube,
            core_cube=core_cube,
            m=m,
            n=n,
            b=b,
            p=p,
            q=q,
            k=k,
        )

    return k, info.total_rounds, rounds_cube, core_cube


def _build_causal_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        validate: bool = True,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """causal 下三角兼容接口。"""

    return _build_band_matrices(
        k=k,
        m=m,
        n=n,
        b=b,
        p=m,
        q=1,
        validate=validate,
    )


def get_core_loads(core_cube: np.ndarray, k: int) -> np.ndarray:
    """统计每个核实际承担的有效块数量，返回 shape=[k]。"""

    if core_cube.ndim != 3:
        raise ValueError(f"core_cube 必须是三维数组，当前 ndim={core_cube.ndim}")
    if k <= 0:
        raise ValueError("k 必须大于 0")

    return np.asarray(
        [np.count_nonzero(core_cube == core_id) for core_id in range(1, k + 1)],
        dtype=np.int64,
    )


def validate_band_schedule(
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> BandScheduleStats:
    """
    校验完整覆盖、无效块未分配、同行轮次唯一、核编号范围及总任务数。

    校验失败会抛出 AssertionError；成功时返回负载统计。
    """

    _validate_common_parameters(m, n, b, p, q, k)
    expected_shape = (b, m, n)
    if rounds_cube.shape != expected_shape:
        raise ValueError(f"rounds_cube.shape 应为 {expected_shape}，实际为 {rounds_cube.shape}")
    if core_cube.shape != expected_shape:
        raise ValueError(f"core_cube.shape 应为 {expected_shape}，实际为 {core_cube.shape}")

    x_grid = np.arange(1, m + 1, dtype=np.int64)[:, None]
    y_grid = np.arange(1, n + 1, dtype=np.int64)[None, :]
    valid_2d = (1 - q <= x_grid - y_grid) & (x_grid - y_grid <= p - 1)
    valid_mask = np.broadcast_to(valid_2d, expected_shape)

    assigned_round = rounds_cube >= 1
    assigned_core = core_cube >= 1

    if not np.array_equal(assigned_round, valid_mask):
        missing = np.argwhere(valid_mask & ~assigned_round)
        unexpected = np.argwhere(~valid_mask & assigned_round)
        raise AssertionError(
            "rounds_cube 覆盖错误："
            f"missing={missing[:8].tolist()}, unexpected={unexpected[:8].tolist()}"
        )

    if not np.array_equal(assigned_core, valid_mask):
        missing = np.argwhere(valid_mask & ~assigned_core)
        unexpected = np.argwhere(~valid_mask & assigned_core)
        raise AssertionError(
            "core_cube 覆盖错误："
            f"missing={missing[:8].tolist()}, unexpected={unexpected[:8].tolist()}"
        )

    if np.any(core_cube[valid_mask] > k):
        bad = np.argwhere(valid_mask & (core_cube > k))
        raise AssertionError(f"发现超过 k 的非法核编号：{bad[:8].tolist()}")

    # 对每个 batch 的每一行，全部有效列的 round_id 必须互不相同。
    for batch0 in range(b):
        for x0 in range(m):
            row_rounds = rounds_cube[batch0, x0]
            row_rounds = row_rounds[row_rounds >= 1]
            if row_rounds.size != np.unique(row_rounds).size:
                values, counts = np.unique(row_rounds, return_counts=True)
                duplicate_rounds = values[counts > 1]
                raise AssertionError(
                    "同行出现重复轮次："
                    f"batch={batch0 + 1}, x={x0 + 1}, "
                    f"duplicate_rounds={duplicate_rounds.tolist()}"
                )

    # 对每个 batch/round，行坐标也必须唯一。该检查与上面等价，但更贴近执行时冲突语义。
    max_round = int(rounds_cube.max(initial=-1))
    for batch0 in range(b):
        for round_id in range(1, max_round + 1):
            positions = np.argwhere(rounds_cube[batch0] == round_id)
            if positions.size == 0:
                continue
            rows = positions[:, 0]
            if rows.size != np.unique(rows).size:
                raise AssertionError(
                    "同一 batch/round 出现同行冲突："
                    f"batch={batch0 + 1}, round={round_id}, rows={(rows + 1).tolist()}"
                )

    valid_per_batch = count_valid_blocks_per_batch(m, n, p, q)
    expected_total = b * valid_per_batch
    actual_total = int(np.count_nonzero(valid_mask))
    if actual_total != expected_total:
        raise AssertionError(
            f"有效块计数错误：expected={expected_total}, actual={actual_total}"
        )

    core_loads = get_core_loads(core_cube, k)
    return BandScheduleStats(
        valid_blocks_per_batch=valid_per_batch,
        total_valid_blocks=expected_total,
        min_core_load=int(core_loads.min()),
        max_core_load=int(core_loads.max()),
        core_loads=tuple(int(v) for v in core_loads),
    )


def verify_determinism(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
) -> bool:
    """使用相同参数构建两次矩阵，确认结果完全一致。"""

    first = _build_band_matrices(k, m, n, b, p, q, validate=True)
    second = _build_band_matrices(k, m, n, b, p, q, validate=True)
    return np.array_equal(first[2], second[2]) and np.array_equal(first[3], second[3])


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def _line_segment_for_offset(m: int, n: int, offset: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    返回 x-y=offset 在矩阵单元中心范围内的可视化线段。

    matplotlib 横轴为列索引 y-1，纵轴为行索引 x-1，因此线方程为 row-col=offset。
    """

    points = []
    # col = 0 / n-1
    for col in (0, n - 1):
        row = col + offset
        if 0 <= row <= m - 1:
            points.append((float(col), float(row)))
    # row = 0 / m-1
    for row in (0, m - 1):
        col = row - offset
        if 0 <= col <= n - 1:
            points.append((float(col), float(row)))

    unique = []
    for point in points:
        if point not in unique:
            unique.append(point)
    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def visualize_band_schedule(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        batches_per_figure: int = 2,
        ncols: int = 2,
        annotate_round: bool = True,
        draw_boundaries: bool = True,
        save_dir: Optional[str | Path] = None,
        dpi: int = 220,
        show: bool = True,
) -> Dict[str, object]:
    """分页可视化通用斜带确定性分核。"""

    k, total_rounds, rounds_cube, core_cube = _build_band_matrices(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        validate=True,
    )
    stats = validate_band_schedule(rounds_cube, core_cube, m, n, b, p, q, k)

    print("所需核数:", k)
    print("确定性计算总轮次:", total_rounds)
    print("单 batch 有效块数:", stats.valid_blocks_per_batch)
    print("总有效块数:", stats.total_valid_blocks)
    print("各核有效任务数:", list(stats.core_loads))
    print("核负载范围:", (stats.min_core_load, stats.max_core_load))

    palette = ["#F3F4F6"] + [
        "#0B84A5", "#EBC262", "#6F4E7C", "#9DD866", "#CA472F",
        "#FFA056", "#8DDDD0", "#BFB5FF", "#3C5488", "#F39C12",
        "#27AE60", "#D35400", "#16A085", "#7F8C8D", "#2E86C1",
        "#E74C3C", "#8E44AD", "#2ECC71", "#34495E", "#F1C40F",
    ]
    if k + 1 > len(palette):
        extra = plt.get_cmap("tab20", k + 1 - len(palette)).colors
        palette.extend(extra)
    cmap = ListedColormap(palette[:k + 1], name="band_schedule")

    batches = list(range(1, b + 1))
    page_size = max(1, batches_per_figure)
    total_pages = ceil(len(batches) / page_size)
    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    cell_w = 0.42
    cell_h = 0.34
    axes_w = max(5.2, n * cell_w)
    axes_h = max(5.0, m * cell_h)
    label_font = 7 if max(m, n) <= 32 else 6
    annot_font = 8 if max(m, n) <= 32 else 6

    for page_idx in range(total_pages):
        page_batches = batches[page_idx * page_size:(page_idx + 1) * page_size]
        cols = min(max(1, ncols), len(page_batches))
        rows = ceil(len(page_batches) / cols)

        fig_w = cols * axes_w + 1.2
        fig_h = rows * axes_h + 1.4
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

        for local_idx, batch_id in enumerate(page_batches):
            row_slot = local_idx // cols
            col_slot = local_idx % cols
            ax = axes[row_slot][col_slot]

            round_mat = rounds_cube[batch_id - 1]
            core_mat = core_cube[batch_id - 1]
            display_core = np.where(core_mat >= 1, core_mat, 0)

            ax.imshow(
                display_core,
                origin="upper",
                cmap=cmap,
                vmin=0,
                vmax=max(k, 1),
                aspect="equal",
                interpolation="nearest",
            )

            ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="#D1D5DB", linestyle="-", linewidth=0.45)
            ax.tick_params(which="minor", length=0)

            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(m))
            ax.set_xticklabels(np.arange(1, n + 1), fontsize=label_font)
            ax.set_yticklabels(np.arange(1, m + 1), fontsize=label_font)
            ax.tick_params(axis="x", pad=2)
            ax.tick_params(axis="y", pad=2)

            if annotate_round:
                for i in range(m):
                    for j in range(n):
                        rid = int(round_mat[i, j])
                        if rid >= 1:
                            ax.text(
                                j,
                                i,
                                str(rid),
                                ha="center",
                                va="center",
                                fontsize=annot_font,
                                fontweight="bold",
                                color="black",
                                bbox=dict(
                                    boxstyle="round,pad=0.10",
                                    facecolor="white",
                                    alpha=0.72,
                                    edgecolor="none",
                                ),
                            )

            if draw_boundaries:
                for offset, label in ((p - 1, "p boundary"), (1 - q, "q boundary")):
                    segment = _line_segment_for_offset(m, n, offset)
                    if segment is not None:
                        (x1, y1), (x2, y2) = segment
                        ax.plot([x1, x2], [y1, y2], linewidth=1.5, label=label)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    # p=q=1 时两条边界重合，只保留一个图例项。
                    unique = dict(zip(labels, handles))
                    ax.legend(unique.values(), unique.keys(), fontsize=6, loc="upper right")

            ax.set_title(f"batch={batch_id}", fontsize=10, pad=8)

        for idx in range(len(page_batches), rows * cols):
            row_slot = idx // cols
            col_slot = idx % cols
            axes[row_slot][col_slot].axis("off")

        fig.suptitle(
            f"Band Schedule | k={k}, m={m}, n={n}, b={b}, p={p}, q={q} "
            f"| page {page_idx + 1}/{total_pages}",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))

        if save_path is not None:
            fig.savefig(
                save_path / f"FAG_sparse04_band_p{p}_q{q}_page_{page_idx + 1}.png",
                dpi=dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

    return {
        "k": k,
        "total_rounds": total_rounds,
        "rounds_cube": rounds_cube,
        "core_cube": core_cube,
        "stats": stats,
    }


def visualize_causal_schedule(
        k: int,
        m: int,
        n: int,
        b: int,
        **kwargs: object,
) -> Dict[str, object]:
    """causal 下三角可视化兼容接口。"""

    return visualize_band_schedule(k=k, m=m, n=n, b=b, p=m, q=1, **kwargs)


# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------


def _demo() -> None:
    k, m, n, b = 32, 32, 32, 8
    p, q = 16, 4

    output_dir = Path(__file__).parent / "outputs_band"
    result = visualize_band_schedule(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        batches_per_figure=2,
        ncols=2,
        save_dir=output_dir,
        dpi=260,
        show=False,
    )
    print("确定性复验:", verify_determinism(k, m, n, b, p, q))
    print("输出目录:", output_dir)
    print("结果键:", list(result.keys()))


if __name__ == "__main__":
    _demo()
```
