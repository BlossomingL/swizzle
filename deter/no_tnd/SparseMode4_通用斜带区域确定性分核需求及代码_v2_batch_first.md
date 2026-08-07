# SparseMode4 通用斜带区域确定性分核需求及代码：Batch-First 方案

> **文档版本：v2.0-batch-first**
>
> 本文档给出与原“跨 batch 全局均衡”方案并列的另一种调度方案：所有核优先协同完成一个 batch 内的 `[m,n]` 有效区域，当前 batch 完成后才进入下一个 batch。
>
> 文档包含完整需求、调度公式、方案取舍、验收条件和完整 Python 实现代码；不附测试代码。

---

## 1. 需求背景

每个 batch 包含一个形状为 `[m,n]` 的任务块矩阵。矩阵坐标采用 1-based：

- `x`：行编号，从上到下递增；
- `y`：列编号，从左到右递增。

参数 `p` 和 `q` 规定有效斜带区域，有效块满足：

$$
1-q \le x-y \le p-1
$$

其中：

- `p` 表示第一列从左上角向下连续有效的块数；
- `q` 表示第一行从左上角向右连续有效的块数；
- 两条 45 度边界线上的块均有效。

当：

$$
p=m,\qquad q=1
$$

有效条件退化为：

$$
y\le x
$$

即 causal 下三角场景。

---

## 2. 新方案目标

新增 **Batch-First 调度方案**，优先级如下：

1. 所有核只处理当前 batch；
2. 当前 batch 的 `[m,n]` 有效区域全部完成后，才进入下一个 batch；
3. 同一个全局轮次不能同时包含多个 batch 的任务；
4. 同一 batch、同一轮次内，同一行最多出现一个有效任务；
5. 每个有效块恰好分配一次；
6. 分核结果由参数和 `(core_id, round_id)` 唯一决定；
7. 在满足 batch 优先的前提下，尽量改善核间负载。

该方案主要解决原方案中单独观察某个 batch 时，轮次可能从 `19` 跳到 `58` 的问题。Batch-First 方案使每个 batch 使用一个连续、互不交叉的全局轮次区间。

---

## 3. 与原方案的区别

| 对比项              | 原全局均衡方案                  | Batch-First 方案        |
| ------------------- | ------------------------------- | ----------------------- |
| 逻辑列排序          | 不同 batch 的列可混合进同一列组 | 每个 batch 独立切分列组 |
| 单轮 batch 数       | 可能涉及不同 batch              | 最多只涉及一个 batch    |
| batch 轮次范围      | 可能离散                        | 严格连续                |
| 核间负载            | 通常更均衡                      | 可能更不均衡            |
| batch 局部性        | 较弱                            | 更强                    |
| `effective_n < k` | 可用其他 batch 填满空核         | 多余核会空闲            |
| 适用目标            | 优先整体负载均衡                | 优先尽快完成单个 batch  |

对于示例参数：

```text
k=32, m=32, n=32, b=8, p=16, q=4
```

两种方案的总轮次数均为 `152`，但负载范围不同：

| 方案             | 最小核负载 | 最大核负载 |
| ---------------- | ---------: | ---------: |
| 原全局均衡方案   |        119 |        122 |
| Batch-First 方案 |        104 |        139 |

因此，新方案是以更强的 batch 局部性换取部分负载均衡能力。

---

## 4. 有效区域定义

有效块统一满足：

$$
1-q \le x-y \le p-1
$$

对第 `x` 行，有效列范围为：

$$
y_{\min}(x)=\max(1,x-p+1)
$$

$$
y_{\max}(x)=\min(n,x+q-1)
$$

对第 `y` 列，有效行范围为：

$$
x_{\min}(y)=\max(1,y-q+1)
$$

$$
x_{\max}(y)=\min(m,y+p-1)
$$

---

## 5. 调度基本单位

### 5.1 有效列数

根据：

$$
x-y\ge 1-q
$$

以及：

$$
x\le m
$$

可得：

$$
y\le m+q-1
$$

因此真正可能包含有效块的列数为：

$$
effective\_n=\min(n,m+q-1)
$$

### 5.2 斜对角阶段数

斜对角偏移定义为：

$$
d=x-y
$$

其取值范围为：

$$
d=1-q,2-q,\ldots,p-1
$$

阶段总数为：

$$
D=p+q-1
$$

### 5.3 单 batch 列组数

每个列组最多包含 `k` 个物理列，因此单 batch 的列组数为：

$$
G=\left\lceil\frac{effective\_n}{k}\right\rceil
$$

### 5.4 单 batch 轮次数

每个列组需要遍历全部 `D` 个斜对角阶段，因此：

$$
R_{batch}=G\cdot D
$$

### 5.5 全局总轮次数

所有 batch 串行推进：

$$
R_{total}=b\cdot R_{batch}
$$

---

## 6. 全局轮次到 batch 的映射

设：

$$
r=round\_id-1
$$

当前 batch 的 0-based 编号为：

$$
b_0=\left\lfloor\frac{r}{R_{batch}}\right\rfloor
$$

当前 batch 内部的局部轮次为：

$$
r_b=r\bmod R_{batch}
$$

因此真实 batch 编号为：

$$
batch\_id=b_0+1
$$

第 `i` 个 batch 使用的全局轮次闭区间为：

$$
[(i-1)R_{batch}+1,\ iR_{batch}]
$$

这些区间连续且互不重叠。

---

## 7. batch 内部列组和斜对角映射

列组编号为：

$$
g=\left\lfloor\frac{r_b}{D}\right\rfloor
$$

斜对角阶段编号为：

$$
t=r_b\bmod D
$$

当前斜对角偏移为：

$$
d=1-q+t
$$

对于当前核分配到的列 `y`，行坐标为：

$$
x=y+d
$$

若 `(x,y)` 超出矩阵或不满足有效条件，则当前核在该轮返回 `None`。

---

## 8. 为什么保证同行无冲突

对固定的 `batch_id` 和 `round_id`，斜对角偏移 `d` 固定。

两个核处理不同列：

$$
y_1\ne y_2
$$

对应行坐标：

$$
x_1=y_1+d
$$

$$
x_2=y_2+d
$$

因此：

$$
y_1\ne y_2\Longrightarrow x_1\ne x_2
$$

所以同一 batch、同一轮次内，不同核不会处理同一行。

不同列组使用不同的全局轮次区间，因此同一行在不同列组中的任务也不会获得相同轮次编号。

---

## 9. 核到列的映射

当前 batch 的第 `g` 个列组包含：

$$
y\in[gk+1,\ \min((g+1)k,effective\_n)]
$$

为了改善边界列和最后一个不满列组造成的负载倾斜，采用：

1. 相邻列组正序/逆序蛇形映射；
2. 每进入一个新 batch，对核映射做确定性旋转；
3. 每两个列组额外旋转一次。

这些旋转只改变“哪个核负责哪个列”，不会改变：

- 当前轮次所属 batch；
- 当前轮次的斜对角偏移；
- 有效区域定义；
- 同行无冲突性质。

---

## 10. 轮次连续性示例

参数：

```text
k=32, m=32, n=32, b=8, p=16, q=4
```

有：

$$
D=p+q-1=19
$$

$$
G=\left\lceil\frac{32}{32}\right\rceil=1
$$

$$
R_{batch}=19
$$

各 batch 轮次区间为：

| batch | 全局轮次区间 |
| ----: | -----------: |
|     1 |        1～19 |
|     2 |       20～38 |
|     3 |       39～57 |
|     4 |       58～76 |
|     5 |       77～95 |
|     6 |      96～114 |
|     7 |     115～133 |
|     8 |     134～152 |

因此，查看 `batch=1` 时只会出现 `1～19` 范围内的轮次，不会再出现 `19 → 58` 这种跨 batch 的跳跃。

需要注意：某一物理列在靠近矩阵边界时，部分斜对角阶段可能无有效任务，因此单列内部仍可能跳过若干无效阶段。这是几何边界导致的空轮，不是其他 batch 插入造成的跳跃。

---

## 11. 接口要求

### 11.1 调度静态信息

```python
def get_batch_first_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> BatchFirstScheduleInfo:
    ...
```

### 11.2 位置计算

```python
def get_band_batch_first_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Tuple[int, int, int]]:
    ...
```

返回：

```text
(batch_id, x, y)
```

当前核在当前轮次无有效任务时返回 `None`。

### 11.3 batch 轮次范围

```python
def get_batch_round_range(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
        batch_id: int,
) -> Tuple[int, int]:
    ...
```

### 11.4 矩阵构建

```python
def build_batch_first_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        validate: bool = True,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    ...
```

### 11.5 causal 兼容接口

```python
def get_causal_batch_first_position(
        m: int,
        n: int,
        b: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Tuple[int, int, int]]:
    ...
```

该接口固定使用：

```text
p=m, q=1
```

---

## 12. 验收条件

### 12.1 完整覆盖

所有满足：

$$
1-q\le x-y\le p-1
$$

的块必须被分配一次。

### 12.2 不重复

同一个 `(batch_id,x,y)` 不能被不同核或不同轮次重复分配。

### 12.3 无效块不执行

所有无效位置在 `rounds_cube` 和 `core_cube` 中保持 `-1`。

### 12.4 同行轮次唯一

对固定的 `batch_id` 和 `x`，所有有效列的 `round_id` 必须互不相同。

### 12.5 单轮 batch 唯一

对任意全局 `round_id`，实际执行任务的 `batch_id` 种类最多为 1。

### 12.6 batch 顺序严格递增

只有当前一个 batch 的全部 `R_batch` 轮结束后，才允许进入下一个 batch。

### 12.7 确定性

相同参数重复构建，`rounds_cube` 和 `core_cube` 必须完全一致。

### 12.8 causal 兼容

当：

```text
p=m, q=1
```

有效区域必须严格等价于 causal 下三角。

---

## 13. 验证结果

当前实现已完成以下验证：

- Python 语法编译检查通过；
- 示例参数完整覆盖检查通过；
- 同行轮次唯一检查通过；
- 单轮 batch 唯一检查通过；
- batch 连续轮次区间检查通过；
- 两次构建的确定性检查通过；
- 18,816 组小尺寸参数组合校验通过。

---

## 14. 当前完整实现代码

实现文件：`FAG_sparse03_band_batch_first_deter.py`

```python
"""
FlashAttentionScoreGrad SparseMode4 通用斜带区域确定性分核：Batch-First 方案。

目标：
1. 所有核优先协同完成当前 batch 的 [m, n] 有效区域；
2. 当前 batch 完成后，才进入下一个 batch；
3. 同一 batch、同一轮次内，同一行最多出现一个有效任务；
4. (core_id, round_id) 可直接、确定性地反算 (batch_id, x, y)。

矩阵坐标采用 1-based：
    1 <= x <= m, 1 <= y <= n

有效块满足：
    1 - q <= x - y <= p - 1

当 p=m、q=1 时，退化为 causal 下三角区域 y <= x。
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
class BatchFirstScheduleInfo:
    """Batch-First 调度静态信息。"""

    k: int
    m: int
    n: int
    b: int
    p: int
    q: int
    effective_n: int
    diagonal_count: int
    column_groups_per_batch: int
    rounds_per_batch: int
    total_rounds: int
    rotation_stride: int


@dataclass(frozen=True)
class BatchFirstScheduleStats:
    """调度正确性与负载统计。"""

    valid_blocks_per_batch: int
    total_valid_blocks: int
    min_core_load: int
    max_core_load: int
    core_loads: Tuple[int, ...]


# ---------------------------------------------------------------------------
# 参数与有效区域
# ---------------------------------------------------------------------------


def _validate_common_parameters(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> None:
    """校验公共输入参数。"""

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
    """判断 1-based 坐标 (x, y) 是否属于 p/q 指定的有效斜带。"""

    return (
        1 <= x <= m
        and 1 <= y <= n
        and 1 - q <= x - y <= p - 1
    )


def count_valid_blocks_per_batch(m: int, n: int, p: int, q: int) -> int:
    """计算单个 batch 内的有效块数量。"""

    _validate_common_parameters(m, n, 1, p, q, 1)
    total = 0
    for x in range(1, m + 1):
        y_min = max(1, x - p + 1)
        y_max = min(n, x + q - 1)
        if y_min <= y_max:
            total += y_max - y_min + 1
    return total


def _find_coprime_rotation_stride(k: int) -> int:
    """选择与 k 互质的确定性核旋转步长。"""

    if k == 1:
        return 0

    stride = k // 2 + 1
    while gcd(stride, k) != 1:
        stride += 1
    return stride % k


def get_batch_first_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> BatchFirstScheduleInfo:
    """计算 Batch-First 调度静态信息。"""

    _validate_common_parameters(m, n, b, p, q, k)

    # 根据 x-y >= 1-q 且 x<=m，可得 y<=m+q-1。
    # 因此右侧超过该范围的列一定没有有效块。
    effective_n = min(n, m + q - 1)

    # 每个列组依次遍历全部有效斜对角偏移：
    # d = 1-q, 2-q, ..., p-1。
    diagonal_count = p + q - 1

    # 每个 batch 独立按 k 列一组切分，不与其他 batch 拼组。
    column_groups_per_batch = ceil(effective_n / k)
    rounds_per_batch = column_groups_per_batch * diagonal_count
    total_rounds = b * rounds_per_batch

    return BatchFirstScheduleInfo(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        effective_n=effective_n,
        diagonal_count=diagonal_count,
        column_groups_per_batch=column_groups_per_batch,
        rounds_per_batch=rounds_per_batch,
        total_rounds=total_rounds,
        rotation_stride=_find_coprime_rotation_stride(k),
    )


def get_batch_round_range(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
        batch_id: int,
) -> Tuple[int, int]:
    """返回指定 batch 对应的全局轮次闭区间。"""

    info = get_batch_first_schedule_info(m, n, b, p, q, k)
    if not isinstance(batch_id, int):
        raise TypeError("batch_id 必须是 int")
    if not 1 <= batch_id <= b:
        raise ValueError(f"batch_id 必须满足 1 <= batch_id <= b，当前为 {batch_id}")

    start = (batch_id - 1) * info.rounds_per_batch + 1
    end = batch_id * info.rounds_per_batch
    return start, end


def get_batch_first_max_rounds(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> int:
    """返回 Batch-First 方案的全局总轮次数。"""

    return get_batch_first_schedule_info(m, n, b, p, q, k).total_rounds


# ---------------------------------------------------------------------------
# 核心确定性映射
# ---------------------------------------------------------------------------


def _get_batch_first_position_from_info(
        info: BatchFirstScheduleInfo,
        core_id: int,
        round_id: int,
) -> Optional[Position]:
    """使用预计算 info，由 (core_id, round_id) 直接计算任务位置。"""

    if round_id > info.total_rounds:
        return None

    round0 = round_id - 1

    # 全局轮次先定位 batch。一个 batch 的全部 rounds_per_batch 轮完成后，
    # 才会进入下一个 batch。
    batch0 = round0 // info.rounds_per_batch
    round_in_batch = round0 % info.rounds_per_batch

    group_in_batch = round_in_batch // info.diagonal_count
    diagonal_id = round_in_batch % info.diagonal_count

    group_start_y0 = group_in_batch * info.k
    group_size = min(info.k, info.effective_n - group_start_y0)
    if group_size <= 0:
        return None

    # 同一个 batch 内，相邻列组采用正序/逆序蛇形分配。
    # 每进入一个新 batch，再按 rotation_stride 旋转核映射：
    # - 不改变“所有核只处理当前 batch”的约束；
    # - 避免不满列组和边界重列永久落到同一批核。
    pair_rotation = (
        batch0 * info.rotation_stride
        + (group_in_batch // 2) * info.rotation_stride
    ) % info.k

    permuted_core = ((core_id - 1) - pair_rotation) % info.k
    if group_in_batch % 2 == 0:
        local_slot = permuted_core
    else:
        local_slot = info.k - 1 - permuted_core

    if local_slot >= group_size:
        return None

    y = group_start_y0 + local_slot + 1

    # 固定 round 内，对当前 batch 的所有核，diagonal 均相同。
    # 由于 x = y + diagonal，不同 y 一定得到不同 x，因此同行无冲突。
    diagonal = 1 - info.q + diagonal_id
    x = y + diagonal

    if not is_valid_block(x, y, info.m, info.n, info.p, info.q):
        return None

    return batch0 + 1, x, y


def get_band_batch_first_position(
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
    Batch-First 位置计算接口。

    返回：
        (batch_id, x, y)，全部采用 1-based。

    当前核在当前轮次无有效任务时返回 None。

    轮次分解：
        rounds_per_batch = ceil(effective_n / k) * (p + q - 1)
        batch0           = (round_id - 1) // rounds_per_batch
        round_in_batch   = (round_id - 1) %  rounds_per_batch
        group            = round_in_batch // (p + q - 1)
        diagonal_id      = round_in_batch %  (p + q - 1)
        d                = 1 - q + diagonal_id
        x                = y + d
    """

    info = get_batch_first_schedule_info(m, n, b, p, q, k)

    if not isinstance(core_id, int) or not isinstance(round_id, int):
        raise TypeError("core_id 和 round_id 必须是 int")
    if not 1 <= core_id <= k:
        raise ValueError(f"core_id 必须满足 1 <= core_id <= k，当前为 {core_id}")
    if round_id < 1:
        raise ValueError(f"round_id 必须大于等于 1，当前为 {round_id}")

    return _get_batch_first_position_from_info(info, core_id, round_id)


def get_causal_batch_first_position(
        m: int,
        n: int,
        b: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    """causal 下三角 Batch-First 兼容接口，等价于 p=m、q=1。"""

    return get_band_batch_first_position(
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
# 构建、校验与统计
# ---------------------------------------------------------------------------


def build_batch_first_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        validate: bool = True,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """构建全部 batch 的 round/core 矩阵。"""

    info = get_batch_first_schedule_info(m, n, b, p, q, k)
    rounds_cube = np.full((b, m, n), -1, dtype=np.int64)
    core_cube = np.full((b, m, n), -1, dtype=np.int64)

    for round_id in range(1, info.total_rounds + 1):
        for core_id in range(1, k + 1):
            pos = _get_batch_first_position_from_info(info, core_id, round_id)
            if pos is None:
                continue

            batch_id, x, y = pos
            index = (batch_id - 1, x - 1, y - 1)
            if rounds_cube[index] != -1 or core_cube[index] != -1:
                raise AssertionError(
                    "检测到重复任务："
                    f"position={(batch_id, x, y)}, "
                    f"old=(round={rounds_cube[index]}, core={core_cube[index]}), "
                    f"new=(round={round_id}, core={core_id})"
                )

            rounds_cube[index] = round_id
            core_cube[index] = core_id

    if validate:
        validate_batch_first_schedule(
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


def get_core_loads(core_cube: np.ndarray, k: int) -> np.ndarray:
    """统计每个核实际执行的有效块数量。"""

    if core_cube.ndim != 3:
        raise ValueError(f"core_cube 必须是三维数组，当前 ndim={core_cube.ndim}")
    if k <= 0:
        raise ValueError("k 必须大于 0")

    return np.asarray(
        [np.count_nonzero(core_cube == core_id) for core_id in range(1, k + 1)],
        dtype=np.int64,
    )


def validate_batch_first_schedule(
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> BatchFirstScheduleStats:
    """
    校验：
    1. 有效块完整覆盖且仅分配一次；
    2. 无效块不参与计算；
    3. 同一 batch 的同一行不存在重复 round；
    4. 一个全局 round 最多属于一个 batch；
    5. batch 的全局轮次区间严格连续且按 batch_id 递增。
    """

    info = get_batch_first_schedule_info(m, n, b, p, q, k)
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
        raise AssertionError(f"发现非法核编号：{bad[:8].tolist()}")

    # 同一 batch、同一行的所有有效列必须使用不同 round。
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

    # Batch-First 专属检查：每个 batch 只能使用自己的连续轮次区间。
    for batch0 in range(b):
        start = batch0 * info.rounds_per_batch + 1
        end = (batch0 + 1) * info.rounds_per_batch
        batch_rounds = rounds_cube[batch0]
        assigned = batch_rounds[batch_rounds >= 1]
        if assigned.size == 0:
            raise AssertionError(f"batch={batch0 + 1} 没有任何有效任务")
        if np.any((assigned < start) | (assigned > end)):
            bad = assigned[(assigned < start) | (assigned > end)]
            raise AssertionError(
                f"batch={batch0 + 1} 使用了区间外轮次："
                f"expected=[{start}, {end}], bad={bad[:8].tolist()}"
            )

    # 一个全局 round 中，所有实际任务必须来自同一个 batch。
    for round_id in range(1, info.total_rounds + 1):
        batch_ids = np.argwhere(rounds_cube == round_id)[:, 0]
        if batch_ids.size == 0:
            continue
        unique_batches = np.unique(batch_ids)
        if unique_batches.size != 1:
            raise AssertionError(
                f"round={round_id} 同时出现多个 batch："
                f"{(unique_batches + 1).tolist()}"
            )

        expected_batch0 = (round_id - 1) // info.rounds_per_batch
        if int(unique_batches[0]) != expected_batch0:
            raise AssertionError(
                f"round={round_id} 的 batch 顺序错误："
                f"expected={expected_batch0 + 1}, actual={int(unique_batches[0]) + 1}"
            )

    valid_per_batch = count_valid_blocks_per_batch(m, n, p, q)
    total_valid = b * valid_per_batch
    if int(np.count_nonzero(valid_mask)) != total_valid:
        raise AssertionError("有效块总数计算不一致")

    core_loads = get_core_loads(core_cube, k)
    return BatchFirstScheduleStats(
        valid_blocks_per_batch=valid_per_batch,
        total_valid_blocks=total_valid,
        min_core_load=int(core_loads.min()),
        max_core_load=int(core_loads.max()),
        core_loads=tuple(int(v) for v in core_loads),
    )


def verify_batch_first_determinism(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
) -> bool:
    """使用相同参数构建两次矩阵，验证结果完全一致。"""

    first = build_batch_first_matrices(k, m, n, b, p, q, validate=True)
    second = build_batch_first_matrices(k, m, n, b, p, q, validate=True)
    return np.array_equal(first[2], second[2]) and np.array_equal(first[3], second[3])


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def visualize_batch_first_schedule(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        batches_per_figure: int = 2,
        ncols: int = 2,
        annotate_round: bool = True,
        save_dir: Optional[str | Path] = None,
        dpi: int = 220,
        show: bool = True,
) -> Dict[str, object]:
    """分页可视化 Batch-First 调度。"""

    info = get_batch_first_schedule_info(m, n, b, p, q, k)
    k, total_rounds, rounds_cube, core_cube = build_batch_first_matrices(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        validate=True,
    )
    stats = validate_batch_first_schedule(
        rounds_cube, core_cube, m, n, b, p, q, k
    )

    print("调度策略: batch-first")
    print("所需核数:", k)
    print("每个 batch 轮次数:", info.rounds_per_batch)
    print("总轮次数:", total_rounds)
    print("单 batch 有效块数:", stats.valid_blocks_per_batch)
    print("总有效块数:", stats.total_valid_blocks)
    print("各核有效任务数:", list(stats.core_loads))
    print("核负载范围:", (stats.min_core_load, stats.max_core_load))
    for batch_id in range(1, b + 1):
        print(f"batch={batch_id} 轮次区间:", get_batch_round_range(m, n, b, p, q, k, batch_id))

    palette = ["#F3F4F6"] + [
        "#0B84A5", "#EBC262", "#6F4E7C", "#9DD866", "#CA472F",
        "#FFA056", "#8DDDD0", "#BFB5FF", "#3C5488", "#F39C12",
        "#27AE60", "#D35400", "#16A085", "#7F8C8D", "#2E86C1",
        "#E74C3C", "#8E44AD", "#2ECC71", "#34495E", "#F1C40F",
    ]
    if k + 1 > len(palette):
        extra = plt.get_cmap("tab20", k + 1 - len(palette)).colors
        palette.extend(extra)
    cmap = ListedColormap(palette[:k + 1], name="batch_first_band")

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

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * axes_w + 1.2, rows * axes_h + 1.4),
            squeeze=False,
        )

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

            round_range = get_batch_round_range(m, n, b, p, q, k, batch_id)
            ax.set_title(
                f"batch={batch_id}, rounds={round_range[0]}-{round_range[1]}",
                fontsize=10,
                pad=8,
            )

        for idx in range(len(page_batches), rows * cols):
            axes[idx // cols][idx % cols].axis("off")

        fig.suptitle(
            f"Batch-First Band Schedule | k={k}, m={m}, n={n}, b={b}, "
            f"p={p}, q={q} | page {page_idx + 1}/{total_pages}",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))

        if save_path is not None:
            fig.savefig(
                save_path / f"FAG_sparse04_batch_first_p{p}_q{q}_page_{page_idx + 1}.png",
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
        "rounds_per_batch": info.rounds_per_batch,
        "rounds_cube": rounds_cube,
        "core_cube": core_cube,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# 示例
# ---------------------------------------------------------------------------


def _demo() -> None:
    k, m, n, b = 32, 32, 32, 8
    p, q = 16, 4

    output_dir = Path(__file__).parent / "outputs_batch_first"
    result = visualize_batch_first_schedule(
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

    print("确定性复验:", verify_batch_first_determinism(k, m, n, b, p, q))
    print("输出目录:", output_dir)
    print("结果键:", list(result.keys()))


if __name__ == "__main__":
    _demo()
```
