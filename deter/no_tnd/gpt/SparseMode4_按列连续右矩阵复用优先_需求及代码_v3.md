# SparseMode4 按列连续、右矩阵复用优先的确定性分核方案及代码

## 1. 问题背景

当前分列组交错方案虽然减少了单核有效任务之间的轮次空洞，但会让同一个核在相邻轮次中频繁切换列组。例如一个核可能在：

```text
y=1 -> y=64 -> y=1 -> y=64
```

之间逐轮切换。

对于按列分核的 MatMul 场景，列坐标 `y` 往往决定右矩阵块。逐轮切换列会降低右矩阵驻留和复用机会，并增加数据搬运、流水线切换和工作集抖动。

现要求：

1. 同一个 `(batch, y)` 列始终归属于唯一 core；
2. 一个 core 应尽量连续完成当前列，再切换到下一列；
3. 列内的 `x` 尽量按照递增 1 或递减 1 的顺序处理；
4. 保持同行互斥、同核单任务、完整覆盖和确定性；
5. 不引入随 `m` 增长的全局 matching 表；
6. 总轮数不应高于现有分列组解析方案。

---

## 2. 结论

可以实现，但需要明确两个层次的“连续”。

### 2.1 列归属和右矩阵复用可以严格保证

默认模式下，一个列组的全部 local round 连续执行。

因此在同一个列组内：

- 一个 core 不会切换到另一列；
- 同一列对应的右矩阵块可以跨多个连续 round 复用；
- 完成当前列组后才切换下一列组。

### 2.2 所有列都完全单调不可同时保证

在一个满 `k × k` 列组中，如果要求：

- 只使用 `k` 轮；
- 每轮 `k` 核全活跃；
- 同一行在同一轮不能重复；
- 每列都严格从 `x=1` 到 `x=k` 单调递增或递减；

则一般不可能同时满足。

原因是：满列的单调递增映射只能近似使用 `round=x`，单调递减只能近似使用 `round=k+1-x`。若大量列采用相同映射，同一 round 会命中相同行，违反确定性同行互斥。

因此，本方案采用：

- 每列固定 owner；
- 每列连续驻留；
- 每列最多一次循环回绕；
- 对 snake 方向上的锚点列保证严格单调。

对于用户关心的第 0 核边界列，可以实现完全连续。

---

## 3. 核心方案

### 3.1 列组划分

有效列数：

$$
effectiveN = \min(n, m+q-1)
$$

每 `k` 列组成一个列组：

$$
group(y)=\left\lfloor\frac{y-1}{k}\right\rfloor
$$

每组最多 `k` 列，每个 core 在一个列组内最多拥有一列。

### 3.2 固定列 owner

沿用列组 snake、group rotation 和 batch rotation。

同一个 `(batch,y)` 的 owner core 一旦确定，与：

- `x`
- `round`
- local round

均无关。

因此整列不会跨核。

### 3.3 列组理论最小轮数

每个列组构造成“行节点—owner core 节点”的二分多重图。

列组轮数取最大度：

$$
\Delta_g =
\max(
\text{组内最大行任务数},
\text{组内最大列高度}
)
$$

每组使用恰好 `delta_g` 轮，不高于旧方案的 `p+q-1`。

### 3.4 偶数列组正向解析

令：

$$
d=x-y
$$

偶数列组使用：

$$
localRound=(d-phase_g)\bmod\Delta_g
$$

`phase_g` 选择该组第一列的有效下界，使锚点列按 `x` 递增。

### 3.5 奇数列组反向解析

奇数列组使用：

$$
localRound=(phase_g-d)\bmod\Delta_g
$$

`phase_g` 选择该组最后一列的有效上界，使 snake 后映射给边界 core 的锚点列按 `x` 递减。

### 3.6 默认采用完整列组连续执行

batch 内轮次顺序为：

```text
group 0: local 0, 1, 2, ..., delta_0-1
group 1: local 0, 1, 2, ..., delta_1-1
...
```

这样一个 core 在整个列组期间固定使用同一列，右矩阵复用最好。

---

## 4. 可选折中：round chunk

实现提供参数：

```text
round_chunk
```

含义：

- `round_chunk <= 0`：完整列组连续执行，右矩阵复用最佳；
- `round_chunk = 4/8/...`：连续处理若干 local round 后切换列组；
- 总轮数不变。

例如 `round_chunk=8`：

```text
g0 local 0..7
g1 local 0..7
g0 local 8..15
g1 local 8..15
...
```

推荐优先使用：

```text
round_chunk = 0
```

只有在设备测试表明跨列组串行产生其他流水问题时，再测试 4 或 8。

---

## 5. 关键场景结果

参数：

```text
k=32
m=32
n=64
b=8
p=32
q=63
```

计算结果：

```text
effectiveN       = 64
group_count      = 2
group_deltas     = [32, 32]
rounds_per_batch = 64
total_rounds     = 512
full_rounds      = 504
partial_rounds   = 8
empty_rounds     = 0
slot_utilization = 99.951172%
```

0-based 第 0 核对应代码中的 `core_id=1`。

batch 1 的执行顺序：

```text
r1  -> (x=1,  y=1)
r2  -> (x=2,  y=1)
...
r32 -> (x=32, y=1)

r33 -> (x=32, y=64)
r34 -> (x=31, y=64)
...
r63 -> (x=2,  y=64)
r64 -> idle
```

其中 `(x=1,y=64)` 无效，因为：

$$
x-y=-63 < 1-q=-62
$$

因此 `r64` 空闲是有效区域本身造成的，不是排布碎片化。

---

## 6. 与逐轮交错方案对比

| 项目 | 逐轮列组交错 | 本方案 |
|---|---|---|
| 总轮数 | 64/批 | 64/批 |
| 槽位利用率 | 99.951% | 99.951% |
| 固定列 owner | 是 | 是 |
| 相邻轮次切换列组 | 几乎每轮 | 一个列组结束后 |
| 右矩阵连续复用 | 较差 | 最好 |
| 锚点列 `x` 顺序 | 来回切换 | 严格递增/递减 |
| 全局 matching 表 | 不需要 | 不需要 |

---

## 7. Kernel 迁移建议

Kernel 侧只需增加或调整以下信息：

```cpp
struct GroupInfo {
    int64_t delta;
    int64_t direction;  // +1 or -1
    int64_t phase;
};
```

偶数组：

```cpp
residue = FloorMod(phase + localRound, delta);
```

奇数组：

```cpp
residue = FloorMod(phase - localRound, delta);
```

再在列的有效连续 `d` 区间中查找唯一满足：

```text
d mod delta == residue
```

的 `d`。

默认轮次状态机使用 group-major：

```text
group 0 全部 local round
group 1 全部 local round
...
```

不再使用逐 local-round 交错状态机。

---

## 8. 验证结果

当前 Python 实现已验证：

- 有效块完整覆盖；
- 无重复分配；
- 无效块不分配；
- 同一 batch、同一 round 无同行冲突；
- 同一 core、同一 round 最多一个任务；
- 同一 `(batch,y)` 始终固定 owner；
- 代表场景轮数和利用率不变；
- 42,336 组小尺寸参数组合通过。

---

## 9. 完整 Python 代码

```python
from __future__ import annotations

"""
SparseMode4 Batch-First 按列连续（Column-Contiguous）确定性分核参考实现。

设计目标
--------
1. 严格按列分核：同一个 (batch, y) 的全部有效块始终归属于同一个 core。
2. 右矩阵复用优先：同一 core 在一个列组内连续处理同一列，不逐轮切换列组。
3. 列内顺序尽量连续：
   - 偶数列组的锚点列按 x 递增；
   - 奇数列组的锚点列按 x 递减；
   - 其他列最多发生一次循环回绕，这是满核、同行互斥和最小轮数共同约束下不可完全避免的。
4. 保持确定性：
   - 同一 round、同一 batch、同一行最多一个任务；
   - 同一 core、同一 round 最多一个任务；
   - 不使用动态抢占或运行时 matching。
5. 不保存随 m 增长的二维 matching 表。
6. 支持可选 round_chunk：
   - round_chunk <= 0：完整列组连续执行，右矩阵复用最佳；
   - round_chunk > 0：每次连续执行若干 local round，再切换列组，作为局部性与时间交错的折中。

核心思想
--------
每个列组最多包含 k 列，每列固定绑定一个 owner core。
组内使用解析边着色：

偶数列组：
    local_round = (d - phase) mod delta

奇数列组：
    local_round = (phase - d) mod delta

其中 d = x - y，delta 为该列组二分图最大度。
phase 选择列组蛇形方向上的锚点列边界，使：
- group 0 的第一个 owner 列从上到下连续；
- group 1 的第一个 owner 列从下到上连续；
- 后续列组交替。

在默认 group-major 模式下，同一个 core 在 delta 个组内轮次中不会切换到另一列，
有利于在 MatMul 中保留该列对应的右矩阵块。
"""

from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

Position = Tuple[int, int, int]
RoundMeta = Tuple[int, int]  # (group_id0, local_round0)


@dataclass(frozen=True)
class GroupInfo:
    group_id0: int
    y_start: int
    y_end: int
    group_size: int
    delta: int
    direction: int  # +1: local round 对应 d 递增；-1: d 递减
    phase: int      # 模 delta 的相位


@dataclass(frozen=True)
class ScheduleInfo:
    k: int
    m: int
    n: int
    b: int
    p: int
    q: int
    effective_n: int
    diagonal_count: int
    group_count: int
    groups: Tuple[GroupInfo, ...]
    rounds_per_batch: int
    total_rounds: int
    rotation_stride: int
    round_chunk: int
    round_meta: Tuple[RoundMeta, ...]


def _validate_parameters(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
        round_chunk: int,
) -> None:
    values = {"m": m, "n": n, "b": b, "p": p, "q": q, "k": k}
    for name, value in values.items():
        if not isinstance(value, int):
            raise TypeError(f"{name} 必须是 int，当前为 {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0，当前为 {value}")
    if not isinstance(round_chunk, int):
        raise TypeError("round_chunk 必须是 int")
    if p > m:
        raise ValueError(f"p 必须满足 1 <= p <= m，当前 p={p}, m={m}")
    if q > n:
        raise ValueError(f"q 必须满足 1 <= q <= n，当前 q={q}, n={n}")


def is_valid_block(
        x: int,
        y: int,
        m: int,
        n: int,
        p: int,
        q: int,
) -> bool:
    return (
        1 <= x <= m
        and 1 <= y <= n
        and 1 - q <= x - y <= p - 1
    )


def _find_coprime_rotation_stride(k: int) -> int:
    if k == 1:
        return 0
    stride = k // 2 + 1
    while gcd(stride, k) != 1:
        stride += 1
    return stride % k


def _column_d_range(y: int, m: int, p: int, q: int) -> Tuple[int, int]:
    """返回列 y 的有效 d=x-y 连续区间。"""
    d_lo = max(1 - q, 1 - y)
    d_hi = min(p - 1, m - y)
    return d_lo, d_hi


def _column_degree(y: int, m: int, p: int, q: int) -> int:
    d_lo, d_hi = _column_d_range(y, m, p, q)
    return max(0, d_hi - d_lo + 1)


def _compute_group_delta(
        y_start: int,
        y_end: int,
        m: int,
        p: int,
        q: int,
) -> int:
    """
    列组二分多重图最大度：
      delta_g = max(最大行度, 最大 owner-core 度)

    每个组最多 k 列，固定列 owner 后，每个 core 在该组最多拥有一列。
    """
    group_size = y_end - y_start + 1
    max_row_degree = min(group_size, p + q - 1)
    max_core_degree = max(
        _column_degree(y, m, p, q)
        for y in range(y_start, y_end + 1)
    )
    return max(max_row_degree, max_core_degree)


def _build_group_info(
        group_id0: int,
        y_start: int,
        y_end: int,
        m: int,
        p: int,
        q: int,
) -> GroupInfo:
    delta = _compute_group_delta(y_start, y_end, m, p, q)

    if group_id0 % 2 == 0:
        # 偶数组按 y 正向映射。锚点为 y_start，使该锚点列从 x_lo 向 x_hi 递增。
        anchor_y = y_start
        anchor_d_lo, _ = _column_d_range(anchor_y, m, p, q)
        direction = +1
        phase = anchor_d_lo % delta
    else:
        # 奇数组按 y 反向 snake。锚点为 y_end，使该锚点列从 x_hi 向 x_lo 递减。
        anchor_y = y_end
        _, anchor_d_hi = _column_d_range(anchor_y, m, p, q)
        direction = -1
        phase = anchor_d_hi % delta

    return GroupInfo(
        group_id0=group_id0,
        y_start=y_start,
        y_end=y_end,
        group_size=y_end - y_start + 1,
        delta=delta,
        direction=direction,
        phase=phase,
    )


def _build_round_meta(
        groups: Sequence[GroupInfo],
        round_chunk: int,
) -> Tuple[RoundMeta, ...]:
    """
    构造 batch 内轮次顺序。

    round_chunk <= 0:
        group-major，完整执行一个列组后再切换。右矩阵复用最佳。

    round_chunk > 0:
        以 local-round chunk 为单位交错列组。例如 chunk=8：
          g0[0:8], g1[0:8], g0[8:16], g1[8:16], ...
        总轮数不变。
    """
    if round_chunk <= 0:
        return tuple(
            (group.group_id0, local_round0)
            for group in groups
            for local_round0 in range(group.delta)
        )

    max_delta = max(group.delta for group in groups)
    result: List[RoundMeta] = []
    for chunk_start in range(0, max_delta, round_chunk):
        chunk_end = chunk_start + round_chunk
        for group in groups:
            local_end = min(chunk_end, group.delta)
            for local_round0 in range(chunk_start, local_end):
                result.append((group.group_id0, local_round0))
    return tuple(result)


def build_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
        round_chunk: int = 0,
) -> ScheduleInfo:
    _validate_parameters(m, n, b, p, q, k, round_chunk)

    effective_n = min(n, m + q - 1)
    group_count = ceil(effective_n / k)

    groups: List[GroupInfo] = []
    for group_id0 in range(group_count):
        y_start = group_id0 * k + 1
        y_end = min(effective_n, y_start + k - 1)
        groups.append(
            _build_group_info(
                group_id0=group_id0,
                y_start=y_start,
                y_end=y_end,
                m=m,
                p=p,
                q=q,
            )
        )

    round_meta = _build_round_meta(groups, round_chunk)
    rounds_per_batch = sum(group.delta for group in groups)
    if len(round_meta) != rounds_per_batch:
        raise AssertionError(
            f"round_meta 数量错误: {len(round_meta)} != {rounds_per_batch}"
        )

    return ScheduleInfo(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        effective_n=effective_n,
        diagonal_count=p + q - 1,
        group_count=group_count,
        groups=tuple(groups),
        rounds_per_batch=rounds_per_batch,
        total_rounds=b * rounds_per_batch,
        rotation_stride=_find_coprime_rotation_stride(k),
        round_chunk=round_chunk,
        round_meta=round_meta,
    )


def _group_rotation(
        info: ScheduleInfo,
        batch0: int,
        group_id0: int,
) -> int:
    return (
        batch0 * info.rotation_stride
        + (group_id0 // 2) * info.rotation_stride
    ) % info.k


def _core_to_group_local_slot(
        info: ScheduleInfo,
        batch0: int,
        group_id0: int,
        core0: int,
) -> int:
    rotation = _group_rotation(info, batch0, group_id0)
    permuted_core = (core0 - rotation) % info.k
    if group_id0 % 2 == 0:
        return permuted_core
    return info.k - 1 - permuted_core


def get_column_owner_core(
        info: ScheduleInfo,
        batch_id: int,
        y: int,
) -> int:
    """返回物理列 (batch_id, y) 固定归属的 1-based core。"""
    if not 1 <= batch_id <= info.b:
        raise ValueError("batch_id 越界")
    if not 1 <= y <= info.effective_n:
        raise ValueError("y 越界或该列没有有效块")

    batch0 = batch_id - 1
    group_id0 = (y - 1) // info.k
    local_slot = (y - 1) % info.k
    rotation = _group_rotation(info, batch0, group_id0)

    if group_id0 % 2 == 0:
        permuted_core = local_slot
    else:
        permuted_core = info.k - 1 - local_slot

    return (permuted_core + rotation) % info.k + 1


def _find_d_for_residue(
        y: int,
        residue: int,
        delta: int,
        m: int,
        p: int,
        q: int,
) -> Optional[int]:
    """
    在列 y 的有效连续 d 区间中寻找唯一满足 d mod delta = residue 的 d。

    因为列有效高度 <= delta，所以解最多一个。
    """
    d_lo, d_hi = _column_d_range(y, m, p, q)
    if d_lo > d_hi:
        return None

    d = d_lo + ((residue - d_lo) % delta)
    if d > d_hi:
        return None
    return d


def _group_residue(group: GroupInfo, local_round0: int) -> int:
    if group.direction > 0:
        return (group.phase + local_round0) % group.delta
    return (group.phase - local_round0) % group.delta


def get_column_contiguous_position_from_info(
        info: ScheduleInfo,
        core_id: int,
        round_id: int,
) -> Optional[Position]:
    """
    根据 1-based (core_id, round_id) 确定性反算 (batch_id, x, y)。

    同一个列组连续轮次内，一个 core 固定处理同一列。
    """
    if not 1 <= core_id <= info.k:
        raise ValueError(f"core_id 必须满足 1 <= core_id <= {info.k}")
    if round_id < 1:
        raise ValueError("round_id 必须 >= 1")
    if round_id > info.total_rounds:
        return None

    round0 = round_id - 1
    batch0 = round0 // info.rounds_per_batch
    round_in_batch0 = round0 % info.rounds_per_batch

    group_id0, local_round0 = info.round_meta[round_in_batch0]
    group = info.groups[group_id0]

    local_slot = _core_to_group_local_slot(
        info=info,
        batch0=batch0,
        group_id0=group_id0,
        core0=core_id - 1,
    )
    if local_slot >= group.group_size:
        return None

    y = group.y_start + local_slot
    residue = _group_residue(group, local_round0)
    d = _find_d_for_residue(
        y=y,
        residue=residue,
        delta=group.delta,
        m=info.m,
        p=info.p,
        q=info.q,
    )
    if d is None:
        return None

    x = y + d
    if not is_valid_block(x, y, info.m, info.n, info.p, info.q):
        raise AssertionError("解析公式生成了无效块")

    return batch0 + 1, x, y


def get_band_column_contiguous_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int,
        round_chunk: int = 0,
) -> Optional[Position]:
    info = build_schedule_info(m, n, b, p, q, k, round_chunk)
    return get_column_contiguous_position_from_info(info, core_id, round_id)


def build_schedule_matrices(
        info: ScheduleInfo,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounds_cube = np.full(
        (info.b, info.m, info.n),
        -1,
        dtype=np.int64,
    )
    core_cube = np.full(
        (info.b, info.m, info.n),
        -1,
        dtype=np.int64,
    )
    active_cores_per_round = np.zeros(
        info.total_rounds,
        dtype=np.int64,
    )

    for round_id in range(1, info.total_rounds + 1):
        seen_rows = set()
        for core_id in range(1, info.k + 1):
            pos = get_column_contiguous_position_from_info(
                info,
                core_id,
                round_id,
            )
            if pos is None:
                continue

            batch_id, x, y = pos
            row_key = (batch_id, x)
            if row_key in seen_rows:
                raise AssertionError(
                    f"同行冲突: round={round_id}, batch={batch_id}, x={x}"
                )
            seen_rows.add(row_key)

            idx = (batch_id - 1, x - 1, y - 1)
            if rounds_cube[idx] != -1:
                raise AssertionError(
                    f"任务重复分配: batch={batch_id}, x={x}, y={y}"
                )

            rounds_cube[idx] = round_id
            core_cube[idx] = core_id
            active_cores_per_round[round_id - 1] += 1

    return rounds_cube, core_cube, active_cores_per_round


def _column_active_trace(
        rounds_cube: np.ndarray,
        batch0: int,
        y0: int,
) -> Tuple[List[int], List[int]]:
    entries: List[Tuple[int, int]] = []
    for x0 in range(rounds_cube.shape[1]):
        rid = int(rounds_cube[batch0, x0, y0])
        if rid >= 1:
            entries.append((rid, x0 + 1))
    entries.sort()
    return [item[0] for item in entries], [item[1] for item in entries]


def _count_direction_changes(values: Sequence[int]) -> int:
    if len(values) < 3:
        return 0
    signs: List[int] = []
    for left, right in zip(values, values[1:]):
        diff = right - left
        if diff > 0:
            signs.append(1)
        elif diff < 0:
            signs.append(-1)
    return sum(
        1
        for left, right in zip(signs, signs[1:])
        if left != right
    )


def validate_schedule(
        info: ScheduleInfo,
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
) -> dict:
    expected = 0

    for batch_id in range(1, info.b + 1):
        for x in range(1, info.m + 1):
            row_rounds: List[int] = []
            for y in range(1, info.n + 1):
                valid = is_valid_block(
                    x,
                    y,
                    info.m,
                    info.n,
                    info.p,
                    info.q,
                )
                rid = int(rounds_cube[batch_id - 1, x - 1, y - 1])
                cid = int(core_cube[batch_id - 1, x - 1, y - 1])

                if valid:
                    expected += 1
                    if rid < 1 or not 1 <= cid <= info.k:
                        raise AssertionError(
                            f"有效块未覆盖: {(batch_id, x, y)}"
                        )
                    row_rounds.append(rid)

                    owner = get_column_owner_core(info, batch_id, y)
                    if cid != owner:
                        raise AssertionError(
                            "列 owner 发生变化: "
                            f"{(batch_id, x, y)}, got={cid}, expected={owner}"
                        )
                elif rid != -1 or cid != -1:
                    raise AssertionError(
                        f"无效块被分配: {(batch_id, x, y)}"
                    )

            if len(row_rounds) != len(set(row_rounds)):
                raise AssertionError(
                    f"同行 round 重复: batch={batch_id}, x={x}"
                )

    assigned = int(np.count_nonzero(rounds_cube >= 1))
    if assigned != expected:
        raise AssertionError(
            f"覆盖数量不一致: assigned={assigned}, expected={expected}"
        )

    # 同一物理列必须始终由唯一 core 负责。
    for batch0 in range(info.b):
        for y0 in range(info.effective_n):
            core_values = core_cube[batch0, :, y0]
            core_values = core_values[core_values >= 1]
            if len(core_values) and len(np.unique(core_values)) != 1:
                raise AssertionError(
                    f"同一列跨核: batch={batch0 + 1}, y={y0 + 1}"
                )

    # 统计列内局部性。
    column_switch_count_per_core: List[int] = []
    max_round_gap_per_core: List[int] = []
    for core_id in range(1, info.k + 1):
        trace: List[Tuple[int, int, int]] = []
        for round_id in range(1, info.total_rounds + 1):
            pos = get_column_contiguous_position_from_info(
                info,
                core_id,
                round_id,
            )
            if pos is not None:
                batch_id, _, y = pos
                trace.append((round_id, batch_id, y))

        switches = sum(
            1
            for left, right in zip(trace, trace[1:])
            if left[1:] != right[1:]
        )
        gaps = [
            right[0] - left[0]
            for left, right in zip(trace, trace[1:])
        ]
        column_switch_count_per_core.append(switches)
        max_round_gap_per_core.append(max(gaps, default=0))

    anchor_examples = []
    for group in info.groups:
        anchor_y = group.y_start if group.direction > 0 else group.y_end
        rounds, xs = _column_active_trace(rounds_cube, 0, anchor_y - 1)
        anchor_examples.append({
            "group": group.group_id0,
            "anchor_y": anchor_y,
            "direction": "ascending" if group.direction > 0 else "descending",
            "rounds": rounds,
            "x_sequence": xs,
            "direction_changes": _count_direction_changes(xs),
        })

    return {
        "effective_n": info.effective_n,
        "group_count": info.group_count,
        "group_deltas": [group.delta for group in info.groups],
        "rounds_per_batch": info.rounds_per_batch,
        "total_rounds": info.total_rounds,
        "valid_blocks": assigned,
        "max_column_switches_per_core": max(
            column_switch_count_per_core,
            default=0,
        ),
        "max_active_round_gap_per_core": max(
            max_round_gap_per_core,
            default=0,
        ),
        "anchor_examples": anchor_examples,
    }


def summarize_rounds(
        info: ScheduleInfo,
        active: np.ndarray,
) -> dict:
    nonzero = active[active > 0]
    return {
        "full_rounds": int(np.count_nonzero(active == info.k)),
        "partial_rounds": int(
            np.count_nonzero((active > 0) & (active < info.k))
        ),
        "empty_rounds": int(np.count_nonzero(active == 0)),
        "avg_active_compute_round": (
            float(nonzero.mean()) if len(nonzero) else 0.0
        ),
        "slot_utilization": (
            float(active.sum() / (len(active) * info.k))
            if len(active)
            else 0.0
        ),
    }


def print_core_trace(
        info: ScheduleInfo,
        core_id: int,
        batch_id: Optional[int] = None,
) -> None:
    if not 1 <= core_id <= info.k:
        raise ValueError("core_id 越界")

    print(f"\ncore {core_id} trace")
    for round_id in range(1, info.total_rounds + 1):
        current_batch = (
            (round_id - 1) // info.rounds_per_batch + 1
        )
        if batch_id is not None and current_batch != batch_id:
            continue

        pos = get_column_contiguous_position_from_info(
            info,
            core_id,
            round_id,
        )
        if pos is None:
            print(f"r{round_id:4d} -> idle")
        else:
            b_id, x, y = pos
            print(
                f"r{round_id:4d} -> "
                f"(b={b_id}, x={x:2d}, y={y:2d})"
            )


def visualize_schedule(
        info: ScheduleInfo,
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        batches_per_figure: int = 2,
        ncols: int = 2,
        annotate_round: bool = True,
        save_dir: Optional[str] = None,
        dpi: int = 220,
        show: bool = False,
) -> List[Path]:
    """保持最初脚本的分核图片风格。"""
    palette = ['#F3F4F6'] + [
        '#0B84A5', '#EBC262', '#6F4E7C', '#9DD866', '#CA472F',
        '#FFA056', '#8DDDD0', '#BFB5FF', '#3C5488', '#F39C12',
        '#27AE60', '#D35400', '#16A085', '#7F8C8D', '#2E86C1',
        '#E74C3C', '#8E44AD', '#2ECC71', '#34495E', '#F1C40F'
    ]
    if info.k + 1 > len(palette):
        extra = plt.get_cmap(
            'tab20',
            info.k + 1 - len(palette),
        ).colors
        palette.extend(extra)
    cmap = ListedColormap(
        palette[:info.k + 1],
        name='dense_clean',
    )

    batches = list(range(1, info.b + 1))
    page_size = max(1, batches_per_figure)
    total_pages = ceil(len(batches) / page_size)
    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    cell_w = 0.42
    cell_h = 0.34
    axes_w = max(5.2, info.n * cell_w)
    axes_h = max(5.0, info.m * cell_h)
    label_font = 7 if max(info.m, info.n) <= 32 else 6
    annot_font = 8 if max(info.m, info.n) <= 32 else 6
    saved: List[Path] = []

    for page_idx in range(total_pages):
        page_batches = batches[
            page_idx * page_size:(page_idx + 1) * page_size
        ]
        cols = min(max(1, ncols), len(page_batches))
        rows = ceil(len(page_batches) / cols)

        fig_w = cols * axes_w + 1.2
        fig_h = rows * axes_h + 1.4
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(fig_w, fig_h),
            squeeze=False,
        )

        for local_idx, batch_id in enumerate(page_batches):
            rr = local_idx // cols
            cc = local_idx % cols
            ax = axes[rr][cc]

            round_mat = rounds_cube[batch_id - 1]
            core_mat = core_cube[batch_id - 1]
            display_core = core_mat.copy()
            display_core[display_core < 0] = 0

            ax.imshow(
                display_core,
                origin='upper',
                cmap=cmap,
                vmin=0,
                vmax=max(info.k, 1),
                aspect='equal',
                interpolation='nearest',
            )

            ax.set_xticks(
                np.arange(info.n + 1) - 0.5,
                minor=True,
            )
            ax.set_yticks(
                np.arange(info.m + 1) - 0.5,
                minor=True,
            )
            ax.grid(
                which='minor',
                color='#D1D5DB',
                linestyle='-',
                linewidth=0.45,
            )
            ax.tick_params(which='minor', length=0)

            ax.set_xticks(np.arange(info.n))
            ax.set_yticks(np.arange(info.m))
            ax.set_xticklabels(
                np.arange(1, info.n + 1),
                fontsize=label_font,
            )
            ax.set_yticklabels(
                np.arange(1, info.m + 1),
                fontsize=label_font,
            )
            ax.tick_params(axis='x', pad=2)
            ax.tick_params(axis='y', pad=2)

            if annotate_round:
                for i in range(info.m):
                    for j in range(info.n):
                        rid = round_mat[i, j]
                        if rid >= 1:
                            ax.text(
                                j,
                                i,
                                str(rid),
                                ha='center',
                                va='center',
                                fontsize=annot_font,
                                fontweight='bold',
                                color='black',
                                bbox=dict(
                                    boxstyle='round,pad=0.10',
                                    facecolor='white',
                                    alpha=0.72,
                                    edgecolor='none',
                                ),
                            )

            ax.set_title(
                f'batch={batch_id}',
                fontsize=10,
                pad=8,
            )

        for idx in range(len(page_batches), rows * cols):
            rr = idx // cols
            cc = idx % cols
            axes[rr][cc].axis('off')

        fig.suptitle(
            "Column-Contiguous Schedule | "
            f"k={info.k}, m={info.m}, n={info.n}, "
            f"b={info.b}, p={info.p}, q={info.q} | "
            f"page {page_idx + 1}/{total_pages}",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))

        if save_path:
            output = save_path / (
                "FAG_sparse04_column_contiguous_"
                f"page_{page_idx + 1}.png"
            )
            fig.savefig(
                output,
                dpi=dpi,
                bbox_inches='tight',
            )
            saved.append(output)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SparseMode4 按列连续确定性分核参考实现"
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--b", type=int, default=8)
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--q", type=int, default=63)
    parser.add_argument(
        "--round-chunk",
        type=int,
        default=0,
        help="<=0 表示完整列组连续；正数表示按该 local-round 块大小交错列组",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=1,
        help="打印指定 1-based core 的 batch 1 轨迹；<=0 不打印",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs_column_contiguous",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    info = build_schedule_info(
        m=args.m,
        n=args.n,
        b=args.b,
        p=args.p,
        q=args.q,
        k=args.k,
        round_chunk=args.round_chunk,
    )
    rounds_cube, core_cube, active = build_schedule_matrices(info)
    validation = validate_schedule(
        info,
        rounds_cube,
        core_cube,
    )
    round_summary = summarize_rounds(info, active)

    print("schedule info:")
    print(f"  effective_n       = {info.effective_n}")
    print(f"  group_count       = {info.group_count}")
    print(f"  group_deltas      = {[g.delta for g in info.groups]}")
    print(f"  rounds_per_batch  = {info.rounds_per_batch}")
    print(f"  total_rounds      = {info.total_rounds}")
    print(f"  round_chunk       = {info.round_chunk}")
    print(f"  valid_blocks      = {validation['valid_blocks']}")
    print(f"  full_rounds       = {round_summary['full_rounds']}")
    print(f"  partial_rounds    = {round_summary['partial_rounds']}")
    print(f"  empty_rounds      = {round_summary['empty_rounds']}")
    print(
        "  slot_utilization  = "
        f"{round_summary['slot_utilization']:.6%}"
    )

    print("\nanchor column sequences in batch 1:")
    for item in validation["anchor_examples"]:
        print(
            f"  group={item['group']}, "
            f"y={item['anchor_y']}, "
            f"direction={item['direction']}, "
            f"x={item['x_sequence']}"
        )

    if args.core > 0:
        print_core_trace(
            info,
            core_id=args.core,
            batch_id=1,
        )

    if not args.no_plot:
        saved = visualize_schedule(
            info,
            rounds_cube,
            core_cube,
            batches_per_figure=2,
            ncols=2,
            annotate_round=True,
            save_dir=args.save_dir,
            dpi=260,
            show=args.show,
        )
        for path in saved:
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
```
