"""
FlashAttentionScoreGrad 通用斜带区域确定性分核：Compact Batch-First 方案。

与 band_batch_first.py 的区别：

1. 仍然严格按 batch 串行；一个全局 round 最多只包含一个 batch。
2. 不再把每条斜对角线固定为一个 round。
3. 完全沿用原脚本的列组、蛇形和 batch 旋转规则；同一 batch 内固定列 y 的
   所有有效块始终归属于同一个核。
4. 在列归属固定后，将有效块视为“行/核”二分多重图的边，把每个 round 构造
   成一个 matching：同一 round 内同行不重复、同核最多执行一个任务。
5. 单 batch round 数达到按列分核约束下的理论下界：

       max(最大行任务数, 最大核任务数)

6. (core_id, round_id) 通过确定性、缓存的 schedule info O(1) 查询
   (batch_id, x, y)。当前实现用于算法验证；迁移到 Kernel 时可选择由 Host 下发
   round table，或进一步将 table 生成过程改写为 Kernel 可接受的公式。

矩阵坐标采用 1-based：
    1 <= x <= m, 1 <= y <= n

有效块满足：
    1 - q <= x - y <= p - 1

当 p=m、q=1 时，退化为 causal 下三角区域 y <= x。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

Position = Tuple[int, int, int]
Edge = Tuple[int, int]
CoreRound = Tuple[Optional[Edge], ...]


@dataclass(frozen=True)
class CompactBatchFirstScheduleInfo:
    """Compact Batch-First 调度的静态信息。"""

    k: int
    m: int
    n: int
    b: int
    p: int
    q: int
    effective_n: int
    valid_blocks_per_batch: int
    maximum_degree: int
    capacity_round_lower_bound: int
    rounds_per_batch: int
    total_rounds: int
    legacy_rounds_per_batch: int
    legacy_total_rounds: int
    rotation_stride: int
    # 采用 batch=1 的基础列归属；其他 batch 只对 core 维做整体旋转。
    round_core_edges: Tuple[CoreRound, ...]


@dataclass(frozen=True)
class CompactBatchFirstScheduleStats:
    """正确性、核负载和时间维度利用率统计。"""

    valid_blocks_per_batch: int
    total_valid_blocks: int
    minimum_rounds_per_batch: int
    rounds_per_batch: int
    total_rounds: int
    legacy_total_rounds: int
    saved_rounds: int
    round_reduction_ratio: float
    min_core_load: int
    max_core_load: int
    core_loads: Tuple[int, ...]
    empty_round_count: int
    nonempty_round_count: int
    full_core_round_count: int
    partial_core_round_count: int
    min_active_cores: int
    max_active_cores: int
    active_cores_per_round: Tuple[int, ...]
    slot_utilization: float
    round_efficiency: float
    max_core_task_gap: int


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


def _build_valid_edges(m: int, n: int, p: int, q: int) -> Tuple[Edge, ...]:
    """按 (x, y) 字典序生成单 batch 的全部有效边。"""

    edges: List[Edge] = []
    for x in range(1, m + 1):
        y_min = max(1, x - p + 1)
        y_max = min(n, x + q - 1)
        for y in range(y_min, y_max + 1):
            edges.append((x, y))
    return tuple(edges)


def count_valid_blocks_per_batch(m: int, n: int, p: int, q: int) -> int:
    """计算单 batch 有效块数量。"""

    _validate_common_parameters(m, n, 1, p, q, 1)
    return len(_build_valid_edges(m, n, p, q))


def _find_coprime_rotation_stride(k: int) -> int:
    """选择与 k 互质的确定性核旋转步长。"""

    if k == 1:
        return 0
    stride = k // 2 + 1
    while gcd(stride, k) != 1:
        stride += 1
    return stride % k


def _get_base_column_owner_core0(
        y: int,
        k: int,
        rotation_stride: int,
) -> int:
    """
    返回 batch=1 时物理列 y 的固定 owner core（0-based）。

    这里严格复用原 Batch-First 脚本的规则：每 k 列一组、相邻组蛇形，
    每两个列组旋转一次。其他 batch 在此基础上整体旋转。
    """

    y0 = y - 1
    group = y0 // k
    local_slot = y0 % k
    group_rotation = ((group // 2) * rotation_stride) % k
    if group % 2 == 0:
        permuted_core = local_slot
    else:
        permuted_core = k - 1 - local_slot
    return (permuted_core + group_rotation) % k


def get_batch_first_column_owner_core(
        k: int,
        batch_id: int,
        y: int,
) -> int:
    """公开查询某个 batch 中物理列 y 的固定 owner core，返回值为 1-based。"""

    for name, value in (("k", k), ("batch_id", batch_id), ("y", y)):
        if not isinstance(value, int):
            raise TypeError(f"{name} 必须是 int，实际为 {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0，当前为 {value}")
    rotation_stride = _find_coprime_rotation_stride(k)
    base_core0 = _get_base_column_owner_core0(y, k, rotation_stride)
    return (base_core0 + (batch_id - 1) * rotation_stride) % k + 1


def _find_perfect_matching(counts: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """在正则二分多重图的支撑图上确定性寻找一个 perfect matching。"""

    vertex_count = len(counts)
    matched_left_for_right = [-1] * vertex_count

    def augment(left: int, visited_right: List[bool]) -> bool:
        for right in range(vertex_count):
            if counts[left][right] <= 0 or visited_right[right]:
                continue
            visited_right[right] = True
            previous_left = matched_left_for_right[right]
            if previous_left == -1 or augment(previous_left, visited_right):
                matched_left_for_right[right] = left
                return True
        return False

    for left in range(vertex_count):
        if not augment(left, [False] * vertex_count):
            raise AssertionError(f"无法为 left={left} 找到 perfect matching")

    matched_right_for_left = [-1] * vertex_count
    for right, left in enumerate(matched_left_for_right):
        matched_right_for_left[left] = right
    if any(right < 0 for right in matched_right_for_left):
        raise AssertionError("perfect matching 不完整")
    return tuple(matched_right_for_left)


def _build_compact_round_core_edges(
        edges: Sequence[Edge],
        m: int,
        k: int,
        rotation_stride: int,
) -> Tuple[Tuple[CoreRound, ...], int, int]:
    """
    在固定按列分核后，构造最少轮数的 row-core matching。

    每个有效块 (x, y) 先按原脚本规则确定 owner core，于是任务成为二分
    多重图中的一条边：左端点是 row x，右端点是 owner core；同一 row-core
    之间允许存在来自不同列的多条边。

    将图补成 delta-正则二分多重图，再逐次抽取 perfect matching。二分多重图
    的边色数等于最大度 delta，因此轮数正好是：

        max(最大行任务数, 最大核任务数)

    这也是固定列归属约束下的理论最少轮数。
    """

    if not edges:
        raise ValueError("有效区域不能为空")

    vertex_count = max(m, k)
    counts = [[0] * vertex_count for _ in range(vertex_count)]
    real_columns: Dict[Tuple[int, int], List[int]] = {}
    left_degree = [0] * vertex_count
    right_degree = [0] * vertex_count

    for x, y in edges:
        left = x - 1
        right = _get_base_column_owner_core0(y, k, rotation_stride)
        counts[left][right] += 1
        left_degree[left] += 1
        right_degree[right] += 1
        real_columns.setdefault((left, right), []).append(y)

    # pop() 时按 y 递增取出，保证完全确定。
    for columns in real_columns.values():
        columns.sort(reverse=True)

    delta = max(max(left_degree), max(right_degree))
    capacity_lower_bound = ceil(len(edges) / k)
    if delta < capacity_lower_bound:
        raise AssertionError("最大核度数不应小于容量轮数下界")

    # 补 dummy 多重边，将左右两侧所有 vertex 的度数补到 delta。
    left_deficit = [delta - degree for degree in left_degree]
    right_deficit = [delta - degree for degree in right_degree]
    left = 0
    right = 0
    while left < vertex_count and right < vertex_count:
        while left < vertex_count and left_deficit[left] == 0:
            left += 1
        while right < vertex_count and right_deficit[right] == 0:
            right += 1
        if left == vertex_count or right == vertex_count:
            break
        amount = min(left_deficit[left], right_deficit[right])
        counts[left][right] += amount
        left_deficit[left] -= amount
        right_deficit[right] -= amount

    if any(left_deficit) or any(right_deficit):
        raise AssertionError("正则化后仍存在未补齐的顶点度数")

    round_core_edges: List[CoreRound] = []
    emitted_edges: List[Edge] = []
    for _ in range(delta):
        matching = _find_perfect_matching(counts)
        core_edges: List[Optional[Edge]] = [None] * k
        for matched_left, matched_right in enumerate(matching):
            counts[matched_left][matched_right] -= 1
            pair = (matched_left, matched_right)
            columns = real_columns.get(pair)
            if matched_left < m and matched_right < k and columns:
                y = columns.pop()
                edge = (matched_left + 1, y)
                core_edges[matched_right] = edge
                emitted_edges.append(edge)
        round_core_edges.append(tuple(core_edges))

    if any(count for row in counts for count in row):
        raise AssertionError("抽取全部 matching 后仍残留多重边")
    if any(columns for columns in real_columns.values()):
        raise AssertionError("仍有真实任务未被分配到 round")
    if len(emitted_edges) != len(edges) or set(emitted_edges) != set(edges):
        raise AssertionError("row-core 边着色后出现有效块遗漏或重复")

    return tuple(round_core_edges), delta, capacity_lower_bound


@lru_cache(maxsize=128)
def get_compact_batch_first_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> CompactBatchFirstScheduleInfo:
    """计算并缓存 Compact Batch-First 调度静态信息。"""

    _validate_common_parameters(m, n, b, p, q, k)
    edges = _build_valid_edges(m, n, p, q)
    rotation_stride = _find_coprime_rotation_stride(k)
    round_core_edges, maximum_degree, capacity_lower_bound = (
        _build_compact_round_core_edges(edges, m, k, rotation_stride)
    )

    effective_n = min(n, m + q - 1)
    legacy_rounds_per_batch = ceil(effective_n / k) * (p + q - 1)
    rounds_per_batch = len(round_core_edges)

    return CompactBatchFirstScheduleInfo(
        k=k,
        m=m,
        n=n,
        b=b,
        p=p,
        q=q,
        effective_n=effective_n,
        valid_blocks_per_batch=len(edges),
        maximum_degree=maximum_degree,
        capacity_round_lower_bound=capacity_lower_bound,
        rounds_per_batch=rounds_per_batch,
        total_rounds=b * rounds_per_batch,
        legacy_rounds_per_batch=legacy_rounds_per_batch,
        legacy_total_rounds=b * legacy_rounds_per_batch,
        rotation_stride=rotation_stride,
        round_core_edges=round_core_edges,
    )


# 保留接近原脚本的命名，便于替换调用。
get_batch_first_schedule_info = get_compact_batch_first_schedule_info


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

    info = get_compact_batch_first_schedule_info(m, n, b, p, q, k)
    if not isinstance(batch_id, int):
        raise TypeError("batch_id 必须是 int")
    if not 1 <= batch_id <= b:
        raise ValueError(f"batch_id 必须满足 1 <= batch_id <= b，当前为 {batch_id}")
    return (
        (batch_id - 1) * info.rounds_per_batch + 1,
        batch_id * info.rounds_per_batch,
    )


def get_batch_first_max_rounds(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> int:
    """返回 Compact Batch-First 全局总轮数。"""

    return get_compact_batch_first_schedule_info(m, n, b, p, q, k).total_rounds


# ---------------------------------------------------------------------------
# 核心确定性映射
# ---------------------------------------------------------------------------


def _get_compact_batch_first_position_from_info(
        info: CompactBatchFirstScheduleInfo,
        core_id: int,
        round_id: int,
) -> Optional[Position]:
    """使用缓存的 schedule info 查询 (core_id, round_id) 对应任务。"""

    if round_id > info.total_rounds:
        return None

    round0 = round_id - 1
    batch0 = round0 // info.rounds_per_batch
    round_in_batch = round0 % info.rounds_per_batch
    # 原脚本按 batch 对整套列 owner 做统一旋转；round 内不再改变列归属。
    batch_rotation = (batch0 * info.rotation_stride) % info.k
    base_core0 = ((core_id - 1) - batch_rotation) % info.k
    edge = info.round_core_edges[round_in_batch][base_core0]
    if edge is None:
        return None

    x, y = edge
    return batch0 + 1, x, y


def get_band_compact_batch_first_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    """Compact Batch-First 的公开位置查询接口。"""

    info = get_compact_batch_first_schedule_info(m, n, b, p, q, k)
    if not isinstance(core_id, int) or not isinstance(round_id, int):
        raise TypeError("core_id 和 round_id 必须是 int")
    if not 1 <= core_id <= k:
        raise ValueError(f"core_id 必须满足 1 <= core_id <= k，当前为 {core_id}")
    if round_id < 1:
        raise ValueError(f"round_id 必须大于等于 1，当前为 {round_id}")
    return _get_compact_batch_first_position_from_info(info, core_id, round_id)


# 与原脚本兼容的别名。
get_band_batch_first_position = get_band_compact_batch_first_position


def get_causal_batch_first_position(
        m: int,
        n: int,
        b: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    """causal 兼容接口，等价于 p=m、q=1。"""

    return get_band_compact_batch_first_position(
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


def build_compact_batch_first_matrices(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        validate: bool = True,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """构建全部 batch 的 round/core 矩阵。"""

    info = get_compact_batch_first_schedule_info(m, n, b, p, q, k)
    rounds_cube = np.full((b, m, n), -1, dtype=np.int64)
    core_cube = np.full((b, m, n), -1, dtype=np.int64)

    for round_id in range(1, info.total_rounds + 1):
        for core_id in range(1, k + 1):
            pos = _get_compact_batch_first_position_from_info(
                info, core_id, round_id
            )
            if pos is None:
                continue
            batch_id, x, y = pos
            index = (batch_id - 1, x - 1, y - 1)
            if rounds_cube[index] != -1:
                raise AssertionError(
                    "检测到重复任务："
                    f"position={(batch_id, x, y)}, "
                    f"old=(round={rounds_cube[index]}, core={core_cube[index]}), "
                    f"new=(round={round_id}, core={core_id})"
                )
            rounds_cube[index] = round_id
            core_cube[index] = core_id

    if validate:
        validate_compact_batch_first_schedule(
            rounds_cube, core_cube, m, n, b, p, q, k
        )
    return k, info.total_rounds, rounds_cube, core_cube


# 与原脚本兼容的别名。
build_batch_first_matrices = build_compact_batch_first_matrices


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


def _get_max_core_task_gap(
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        k: int,
) -> int:
    """返回任一核相邻有效任务之间的最大 round 间隔。"""

    maximum_gap = 0
    for core_id in range(1, k + 1):
        rounds = np.sort(rounds_cube[core_cube == core_id])
        if rounds.size >= 2:
            maximum_gap = max(maximum_gap, int(np.diff(rounds).max()))
    return maximum_gap


def validate_compact_batch_first_schedule(
        rounds_cube: np.ndarray,
        core_cube: np.ndarray,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> CompactBatchFirstScheduleStats:
    """
    校验完整覆盖、无重复、同行/同列轮次唯一、单轮 batch 唯一、核唯一，
    并返回时间维度利用率统计。
    """

    info = get_compact_batch_first_schedule_info(m, n, b, p, q, k)
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
        raise AssertionError("发现超过 k 的非法核编号")

    # 同一 batch、同一行或同一列中不能出现重复 round；同一列必须固定归属同一核。
    for batch0 in range(b):
        for x0 in range(m):
            row_rounds = rounds_cube[batch0, x0]
            row_rounds = row_rounds[row_rounds >= 1]
            if row_rounds.size != np.unique(row_rounds).size:
                raise AssertionError(
                    f"同行出现重复轮次：batch={batch0 + 1}, x={x0 + 1}"
                )
        for y0 in range(n):
            column_rounds = rounds_cube[batch0, :, y0]
            column_rounds = column_rounds[column_rounds >= 1]
            if column_rounds.size != np.unique(column_rounds).size:
                raise AssertionError(
                    f"同列出现重复轮次：batch={batch0 + 1}, y={y0 + 1}"
                )
            column_cores = core_cube[batch0, :, y0]
            column_cores = column_cores[column_cores >= 1]
            if column_cores.size == 0:
                continue
            unique_column_cores = np.unique(column_cores)
            if unique_column_cores.size != 1:
                raise AssertionError(
                    "同一列被拆分到多个核："
                    f"batch={batch0 + 1}, y={y0 + 1}, "
                    f"cores={unique_column_cores.tolist()}"
                )
            base_core0 = _get_base_column_owner_core0(
                y0 + 1, k, info.rotation_stride
            )
            expected_core = (
                base_core0 + batch0 * info.rotation_stride
            ) % k + 1
            if int(unique_column_cores[0]) != expected_core:
                raise AssertionError(
                    "列 owner 不符合原 Batch-First 映射："
                    f"batch={batch0 + 1}, y={y0 + 1}, "
                    f"expected_core={expected_core}, "
                    f"actual_core={int(unique_column_cores[0])}"
                )

    active_cores_per_round: List[int] = []
    for round_id in range(1, info.total_rounds + 1):
        locations = np.argwhere(rounds_cube == round_id)
        if locations.size == 0:
            active_cores_per_round.append(0)
            continue

        batch_ids = np.unique(locations[:, 0])
        if batch_ids.size != 1:
            raise AssertionError(
                f"round={round_id} 同时出现多个 batch：{(batch_ids + 1).tolist()}"
            )
        expected_batch0 = (round_id - 1) // info.rounds_per_batch
        if int(batch_ids[0]) != expected_batch0:
            raise AssertionError(
                f"round={round_id} batch 顺序错误："
                f"expected={expected_batch0 + 1}, actual={int(batch_ids[0]) + 1}"
            )

        round_cores = core_cube[rounds_cube == round_id]
        if round_cores.size != np.unique(round_cores).size:
            raise AssertionError(f"round={round_id} 同一个核被分配多个任务")
        if round_cores.size > k:
            raise AssertionError(f"round={round_id} 有效任务数超过 k")
        active_cores_per_round.append(int(round_cores.size))

    # 每个 batch 严格占用自己的连续 round 区间。
    for batch0 in range(b):
        start = batch0 * info.rounds_per_batch + 1
        end = (batch0 + 1) * info.rounds_per_batch
        assigned = rounds_cube[batch0][rounds_cube[batch0] >= 1]
        if assigned.size == 0 or np.any((assigned < start) | (assigned > end)):
            raise AssertionError(
                f"batch={batch0 + 1} 未严格落在 round 区间 [{start}, {end}]"
            )

    total_valid = int(np.count_nonzero(valid_mask))
    expected_total_valid = b * info.valid_blocks_per_batch
    if total_valid != expected_total_valid:
        raise AssertionError(
            f"有效块总数不一致：expected={expected_total_valid}, actual={total_valid}"
        )

    active = np.asarray(active_cores_per_round, dtype=np.int64)
    nonempty_count = int(np.count_nonzero(active))
    empty_count = int(active.size - nonempty_count)
    full_count = int(np.count_nonzero(active == k))
    partial_count = int(np.count_nonzero((active > 0) & (active < k)))
    core_loads = get_core_loads(core_cube, k)
    minimum_rounds = max(
        info.maximum_degree,
        info.capacity_round_lower_bound,
    )

    return CompactBatchFirstScheduleStats(
        valid_blocks_per_batch=info.valid_blocks_per_batch,
        total_valid_blocks=expected_total_valid,
        minimum_rounds_per_batch=minimum_rounds,
        rounds_per_batch=info.rounds_per_batch,
        total_rounds=info.total_rounds,
        legacy_total_rounds=info.legacy_total_rounds,
        saved_rounds=info.legacy_total_rounds - info.total_rounds,
        round_reduction_ratio=(
            1.0 - info.total_rounds / info.legacy_total_rounds
        ),
        min_core_load=int(core_loads.min()),
        max_core_load=int(core_loads.max()),
        core_loads=tuple(int(value) for value in core_loads),
        empty_round_count=empty_count,
        nonempty_round_count=nonempty_count,
        full_core_round_count=full_count,
        partial_core_round_count=partial_count,
        min_active_cores=int(active[active > 0].min()),
        max_active_cores=int(active.max()),
        active_cores_per_round=tuple(int(value) for value in active),
        slot_utilization=expected_total_valid / (k * nonempty_count),
        round_efficiency=minimum_rounds / info.rounds_per_batch,
        max_core_task_gap=_get_max_core_task_gap(rounds_cube, core_cube, k),
    )


# 与原脚本兼容的别名。
validate_batch_first_schedule = validate_compact_batch_first_schedule


def verify_batch_first_determinism(
        k: int,
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
) -> bool:
    """使用相同参数构建两次，验证 round/core 矩阵完全一致。"""

    first = build_compact_batch_first_matrices(k, m, n, b, p, q, validate=True)
    # 清理缓存后重新生成，避免仅验证同一缓存对象。
    get_compact_batch_first_schedule_info.cache_clear()
    second = build_compact_batch_first_matrices(k, m, n, b, p, q, validate=True)
    return (
        np.array_equal(first[2], second[2])
        and np.array_equal(first[3], second[3])
    )


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
    """分页可视化 Compact Batch-First 调度。"""

    info = get_compact_batch_first_schedule_info(m, n, b, p, q, k)
    k, total_rounds, rounds_cube, core_cube = build_compact_batch_first_matrices(
        k, m, n, b, p, q, validate=True
    )
    stats = validate_compact_batch_first_schedule(
        rounds_cube, core_cube, m, n, b, p, q, k
    )

    print("调度策略: compact-batch-first")
    print("所需核数:", k)
    print("最大行/核度数:", info.maximum_degree)
    print("容量轮数下界:", info.capacity_round_lower_bound)
    print("每个 batch 轮次数:", info.rounds_per_batch)
    print("原 Batch-First 每 batch 轮次数:", info.legacy_rounds_per_batch)
    print("总轮次数:", total_rounds)
    print("原 Batch-First 总轮次数:", info.legacy_total_rounds)
    print("轮次降低比例:", f"{stats.round_reduction_ratio:.2%}")
    print("槽位利用率:", f"{stats.slot_utilization:.2%}")
    print("满核/部分核/空轮:", (
        stats.full_core_round_count,
        stats.partial_core_round_count,
        stats.empty_round_count,
    ))
    print("各核有效任务数:", list(stats.core_loads))
    print("核负载范围:", (stats.min_core_load, stats.max_core_load))
    print("单核最大任务间隔:", stats.max_core_task_gap)

    palette = ["#F3F4F6"] + [
        "#0B84A5", "#EBC262", "#6F4E7C", "#9DD866", "#CA472F",
        "#FFA056", "#8DDDD0", "#BFB5FF", "#3C5488", "#F39C12",
        "#27AE60", "#D35400", "#16A085", "#7F8C8D", "#2E86C1",
        "#E74C3C", "#8E44AD", "#2ECC71", "#34495E", "#F1C40F",
    ]
    if k + 1 > len(palette):
        palette.extend(plt.get_cmap("tab20", k + 1 - len(palette)).colors)
    cmap = ListedColormap(palette[:k + 1], name="compact_batch_first_band")

    batches = list(range(1, b + 1))
    page_size = max(1, batches_per_figure)
    total_pages = ceil(len(batches) / page_size)
    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    axes_w = max(5.2, n * 0.42)
    axes_h = max(5.0, m * 0.34)
    label_font = 7 if max(m, n) <= 32 else 6
    annot_font = 8 if max(m, n) <= 32 else 6

    for page_idx in range(total_pages):
        page_batches = batches[page_idx * page_size:(page_idx + 1) * page_size]
        cols = min(max(1, ncols), len(page_batches))
        rows = ceil(len(page_batches) / cols)
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * axes_w + 1.2, rows * axes_h + 1.4),
            squeeze=False,
        )

        for local_idx, batch_id in enumerate(page_batches):
            ax = axes[local_idx // cols][local_idx % cols]
            round_mat = rounds_cube[batch_id - 1]
            core_mat = core_cube[batch_id - 1]
            ax.imshow(
                np.where(core_mat >= 1, core_mat, 0),
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
                for x0 in range(m):
                    for y0 in range(n):
                        round_id = int(round_mat[x0, y0])
                        if round_id >= 1:
                            ax.text(
                                y0, x0, str(round_id),
                                ha="center", va="center",
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

            start, end = get_batch_round_range(m, n, b, p, q, k, batch_id)
            ax.set_title(
                f"batch={batch_id}, rounds={start}-{end}",
                fontsize=10,
                pad=8,
            )

        for index in range(len(page_batches), rows * cols):
            axes[index // cols][index % cols].axis("off")

        fig.suptitle(
            f"Compact Batch-First | k={k}, m={m}, n={n}, b={b}, "
            f"p={p}, q={q} | page {page_idx + 1}/{total_pages}",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        if save_path is not None:
            fig.savefig(
                save_path / f"FAG_band_compact_batch_first_p{p}_q{q}_page_{page_idx + 1}.png",
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


def _demo() -> None:
    # k, m, n, b = 32, 32, 32, 8
    # p, q = 31, 31

    # k, m, n, b = 32, 32, 64, 8
    # p, q = 32, 63

    k, m, n, b = 32, 32, 32, 8
    p, q = 32, 1
    output_dir = Path(__file__).parent / "outputs_batch_first_compact"
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
