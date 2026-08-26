#!/usr/bin/env python3
"""FAG arch35 非 TND 确定性 BAND 三模式调度的算子同构参考实现。

本文件以当前算子实现为准：

* Host: ``SelectDeterBandSchedule``；
* Kernel: ``CalBandDeterIndex`` / ``CalDeterMaxLoopNum``；
* 模式不是固定几何优先级，而是从 BAND 基线开始，仅在候选轮次严格更小时切换；
* CAUSAL 支持偶数 Batch 配对段以及奇数末尾的单 Batch 确定性尾段。

坐标和核号均使用 1-based，便于逐句对照 Kernel。
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional


Coordinate = tuple[int, int, int]  # (batch, s1_idx, s2_idx), 1-based
RIGHT_DOWN_CAUSAL_SPARSE_MODE = 3


class DeterBandScheduleMode(IntEnum):
    DISABLED = 0
    CAUSAL = 1
    DENSE = 2
    BAND = 3


@dataclass(frozen=True)
class ScheduleResult:
    mode: DeterBandScheduleMode
    max_round: int
    k: int
    m: int
    n: int
    p: int
    q: int
    b: int
    band_blocks: int = 0
    band_round: Optional[int] = None
    dense_round: Optional[int] = None
    causal_round: Optional[int] = None
    pair_round: int = 0
    tail_round: int = 0
    cols_per_batch: int = 0
    hybrid_pair_count: int = 0

    @property
    def mode_name(self) -> str:
        return self.mode.name


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    errors: tuple[str, ...]
    schedule: ScheduleResult
    coordinate_count: int
    loads: tuple[int, ...]


def ceil_div(dividend: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return dividend // divisor + int(dividend % divisor != 0)


def keep_legacy_right_down_swizzle(sparse_mode: int, legacy_split_by_block_idx: bool) -> bool:
    """镜像 Host 在混合三模式选择前保留远端 RIGHT_DOWN_CAUSAL 路径的条件。"""
    return sparse_mode == RIGHT_DOWN_CAUSAL_SPARSE_MODE and legacy_split_by_block_idx


def calc_legacy_right_down_round(k: int, m: int, n: int, b: int) -> int:
    """镜像 Kernel ``scheduleMode=DISABLED`` 的 RIGHT_DOWN_CAUSAL 最大轮次公式。"""
    if min(k, m, n, b) <= 0:
        return 0
    if m < n:
        causal_m = m
        causal_n = 2 * n - m + 3
    else:
        causal_m = n + 1
        causal_n = n + 2
    return max(causal_m * ceil_div(causal_n * (b // 2), k), causal_n)


def normalize_deter_band_schedule_params(m: int, n: int, p: int, q: int) -> tuple[int, int, int, int]:
    """镜像 Host ``NormalizeDeterBandScheduleParams``。"""
    p = min(p, m)
    q = min(q, n)
    if p < 0:
        return m, n + p, 1, p + q
    if q < 0:
        return m + q, n, p + q, 1
    return m, n, p, q


def seg_params(m: int, n: int, p: int, q: int) -> dict[str, int]:
    """镜像 ``GenBandHybridInfo`` 的 clamp、裁剪和分段。"""
    p = min(p, m)
    q = min(q, n)
    m = min(m, n + p - 1)
    n = min(n, m + q - 1)
    if p + q <= m:
        l1 = q - 1
        l2 = min(n - q + 1, m + 2 - p - q)
        l3 = max(0, min(p + n - m - 1, p + q - 2))
        band_blocks = (
            (2 * p - 2 + q) * l1 // 2
            + (p + q - 1) * l2
            + (p + q - 2) * l3
            - l3 * (l3 - 1) // 2
        )
    else:
        l1 = m - p
        l2 = p + q - m
        l3 = min(n - q, m - 1)
        band_blocks = (p + m - 1) * l1 // 2 + m * l2 + (2 * m - 1 - l3) * l3 // 2
    pair_count = max(0, min(l1, l3 - p + 1))
    return {
        "m": m,
        "n": n,
        "p": p,
        "q": q,
        "L1": l1,
        "L2": l2,
        "L3": l3,
        "n_seg": l1 + l2 + l3,
        "slot": p + q - 1,
        "band_blocks": band_blocks,
        "hybrid_pair_count": pair_count,
        "cols_per_batch": l1 + l2 + l3 - pair_count,
    }


def calc_causal_single_batch_round(k: int, causal_size: int) -> int:
    """镜像 Host 尾段轮次上界，与 ``CalCausalSingleBatchDeterIndex`` 一致。"""
    if k <= 0 or causal_size <= 0:
        return 0
    group_count = causal_size // (2 * k)
    group_round = (2 * causal_size + 1) * group_count - 2 * k * group_count * group_count
    remain = causal_size - 2 * k * group_count
    if remain <= k:
        return group_round + remain
    return group_round + max(remain, 2 * remain - 2 * k + 1)


def select_deter_band_schedule(k: int, m: int, n: int, p: int, q: int, b: int) -> ScheduleResult:
    """镜像算子当前完整调用：先 Normalize，再执行 ``SelectDeterBandSchedule``。"""
    m, n, p, q = normalize_deter_band_schedule_params(m, n, p, q)
    disabled = ScheduleResult(DeterBandScheduleMode.DISABLED, 0, k, m, n, p, q, b)
    if min(k, m, n, p, q, b) <= 0:
        return disabled

    params = seg_params(m, n, p, q)
    m, n, p, q = (params[name] for name in ("m", "n", "p", "q"))
    band_blocks = params["band_blocks"]
    cols_per_batch = params["cols_per_batch"]
    pair_count = params["hybrid_pair_count"]
    slot = params["slot"]

    band_round = ceil_div(b * cols_per_batch, k) * slot
    mode = DeterBandScheduleMode.BAND
    max_round = band_round
    dense_round: Optional[int] = None
    causal_round: Optional[int] = None
    causal_pair_round = 0
    causal_tail_round = 0

    # Dense 候选：所有 k 核必须保持活跃，且同 Batch 同轮列数不能绕 m 后撞同行。
    dense_k = min(k, m * b)
    if dense_k == k and min(dense_k, n) <= m:
        dense_round = ceil_div(n * b, dense_k) * m
        if dense_round < max_round:  # 算子使用严格小于，平局保持先前模式。
            mode = DeterBandScheduleMode.DENSE
            max_round = dense_round

    # Causal 候选：当前算子只启用 lower embedding，不启用 upper 转置。
    lower_size = m + q - 1
    upper_size = n + p - 1
    lower_waste = lower_size * (lower_size + 1) // 2 - band_blocks
    upper_waste = upper_size * (upper_size + 1) // 2 - band_blocks
    use_lower_causal = (
        band_blocks > 0
        and lower_waste >= 0
        and lower_waste <= upper_waste
        and lower_waste <= (band_blocks - 1) // 10
    )
    if use_lower_causal:
        causal_pair_count = b // 2
        causal_k = min(k, lower_size * causal_pair_count)
        if causal_pair_count > 0 and causal_k == k and causal_k <= lower_size:
            causal_pair_round = ceil_div((lower_size + 1) * causal_pair_count, causal_k) * lower_size
            causal_tail_round = 0 if b % 2 == 0 else calc_causal_single_batch_round(k, lower_size)
            causal_round = causal_pair_round + causal_tail_round
            if causal_round < max_round:
                mode = DeterBandScheduleMode.CAUSAL
                max_round = causal_round

    return ScheduleResult(
        mode=mode,
        max_round=max_round,
        k=k,
        m=m,
        n=n,
        p=p,
        q=q,
        b=b,
        band_blocks=band_blocks,
        band_round=band_round,
        dense_round=dense_round,
        causal_round=causal_round,
        pair_round=causal_pair_round,
        tail_round=causal_tail_round,
        cols_per_batch=cols_per_batch,
        hybrid_pair_count=pair_count,
    )


def _valid_band(schedule: ScheduleResult, x: int, y: int) -> bool:
    return (
        1 <= x <= schedule.m
        and 1 <= y <= schedule.n
        and max(1, y - schedule.q + 1) <= x <= min(schedule.m, y + schedule.p - 1)
    )


def cal_dense_swizzle_index(k: int, m: int, n: int, b: int, j: int, r: int) -> Optional[Coordinate]:
    """镜像 Kernel ``CalDenseSwizzleIndex``。"""
    if min(k, m, n, b, j, r) <= 0:
        return None
    active_k = min(k, b * m)
    if j > active_k:
        return None
    zero_j = j - 1
    zero_r = r - 1
    column = (zero_r // m) * active_k + zero_j
    if column >= n * b:
        return None
    batch, y = divmod(column, n)
    x = (y + zero_r) % m
    return batch + 1, x + 1, y + 1


def cal_causal_swizzle_index(k: int, size: int, b: int, j: int, r: int) -> Optional[Coordinate]:
    """镜像方阵路径 ``CalCausalSwizzleIndex``，输入 b 必须是已配对的偶数部分。"""
    result = cal_dense_swizzle_index(k, size, size + 1, b // 2, j, r)
    if result is None:
        return None
    pair_id, x, y = result
    if y >= x + 1:
        return pair_id * 2, size + 1 - x, size - y + 2
    return pair_id * 2 - 1, x, y


def _causal_g2k_single(k: int, j: int, local_round: int, size: int, offset: int) -> tuple[int, int]:
    if j % 2 == 1:
        if local_round <= size - j + 1:
            y = j + offset
            x = y + local_round - 1
        else:
            y = 2 * k + 1 - j + offset
            x = y + 2 * size - 2 * k + 1 - local_round
    else:
        if local_round >= size - 2 * k + 1 + j:
            y = j + offset
            x = y + 2 * size - 2 * k + 1 - local_round
        else:
            y = 2 * k + 1 - j + offset
            x = y + local_round - 1
    return x, y


def _causal_no_rec_single(k: int, size: int, j: int, r: int) -> Optional[tuple[int, int]]:
    if j > size // 2 + 1:
        return None
    if j % 2 == 1:
        if r + j <= size + 1:
            x, y = r + j - 1, j
        else:
            x, y = 2 * size + 2 - j - r, size + 3 - j - size % 2
    else:
        if j <= r + 1 - size % 2:
            x, y = size + j - r - 1 + size % 2, j
        else:
            x, y = size + 2 + r - j - size % 2, size + 3 - j - size % 2
    return (x, y) if 1 <= y <= x <= size else None


def _causal_rec_single(k: int, m: int, n: int, j: int, r: int) -> Optional[tuple[int, int]]:
    if 2 * k < m + 1 and k < n:
        x, y = _causal_g2k_single(k, j, r, m, 0)
        return (x, y) if 1 <= y <= n and y <= x <= m else None
    result = _causal_no_rec_single(k, m, j, r)
    if result is None:
        return None
    x, y = result
    return result if y <= n else None


def cal_causal_single_batch_deter_index(k: int, size: int, j: int, r: int) -> Optional[tuple[int, int]]:
    """镜像 ``CalCausalSingleBatchDeterIndex``，返回单尾 Batch 的 ``(x,y)``。"""
    if min(k, size, j, r) <= 0 or j > k:
        return None
    if k >= size // 2 + 1:
        return _causal_rec_single(k, size, size, j, r)

    group_count = size // (2 * k)
    group_bound = (2 * size + 1) * group_count - 2 * k * group_count * group_count
    if r <= group_bound:
        low, high = 1, group_count
        while low < high:
            middle = (low + high) // 2
            prefix = (2 * size + 1) * middle - 2 * k * middle * middle
            if r <= prefix:
                high = middle
            else:
                low = middle + 1
        group = low
        previous = (2 * size + 1) * (group - 1) - 2 * k * (group - 1) * (group - 1)
        offset = 2 * k * (group - 1)
        group_size = size - offset
        return _causal_g2k_single(k, j, r - previous, group_size, offset)

    full_groups = size // k
    remainder = size % k
    has_k_group = full_groups % 2
    remaining_columns = has_k_group * k + remainder
    if remaining_columns <= 0:
        return None
    local_round = r - group_bound
    tail_size = remaining_columns
    offset = 2 * k * group_count
    if has_k_group == 0 and j <= remaining_columns:
        x = offset + j + local_round - 1
        y = offset + j
        return (x, y) if y <= x <= size else None
    if has_k_group == 1:
        result = _causal_rec_single(k, tail_size, tail_size, j, local_round)
        if result is None:
            return None
        x, y = result
        return x + offset, y + offset
    return None


def cal_band_hybrid_index(schedule: ScheduleResult, j: int, r: int) -> Optional[Coordinate]:
    """镜像 Kernel ``CalBandHybridIndex``。"""
    params = seg_params(schedule.m, schedule.n, schedule.p, schedule.q)
    slot = params["slot"]
    cols_per_batch = params["cols_per_batch"]
    if not (1 <= j <= schedule.k) or r < 1 or slot <= 0 or cols_per_batch <= 0:
        return None
    layer, zero_local_round = divmod(r - 1, slot)
    local_round = zero_local_round + 1
    global_column = layer * schedule.k + j
    if global_column > schedule.b * cols_per_batch:
        return None
    batch = (global_column - 1) // cols_per_batch + 1
    local_column = (global_column - 1) % cols_per_batch + 1
    l1, l2, l3 = params["L1"], params["L2"], params["L3"]
    pair_count = params["hybrid_pair_count"]

    def make(y: int) -> Optional[Coordinate]:
        x = y + local_round - schedule.q
        return (batch, x, y) if _valid_band(schedule, x, y) else None

    if local_column <= pair_count:
        result = make(local_column)
        if result is not None:
            return result
        seg3_index = schedule.p + local_column - 1
        return make(l1 + l2 + seg3_index)
    if local_column <= pair_count + l2:
        return make(l1 + local_column - pair_count)

    single_index = local_column - pair_count - l2
    unpaired_seg1 = l1 - pair_count
    if single_index <= unpaired_seg1:
        return make(pair_count + single_index)
    unpaired_seg3_index = single_index - unpaired_seg1
    unpaired_seg3_prefix = min(schedule.p - 1, l3)
    seg3_index = (
        unpaired_seg3_index
        if unpaired_seg3_index <= unpaired_seg3_prefix
        else unpaired_seg3_index + pair_count
    )
    return make(l1 + l2 + seg3_index)


def schedule_coordinate(schedule: ScheduleResult, j: int, r: int) -> Optional[Coordinate]:
    """按 Host 已选模式镜像 Kernel 坐标分发，并执行 BAND 有效区过滤。"""
    if schedule.mode == DeterBandScheduleMode.DISABLED or not (1 <= r <= schedule.max_round):
        return None
    if schedule.mode == DeterBandScheduleMode.BAND:
        return cal_band_hybrid_index(schedule, j, r)
    if schedule.mode == DeterBandScheduleMode.DENSE:
        result = cal_dense_swizzle_index(schedule.k, schedule.m, schedule.n, schedule.b, j, r)
        if result is None:
            return None
        batch, x, y = result
        return result if _valid_band(schedule, x, y) else None

    causal_size = schedule.m + schedule.q - 1
    pair_count = schedule.b // 2
    paired_batch = pair_count * 2
    if schedule.b % 2 == 1 and r > schedule.pair_round:
        result = cal_causal_single_batch_deter_index(schedule.k, causal_size, j, r - schedule.pair_round)
        if result is None:
            return None
        x_causal, y = result
        batch = schedule.b
    else:
        result = cal_causal_swizzle_index(schedule.k, causal_size, paired_batch, j, r)
        if result is None:
            return None
        batch, x_causal, y = result
    x = x_causal - (schedule.q - 1)
    return (batch, x, y) if _valid_band(schedule, x, y) else None


def calc_pos_hybrid(k: int, m: int, n: int, p: int, q: int, b: int, j: int, r: int):
    """兼容旧入口，返回 ``(w,(x,y))``。"""
    result = schedule_coordinate(select_deter_band_schedule(k, m, n, p, q, b), j, r)
    return None if result is None else (result[0], (result[1], result[2]))


def verify_case(k: int, m: int, n: int, p: int, q: int, b: int) -> VerifyResult:
    schedule = select_deter_band_schedule(k, m, n, p, q, b)
    errors: list[str] = []
    if schedule.mode == DeterBandScheduleMode.DISABLED:
        return VerifyResult(True, (), schedule, 0, tuple(0 for _ in range(max(k, 0))))

    seen: dict[Coordinate, tuple[int, int]] = {}
    row_round: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    column_cores: dict[tuple[int, int], set[int]] = defaultdict(set)
    loads = [0] * k
    for core in range(1, k + 1):
        for round_id in range(1, schedule.max_round + 1):
            coordinate = schedule_coordinate(schedule, core, round_id)
            if coordinate is None:
                continue
            if coordinate in seen:
                errors.append(f"重复坐标 {coordinate}: {seen[coordinate]} 与 {(core, round_id)}")
            else:
                seen[coordinate] = (core, round_id)
            batch, x, y = coordinate
            row_round[(batch, x, round_id)].append(core)
            column_cores[(batch, y)].add(core)
            loads[core - 1] += 1
        if schedule_coordinate(schedule, core, schedule.max_round + 1) is not None:
            errors.append(f"core {core} 在 maxRound 之后仍返回有效坐标")

    expected = {
        (batch, x, y)
        for batch in range(1, b + 1)
        for y in range(1, schedule.n + 1)
        for x in range(max(1, y - schedule.q + 1), min(schedule.m, y + schedule.p - 1) + 1)
    }
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing:
        errors.append(f"缺失坐标 {len(missing)} 个")
    if extra:
        errors.append(f"越界坐标 {len(extra)} 个")
    collisions = [key for key, cores in row_round.items() if len(cores) > 1]
    if collisions:
        errors.append(f"同 Batch 同行同轮冲突 {len(collisions)} 处")
    split_columns = [key for key, cores in column_cores.items() if len(cores) > 1]
    if split_columns:
        errors.append(f"同一列跨核 {len(split_columns)} 列")
    return VerifyResult(not errors, tuple(errors), schedule, len(seen), tuple(loads))


def verify(k: int, m: int, n: int, p: int, q: int, b: int, verbose: bool = True):
    """兼容旧入口。"""
    result = verify_case(k, m, n, p, q, b)
    if verbose:
        status = "PASS" if result.ok else "FAIL"
        loads = result.loads or (0,)
        print(
            f"{status} mode={result.schedule.mode_name} maxRound={result.schedule.max_round} "
            f"coordinates={result.coordinate_count} load=[{min(loads)},{max(loads)}]"
        )
        for error in result.errors:
            print(f"  {error}")
    return result.ok, list(result.errors)


def run_full_test() -> None:
    assert keep_legacy_right_down_swizzle(RIGHT_DOWN_CAUSAL_SPARSE_MODE, True)
    assert not keep_legacy_right_down_swizzle(RIGHT_DOWN_CAUSAL_SPARSE_MODE, False)
    assert not keep_legacy_right_down_swizzle(4, True)
    assert calc_legacy_right_down_round(28, 33, 49, 17) == 660
    assert calc_legacy_right_down_round(28, 33, 49, 18) == 726

    branch_cases = [
        ((28, 55, 55, 55, 1, 13), DeterBandScheduleMode.CAUSAL, 715),
        ((28, 55, 55, 55, 1, 12), DeterBandScheduleMode.CAUSAL, 660),
        ((28, 55, 55, 55, 55, 2), DeterBandScheduleMode.DENSE, 220),
        ((28, 55, 55, 3, 3, 2), DeterBandScheduleMode.BAND, 20),
    ]
    checked_coordinates = 0
    for arguments, expected_mode, expected_round in branch_cases:
        result = verify_case(*arguments)
        assert result.ok, (arguments, result.errors)
        assert result.schedule.mode == expected_mode, (arguments, result.schedule)
        assert result.schedule.max_round == expected_round, (arguments, result.schedule)
        checked_coordinates += result.coordinate_count

    checked_cases = len(branch_cases)
    for k in (1, 2, 3, 4, 7, 8, 12, 16):
        for size in (4, 5, 8, 15, 23, 31):
            for b in (2, 3, 5):
                for q in (1, 2):
                    result = verify_case(k, size, size, size, q, b)
                    assert result.ok, ((k, size, size, size, q, b), result.errors)
                    checked_cases += 1
                    checked_coordinates += result.coordinate_count
    print(f"PASS full-test cases={checked_cases} coordinates={checked_coordinates}")


def visualize_grouped(
    k: int,
    m: int,
    n: int,
    p: int,
    q: int,
    b: int,
    title: str = "",
    output_path: Optional[str | Path] = None,
    cell_size: float = 0.28,
    dpi: int = 220,
    batches_per_image: int = 2,
) -> list[Path]:
    """按算子实际分支生成核号/轮次图；绘图依赖仅在调用时加载。"""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    schedule = select_deter_band_schedule(k, m, n, p, q, b)
    mapping: list[tuple[int, int, int, int, int]] = []
    for core in range(1, k + 1):
        for round_id in range(1, schedule.max_round + 1):
            result = schedule_coordinate(schedule, core, round_id)
            if result is not None:
                batch, x, y = result
                mapping.append((batch, core, round_id, x, y))

    base_path = Path(output_path) if output_path else Path(__file__).with_name(
        f"band_hybrid_{schedule.mode_name.lower()}_k{k}_m{m}_n{n}_p{p}_q{q}_b{b}.png"
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the core-color convention aligned with the Dense TND visualizer.
    color_pool = (
        list(plt.get_cmap("tab20", 20).colors)
        + list(plt.get_cmap("tab20b", 20).colors)
        + list(plt.get_cmap("tab20c", 20).colors)
    )
    color_order = np.random.RandomState(42).permutation(len(color_pool))
    selected_colors = [color_pool[index] for index in color_order[:max(k, 1)]]
    cmap = ListedColormap(selected_colors, name=f"shuffled{len(selected_colors)}")
    max_round_digits = max(1, len(str(schedule.max_round)))
    cell_px = max(cell_size * dpi, 56, max_round_digits * 16 + 18)
    text_size = max(6, min(12, cell_px / max(4.5, max_round_digits * 1.4)))
    output_files: list[Path] = []
    for group_index, start_batch in enumerate(range(1, b + 1, batches_per_image), start=1):
        group = list(range(start_batch, min(start_batch + batches_per_image, b + 1)))
        columns = min(2, len(group))
        rows = math.ceil(len(group) / columns)
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(
                max(4.5, schedule.n * cell_px / dpi) * columns,
                max(4.5, schedule.m * cell_px / dpi) * rows,
            ),
            dpi=dpi,
            squeeze=False,
        )
        for local_index, batch in enumerate(group):
            ax = axes[local_index // columns][local_index % columns]
            core_map = np.zeros((schedule.m, schedule.n), dtype=int)
            round_map = np.zeros((schedule.m, schedule.n), dtype=int)
            for current_batch, core, round_id, x, y in mapping:
                if current_batch == batch:
                    core_map[x - 1, y - 1] = core
                    round_map[x - 1, y - 1] = round_id
            ax.imshow(
                np.ma.array(core_map, mask=core_map == 0),
                origin="upper",
                cmap=cmap,
                vmin=1,
                vmax=k,
                aspect="equal",
                interpolation="none",
            )
            ax.set_xticks(np.arange(schedule.n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(schedule.m + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.35)
            ax.tick_params(which="minor", length=0)
            if schedule.m <= 40 and schedule.n <= 40:
                ax.set_xticks(np.arange(schedule.n))
                ax.set_yticks(np.arange(schedule.m))
                ax.set_xticklabels(np.arange(1, schedule.n + 1))
                ax.set_yticklabels(np.arange(1, schedule.m + 1))
                ax.tick_params(axis="both", which="major", labelsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(f"Batch {batch}")
            ax.set_xlabel("s2 block")
            ax.set_ylabel("s1 block")
            for x in range(schedule.m):
                for y in range(schedule.n):
                    if round_map[x, y] > 0:
                        ax.text(
                            y,
                            x,
                            str(round_map[x, y]),
                            ha="center",
                            va="center",
                            fontsize=text_size,
                            color="white",
                            fontfamily="Consolas",
                        )
        for index in range(len(group), rows * columns):
            axes[index // columns][index % columns].axis("off")
        fig.suptitle(
            f"{title or 'FAG deterministic BAND schedule'} | mode={schedule.mode_name} "
            f"maxRound={schedule.max_round} | k,m,n,p,q,b={k},{m},{n},{p},{q},{b}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = base_path.with_name(f"{base_path.stem}_group{group_index:02d}{base_path.suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        output_files.append(path)
    return output_files


def visualize(*args, **kwargs):
    files = visualize_grouped(*args, batches_per_image=kwargs.pop("batches_per_image", args[5] if len(args) > 5 else 2), **kwargs)
    return files[0] if len(files) == 1 else files


def _display_units(char: str) -> float:
    return 2.0 if unicodedata.east_asian_width(char) in ("W", "F") else 1.0


def _wrap_visual(text: str, width: float) -> list[str]:
    if not text:
        return [""]
    result: list[str] = []
    remaining = text
    while remaining:
        used = 0.0
        split_at = 0
        last_space = -1
        for index, char in enumerate(remaining):
            if used + _display_units(char) > width:
                break
            used += _display_units(char)
            split_at = index + 1
            if char.isspace():
                last_space = split_at
        else:
            result.append(remaining)
            break
        if last_space > 0 and split_at - last_space < 20:
            split_at = last_space
        split_at = max(split_at, 1)
        result.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return result


def render_pdf(output: Path, preview_dir: Optional[Path] = None) -> Path:
    """生成与本脚本/当前算子一致的固定 A4 方案文档。"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    font_paths = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]
    font_path = next((path for path in font_paths if path.exists()), None)
    font = FontProperties(fname=str(font_path)) if font_path else FontProperties(family="sans-serif")
    requested = select_deter_band_schedule(28, 55, 55, 55, 1, 13)
    blocks: list[tuple[str, str]] = [
        ("title", "FAG arch35 非 TND 确定性 BAND 三模式调度（算子同步版）"),
        ("body", "版本：v3.1    日期：2026-08-17"),
        ("body", "对应实现：SelectDeterBandSchedule / CalBandDeterIndex / CalDeterMaxLoopNum"),
        ("h2", "1. 结论"),
        ("body", "分支不是按 Causal、Dense、Band 的固定几何优先级选择。Host 先构造 BAND 基线，再依次评估 DENSE 和 CAUSAL；候选轮次只有严格小于当前最优轮次时才替换。平局保持先前模式。"),
        ("code", "baseline = BAND\nif denseRound  < bestRound: mode = DENSE\nif causalRound < bestRound: mode = CAUSAL"),
        ("h2", "2. 外围入口条件"),
        ("body", "本方案只描述已经进入确定性 DETER_BAND 调度选择后的三模式。算子外围还要求 enableSwizzle、非 TND、splitAxis=BN2GS1S2、两侧 Cube 基本块相等、sparseType 可支持。"),
        ("code", "isDeterministic\ndeterSparseType == DETER_BAND\ng == 1\ncoreNum == 2 * aicNum\nactualBatch = (b-tailZeroCount)*n2 > 0"),
        ("h3", "2.1 sparseMode=3 的远端兼容优先级"),
        ("body", "RIGHT_DOWN_CAUSAL 已满足远端原 swizzle 全部条件时，直接保留旧路径，不进入三模式选择器。Host 保持调度模式 DISABLED，且 isSplitByBlockIdx=true。Kernel 继续调用 CalCausalSwizzleIndex 和配套旧轮次公式，避免重新选择为 DENSE 后发生性能回退。"),
        ("code", "legacyRightDown = sparseMode==3 and oldIsSplitByBlockIdx\nif legacyRightDown:\n  scheduleMode=DISABLED; isSplitByBlockIdx=true\n  kernelDispatch=CalCausalSwizzleIndex\nelse:\n  evaluate BAND / DENSE / CAUSAL"),
        ("body", "旧入口还要求原始 b*n2 为偶数。若示例 B=17 是扣除 tailZero 后的 actualBatch，而原始 Batch 为 18，则旧路径仍成立；若原始 b*n2 本身就是 17，远端条件不会开启该 swizzle。m=33、n=49 时旧公式使用 causalM=33、causalN=68。"),
        ("h2", "3. 参数归一化"),
        ("code", "p = min(p,m); q = min(q,n)\nif p < 0: (m,n,p,q) = (m,n+p,1,p+q)\nif q < 0: (m,n,p,q) = (m+q,n,p+q,1)\n\np=min(p,m); q=min(q,n)\nm=min(m,n+p-1); n=min(n,m+q-1)"),
        ("body", "Python 的 normalize_deter_band_schedule_params() 与 seg_params() 按相同顺序执行，负 token 转换、clamp 和无效空行列裁剪均与算子一致。"),
        ("h2", "4. BAND 基线"),
        ("body", "根据 p+q 与 m 的关系计算 L1/L2/L3、有效块数 bandBlocks、可配对列数 hybridPairCount，以及每个 Batch 的虚拟列数。"),
        ("code", "colsPerBatch = L1+L2+L3-hybridPairCount\nbandSlot    = p+q-1\nbandRound   = ceil(b*colsPerBatch/k)*bandSlot"),
        ("h2", "5. DENSE 候选"),
        ("code", "denseK = min(k,m*b)\neligible = denseK==k and min(denseK,n)<=m\ndenseRound = ceil(n*b/denseK)*m\n仅当 denseRound < bestRound 时选择 DENSE"),
        ("body", "因此 DENSE 与 BAND 轮次相等时仍保持 BAND，这正是旧 Python 与算子容易不一致的边界。"),
        ("h2", "6. CAUSAL 候选"),
        ("body", "当前算子只采用 lower causal embedding。upperWaste 即使更小也不会转置使用，因为会破坏原始列固定归核约束。"),
        ("code", "lowerSize  = m+q-1\nupperSize  = n+p-1\nlowerWaste = C(lowerSize)-bandBlocks\nuseLower = bandBlocks>0 and 0<=lowerWaste<=upperWaste\n           and lowerWaste<=floor((bandBlocks-1)/10)"),
        ("body", "CAUSAL 还要求至少存在一个 Batch pair、配对段不减少活跃核数，并满足 causalK<=lowerSize。"),
        ("code", "pairCount = b//2\ncausalK   = min(k,lowerSize*pairCount)\neligible  = pairCount>0 and causalK==k and causalK<=lowerSize\npairRound = ceil((lowerSize+1)*pairCount/causalK)*lowerSize"),
        ("h3", "6.1 奇数尾 Batch"),
        ("body", "b 为奇数时，前 b-1 个 Batch 继续成对拼接；最后一个 Batch 调用 CalCausalSingleBatchDeterIndex。该函数保持同行同轮互斥和整列单核归属。"),
        ("code", "groupCount = lowerSize//(2*k)\ngroupRound = (2*lowerSize+1)*groupCount-2*k*groupCount^2\nremain = lowerSize-2*k*groupCount\ntailRound = groupRound + (remain if remain<=k\n             else max(remain,2*remain-2*k+1))\ncausalRound = pairRound+tailRound"),
        ("h2", "7. 目标 case 的实际分支"),
        ("code", "k,m,n,p,q,b = 28,55,55,55,1,13\nBAND  = 1430\nDENSE = 1430（平局，不切换）\nCAUSAL pair = 660\nCAUSAL tail = 55\nCAUSAL total = 715"),
        ("body", f"最终 mode={requested.mode_name}，maxRound={requested.max_round}。有效任务数为 20,020，28×715 也为 20,020；每核负载 715，逻辑浪费轮次为 0。"),
        ("h2", "8. Kernel 坐标分发"),
        ("code", "r <= pairRound:\n  CalCausalSwizzleIndex(..., pairedBatch, r)\nr > pairRound and b is odd:\n  CalCausalSingleBatchDeterIndex(..., r-pairRound)\n  batchId = b\n最后执行 x -= q-1 和 BAND 有效区过滤"),
        ("body", "Kernel 的 CAUSAL 总轮次直接读取 Host 序列化的 deterMaxRound；pairRound 在 Kernel 按同一公式计算，因此阶段边界和总轮次均与 Host 对齐。"),
        ("h2", "9. Python 验证项"),
        ("body", "验证包括：分支和 maxRound、坐标全覆盖、无重复/越界、maxRound 后无有效任务、同 Batch 同行同轮无冲突、同一实际列不跨核、偶数配对段和奇数尾段的边界连续。"),
        ("code", "python band_hybrid.py --full-test\npython band_hybrid.py --case 28 55 55 55 1 13\npython band_hybrid.py --render-pdf band_hybrid.pdf"),
        ("h2", "10. 回退说明"),
        ("body", "Python 的 DISABLED 只表示三模式选择器未启用；完整算子还会根据 isSplitByBlockIdx 和原有确定性路径回退。脚本的 k/m/n/p/q/b 分支判断以外围条件已经满足为前提。"),
    ]

    page_size = (8.27, 11.69)
    left, right, top, bottom = 0.072, 0.928, 0.942, 0.062
    styles = {
        "title": (18.0, 0.036, "#17365d", 62.0, "bold", 0.010),
        "h2": (13.0, 0.028, "#1f4e79", 82.0, "bold", 0.006),
        "h3": (10.8, 0.023, "#2f5597", 96.0, "bold", 0.004),
        "body": (8.8, 0.019, "#202020", 108.0, "normal", 0.005),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        figure = None
        axis = None
        page = 0
        y = top

        def save_page() -> None:
            if figure is None:
                return
            pdf.savefig(figure)
            if preview_dir:
                figure.savefig(preview_dir / f"page-{page:02d}.png", dpi=120)
            plt.close(figure)

        def new_page():
            nonlocal figure, axis, page, y
            if figure is not None:
                save_page()
            page += 1
            figure = plt.figure(figsize=page_size, facecolor="white")
            axis = figure.add_axes((0, 0, 1, 1))
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.axis("off")
            if page > 1:
                axis.text(left, 0.973, "FAG 确定性 BAND 三模式调度（算子同步版）", ha="left", va="top",
                          fontsize=7.5, color="#6b7280", fontproperties=font)
                axis.plot((left, right), (0.958, 0.958), color="#d7dde5", linewidth=0.6)
            axis.plot((left, right), (0.047, 0.047), color="#d7dde5", linewidth=0.5)
            axis.text(0.5, 0.027, f"— {page} —", ha="center", va="center", fontsize=7.5,
                      color="#777777", fontproperties=font)
            y = top

        new_page()
        for kind, content in blocks:
            if kind == "code":
                lines = [line for source_line in content.splitlines() for line in _wrap_visual(source_line, 111)]
                height = 0.010 * 2 + 0.0175 * max(len(lines), 1)
                if y - height < bottom:
                    new_page()
                axis.add_patch(Rectangle((left, y - height), right - left, height, facecolor="#f4f6f8",
                                         edgecolor="#d5dbe3", linewidth=0.6))
                text_y = y - 0.010
                for line in lines:
                    props = font.copy()
                    props.set_size(8.1)
                    axis.text(left + 0.012, text_y, line, ha="left", va="top", color="#263238",
                              fontproperties=props)
                    text_y -= 0.0175
                y -= height + 0.009
                continue
            size, line_height, color, width, weight, after = styles[kind]
            lines = _wrap_visual(content, width)
            required = line_height * len(lines) + after + (0.035 if kind in ("h2", "h3") else 0)
            if y - required < bottom:
                new_page()
            for line in lines:
                props = font.copy()
                props.set_size(size)
                props.set_weight(weight)
                axis.text(left, y, line, ha="left", va="top", color=color, fontproperties=props)
                y -= line_height
            y -= after
        save_page()
    return output


def _print_case(schedule: ScheduleResult) -> None:
    print(f"mode={schedule.mode_name} maxRound={schedule.max_round}")
    print(f"effective(m,n,p,q)=({schedule.m},{schedule.n},{schedule.p},{schedule.q})")
    print(
        f"candidateRounds: BAND={schedule.band_round} DENSE={schedule.dense_round} "
        f"CAUSAL={schedule.causal_round} (pair={schedule.pair_round}, tail={schedule.tail_round})"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=6, type=int, metavar=("K", "M", "N", "P", "Q", "B"),
                        default=(32, 32, 64, 32, 32, 32))
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--render-pdf", nargs="?", const=str(here / "band_hybrid.pdf"))
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    case = tuple(args.case)
    result = verify_case(*case)
    _print_case(result.schedule)
    print(f"verify={'PASS' if result.ok else 'FAIL'} coordinates={result.coordinate_count}")
    if result.loads:
        print(f"load=[{min(result.loads)},{max(result.loads)}] total={sum(result.loads)}")
    for error in result.errors:
        print(f"  {error}")
    if not result.ok:
        raise SystemExit(1)
    if args.full_test:
        run_full_test()
    if args.visualize:
        for path in visualize_grouped(*case, output_path=here / "outputs" / "band_hybrid.png"):
            print(path.resolve())
    if args.render_pdf:
        print(render_pdf(Path(args.render_pdf), args.preview_dir).resolve())


if __name__ == "__main__":
    main()
