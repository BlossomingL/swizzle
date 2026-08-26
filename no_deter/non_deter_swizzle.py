#!/usr/bin/env python3
"""FAG 非确定性 Dense/Causal 严格按列分核参考实现。

“按列分核”的硬约束：同一个实际列 (batch, s2_idx) 的全部 s1 基本块
只能由一个 core 处理。坐标和核号均使用 0-based。

运行：
    python non_deter_swizzle.py
    python non_deter_swizzle.py --full-test --demo
    python non_deter_swizzle.py --plot-dir figures
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


@dataclass(frozen=True, slots=True)
class Coordinate:
    batch: int
    s1_idx: int
    s2_idx: int


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    mode: str
    cases: int
    coordinates: int
    columns: int
    max_load_skew_ratio: float


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


def _check_common(k: int, m: int, b: int, core_id: int, local_step: int) -> None:
    if k <= 0 or m <= 0 or b <= 0:
        raise ValueError("k, m and b must all be positive")
    if not 0 <= core_id < k:
        raise ValueError(f"core_id must be in [0, {k}), got {core_id}")
    if local_step < 0:
        raise ValueError("local_step must be non-negative")


def _owned_descriptor_count(total_descriptors: int, k: int, core_id: int) -> int:
    """descriptor 按 core_id, core_id+k, ... 轮转分配。"""
    if total_descriptors <= core_id:
        return 0
    return (total_descriptors - 1 - core_id) // k + 1


# ---------------------------------------------------------------------------
# Dense：分核基本单元是完整的实际列 (batch, s2_idx)。
# ---------------------------------------------------------------------------


def dense_total_tasks(m: int, n: int, b: int) -> int:
    if m <= 0 or n <= 0 or b <= 0:
        raise ValueError("m, n and b must all be positive")
    return b * m * n


def dense_core_task_count(k: int, m: int, n: int, b: int, core_id: int) -> int:
    _check_common(k, m, b, core_id, 0)
    if n <= 0:
        raise ValueError("n must be positive")
    owned_columns = _owned_descriptor_count(b * n, k, core_id)
    return owned_columns * m


def dense_max_local_steps(k: int, m: int, n: int, b: int) -> int:
    """所有核可共用的 Dense 循环上界。"""
    if min(k, m, n, b) <= 0:
        raise ValueError("k, m, n and b must all be positive")
    return ceil_div(b * n, k) * m


def dense_column_owner(k: int, n: int, b: int, batch: int, s2_idx: int) -> int:
    """返回 Dense 实际列的唯一 owner core。"""
    if k <= 0 or n <= 0 or b <= 0 or not 0 <= batch < b or not 0 <= s2_idx < n:
        raise ValueError("invalid Dense column owner arguments")
    return (batch * n + s2_idx) % k


def dense_position(
    k: int,
    m: int,
    n: int,
    b: int,
    core_id: int,
    local_step: int,
) -> Optional[Coordinate]:
    """Dense 严格按列映射。

    一个核先完整遍历当前列的 m 个 s1 块，再领取下一列。
    column_id = local_column_id * k + core_id 保证列 owner 唯一。
    """
    _check_common(k, m, b, core_id, local_step)
    if n <= 0:
        raise ValueError("n must be positive")

    local_column_id, s1_idx = divmod(local_step, m)
    column_id = local_column_id * k + core_id
    if column_id >= b * n:
        return None
    batch, s2_idx = divmod(column_id, n)
    return Coordinate(batch, s1_idx, s2_idx)


# ---------------------------------------------------------------------------
# Causal：m == n，实际有效区域为 s2_idx <= s1_idx。
# 偶数部分将两个 batch 拼成 m*(m+1) 矩形，但按“虚拟整列”归核；
# 一个虚拟列最多包含两个实际列片段，这两个实际列也都只属于该核。
# 奇数尾 batch 的实际列按轮转方式完整归核。
# ---------------------------------------------------------------------------


def causal_total_tasks(m: int, b: int) -> int:
    if m <= 0 or b <= 0:
        raise ValueError("m and b must both be positive")
    return b * m * (m + 1) // 2


def _triangle_column_prefix(m: int, s2_idx: int) -> int:
    return s2_idx * (2 * m - s2_idx + 1) // 2


def _causal_pair_coordinate(
    m: int,
    virtual_column_id: int,
    virtual_s1: int,
) -> Coordinate:
    """将一个配对矩形的虚拟列坐标映射回两个实际下三角 batch。"""
    pair_id, virtual_s2 = divmod(virtual_column_id, m + 1)
    if virtual_s2 <= virtual_s1:
        return Coordinate(pair_id * 2, virtual_s1, virtual_s2)
    return Coordinate(pair_id * 2 + 1, m - 1 - virtual_s1, m - virtual_s2)


def _tail_owner_start(pair_columns: int, k: int) -> int:
    """把尾三角最长列优先交给配对区中少拿一列的核。"""
    return pair_columns % k


def _tail_residue(pair_columns: int, k: int, core_id: int) -> int:
    return (core_id - _tail_owner_start(pair_columns, k)) % k


def _tail_column_count(m: int, k: int, residue: int) -> int:
    return ceil_div(m - residue, k) if residue < m else 0


def _tail_task_prefix(column_count: int, m: int, k: int, residue: int) -> int:
    """该核前 column_count 个尾三角列的总任务数。"""
    return column_count * (m - residue) - k * column_count * (column_count - 1) // 2


def _tail_core_task_count(m: int, k: int, pair_columns: int, core_id: int) -> int:
    residue = _tail_residue(pair_columns, k, core_id)
    count = _tail_column_count(m, k, residue)
    return _tail_task_prefix(count, m, k, residue)


def _decode_tail_task(
    task_id: int,
    m: int,
    k: int,
    pair_columns: int,
    core_id: int,
) -> tuple[int, int]:
    """反解该核奇数尾 batch 中按完整列拼接的 task_id。"""
    residue = _tail_residue(pair_columns, k, core_id)
    column_count = _tail_column_count(m, k, residue)
    total = _tail_task_prefix(column_count, m, k, residue)
    if not 0 <= task_id < total:
        raise ValueError("tail task_id is out of range")

    # prefix(q) = q*(m-residue) - k*q*(q-1)/2。
    # 使用较小二次根给出候选，再用精确整数前缀做常数次边界修正。
    length0 = m - residue
    coefficient = 2 * length0 + k
    discriminant = coefficient * coefficient - 8 * k * task_id
    local_column = (coefficient - math.isqrt(discriminant)) // (2 * k)
    local_column = min(max(local_column, 0), column_count - 1)
    while local_column > 0 and _tail_task_prefix(local_column, m, k, residue) > task_id:
        local_column -= 1
    while (
        local_column + 1 < column_count
        and _tail_task_prefix(local_column + 1, m, k, residue) <= task_id
    ):
        local_column += 1

    s2_idx = residue + local_column * k
    offset = task_id - _tail_task_prefix(local_column, m, k, residue)
    return s2_idx + offset, s2_idx


def causal_core_task_count(k: int, m: int, b: int, core_id: int) -> int:
    _check_common(k, m, b, core_id, 0)
    pair_columns = (b // 2) * (m + 1)
    paired_tasks = _owned_descriptor_count(pair_columns, k, core_id) * m
    tail_tasks = _tail_core_task_count(m, k, pair_columns, core_id) if b % 2 else 0
    return paired_tasks + tail_tasks


def causal_max_local_steps(k: int, m: int, b: int) -> int:
    if min(k, m, b) <= 0:
        raise ValueError("k, m and b must all be positive")
    return max(causal_core_task_count(k, m, b, core_id) for core_id in range(k))


def causal_column_owner(k: int, m: int, b: int, batch: int, s2_idx: int) -> int:
    """返回 Causal 实际列的唯一 owner core。"""
    if k <= 0 or m <= 0 or b <= 0 or not 0 <= batch < b or not 0 <= s2_idx < m:
        raise ValueError("invalid Causal column owner arguments")
    pair_id = batch // 2
    pair_columns = (b // 2) * (m + 1)
    if batch < (b // 2) * 2:
        virtual_s2 = s2_idx if batch % 2 == 0 else m - s2_idx
        return (pair_id * (m + 1) + virtual_s2) % k
    return (s2_idx + _tail_owner_start(pair_columns, k)) % k


def causal_position(
    k: int,
    m: int,
    b: int,
    core_id: int,
    local_step: int,
) -> Optional[Coordinate]:
    """Causal 严格按实际列归核，支持奇偶 batch。"""
    _check_common(k, m, b, core_id, local_step)
    pair_columns = (b // 2) * (m + 1)
    owned_pair_columns = _owned_descriptor_count(pair_columns, k, core_id)
    paired_task_count = owned_pair_columns * m

    if local_step < paired_task_count:
        local_column_id, virtual_s1 = divmod(local_step, m)
        virtual_column_id = local_column_id * k + core_id
        return _causal_pair_coordinate(m, virtual_column_id, virtual_s1)

    if b % 2 == 0:
        return None

    tail_task_id = local_step - paired_task_count
    tail_total = _tail_core_task_count(m, k, pair_columns, core_id)
    if tail_task_id >= tail_total:
        return None
    s1_idx, s2_idx = _decode_tail_task(tail_task_id, m, k, pair_columns, core_id)
    return Coordinate(b - 1, s1_idx, s2_idx)


def iter_core_tasks(
    position_fn: Callable[..., Optional[Coordinate]],
    loop_limit: int,
    k: int,
    core_id: int,
    **position_args: int,
) -> Iterable[tuple[int, Coordinate]]:
    for local_step in range(loop_limit):
        coordinate = position_fn(k=k, core_id=core_id, local_step=local_step, **position_args)
        if coordinate is not None:
            yield local_step, coordinate


def _validate_one(mode: str, k: int, m: int, n: int, b: int) -> tuple[int, int, int]:
    if mode == "dense":
        total = dense_total_tasks(m, n, b)
        loop_limit = dense_max_local_steps(k, m, n, b)
        position_fn = dense_position
        core_count_fn = lambda core: dense_core_task_count(k, m, n, b, core)
        args = {"m": m, "n": n, "b": b}
    elif mode == "causal":
        if m != n:
            raise ValueError("causal reference implementation requires m == n")
        total = causal_total_tasks(m, b)
        loop_limit = causal_max_local_steps(k, m, b)
        position_fn = causal_position
        core_count_fn = lambda core: causal_core_task_count(k, m, b, core)
        args = {"m": m, "b": b}
    else:
        raise ValueError(f"unknown mode: {mode}")

    seen: dict[Coordinate, tuple[int, int]] = {}
    column_owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    loads: list[int] = []
    for core_id in range(k):
        load = 0
        for local_step in range(loop_limit):
            coordinate = position_fn(k=k, core_id=core_id, local_step=local_step, **args)
            if coordinate is None:
                continue
            if not (0 <= coordinate.batch < b and 0 <= coordinate.s1_idx < m and 0 <= coordinate.s2_idx < n):
                raise AssertionError(("out_of_range", mode, k, m, n, b, coordinate))
            if mode == "causal" and coordinate.s2_idx > coordinate.s1_idx:
                raise AssertionError(("outside_causal", k, m, b, coordinate))
            if coordinate in seen:
                raise AssertionError(
                    ("duplicate", mode, k, m, n, b, coordinate, seen[coordinate], (core_id, local_step))
                )
            seen[coordinate] = (core_id, local_step)
            column_owners[(coordinate.batch, coordinate.s2_idx)].add(core_id)
            expected_owner = (
                dense_column_owner(k, n, b, coordinate.batch, coordinate.s2_idx)
                if mode == "dense"
                else causal_column_owner(k, m, b, coordinate.batch, coordinate.s2_idx)
            )
            if expected_owner != core_id:
                raise AssertionError(
                    ("wrong_column_owner", mode, k, m, n, b, coordinate, core_id, expected_owner)
                )
            load += 1

        expected_core_load = core_count_fn(core_id)
        if load != expected_core_load:
            raise AssertionError(("core_load", mode, k, m, n, b, core_id, load, expected_core_load))
        if position_fn(k=k, core_id=core_id, local_step=loop_limit, **args) is not None:
            raise AssertionError(("loop_bound_too_small", mode, k, m, n, b, core_id))
        loads.append(load)

    if len(seen) != total:
        raise AssertionError(("incomplete", mode, k, m, n, b, len(seen), total))

    expected_columns = b * n
    if len(column_owners) != expected_columns:
        raise AssertionError(("column_incomplete", mode, k, m, n, b, len(column_owners), expected_columns))
    split_columns = {column: owners for column, owners in column_owners.items() if len(owners) != 1}
    if split_columns:
        raise AssertionError(("column_split", mode, k, m, n, b, list(split_columns.items())[:4]))

    skew = max(loads) - min(loads)
    # 严格按列后，负载最小粒度是列；轮转和尾列旋转使差值不超过一个最长列 m。
    if skew > m:
        raise AssertionError(("load_skew", mode, k, m, n, b, loads))
    return total, expected_columns, skew


def _has_same_row_in_one_step(mode: str, k: int, m: int, n: int, b: int) -> bool:
    loop_limit = dense_max_local_steps(k, m, n, b) if mode == "dense" else causal_max_local_steps(k, m, b)
    for local_step in range(loop_limit):
        rows: set[tuple[int, int]] = set()
        for core_id in range(k):
            coordinate = (
                dense_position(k, m, n, b, core_id, local_step)
                if mode == "dense"
                else causal_position(k, m, b, core_id, local_step)
            )
            if coordinate is None:
                continue
            key = (coordinate.batch, coordinate.s1_idx)
            if key in rows:
                return True
            rows.add(key)
    return False


def column_reuse_ratio(mode: str, k: int, m: int, n: int, b: int) -> float:
    """单核相邻任务复用同一实际 (batch, s2_idx) 的比例。"""
    if mode == "dense":
        loop_limit = dense_max_local_steps(k, m, n, b)
        position_fn = dense_position
        args = {"m": m, "n": n, "b": b}
    else:
        loop_limit = causal_max_local_steps(k, m, b)
        position_fn = causal_position
        args = {"m": m, "b": b}

    same_column = 0
    transitions = 0
    for core_id in range(k):
        previous: Optional[Coordinate] = None
        for _, coordinate in iter_core_tasks(position_fn, loop_limit, k, core_id, **args):
            if previous is not None:
                transitions += 1
                same_column += int(
                    previous.batch == coordinate.batch and previous.s2_idx == coordinate.s2_idx
                )
            previous = coordinate
    return same_column / transitions if transitions else 1.0


def run_self_test(full: bool = False) -> list[ValidationSummary]:
    """穷举与随机验证覆盖、唯一性、严格列 owner 和负载上界。"""
    counters = {
        "dense": {"cases": 0, "coordinates": 0, "columns": 0, "max_ratio": 0.0},
        "causal": {"cases": 0, "coordinates": 0, "columns": 0, "max_ratio": 0.0},
    }

    def check(mode: str, k: int, m: int, n: int, b: int) -> None:
        total, columns, skew = _validate_one(mode, k, m, n, b)
        item = counters[mode]
        item["cases"] += 1
        item["coordinates"] += total
        item["columns"] += columns
        item["max_ratio"] = max(item["max_ratio"], skew / m)

    small_limit = 12 if full else 9
    for k in range(1, 9):
        for m in range(1, small_limit + 1):
            for n in range(1, small_limit + 1):
                for b in range(1, 5):
                    check("dense", k, m, n, b)
        for m in range(1, small_limit + 3):
            for b in range(1, 6):
                check("causal", k, m, m, b)

    rng = random.Random(20260813)
    random_cases = 360 if full else 300
    dimension_limit = 128 if full else 96
    for _ in range(random_cases):
        k = rng.choice((16, 20, 24, 32, 48))
        m = rng.randint(1, dimension_limit)
        n = rng.randint(1, dimension_limit)
        b = rng.randint(1, 8)
        check("dense", k, m, n, b)
        check("causal", k, m, m, b)

    # 非确定性允许同行并发；列 owner 唯一不等于同行互斥。
    if not _has_same_row_in_one_step("dense", 8, 8, 8, 1):
        raise AssertionError("dense schedule unexpectedly retains same-row exclusion")
    if not _has_same_row_in_one_step("causal", 4, 8, 8, 2):
        raise AssertionError("causal schedule unexpectedly retains same-row exclusion")

    # 奇数尾三角的整数反解检查到大 m，覆盖各核的列首/列中/列尾任务。
    for m in (1, 2, 3, 127, 128, 129, 4096, 1_000_000):
        for k in (1, 2, 7, 32, 48):
            pair_columns = 3 * (m + 1)
            for core_id in range(k):
                total = _tail_core_task_count(m, k, pair_columns, core_id)
                if total == 0:
                    continue
                for task_id in {0, total // 2, total - 1}:
                    s1_idx, s2_idx = _decode_tail_task(task_id, m, k, pair_columns, core_id)
                    owner = (s2_idx + _tail_owner_start(pair_columns, k)) % k
                    if not (0 <= s2_idx <= s1_idx < m and owner == core_id):
                        raise AssertionError(
                            ("large_tail_inverse", m, k, core_id, task_id, s1_idx, s2_idx, owner)
                        )

    # 仅检查解析负载公式，可低成本覆盖更宽的 k/m/b 组合。
    for k in range(1, 65):
        for m in range(1, 257):
            for b in range(1, 10):
                loads = [causal_core_task_count(k, m, b, core_id) for core_id in range(k)]
                if max(loads) - min(loads) > m:
                    raise AssertionError(("analytic_causal_load_skew", k, m, b, loads))

    return [
        ValidationSummary(
            mode=mode,
            cases=counters[mode]["cases"],
            coordinates=counters[mode]["coordinates"],
            columns=counters[mode]["columns"],
            max_load_skew_ratio=counters[mode]["max_ratio"],
        )
        for mode in ("dense", "causal")
    ]


def _build_plot_data(
    mode: str,
    k: int,
    m: int,
    n: int,
    b: int,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    core_grid = [[[-1 for _ in range(n)] for _ in range(m)] for _ in range(b)]
    step_grid = [[[-1 for _ in range(n)] for _ in range(m)] for _ in range(b)]
    if mode == "dense":
        loop_limit = dense_max_local_steps(k, m, n, b)
        position_fn = dense_position
        args = {"m": m, "n": n, "b": b}
    else:
        loop_limit = causal_max_local_steps(k, m, b)
        position_fn = causal_position
        args = {"m": m, "b": b}
    for core_id in range(k):
        for local_step, coordinate in iter_core_tasks(position_fn, loop_limit, k, core_id, **args):
            core_grid[coordinate.batch][coordinate.s1_idx][coordinate.s2_idx] = core_id
            step_grid[coordinate.batch][coordinate.s1_idx][coordinate.s2_idx] = local_step
    return core_grid, step_grid


def plot_schedule(mode: str, k: int, m: int, n: int, b: int, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("plot_schedule requires matplotlib and numpy") from error

    if mode == "causal" and m != n:
        raise ValueError("causal plot requires m == n")
    core_grid, step_grid = _build_plot_data(mode, k, m, n, b)
    cols = min(2, b)
    rows = ceil_div(b, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.3 * rows), squeeze=False)
    colors = ["#eeeeee"] + list(plt.get_cmap("tab20").colors) * ceil_div(k, 20)
    cmap = ListedColormap(colors[: k + 1])

    for batch in range(b):
        ax = axes[batch // cols][batch % cols]
        matrix = np.asarray(core_grid[batch], dtype=int) + 1
        ax.imshow(matrix, origin="upper", cmap=cmap, vmin=0, vmax=k, interpolation="nearest")
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", length=0)
        ax.set_xlabel("s2 block (column)")
        ax.set_ylabel("s1 block (row)")
        ax.set_title(f"batch={batch}")
        if max(m, n) <= 18:
            for s1_idx in range(m):
                for s2_idx in range(n):
                    core_id = core_grid[batch][s1_idx][s2_idx]
                    if core_id >= 0:
                        local_step = step_grid[batch][s1_idx][s2_idx]
                        ax.text(
                            s2_idx,
                            s1_idx,
                            f"c{core_id}\nl{local_step}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="black",
                        )
    for index in range(b, rows * cols):
        axes[index // cols][index % cols].axis("off")

    reuse = column_reuse_ratio(mode, k, m, n, b)
    fig.suptitle(
        f"Strict column-owned non-deterministic {mode.upper()} swizzle | "
        f"k={k}, m={m}, n={n}, b={b}, same-column transition={reuse:.1%}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _print_demo() -> None:
    cases = (("dense", 8, 16, 12, 2), ("causal", 8, 16, 16, 3))
    for mode, k, m, n, b in cases:
        total = dense_total_tasks(m, n, b) if mode == "dense" else causal_total_tasks(m, b)
        loop_limit = dense_max_local_steps(k, m, n, b) if mode == "dense" else causal_max_local_steps(k, m, b)
        ratio = column_reuse_ratio(mode, k, m, n, b)
        print(
            f"{mode}: tasks={total}, max_local_steps={loop_limit}, "
            f"same_column_transition={ratio:.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test", action="store_true", help="扩大随机尺寸和穷举范围")
    parser.add_argument("--demo", action="store_true", help="输出代表性用例指标")
    parser.add_argument("--plot-dir", type=Path, help="输出 Dense/Causal 示例图")
    args = parser.parse_args()

    summaries = run_self_test(full=args.full_test)
    for summary in summaries:
        print(
            f"PASS {summary.mode}: cases={summary.cases}, coordinates={summary.coordinates}, "
            f"columns={summary.columns}, max_load_skew/m={summary.max_load_skew_ratio:.2f} <= 1.00"
        )
    print("PASS strict column ownership: every (batch, s2_idx) has exactly one owner core")
    print("PASS relaxed constraint: same-row tasks in one local_step are allowed and observed")

    if args.demo:
        _print_demo()
    if args.plot_dir:
        plot_schedule("dense", 8, 16, 12, 2, args.plot_dir / "dense_k8_m16_n12_b2.png")
        plot_schedule("causal", 8, 16, 16, 3, args.plot_dir / "causal_k8_m16_b3.png")
        print(f"plots written to: {args.plot_dir.resolve()}")


if __name__ == "__main__":
    main()
