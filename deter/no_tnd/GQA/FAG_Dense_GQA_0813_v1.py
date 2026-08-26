from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


MAX_ROUND_GROWTH_PERCENT = 3
MAX_INVALID_PERCENT = 4


@dataclass(frozen=True)
class DenseRoundDecision:
    used_core_num: int
    base_round: int
    selected_round: int
    base_period: int
    selected_period: int
    base_invalid: int
    selected_invalid: int
    adjusted: bool


@dataclass(frozen=True)
class ScheduleStats:
    max_round: int
    valid_blocks: int
    invalid_blocks: int
    duplicate_coordinates: int
    missing_coordinates: int
    dq_round_conflicts: int
    dkv_round_conflicts: int
    split_kv_groups: int
    average_distinct_batches: float
    average_distinct_heads: float


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def get_used_core_num(k: int, m: int, n: int, b: int, g: int) -> int:
    return min(k, b * g * m, b * n)


def get_base_dense_round(k: int, m: int, n: int, b: int, g: int) -> int:
    used_core_num = get_used_core_num(k, m, n, b, g)
    return max(ceil_div(b * n * g, used_core_num), ceil_div(n, m), g)


def select_dense_round(k: int, m: int, n: int, b: int, g: int) -> DenseRoundDecision:
    """Select a nearby R that improves GQA locality with bounded extra work.

    Here b has the same meaning as the kernel argument: B * N2. The script
    argument named N1 maps to the kernel's G = query_heads / kv_heads.
    """
    used_core_num = get_used_core_num(k, m, n, b, g)
    base_round = get_base_dense_round(used_core_num, m, n, b, g)
    total_id = b * n * g
    batch_head_num = b * g
    base_period = batch_head_num // gcd(batch_head_num, base_round)
    base_invalid = used_core_num * base_round - total_id

    selected_round = base_round
    if g > 1 and base_round % g != 0:
        candidate_round = ceil_div(base_round, g) * g
        candidate_period = batch_head_num // gcd(batch_head_num, candidate_round)
        candidate_invalid = used_core_num * candidate_round - total_id

        round_cost_ok = (
            (candidate_round - base_round) * 100
            <= base_round * MAX_ROUND_GROWTH_PERCENT
        )
        invalid_cost_ok = (
            candidate_invalid * 100
            <= used_core_num * candidate_round * MAX_INVALID_PERCENT
        )
        locality_better = candidate_period < base_period and candidate_period < used_core_num
        row_offset_enough = ceil_div(used_core_num, candidate_period) <= m
        if round_cost_ok and invalid_cost_ok and locality_better and row_offset_enough:
            selected_round = candidate_round

    selected_period = batch_head_num // gcd(batch_head_num, selected_round)
    selected_invalid = used_core_num * selected_round - total_id
    return DenseRoundDecision(
        used_core_num=used_core_num,
        base_round=base_round,
        selected_round=selected_round,
        base_period=base_period,
        selected_period=selected_period,
        base_invalid=base_invalid,
        selected_invalid=selected_invalid,
        adjusted=selected_round != base_round,
    )


def get_dense_batch_position(
    m: int,
    n: int,
    k: int,
    b: int,
    N1: int,
    core_id: int,
    round_id: int,
    dense_round: Optional[int] = None,
) -> Optional[Tuple[int, int, int]]:
    k = get_used_core_num(k, m, n, b, N1)
    R = dense_round if dense_round is not None else get_base_dense_round(k, m, n, b, N1)

    if core_id < 1 or core_id > k or round_id < 1 or round_id > R * m:
        return None

    ID = (core_id - 1) * R + ceil_div(round_id, m)
    local_id = round_id % m or m
    if ID > N1 * n * b:
        return None

    N = b * N1
    b_id = ceil_div(ID % N or N, N1)
    y = ceil_div(ID, N)
    w = ID % N1 or N1

    common_divisor = gcd(N, R)
    t1 = R // common_divisor
    t2 = N // common_divisor

    t1_new = t1 * m
    y1 = y % t1_new or t1_new
    offset = ceil_div(y1, t1)

    if t1_new < n:
        n1 = n % t1_new or t1_new
        if y <= n - n1:
            delta = ceil_div(y, t1_new)
            ID += delta
            if ID > (delta - 1) * t2 * m * R + offset * t2 * R:
                ID -= t2 * R

            b_id = ceil_div(ID % N or N, N1)
            w = ID % N1 or N1
            y = ceil_div(ID, N)

    x = local_id + offset - 1
    if x > m:
        x -= m

    return w + (b_id - 1) * N1, x, y


def analyze_schedule(m: int, n: int, k: int, b: int, g: int, dense_round: int) -> ScheduleStats:
    used_core_num = get_used_core_num(k, m, n, b, g)
    max_round = dense_round * m
    expected_block_num = b * g * m * n
    coordinates = set()
    valid_blocks = 0
    duplicate_coordinates = 0
    dq_round_conflicts = 0
    dkv_round_conflicts = 0

    for round_id in range(1, max_round + 1):
        dq_keys = set()
        dkv_keys = set()
        for core_id in range(1, used_core_num + 1):
            position = get_dense_batch_position(
                m, n, used_core_num, b, g, core_id, round_id, dense_round
            )
            if position is None:
                continue

            global_head, x, y = position
            valid_blocks += 1
            if position in coordinates:
                duplicate_coordinates += 1
            coordinates.add(position)

            batch_id = ceil_div(global_head, g)
            dq_key = (global_head, x)
            dkv_key = (batch_id, y)
            if dq_key in dq_keys:
                dq_round_conflicts += 1
            if dkv_key in dkv_keys:
                dkv_round_conflicts += 1
            dq_keys.add(dq_key)
            dkv_keys.add(dkv_key)

    split_kv_groups = sum(
        1
        for boundary in range(dense_round, b * n * g, dense_round)
        if boundary % g != 0
    )

    distinct_batch_sum = 0
    distinct_head_sum = 0
    for local_round in range(1, dense_round + 1):
        batches = set()
        heads = set()
        for core_id in range(1, used_core_num + 1):
            ID = (core_id - 1) * dense_round + local_round
            if ID > b * n * g:
                continue
            global_head = ID % (b * g) or b * g
            batches.add(ceil_div(global_head, g))
            heads.add(global_head)
        distinct_batch_sum += len(batches)
        distinct_head_sum += len(heads)

    total_slots = used_core_num * max_round
    return ScheduleStats(
        max_round=max_round,
        valid_blocks=valid_blocks,
        invalid_blocks=total_slots - valid_blocks,
        duplicate_coordinates=duplicate_coordinates,
        missing_coordinates=expected_block_num - len(coordinates),
        dq_round_conflicts=dq_round_conflicts,
        dkv_round_conflicts=dkv_round_conflicts,
        split_kv_groups=split_kv_groups,
        average_distinct_batches=distinct_batch_sum / dense_round,
        average_distinct_heads=distinct_head_sum / dense_round,
    )


def validate_and_report(m: int, n: int, k: int, b: int, g: int) -> DenseRoundDecision:
    decision = select_dense_round(k, m, n, b, g)
    base_stats = analyze_schedule(m, n, k, b, g, decision.base_round)
    selected_stats = analyze_schedule(m, n, k, b, g, decision.selected_round)

    print(
        f"input: k={k}, m={m}, n={n}, b={b}, G={g}; "
        f"used_core_num={decision.used_core_num}"
    )
    print(
        f"round: base={decision.base_round}, selected={decision.selected_round}, "
        f"adjusted={decision.adjusted}, period={decision.base_period}->{decision.selected_period}"
    )
    for name, stats in (("base", base_stats), ("selected", selected_stats)):
        print(
            f"{name}: max_round={stats.max_round}, valid={stats.valid_blocks}, "
            f"invalid={stats.invalid_blocks}, duplicate={stats.duplicate_coordinates}, "
            f"missing={stats.missing_coordinates}, dq_conflict={stats.dq_round_conflicts}, "
            f"dkv_conflict={stats.dkv_round_conflicts}, split_kv={stats.split_kv_groups}, "
            f"avg_batches={stats.average_distinct_batches:.3f}, "
            f"avg_heads={stats.average_distinct_heads:.3f}"
        )

    assert selected_stats.valid_blocks == b * g * m * n
    assert selected_stats.duplicate_coordinates == 0
    assert selected_stats.missing_coordinates == 0
    assert selected_stats.dq_round_conflicts == 0
    assert selected_stats.dkv_round_conflicts == 0
    return decision


def visualize_dense_schedule(m: int, n: int, k: int, b: int, N1: int, dense_round: int) -> None:
    used_core_num = get_used_core_num(k, m, n, b, N1)
    max_round = dense_round * m
    print(f"visualize: k={used_core_num}, R={dense_round}, max_round={max_round}")

    all_colors = (
        list(plt.get_cmap("tab20", 20).colors)
        + list(plt.get_cmap("tab20b", 20).colors)
        + list(plt.get_cmap("tab20c", 20).colors)
    )
    np.random.seed(42)
    np.random.shuffle(all_colors)
    cmap = ListedColormap(all_colors[:32], name="shuffled32")

    for global_head in range(1, b * N1 + 1):
        rounds_mat = np.zeros((m, n), dtype=int)
        core_mat = np.zeros((m, n), dtype=int)

        for core_id in range(1, used_core_num + 1):
            for round_id in range(1, max_round + 1):
                position = get_dense_batch_position(
                    m, n, used_core_num, b, N1, core_id, round_id, dense_round
                )
                if position is None:
                    continue
                position_head, x, y = position
                if position_head == global_head:
                    rounds_mat[x - 1, y - 1] = round_id
                    core_mat[x - 1, y - 1] = core_id

        cell_px = 40
        dpi = 200
        width_in = max(4.0, n * cell_px / dpi)
        height_in = max(4.0, m * cell_px / dpi)
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        ax.imshow(
            core_mat,
            origin="upper",
            cmap=cmap,
            vmin=1,
            vmax=used_core_num,
            aspect="equal",
            interpolation="none",
        )
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.35)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        max_round_digits = len(str(max_round))
        cell_size_pt = cell_px * 72.0 / dpi
        text_size = max(2.5, min(8.0, cell_size_pt * 0.72 / (max_round_digits * 0.60)))
        for i in range(m):
            for j in range(n):
                round_id = rounds_mat[i, j]
                if round_id > 0:
                    ax.text(
                        j,
                        i,
                        str(round_id),
                        ha="center",
                        va="center",
                        fontsize=text_size,
                        color="white",
                        fontfamily="DejaVu Sans Mono",
                        clip_on=True,
                    )

        ax.set_title(
            f"Dense GQA | k={used_core_num}, R={dense_round}, head={global_head}, size=({m}x{n})"
        )
        ax.set_xlabel("Column (y)")
        ax.set_ylabel("Row (x)")
        plt.tight_layout()
        output_dir = Path(__file__).with_name("figures_v1")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"Dense_GQA_v1_k{used_core_num}_R{dense_round}_m{m}_n{n}_"
            f"b{b}_G{N1}_head{global_head}.png"
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Image saved to: {output_path}")


def parse_args():
    parser = ArgumentParser(description="Validate and visualize the adjusted Dense GQA schedule")
    parser.add_argument("--k", type=int, default=28)
    parser.add_argument("--m", type=int, default=58)
    parser.add_argument("--n", type=int, default=38)
    parser.add_argument("--b", type=int, default=10)
    parser.add_argument("--g", type=int, default=3)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    round_decision = validate_and_report(args.m, args.n, args.k, args.b, args.g)
    if args.visualize:
        visualize_dense_schedule(
            args.m,
            args.n,
            args.k,
            args.b,
            args.g,
            round_decision.selected_round,
        )
