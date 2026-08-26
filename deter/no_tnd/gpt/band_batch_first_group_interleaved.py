from __future__ import annotations

"""
SparseMode4 Batch-First 分列组解析 + 轮次交错参考实现。

目标：
1. 保持原 Kernel 的按列 owner、列组 snake、group/batch rotation；
2. 保持每个列组的解析边着色 local_round = (x - y) mod delta_g；
3. 不保存 k * rounds_per_batch 的全局 matching 表；
4. 将列组从“整组串行”改为“按 local_round 波次交错”，消除组边界造成的轮次大跳变；
5. Python 与 Kernel 只要使用同一交错状态机，即可逐项一致。

注意：这不是复现某个 perfect-matching 库的任意 tie-break 顺序，而是定义一个新的、
可解析、可在 Kernel 中 O(1) 状态推进的 canonical schedule。
"""

from dataclasses import dataclass
from math import ceil, gcd
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import argparse

import numpy as np
import matplotlib.pyplot as plt
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
    max_group_delta: int
    rounds_per_batch: int
    total_rounds: int
    rotation_stride: int
    interleaved_round_meta: Tuple[RoundMeta, ...]


def _validate_parameters(m: int, n: int, b: int, p: int, q: int, k: int) -> None:
    values = {"m": m, "n": n, "b": b, "p": p, "q": q, "k": k}
    for name, value in values.items():
        if not isinstance(value, int):
            raise TypeError(f"{name} 必须是 int，当前为 {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0，当前为 {value}")
    if p > m:
        raise ValueError(f"p 必须满足 1 <= p <= m，当前 p={p}, m={m}")
    if q > n:
        raise ValueError(f"q 必须满足 1 <= q <= n，当前 q={q}, n={n}")


def is_valid_block(x: int, y: int, m: int, n: int, p: int, q: int) -> bool:
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


def _column_valid_x_range(y: int, m: int, p: int, q: int) -> Tuple[int, int]:
    x_lo = max(1, y + 1 - q)
    x_hi = min(m, y + p - 1)
    return x_lo, x_hi


def _column_degree(y: int, m: int, p: int, q: int) -> int:
    x_lo, x_hi = _column_valid_x_range(y, m, p, q)
    return max(0, x_hi - x_lo + 1)


def _compute_group_delta(
        y_start: int,
        y_end: int,
        m: int,
        p: int,
        q: int,
) -> int:
    """
    列组二分多重图的最大度：
      delta_g = max(最大行度, 最大 owner-core 度)

    同一列组中每个 core 最多拥有一列，因此 owner-core 度就是该列有效高度。
    组内某一行的有效列构成连续区间；其最大重叠度为 min(group_size, p+q-1)。
    """
    group_size = y_end - y_start + 1
    max_row_degree = min(group_size, p + q - 1)
    max_core_degree = max(
        _column_degree(y, m, p, q)
        for y in range(y_start, y_end + 1)
    )
    return max(max_row_degree, max_core_degree)


def _build_interleaved_round_meta(groups: Sequence[GroupInfo]) -> Tuple[RoundMeta, ...]:
    """
    local-round-major 交错：

      wave 0: group 0, group 1, ..., group G-1（仅 delta_g > 0）
      wave 1: group 0, group 1, ..., group G-1（仅 delta_g > 1）
      ...

    总条目数严格等于 sum(delta_g)，不增加轮数。
    """
    if not groups:
        return tuple()
    max_delta = max(group.delta for group in groups)
    result: List[RoundMeta] = []
    for local_round0 in range(max_delta):
        for group in groups:
            if local_round0 < group.delta:
                result.append((group.group_id0, local_round0))
    return tuple(result)


def build_schedule_info(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        k: int,
) -> ScheduleInfo:
    _validate_parameters(m, n, b, p, q, k)

    effective_n = min(n, m + q - 1)
    group_count = ceil(effective_n / k)

    groups: List[GroupInfo] = []
    for group_id0 in range(group_count):
        y_start = group_id0 * k + 1
        y_end = min(effective_n, y_start + k - 1)
        delta = _compute_group_delta(y_start, y_end, m, p, q)
        groups.append(GroupInfo(
            group_id0=group_id0,
            y_start=y_start,
            y_end=y_end,
            group_size=y_end - y_start + 1,
            delta=delta,
        ))

    round_meta = _build_interleaved_round_meta(groups)
    rounds_per_batch = sum(group.delta for group in groups)
    if len(round_meta) != rounds_per_batch:
        raise AssertionError("交错轮次条目数与 sum(delta_g) 不一致")

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
        max_group_delta=max(group.delta for group in groups),
        rounds_per_batch=rounds_per_batch,
        total_rounds=b * rounds_per_batch,
        rotation_stride=_find_coprime_rotation_stride(k),
        interleaved_round_meta=round_meta,
    )


def _group_rotation(info: ScheduleInfo, batch0: int, group_id0: int) -> int:
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
    """返回物理列 (batch_id, y) 的固定 1-based owner core。"""
    if not 1 <= batch_id <= info.b:
        raise ValueError("batch_id 越界")
    if not 1 <= y <= info.effective_n:
        raise ValueError("y 越界或该列无有效块")

    batch0 = batch_id - 1
    group_id0 = (y - 1) // info.k
    local_slot = (y - 1) % info.k
    rotation = _group_rotation(info, batch0, group_id0)

    if group_id0 % 2 == 0:
        permuted_core = local_slot
    else:
        permuted_core = info.k - 1 - local_slot

    return (permuted_core + rotation) % info.k + 1


def _find_d_for_local_round(
        y: int,
        local_round0: int,
        delta: int,
        m: int,
        p: int,
        q: int,
) -> Optional[int]:
    """
    在该列有效连续区间 [d_lo, d_hi] 中寻找唯一满足 d mod delta=local_round 的 d。
    因列有效高度 <= delta，解最多一个。
    """
    d_lo = max(1 - q, 1 - y)
    d_hi = min(p - 1, m - y)
    if d_lo > d_hi:
        return None

    d = d_lo + ((local_round0 - d_lo) % delta)
    if d > d_hi:
        return None
    return d


def get_interleaved_position_from_info(
        info: ScheduleInfo,
        core_id: int,
        round_id: int,
) -> Optional[Position]:
    """由 1-based (core_id, round_id) 直接返回 (batch_id, x, y)。"""
    if not 1 <= core_id <= info.k:
        raise ValueError(f"core_id 必须满足 1 <= core_id <= {info.k}")
    if round_id < 1:
        raise ValueError("round_id 必须 >= 1")
    if round_id > info.total_rounds:
        return None

    round0 = round_id - 1
    batch0 = round0 // info.rounds_per_batch
    round_in_batch0 = round0 % info.rounds_per_batch

    group_id0, local_round0 = info.interleaved_round_meta[round_in_batch0]
    group = info.groups[group_id0]

    local_slot = _core_to_group_local_slot(info, batch0, group_id0, core_id - 1)
    if local_slot >= group.group_size:
        return None

    y = group.y_start + local_slot
    d = _find_d_for_local_round(
        y=y,
        local_round0=local_round0,
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


def get_band_group_interleaved_position(
        m: int,
        n: int,
        b: int,
        p: int,
        q: int,
        core_id: int,
        round_id: int,
        k: int,
) -> Optional[Position]:
    info = build_schedule_info(m, n, b, p, q, k)
    return get_interleaved_position_from_info(info, core_id, round_id)


def build_schedule_matrices(info: ScheduleInfo):
    rounds_cube = np.full((info.b, info.m, info.n), -1, dtype=np.int64)
    core_cube = np.full((info.b, info.m, info.n), -1, dtype=np.int64)
    active_cores_per_round = np.zeros(info.total_rounds, dtype=np.int64)

    for round_id in range(1, info.total_rounds + 1):
        seen_rows = set()
        for core_id in range(1, info.k + 1):
            pos = get_interleaved_position_from_info(info, core_id, round_id)
            if pos is None:
                continue
            batch_id, x, y = pos
            key = (batch_id, x)
            if key in seen_rows:
                raise AssertionError(
                    f"同行冲突: round={round_id}, batch={batch_id}, x={x}"
                )
            seen_rows.add(key)

            idx = (batch_id - 1, x - 1, y - 1)
            if rounds_cube[idx] != -1:
                raise AssertionError(f"任务重复分配: {(batch_id, x, y)}")
            rounds_cube[idx] = round_id
            core_cube[idx] = core_id
            active_cores_per_round[round_id - 1] += 1

    return rounds_cube, core_cube, active_cores_per_round


def validate_schedule(info: ScheduleInfo, rounds_cube: np.ndarray, core_cube: np.ndarray) -> dict:
    expected = 0
    for batch_id in range(1, info.b + 1):
        for x in range(1, info.m + 1):
            row_rounds: List[int] = []
            for y in range(1, info.n + 1):
                valid = is_valid_block(x, y, info.m, info.n, info.p, info.q)
                rid = int(rounds_cube[batch_id - 1, x - 1, y - 1])
                cid = int(core_cube[batch_id - 1, x - 1, y - 1])
                if valid:
                    expected += 1
                    if rid < 1 or not 1 <= cid <= info.k:
                        raise AssertionError(f"有效块未覆盖: {(batch_id, x, y)}")
                    row_rounds.append(rid)
                    owner = get_column_owner_core(info, batch_id, y)
                    if cid != owner:
                        raise AssertionError(
                            f"列 owner 变化: {(batch_id, x, y)}, got={cid}, expected={owner}"
                        )
                else:
                    if rid != -1 or cid != -1:
                        raise AssertionError(f"无效块被分配: {(batch_id, x, y)}")

            if len(row_rounds) != len(set(row_rounds)):
                raise AssertionError(f"同行 round 重复: batch={batch_id}, x={x}")

    assigned = int(np.count_nonzero(rounds_cube >= 1))
    if assigned != expected:
        raise AssertionError(f"覆盖数不一致: assigned={assigned}, expected={expected}")

    # 每个物理列只允许一个 owner core。
    for batch0 in range(info.b):
        for y0 in range(info.effective_n):
            values = core_cube[batch0, :, y0]
            values = values[values >= 1]
            if len(values) and len(np.unique(values)) != 1:
                raise AssertionError(f"同一列跨核: batch={batch0+1}, y={y0+1}")

    return {
        "effective_n": info.effective_n,
        "group_count": info.group_count,
        "group_deltas": [g.delta for g in info.groups],
        "rounds_per_batch": info.rounds_per_batch,
        "total_rounds": info.total_rounds,
        "valid_blocks": assigned,
    }


def print_core_trace(info: ScheduleInfo, core_id: int, batch_id: Optional[int] = None) -> None:
    if not 1 <= core_id <= info.k:
        raise ValueError("core_id 越界")
    print(f"\ncore {core_id} trace")
    for round_id in range(1, info.total_rounds + 1):
        pos = get_interleaved_position_from_info(info, core_id, round_id)
        current_batch = (round_id - 1) // info.rounds_per_batch + 1
        if batch_id is not None and current_batch != batch_id:
            continue
        if pos is None:
            print(f"r{round_id:4d} -> idle")
        else:
            b_id, x, y = pos
            print(f"r{round_id:4d} -> (b={b_id}, x={x}, y={y})")


def summarize_rounds(info: ScheduleInfo, active: np.ndarray) -> dict:
    nonzero = active[active > 0]
    return {
        "full_rounds": int(np.count_nonzero(active == info.k)),
        "partial_rounds": int(np.count_nonzero((active > 0) & (active < info.k))),
        "empty_rounds": int(np.count_nonzero(active == 0)),
        "avg_active_compute_round": float(nonzero.mean()) if len(nonzero) else 0.0,
        "slot_utilization": float(active.sum() / (len(active) * info.k)) if len(active) else 0.0,
    }


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
    """保持原脚本的分核图片风格。"""
    palette = ['#F3F4F6'] + [
        '#0B84A5', '#EBC262', '#6F4E7C', '#9DD866', '#CA472F',
        '#FFA056', '#8DDDD0', '#BFB5FF', '#3C5488', '#F39C12',
        '#27AE60', '#D35400', '#16A085', '#7F8C8D', '#2E86C1',
        '#E74C3C', '#8E44AD', '#2ECC71', '#34495E', '#F1C40F'
    ]
    if info.k + 1 > len(palette):
        extra = plt.get_cmap('tab20', info.k + 1 - len(palette)).colors
        palette.extend(extra)
    cmap = ListedColormap(palette[:info.k + 1], name='dense_clean')

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
        page_batches = batches[page_idx * page_size:(page_idx + 1) * page_size]
        cols = min(max(1, ncols), len(page_batches))
        rows = ceil(len(page_batches) / cols)

        fig_w = cols * axes_w + 1.2
        fig_h = rows * axes_h + 1.4
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

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
            ax.set_xticks(np.arange(info.n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(info.m + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='#D1D5DB', linestyle='-', linewidth=0.45)
            ax.tick_params(which='minor', length=0)
            ax.set_xticks(np.arange(info.n))
            ax.set_yticks(np.arange(info.m))
            ax.set_xticklabels(np.arange(1, info.n + 1), fontsize=label_font)
            ax.set_yticklabels(np.arange(1, info.m + 1), fontsize=label_font)
            ax.tick_params(axis='x', pad=2)
            ax.tick_params(axis='y', pad=2)

            if annotate_round:
                for i in range(info.m):
                    for j in range(info.n):
                        rid = round_mat[i, j]
                        if rid >= 1:
                            ax.text(
                                j, i, str(rid),
                                ha='center', va='center',
                                fontsize=annot_font,
                                fontweight='bold', color='black',
                                bbox=dict(
                                    boxstyle='round,pad=0.10',
                                    facecolor='white', alpha=0.72,
                                    edgecolor='none',
                                ),
                            )
            ax.set_title(f'batch={batch_id}', fontsize=10, pad=8)

        for idx in range(len(page_batches), rows * cols):
            rr = idx // cols
            cc = idx % cols
            axes[rr][cc].axis('off')

        fig.suptitle(
            f'Group-Interleaved Schedule | k={info.k}, m={info.m}, n={info.n}, '
            f'b={info.b}, p={info.p}, q={info.q} | page {page_idx + 1}/{total_pages}',
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))

        if save_path:
            out = save_path / f'FAG_sparse04_group_interleaved_page_{page_idx + 1}.png'
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            saved.append(out)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseMode4 分列组解析交错轮次参考实现")
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--b', type=int, default=8)
    parser.add_argument('--p', type=int, default=32)
    parser.add_argument('--q', type=int, default=63)
    parser.add_argument('--core', type=int, default=None)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='outputs_group_interleaved')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()

"""
保留现有 Kernel 的：

固定列 owner；
每 k 列一个列组；
snake、group rotation、batch rotation；
组内解析公式 (x-y) mod delta_g；
不保存大 matching 表。

只把列组执行顺序从：
group 0：local 0,1,2,...,31
group 1：local 0,1,2,...,31

改为：
group 0 local 0
group 1 local 0
group 0 local 1
group 1 local 1
...

对于：

k=32, m=32, n=64, b=8, p=32, q=63

新的 batch 内轮次为：

r1  -> group 0, local 0
r2  -> group 1, local 0
r3  -> group 0, local 1
r4  -> group 1, local 1
...
r63 -> group 0, local 31
r64 -> group 1, local 31

这样：

每 batch 仍是 64 轮；
总任务数和槽位利用率不变；
不再在第 32/33 轮集中切换列组；
Python 和 Kernel 可以使用完全相同的交错规范；
新增 Kernel 状态仅约 16 字节；
不需要 k × rounds_per_batch matching 表。
"""
def main() -> None:
    args = _parse_args()
    info = build_schedule_info(args.m, args.n, args.b, args.p, args.q, args.k)
    rounds_cube, core_cube, active = build_schedule_matrices(info)
    validation = validate_schedule(info, rounds_cube, core_cube)
    stats = summarize_rounds(info, active)

    print('列组轮数:', validation['group_deltas'])
    print('每 batch 轮数:', info.rounds_per_batch)
    print('总轮数:', info.total_rounds)
    print('有效块:', validation['valid_blocks'])
    print('满核轮:', stats['full_rounds'])
    print('部分核轮:', stats['partial_rounds'])
    print('空轮:', stats['empty_rounds'])
    print(f"槽位利用率: {stats['slot_utilization'] * 100:.6f}%")
    print('batch 内前 16 个 (group, local_round):', info.interleaved_round_meta[:16])

    if args.core is not None:
        print_core_trace(info, args.core, args.batch)

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
            print('saved:', path)


if __name__ == '__main__':
    main()