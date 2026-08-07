"""
FlashAttentionScoreGrad SparseMode3 通用斜带区域确定性分核：Batch-First 方案。

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
                save_path / f"FAG_sparse03_batch_first_p{p}_q{q}_page_{page_idx + 1}.png",
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
    p, q = 31, 31

    # k, m, n, b = 32, 32, 32, 8
    # p, q = 32, 1

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