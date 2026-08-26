"""
FlashAttentionScoreGrad SparseMode3 通用斜带区域确定性分核。

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
                save_path / f"FAG_sparse03_band_p{p}_q{q}_page_{page_idx + 1}.png",
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
    # k, m, n, b = 32, 32, 32, 8
    # p, q = 16, 4
    # k, m, n, b = 32, 32, 32, 8
    # p, q = 31, 31
    k, m, n, b = 32, 32, 32, 8
    p, q = 32, 1

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