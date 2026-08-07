from math import ceil, gcd
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def get_dense_batch_position(
        m: int, n: int, b: int,
        core_id: int, round_id: int, k: int
) -> Optional[Tuple[int, int, int]]:
    """
    计算从 (j, r) 得到的全局索引并返回对应的 batch ID 和坐标 (x, y)
    """
    k = min(k, m * b)
    j = core_id - 1
    r = round_id - 1
    if j > k:
        return None

    y1 = r // m * k + j
    if y1 >= n * b:
        return None

    b_id = y1 // n
    y = y1 % n
    x = (y + r) % m
    if x >= m:
        x -= m

    b_id = b_id + 1
    x = x + 1
    y = y + 1

    return (b_id, x, y)


# 对于sparsemode3 如果初始s1 > s2 那么真实的赋值是 s1 = s2，也就是总可以保证 s1 <= s2
# 如果 k <= s1，适用；如果 k > s1，对于真实 s1 = s2 也适用；如果 k > s1 且 s2 > s1，目前不适用
# 处理 b 是偶数，优先拼接思路而不是近似思路

# 1
def get_causal_batch_position(
        m: int, n: int, b: int, core_id: int, round_id: int, k: int
) -> Optional[Tuple[int, int, int]]:
    if m == n:
        n_new = n + 1
    else:
        n_new = (n - m + 2) + (n + 1)
    b_new = b // 2
    res = get_dense_batch_position(m, n_new, b_new, core_id, round_id, k)
    if res is None:
        return None
    global_id, x, y = res

    if m == n:
        if y >= x + 1:
            y = 2 * n - m - y + 2
            x = m + 1 - x
            b_id = 2 * global_id
        else:
            b_id = 2 * global_id - 1
    else:
        n += 1
        # 改动

        if y >= x + n - m + 1:
            #         if y<n :
            #             y = 2*n - m - y
            #             x = x + y - n + m
            #         else:
            #             y = 2*n - m - y
            y = 2 * n - m - y + 2
            x = m + 1 - x
            b_id = 2 * global_id
        else:
            b_id = 2 * global_id - 1

        # 不确定是否需要加
        if (x, y) == (m, n):
            return None

    return (b_id, x, y) # +1


def _build_causal_matrices(k: int, m: int, n: int, b: int):
    """一次性构建所有 batch 的 round/core 矩阵。"""

    # 改种case下的最大轮次
    if m == n:
        m_new, n_new = m, n + 1
    elif m < n:
        m_new, n_new = m, 2 * n - m + 3
    else:
        m_new = n + 1
        n_new = n + 2
    R = max(m_new * ceil(n_new * (b // 2) / k), n_new)

    #     rounds_cube = np.zeros((b, m, n), dtype=int)
    rounds_cube = np.full((b, m, n), -1, dtype=int)
    core_cube = np.full((b, m, n), -1, dtype=int)

    if m > n:
        m_cal = n_cal = n + 1
    else:
        m_cal = m
        n_cal = n


    for core_id in range(1, k + 1):
        for round_id in range(1, R + 1):
            pos = get_causal_batch_position(m_cal, n_cal, b, core_id, round_id, k)
            if pos is None:
                continue
            batch_id, x, y = pos
            if m>n:
                x += (m - n - 1)
            if 1 <= batch_id <= b and 1 <= x <= m and 1 <= y <= n:
                rounds_cube[batch_id - 1, x - 1, y - 1] = round_id
                core_cube[batch_id - 1, x - 1, y - 1] = core_id

    return k, R, rounds_cube, core_cube


def visualize_causal_schedule(
        k: int,
        m: int,
        n: int,
        b: int,
        batches_per_figure: int = 2,
        ncols: int = 2,
        annotate_round: bool = True,
        save_dir: Optional[str] = None,
        dpi: int = 220,
        show: bool = True,
):
    """
    分页网格可视化 dense 调度。
    - 避免 b 很大时输出过长
    - 优化基本块网格和文本观感
    """

    k, R, rounds_cube, core_cube = _build_causal_matrices(k, m, n, b)
    print('所需核数:', k)
    print('确定性计算总轮次:', R)

    palette = ['#F3F4F6'] + [
        '#0B84A5', "#EBC262", '#6F4E7C', '#9DD866', '#CA472F',
        '#FFA056', '#8DDDD0', '#BFB5FF', '#3C5488', '#F39C12',
        '#27AE60', '#D35400', '#16A085', '#7F8C8D', '#2E86C1',
        '#E74C3C', '#8E44AD', '#2ECC71', '#34495E', '#F1C40F'
    ]
    if k + 1 > len(palette):
        extra = plt.get_cmap('tab20', k + 1 - len(palette)).colors
        palette.extend(extra)
    cmap = ListedColormap(palette[:k + 1], name='dense_clean')

    batches = list(range(1, b + 1))
    page_size = max(1, batches_per_figure)
    total_pages = ceil(len(batches) / page_size)
    save_path = Path(save_dir) if save_dir else None
    if save_path:
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
            r = local_idx // cols
            c = local_idx % cols
            ax = axes[r][c]

            round_mat = rounds_cube[batch_id - 1]
            core_mat = core_cube[batch_id - 1]

            ax.imshow(
                core_mat,
                origin='upper',
                cmap=cmap,
                vmin=0,
                vmax=max(k, 1),
                aspect='equal',
                interpolation='nearest',
            )

            ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
            ax.grid(which='minor', color='#D1D5DB', linestyle='-', linewidth=0.45)
            ax.tick_params(which='minor', length=0)

            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(m))
            ax.set_xticklabels(np.arange(1, n + 1), fontsize=label_font)
            ax.set_yticklabels(np.arange(1, m + 1), fontsize=label_font)
            ax.tick_params(axis='x', pad=2)
            ax.tick_params(axis='y', pad=2)

            if annotate_round:
                for i in range(m):
                    for j in range(n):
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
                                bbox=dict(boxstyle='round,pad=0.10', facecolor='white', alpha=0.72, edgecolor='none')
                            )

            ax.set_title(f'batch={batch_id}', fontsize=10, pad=8)

        # 关闭空白子图
        for idx in range(len(page_batches), rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis('off')

        fig.suptitle(
            f'Dense Schedule | k={k}, m={m}, n={n}, b={b} | page {page_idx + 1}/{total_pages}',
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        if save_path:
            fig.savefig(
                save_path / f'FAG_sparse03_deter_0309_v2_page_{page_idx + 1}.png',
                dpi=dpi,
                bbox_inches='tight',
            )
        if show:
            plt.show()
        else:
            plt.close(fig)

### 需求背景 ###
'''
该脚本实现FlashAttentionScoreGrad算子确定性计算，非TND，下三角的场景的分核，本质是将b个[m, n]的任务块（下三角场景在对角线以下有效块）分到k个核上处理，需要满足以下条件：
1. 按列的粒度分核
2. 确定性计算要求：每一行不存在相同的河内编号轮次
3. 负载尽量均衡
'''

# k, m, n, b = 32, 64, 32, 16
k, m, n, b = 32, 32, 32, 8
visualize_causal_schedule(
    k, m, n, b,
    batches_per_figure=2,
    ncols=2,
    save_dir=Path(__file__).parent / 'outputs',
    dpi=260,
    show=False
)
