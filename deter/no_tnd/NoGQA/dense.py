from typing import List, Tuple, Optional
from math import ceil,floor
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def preprocess_dense_schedule(list_m: List[int], list_n: List[int], N1: int, k: int) -> Tuple[int, List[int]]:
    """
    预处理步骤：计算每个 batch 的前缀和 prefix
    需要保证 min(list_m) * N1 >= k
    Returns:
        prefix: 累积数组
        注意最大轮次数(从0开始)R是prefix[-1]
    """
    prefix = [0]
    for m, n in zip(list_m, list_n):
        prefix.append(prefix[-1] + ceil(n*N1/k)*m)
    return prefix


def get_dense_batch_position(
        list_m: List[int], list_n: List[int], N1: int,
        prefix: List[int], core_id: int, round_id: int, k: int
) -> Optional[Tuple[int, int, int]]:
    """
    计算从 (j, r) 得到的全局索引并返回对应的 batch ID 和坐标 (x, y)，传入mode_chosen为1
    """
    if core_id < 0 or round_id < 0:
        return None

    # 查找符合条件的 batch ID
    b = len(list_m)
    w = 0
    while w < b and round_id >= prefix[w + 1]:
        w += 1
    if w >= b:
        return None

    m, n = list_m[w], list_n[w]
    num = m * n
    batch_id = w
    delta = round_id - prefix[w]

    y1 = delta // m * k + core_id

    if y1 >= n * N1:
        return None

    N1_id = y1 % N1
    y = y1 // N1
    x = (y + delta) % m
    if x >= m:
        x -= m

    return (batch_id * N1 + N1_id, x, y)


"""
    若所有batch都满足 m>=n，可以选择这个，直观上轮次更紧密，传入大于1的mode_chosen
"""


def get_dense_batch_position_v2(
        list_m: List[int], list_n: List[int], N1: int,
        prefix: List[int], core_id: int, round_id: int, k: int
) -> Optional[Tuple[int, int, int]]:
    """
    计算从 (j, r) 得到的全局索引并返回对应的 batch ID 和坐标 (x, y)
    """
    if core_id < 0 or round_id < 0:
        return None

    # 查找符合条件的 batch ID
    b = len(list_m)
    w = 0
    while w < b and round_id >= prefix[w + 1]:
        w += 1
    if w >= b:
        return None

    m, n = list_m[w], list_n[w]
    num = m * n
    batch_id = w
    delta = round_id - prefix[w]

    y1 = delta // m * k + core_id

    if y1 >= n * N1:
        return None

    N1_id = y1 // n
    y = y1 % n
    x = (y + delta) % m
    if x >= m:
        x -= m

    return (batch_id * N1 + N1_id, x, y)

def visualize_dense_schedule(
    k: int, list_m: List[int], list_n: List[int],
    N1: int, prefix: List[int], mode_chosen: int
):
    """
    可视化 dense 调度坐标结果（使用 get_dense_batch_position）

    Parameters:
        k: AI core 数
        list_m, list_n: 各 batch 的行数和列数(s1,s2)
        N1: 每个 batch 重复次数
        prefix: 前缀和（来自预处理）
    """
    R = prefix[-1]
    b = len(list_m)

    total_batches = b * N1
    cmap1 = plt.get_cmap('tab20', 20)
    cmap2 = plt.get_cmap('tab20b', 20)
    cmap3 = plt.get_cmap('tab20c', 20)

    # 合并颜色列表（共 60 种）
    all_colors = list(cmap1.colors) + list(cmap2.colors) + list(cmap3.colors)

    # === 简单随机打乱，避免相邻相似 ===
    np.random.seed(42)  # 固定随机种子以复现结果（可删）
    np.random.shuffle(all_colors)

    # 选取前 32 种颜色
    selected_colors = all_colors[:32]
    # === 构建 cmap ===
    cmap = ListedColormap(selected_colors, name='shuffled32')

    for w in range(0, total_batches):
        batch0 = w // N1  # 原始 batch 索引
        m, n = list_m[batch0], list_n[batch0]
        rounds_mat = np.zeros((m, n), dtype=int)
        core_mat = np.zeros((m, n), dtype=int)

        for core_id in range(0, k):
            for round_id in range(0, R):
                # Ensure core_id and round_id are single integers, not lists
                if mode_chosen == 1:
                    pos = get_dense_batch_position(list_m, list_n, N1, prefix, core_id, round_id,k)
                else:
                    pos = get_dense_batch_position_v2(list_m, list_n, N1, prefix, core_id, round_id,k)
                if pos is None:
                    continue
                batch_id, x, y = pos

                # Directly use x and y since indexing starts from 1
                if batch_id == w and 0 <= x < m and 0 <= y < n:
                    rounds_mat[x, y] = round_id  # Direct assignment
                    core_mat[x, y] = core_id  # Direct assignment

        # 绘图
        fig, ax = plt.subplots(figsize=(3, 3 * m / n))

        im = ax.imshow(core_mat, origin='upper', cmap=cmap, vmin=0, vmax=k-1, aspect='auto')

        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', length=0)

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(m))
        ax.set_xticklabels(np.arange(0, n))
        ax.set_yticklabels(np.arange(0, m))

        for i in range(m):
            for j in range(n):
                r_id = rounds_mat[i, j]
                if r_id >= 0:
                    ax.text(j, i, str(r_id), ha='center', va='center', fontsize=10, color='white')

        ax.set_title(f"Dense Schedule | k={k}, batch={w}, size=({m}×{n})")
        ax.set_xlabel("Column (y)")
        ax.set_ylabel("Row (x)")
        plt.tight_layout()
        plt.show()

# Ensure you pass correct arguments to this function when calling it
# list_m = [11, 5, 6,1,4,4]
# list_n = [11, 4, 6,1,4,3]
# k = 10
# N1 = 6

# list_m = [60, 5]
# list_n = [60, 6]
# k = 32
# N1 = 8

list_m = [63, 2]
list_n = [64, 2]
k = 32
N1 = 16

prefix = preprocess_dense_schedule(list_m, list_n,N1,k)

print(prefix)

visualize_dense_schedule(k, list_m, list_n, N1, prefix, 1)