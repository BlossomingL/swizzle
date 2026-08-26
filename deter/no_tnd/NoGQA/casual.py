"""FAG arch35 非 TND DETER_CAUSAL（sparseMode=2）非等长 Swizzle 镜像与验证。

该脚本与算子侧以下实现保持一致：

* Host ``SelectBlockSchedule`` 的 LEFT_UP_CAUSAL 入口；
* Kernel ``CalLeftUpCausalSwizzleMaxRound`` 的轮次公式；
* Kernel ``CalLeftUpCausalSwizzleIndex`` 的逐核逐轮坐标公式。

坐标采用与 Kernel helper 相同的 1-based ``(batch, s1, s2)`` 表示。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


Coordinate = Tuple[int, int, int]
SPARSE_MODE_LEFT_UP_CAUSAL = 2


@dataclass(frozen=True)
class ScheduleInfo:
    k: int
    m: int
    n: int
    b: int
    pair_count: int
    schedule_mode: str
    virtual_m: int
    virtual_n: int
    active_k: int
    max_round: int


@dataclass(frozen=True)
class VerifyResult:
    schedule: ScheduleInfo
    valid_blocks: int
    issued_coordinates: int
    idle_slots: int


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError(f"ceil_div requires value>=0 and divisor>0, got {value=}, {divisor=}")
    return (value + divisor - 1) // divisor


def host_enables_left_up_causal_swizzle(
    *,
    enable_swizzle: bool,
    is_non_tnd: bool,
    is_bn2gs1s2: bool,
    cube_base_equal: bool,
    sparse_type_supported: bool,
    is_deterministic: bool,
    deter_causal: bool,
    sparse_mode: int,
    batch_n2: int,
    g: int,
    s1: int,
    aic_num: int,
) -> bool:
    """镜像 Host 对 sparseMode=2 DETER_CAUSAL 的 Swizzle 外围条件。"""
    can_split = (
        enable_swizzle
        and is_non_tnd
        and is_bn2gs1s2
        and cube_base_equal
        and sparse_type_supported
    )
    left_up_causal = deter_causal and sparse_mode == SPARSE_MODE_LEFT_UP_CAUSAL
    return (
        can_split
        and is_deterministic
        and batch_n2 > 0
        and batch_n2 % 2 == 0
        and g == 1
        and s1 >= aic_num * 128
        and left_up_causal
    )


def cal_dense_swizzle_index(k: int, m: int, n: int, b: int, j: int, r: int) -> Optional[Coordinate]:
    """镜像 Kernel ``CalDenseSwizzleIndex``；调用方负责限制有效核数。"""
    j_zero = j - 1
    r_zero = r - 1
    actual_k = min(k, b * m)
    if j_zero < 0 or j_zero >= actual_k or r_zero < 0:
        return None

    position = (r_zero // m) * actual_k + j_zero
    batch = position // n
    s2 = position % n
    s1 = (s2 + r_zero) % m
    if 0 <= batch < b and 0 <= s1 < m and 0 <= s2 < n:
        return batch + 1, s1 + 1, s2 + 1
    return None


def cal_dense_index(k: int, m: int, n: int, b: int, j: int, r: int) -> Optional[Coordinate]:
    """镜像 Kernel ``CalDenseIndex``，用于 S1>S2 的下方全有效矩形区。"""
    actual_k = min(k, b * m)
    if not (1 <= j <= actual_k) or r < 1:
        return None

    position = ((r - 1) // m) * actual_k + j
    batch = position % b or b
    s2 = ceil_div(position, b)
    s1_base = s2 % m or m
    round_in_column = r % m or m
    s1 = s1_base + round_in_column - 1
    if s1 > m:
        s1 -= m
    if 1 <= batch <= b and 1 <= s1 <= m and 1 <= s2 <= n:
        return batch, s1, s2
    return None


def cal_causal_swizzle_index(k: int, m: int, n: int, b: int, j: int, r: int) -> Optional[Coordinate]:
    """镜像 Kernel ``CalCausalSwizzleIndex``；新路径以 ``m==n`` 调用。"""
    n_new = n + 1 if m == n else (n - m + 2) + (n + 1)
    paired_batch = b // 2
    result = cal_dense_swizzle_index(k, m, n_new, paired_batch, j, r)
    if result is None:
        return None

    pair_id, s1, s2 = result
    if m == n:
        if s2 >= s1 + 1:
            s2 = 2 * n - m - s2 + 2
            s1 = m + 1 - s1
            batch = 2 * pair_id
        else:
            batch = 2 * pair_id - 1
    else:
        if s2 >= s1 + (n + 1) - m + 1:
            s2 = 2 * (n + 1) - m - s2 + 2
            s1 = m + 1 - s1
            batch = 2 * pair_id
        else:
            batch = 2 * pair_id - 1

    if 1 <= batch <= b and 1 <= s1 <= m and 1 <= s2 <= n:
        return batch, s1, s2
    return None


def calc_left_up_causal_swizzle_schedule(k: int, m: int, n: int, b: int) -> ScheduleInfo:
    """镜像 Kernel 最大轮次函数，并返回虚拟矩形参数。"""
    if min(k, m, n, b) <= 0:
        raise ValueError(f"k/m/n/b must be positive, got {(k, m, n, b)}")
    if b % 2 != 0:
        raise ValueError("Host 仅在 b*n2 为正偶数时使能该 Swizzle")

    pair_count = b // 2
    if m <= n:
        schedule_mode = "TRIANGLE_PAIR"
        virtual_m = m
        virtual_n = m + 1
        active_k = min(k, m * pair_count)
    else:
        schedule_mode = "TRAPEZOID_PAIR"
        virtual_m = 2 * m - n + 1
        virtual_n = n
        active_k = min(k, virtual_n * pair_count)
    max_round = virtual_m * ceil_div(virtual_n * pair_count, active_k)
    return ScheduleInfo(
        k=k,
        m=m,
        n=n,
        b=b,
        pair_count=pair_count,
        schedule_mode=schedule_mode,
        virtual_m=virtual_m,
        virtual_n=virtual_n,
        active_k=active_k,
        max_round=max_round,
    )


def cal_left_up_causal_swizzle_index(
    k: int, m: int, n: int, b: int, j: int, r: int
) -> Optional[Coordinate]:
    """镜像 Kernel 新增坐标函数。"""
    schedule = calc_left_up_causal_swizzle_schedule(k, m, n, b)
    return _cal_left_up_causal_swizzle_index(schedule, j, r)


def _cal_left_up_causal_swizzle_index(
    schedule: ScheduleInfo, j: int, r: int
) -> Optional[Coordinate]:
    """复用已计算的轮次参数，坐标公式与公开入口完全相同。"""
    if not (1 <= j <= schedule.k) or not (1 <= r <= schedule.max_round):
        return None

    if schedule.m <= schedule.n:
        if j > schedule.active_k:
            return None
        return cal_causal_swizzle_index(
            schedule.active_k,
            schedule.m,
            schedule.m,
            schedule.b,
            j,
            r,
        )

    if j > schedule.active_k:
        return None
    column_id = ((r - 1) // schedule.virtual_m) * schedule.active_k + j - 1
    column_count = schedule.virtual_n * schedule.pair_count
    if column_id >= column_count:
        return None

    pair_id = column_id // schedule.n + 1
    virtual_s2 = column_id % schedule.n + 1
    virtual_s1 = (r - 1) % schedule.virtual_m + 1
    odd_batch_len = schedule.m - virtual_s2 + 1
    if virtual_s1 <= odd_batch_len:
        return 2 * pair_id - 1, virtual_s2 + virtual_s1 - 1, virtual_s2

    even_batch_offset = virtual_s1 - odd_batch_len
    return 2 * pair_id, schedule.m - even_batch_offset + 1, schedule.n - virtual_s2 + 1


# 保留旧脚本常用入口名，便于已有调用切换到新实现。
def get_dense_batch_position(m: int, n: int, b: int, core_id: int, round_id: int, k: int):
    return cal_dense_swizzle_index(k, m, n, b, core_id, round_id)


def get_causal_batch_position(m: int, n: int, b: int, core_id: int, round_id: int, k: int):
    return cal_left_up_causal_swizzle_index(k, m, n, b, core_id, round_id)


def expected_coordinates(m: int, n: int, b: int) -> set[Coordinate]:
    """sparseMode=2 的块级有效区：S2 块坐标不大于 S1 块坐标。"""
    return {
        (batch, s1, s2)
        for batch in range(1, b + 1)
        for s1 in range(1, m + 1)
        for s2 in range(1, n + 1)
        if s2 <= s1
    }


def iter_schedule(k: int, m: int, n: int, b: int) -> Iterable[Tuple[int, int, Coordinate]]:
    schedule = calc_left_up_causal_swizzle_schedule(k, m, n, b)
    for round_id in range(1, schedule.max_round + 1):
        for core_id in range(1, k + 1):
            coordinate = _cal_left_up_causal_swizzle_index(schedule, core_id, round_id)
            if coordinate is not None:
                yield core_id, round_id, coordinate


def verify_case(k: int, m: int, n: int, b: int) -> VerifyResult:
    schedule = calc_left_up_causal_swizzle_schedule(k, m, n, b)
    records = list(iter_schedule(k, m, n, b))
    coordinates = [coordinate for _, _, coordinate in records]
    expected = expected_coordinates(m, n, b)

    actual = set(coordinates)
    if actual != expected:
        missing = sorted(expected - actual)[:8]
        extra = sorted(actual - expected)[:8]
        raise AssertionError(f"coordinate coverage mismatch: {missing=}, {extra=}")
    if len(coordinates) != len(actual):
        raise AssertionError("the same (batch,s1,s2) coordinate is issued more than once")

    rows_by_round: Dict[int, list[Tuple[int, int]]] = {}
    cores_by_column: Dict[Tuple[int, int], set[int]] = {}
    for core_id, round_id, (batch, s1, s2) in records:
        rows_by_round.setdefault(round_id, []).append((batch, s1))
        cores_by_column.setdefault((batch, s2), set()).add(core_id)
    for round_id, rows in rows_by_round.items():
        if len(rows) != len(set(rows)):
            raise AssertionError(f"duplicate row in round {round_id}: {rows}")
    for column, core_ids in cores_by_column.items():
        if len(core_ids) != 1:
            raise AssertionError(f"column {column} is split across cores: {sorted(core_ids)}")

    for core_id in range(1, k + 1):
        if _cal_left_up_causal_swizzle_index(schedule, core_id, schedule.max_round + 1) is not None:
            raise AssertionError("valid coordinate exists after max_round")

    return VerifyResult(
        schedule=schedule,
        valid_blocks=len(expected),
        issued_coordinates=len(coordinates),
        idle_slots=k * schedule.max_round - len(coordinates),
    )


def run_full_test() -> None:
    checked_cases = 0
    checked_coordinates = 0
    for k in range(1, 17):
        for m in range(1, 17):
            for n in range(1, 17):
                for b in (2, 4, 6, 8):
                    result = verify_case(k, m, n, b)
                    checked_cases += 1
                    checked_coordinates += result.issued_coordinates

    assert host_enables_left_up_causal_swizzle(
        enable_swizzle=True, is_non_tnd=True, is_bn2gs1s2=True, cube_base_equal=True,
        sparse_type_supported=True, is_deterministic=True, deter_causal=True, sparse_mode=2,
        batch_n2=18, g=1, s1=28 * 128, aic_num=28,
    )
    assert not host_enables_left_up_causal_swizzle(
        enable_swizzle=True, is_non_tnd=True, is_bn2gs1s2=True, cube_base_equal=True,
        sparse_type_supported=True, is_deterministic=True, deter_causal=True, sparse_mode=2,
        batch_n2=17, g=1, s1=28 * 128, aic_num=28,
    )
    print(f"PASS full-test cases={checked_cases} coordinates={checked_coordinates}")


def _wrap_visual(text: str, limit: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    current_width = 0
    for char in text:
        width = 2 if ord(char) > 127 else 1
        if current and current_width + width > limit:
            last_space = current.rfind(" ")
            if last_space > len(current) // 2:
                lines.append(current[:last_space])
                current = current[last_space + 1 :] + char
                current_width = sum(2 if ord(item) > 127 else 1 for item in current)
            else:
                lines.append(current)
                current = char
                current_width = width
        else:
            current += char
            current_width += width
    if current:
        lines.append(current)
    return lines


def render_pdf(output: Path, preview_dir: Optional[Path] = None) -> Path:
    """生成与当前 Host/Kernel/Python 一致的 A4 PDF 文档。"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    font_paths = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]
    font_path = next((path for path in font_paths if path.exists()), None)
    font = FontProperties(fname=str(font_path)) if font_path else FontProperties(family="sans-serif")

    short_s1 = verify_case(28, 33, 49, 18)
    long_s1 = verify_case(28, 49, 33, 18)
    blocks: list[tuple[str, str]] = [
        ("title", "FAG arch35 确定性 LEFT_UP_CAUSAL 非等长 Swizzle 方案"),
        ("body", "版本：v1.0    日期：2026-08-18    范围：非 TND、DETER_CAUSAL、sparseMode=2、G=1"),
        ("body", "对应实现：SelectBlockSchedule / CalLeftUpCausalSwizzleMaxRound / CalLeftUpCausalSwizzleIndex"),
        ("h2", "1. 目标与结论"),
        ("body", "原实现仅在 S1==S2 时开启 Causal Swizzle。本方案支持 S1!=S2，并保持按列连续、坐标全覆盖、无重复坐标，以及确定性要求的同 Batch 同行同轮互斥。"),
        ("h2", "2. Host 入口条件"),
        ("code", "enableSwizzle && 非TND && splitAxis==BN2GS1S2\nCubeBaseM==CubeBaseN && sparseType受支持 && isDeterministic\ndeterSparseType==DETER_CAUSAL && sparseMode==LEFT_UP_CAUSAL\n(b*n2)%2==0 && g==1 && s1>=aicNum*128"),
        ("body", "与旧逻辑相比，仅 sparseMode=2 的 DETER_CAUSAL 不再要求原始 S1 与 S2 完全相等。NO_MASK 和 RIGHT_DOWN_CAUSAL 的入口含义保持不变。"),
        ("h2", "3. 有效块几何与按列拼接"),
        ("body", "令 m=S1Outer、n=S2Outer，块级有效条件为 y<=x。当 m<=n 时，S2 超出 m 的列整体无效，继续复用边长 m 的方形 Causal Swizzle。当 m>n 时，将相邻两个 Batch 的下三角梯形按互补列拼成 n 个等高虚拟列；每个虚拟列对应奇数 Batch 的列 y 和偶数 Batch 的列 n+1-y。"),
        ("code", "m>n:\nvirtualM   = 2*m-n+1\nvirtualN   = n\nvirtual col y = odd batch col y + even batch col (n+1-y)"),
        ("body", "每个虚拟列从头到尾固定在同一核，因此两个实际列也分别固定在该核；不存在三角区和尾部区跨核的问题。虚拟列内部前 m-y+1 个位置映射到奇数 Batch，剩余 m-n+y 个位置逆序映射到偶数 Batch。"),
        ("h2", "4. 最大轮次公式"),
        ("code", "pairCount = b/2\n\nif m<=n:\n  virtualM=m; virtualN=m+1\n  activeK=min(k,m*pairCount)\nelse:\n  virtualM=2*m-n+1; virtualN=n\n  activeK=min(k,virtualN*pairCount)\n\nmaxRound=virtualM*ceil(virtualN*pairCount/activeK)"),
        ("h2", "5. Kernel 坐标映射"),
        ("code", "m<=n: CalCausalSwizzleIndex(activeK,m,m,b,j,r)\n\nm>n:\n  columnId=((r-1)/virtualM)*activeK+j-1\n  pairId=columnId/n+1; y=columnId%n+1; xVirtual=(r-1)%virtualM+1\n  xVirtual<=m-y+1: (batch,x,y)=(2*pairId-1,y+xVirtual-1,y)\n  else:             (batch,x,y)=(2*pairId,m-(xVirtual-(m-y+1))+1,n-y+1)"),
        ("body", "在一个 virtualM 轮窗口内，columnId 和物理核保持不变，只有虚拟行递增。这既实现整列连续复用，又使同一 Batch 的不同列在同一轮映射到不同 S1 行。"),
        ("h2", "6. 两类非等长场景"),
        ("body", f"S1Outer<S2Outer 示例 k/m/n/b=28/33/49/18：mode={short_s1.schedule.schedule_mode}，virtualM/virtualN={short_s1.schedule.virtual_m}/{short_s1.schedule.virtual_n}，maxRound={short_s1.schedule.max_round}，有效坐标={short_s1.valid_blocks}，空闲槽位={short_s1.idle_slots}。"),
        ("body", f"S1Outer>S2Outer 示例 k/m/n/b=28/49/33/18：mode={long_s1.schedule.schedule_mode}，virtualM/virtualN={long_s1.schedule.virtual_m}/{long_s1.schedule.virtual_n}，maxRound={long_s1.schedule.max_round}，有效坐标={long_s1.valid_blocks}，空闲槽位={long_s1.idle_slots}。"),
        ("h2", "7. 一致性与正确性校验"),
        ("body", "Python 逐核逐轮镜像 Kernel。每个用例校验：maxRound 后无坐标；生成集合等于 y<=x 的理论有效集合；坐标不重复、不越界；每一轮内 (batch,s1) 唯一；每个 (batch,s2列) 仅归属一个物理核；Host 偶数 Batch 入口约束一致。"),
        ("code", "python casual.py --case 28 33 49 18\npython casual.py --case 28 49 33 18\npython casual.py --full-test\npython casual.py --render-pdf"),
        ("body", "穷举范围：k,m,n∈[1,16]，b∈{2,4,6,8}，共 16,384 组，覆盖 m<n、m==n、m>n 以及有效核数小于物理核数的边界。"),
        ("h2", "8. 保持不变的限制"),
        ("body", "当前仍要求 b*n2 为正偶数、G=1、两侧 Cube 基本块相等、非 TND、BN2GS1S2 模板并满足原 enableSwizzle 和 S1 规模门槛。奇数 Batch、GQA 和不同 CubeBase 不在本次扩展范围内。"),
    ]

    page_size = (8.27, 11.69)
    left, right, top, bottom = 0.072, 0.928, 0.942, 0.062
    styles = {
        "title": (17.0, 0.035, "#17365d", 62, "bold", 0.012),
        "h2": (12.5, 0.027, "#1f4e79", 82, "bold", 0.007),
        "body": (8.8, 0.019, "#202020", 108, "normal", 0.006),
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
                figure.savefig(preview_dir / f"page-{page:02d}.png", dpi=130)
            plt.close(figure)

        def new_page() -> None:
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
                axis.text(left, 0.973, "FAG LEFT_UP_CAUSAL 非等长 Swizzle", ha="left", va="top",
                          fontsize=7.5, color="#6b7280", fontproperties=font)
                axis.plot((left, right), (0.958, 0.958), color="#d7dde5", linewidth=0.6)
            axis.plot((left, right), (0.047, 0.047), color="#d7dde5", linewidth=0.5)
            axis.text(0.5, 0.027, f"— {page} —", ha="center", va="center", fontsize=7.5,
                      color="#777777", fontproperties=font)
            y = top

        new_page()
        for kind, content in blocks:
            if kind == "code":
                lines = [line for source in content.splitlines() for line in _wrap_visual(source, 111)]
                height = 0.020 + 0.0175 * max(len(lines), 1)
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
                y -= height + 0.010
                continue

            size, line_height, color, width, weight, after = styles[kind]
            lines = _wrap_visual(content, width)
            required = line_height * len(lines) + after + (0.025 if kind == "h2" else 0)
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


def print_case(result: VerifyResult) -> None:
    schedule = result.schedule
    print(f"k/m/n/b={schedule.k}/{schedule.m}/{schedule.n}/{schedule.b} maxRound={schedule.max_round}")
    print(
        f"mode={schedule.schedule_mode} pairCount={schedule.pair_count} "
        f"virtualM/N={schedule.virtual_m}/{schedule.virtual_n} activeK={schedule.active_k}"
    )
    print(
        f"verify=PASS validBlocks={result.valid_blocks} issued={result.issued_coordinates} "
        f"idleSlots={result.idle_slots}"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=4, type=int, metavar=("K", "M", "N", "B"), default=(28, 33, 49, 18))
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument(
        "--render-pdf",
        nargs="?",
        const=str(here / "FAG确定性计算_LEFT_UP_CAUSAL非等长Swizzle方案.pdf"),
    )
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    result = verify_case(*args.case)
    print_case(result)
    if args.full_test:
        run_full_test()
    if args.render_pdf:
        print(render_pdf(Path(args.render_pdf), args.preview_dir).resolve())


if __name__ == "__main__":
    main()
