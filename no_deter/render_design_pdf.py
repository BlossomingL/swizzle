#!/usr/bin/env python3
"""将方案 Markdown 渲染为固定 A4 版式的 PDF，并追加 Dense/Causal 示意图。"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


PAGE_SIZE = (8.27, 11.69)
LEFT = 0.072
RIGHT = 0.928
TOP = 0.942
BOTTOM = 0.062
CONTENT_WIDTH = RIGHT - LEFT
DOCUMENT_TITLE = "FAG 非确定性计算严格按列 Swizzle 分核方案（Dense / Causal）"


def find_font() -> FontProperties:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            return FontProperties(fname=str(candidate))
    return FontProperties(family="sans-serif")


def clean_inline(text: str) -> str:
    """移除当前文档用到的轻量 Markdown 行内标记。"""
    return text.replace("**", "").replace("`", "")


def char_units(char: str) -> float:
    if char == "\t":
        return 4.0
    return 2.0 if unicodedata.east_asian_width(char) in ("W", "F") else 1.0


def wrap_visual(text: str, max_units: float, continuation: str = "") -> list[str]:
    """按中英文混排的近似显示宽度换行，避免中文行越过右边距。"""
    if not text:
        return [""]

    result: list[str] = []
    remaining = text.rstrip()
    prefix = ""
    while remaining:
        budget = max_units - sum(char_units(char) for char in prefix)
        used = 0.0
        split_at = 0
        last_space = -1
        for index, char in enumerate(remaining):
            next_used = used + char_units(char)
            if next_used > budget:
                break
            used = next_used
            split_at = index + 1
            if char.isspace():
                last_space = split_at
        else:
            result.append(prefix + remaining)
            break

        if split_at == 0:
            split_at = 1
        elif last_space > 0 and split_at - last_space < 20:
            split_at = last_space
        line = remaining[:split_at].rstrip()
        result.append(prefix + line)
        remaining = remaining[split_at:].lstrip()
        prefix = continuation
    return result or [""]


def split_table_row(line: str) -> list[str]:
    return [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_blocks(markdown: Path) -> list[tuple[str, object]]:
    lines = markdown.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].rstrip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            blocks.append(("code", code_lines))
            index += 1
            continue
        if (
            line.strip().startswith("|")
            and index + 1 < len(lines)
            and is_table_separator(lines[index + 1])
        ):
            rows = [split_table_row(line)]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index]))
                index += 1
            blocks.append(("table", rows))
            continue
        if line.startswith("# "):
            blocks.append(("title", clean_inline(line[2:])))
        elif line.startswith("## "):
            blocks.append(("h2", clean_inline(line[3:])))
        elif line.startswith("### "):
            blocks.append(("h3", clean_inline(line[4:])))
        elif line.startswith("- "):
            blocks.append(("list", "• " + clean_inline(line[2:])))
        elif re.match(r"^\d+\.\s", line):
            blocks.append(("list", clean_inline(line)))
        elif not line:
            blocks.append(("blank", ""))
        else:
            blocks.append(("body", clean_inline(line)))
        index += 1
    return blocks


def render_markdown(
    markdown: Path,
    output: Path,
    figures: list[Path],
    preview_dir: Path | None = None,
) -> None:
    font = find_font()
    styles = {
        "title": (18.0, 0.034, "#17365d", 62.0, "bold"),
        "h2": (13.0, 0.027, "#1f4e79", 82.0, "bold"),
        "h3": (10.8, 0.023, "#2f5597", 96.0, "bold"),
        "body": (8.8, 0.019, "#202020", 108.0, "normal"),
        "list": (8.8, 0.019, "#202020", 104.0, "normal"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        fig = None
        ax = None
        y = TOP
        page = 0

        def save_page() -> None:
            if fig is None:
                return
            # 不使用 bbox_inches="tight"，确保每页都是同样大小的 A4 页面。
            pdf.savefig(fig)
            if preview_dir is not None:
                fig.savefig(preview_dir / f"page-{page:02d}.png", dpi=120)
            plt.close(fig)

        def new_page() -> tuple[object, object, float]:
            nonlocal fig, ax, page
            if fig is not None:
                save_page()
            page += 1
            fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            if page > 1:
                ax.text(
                    LEFT,
                    0.973,
                    DOCUMENT_TITLE,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    color="#6b7280",
                    fontproperties=font,
                )
                ax.plot([LEFT, RIGHT], [0.958, 0.958], color="#d7dde5", linewidth=0.6)
            ax.plot([LEFT, RIGHT], [0.047, 0.047], color="#d7dde5", linewidth=0.5)
            ax.text(
                0.5,
                0.027,
                f"— {page} —",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                fontproperties=font,
            )
            return fig, ax, TOP

        def ensure_space(required: float) -> None:
            nonlocal fig, ax, y
            if y - required < BOTTOM:
                fig, ax, y = new_page()

        def draw_text(kind: str, content: str) -> None:
            nonlocal fig, ax, y
            size, line_height, color, width, weight = styles[kind]
            indent = 0.0
            continuation = ""
            if kind == "list":
                indent = 0.006
                continuation = "    "
            wrapped = wrap_visual(content, width, continuation)
            after = 0.008 if kind in ("title", "h2") else 0.003 if kind == "h3" else 0.0
            keep_next = 0.038 if kind in ("h2", "h3") else 0.0
            ensure_space(line_height * len(wrapped) + after + keep_next)
            for text_line in wrapped:
                props = font.copy()
                props.set_size(size)
                props.set_weight(weight)
                ax.text(
                    LEFT + indent,
                    y,
                    text_line,
                    ha="left",
                    va="top",
                    color=color,
                    fontproperties=props,
                )
                y -= line_height
            y -= after

        def draw_code(source_lines: list[str]) -> None:
            nonlocal fig, ax, y
            rendered: list[str] = []
            for source_line in source_lines:
                rendered.extend(wrap_visual(source_line.expandtabs(4), 111.0, "    "))
            rendered = rendered or [""]
            line_height = 0.0175
            padding = 0.010
            cursor = 0
            while cursor < len(rendered):
                available_lines = int((y - BOTTOM - 2 * padding) / line_height)
                if available_lines < 1:
                    fig, ax, y = new_page()
                    available_lines = int((y - BOTTOM - 2 * padding) / line_height)
                count = min(available_lines, len(rendered) - cursor)
                height = 2 * padding + count * line_height
                ax.add_patch(
                    Rectangle(
                        (LEFT, y - height),
                        CONTENT_WIDTH,
                        height,
                        facecolor="#f4f6f8",
                        edgecolor="#d5dbe3",
                        linewidth=0.6,
                    )
                )
                text_y = y - padding
                for code_line in rendered[cursor : cursor + count]:
                    props = font.copy()
                    props.set_size(8.1)
                    ax.text(
                        LEFT + 0.012,
                        text_y,
                        code_line,
                        ha="left",
                        va="top",
                        color="#263238",
                        fontproperties=props,
                    )
                    text_y -= line_height
                y -= height + 0.008
                cursor += count
                if cursor < len(rendered):
                    fig, ax, y = new_page()

        def draw_table(rows: list[list[str]]) -> None:
            nonlocal fig, ax, y
            if not rows:
                return
            column_count = max(len(row) for row in rows)
            normalized = [row + [""] * (column_count - len(row)) for row in rows]
            if column_count == 2:
                fractions = [0.19, 0.81]
            else:
                fractions = [1.0 / column_count] * column_count
            row_gap = 0.003

            def prepare_row(row: list[str]) -> tuple[list[list[str]], float]:
                wrapped_cells = [
                    wrap_visual(cell, max(10.0, 106.0 * fractions[index] - 3.0))
                    for index, cell in enumerate(row)
                ]
                row_height = max(0.032, 0.017 * max(len(cell) for cell in wrapped_cells) + 0.012)
                return wrapped_cells, row_height

            prepared = [prepare_row(row) for row in normalized]

            def paint_row(row_index: int, wrapped_cells: list[list[str]], height: float) -> None:
                nonlocal y
                x = LEFT
                face = "#dfeaf5" if row_index == 0 else ("#f7f9fb" if row_index % 2 == 0 else "white")
                for column, cell_lines in enumerate(wrapped_cells):
                    width = CONTENT_WIDTH * fractions[column]
                    ax.add_patch(
                        Rectangle(
                            (x, y - height),
                            width,
                            height,
                            facecolor=face,
                            edgecolor="#9aa9b8",
                            linewidth=0.55,
                        )
                    )
                    text_y = y - 0.007
                    for cell_line in cell_lines:
                        props = font.copy()
                        props.set_size(8.0)
                        props.set_weight("bold" if row_index == 0 else "normal")
                        ax.text(
                            x + 0.008,
                            text_y,
                            cell_line,
                            ha="left",
                            va="top",
                            color="#203040",
                            fontproperties=props,
                        )
                        text_y -= 0.017
                    x += width
                y -= height

            header_cells, header_height = prepared[0]
            ensure_space(header_height + (prepared[1][1] if len(prepared) > 1 else 0) + row_gap)
            paint_row(0, header_cells, header_height)
            for row_index, (cells, height) in enumerate(prepared[1:], start=1):
                if y - height < BOTTOM:
                    fig, ax, y = new_page()
                    paint_row(0, header_cells, header_height)
                paint_row(row_index, cells, height)
            y -= 0.010

        fig, ax, y = new_page()
        for kind, content in parse_blocks(markdown):
            if kind == "blank":
                if y < TOP - 0.004:
                    y -= 0.009
                continue
            if kind == "code":
                draw_code(content)
            elif kind == "table":
                draw_table(content)
            else:
                draw_text(kind, content)
        save_page()
        fig = None
        ax = None

        figure_titles = {
            "dense_k8_m16_n12_b2": "Dense：严格按实际列轮转分核示意",
            "causal_k8_m16_b3": "Causal：双 Batch 拼接与奇数尾三角分核示意",
        }
        for figure_path in figures:
            if not figure_path.exists():
                continue
            page += 1
            fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
            canvas = fig.add_axes([0, 0, 1, 1])
            canvas.set_xlim(0, 1)
            canvas.set_ylim(0, 1)
            canvas.axis("off")
            canvas.text(
                LEFT,
                0.95,
                figure_titles.get(figure_path.stem, figure_path.stem),
                ha="left",
                va="top",
                fontsize=13,
                color="#1f4e79",
                fontproperties=font,
                fontweight="bold",
            )
            canvas.plot([LEFT, RIGHT], [0.047, 0.047], color="#d7dde5", linewidth=0.5)
            canvas.text(
                0.5,
                0.027,
                f"— {page} —",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                fontproperties=font,
            )
            image_ax = fig.add_axes([LEFT, 0.09, CONTENT_WIDTH, 0.80])
            image_ax.axis("off")
            image_ax.imshow(plt.imread(figure_path))
            pdf.savefig(fig)
            if preview_dir is not None:
                fig.savefig(preview_dir / f"page-{page:02d}.png", dpi=120)
            plt.close(fig)
            fig = None


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=here / "FAG非确定性Swizzle分核方案_Dense_Causal.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "FAG非确定性Swizzle分核方案_Dense_Causal.pdf",
    )
    parser.add_argument("--figure-dir", type=Path, default=here / "figures")
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()
    figures = [
        args.figure_dir / "dense_k8_m16_n12_b2.png",
        args.figure_dir / "causal_k8_m16_b3.png",
    ]
    render_markdown(args.markdown, args.output, figures, args.preview_dir)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
