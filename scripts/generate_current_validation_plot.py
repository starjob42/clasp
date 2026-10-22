"""Generate the current interim CLASP/FOARL validation-status figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ("TSP", "OP", "JSSP", "CVRP", "MIS")
COLORS = {"CLASP": "#2166ac", "FOARL": "#b2182b"}


def svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start") -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}">{escaped}</text>'
    )


def bar(x: float, y: float, width: float, height: float, color: str, pending: bool) -> str:
    if pending:
        return (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            'fill="#dddddd" stroke="#666666" stroke-dasharray="5,4"/>'
        )
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}"/>'


def result_value(result: dict, key: str) -> tuple[float | None, bool]:
    if result.get("status") != "complete":
        return None, True
    value = result.get(key)
    return (float(value) if value is not None else None), value is None


def build_svg(manifest: dict) -> str:
    width, height = 1100, 650
    left, right = 90, 1030
    plot_width = right - left
    baseline = 255
    gap_baseline = 555
    group_width = plot_width / len(TASKS)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { fill: #222; } .grid { stroke: #dddddd; stroke-width: 1; }</style>',
        svg_text(width / 2, 30, "Interim tuned validation results (eval[100:196])", size=22, anchor="middle"),
        svg_text(width / 2, 54, "Complete values are shown; pending runs are intentionally grey and unfilled.", size=13, anchor="middle"),
    ]

    for axis_y, title, maximum, ticks in (
        (baseline, "Instance feasibility (%)", 100, (0, 25, 50, 75, 100)),
        (gap_baseline, "Mean best-feasible reference gap (%)", 8, (0, 2, 4, 6, 8)),
    ):
        top = axis_y - 170
        out.append(svg_text(left, top - 18, title, size=16))
        for tick in ticks:
            y = axis_y - (tick / maximum) * 170
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
            out.append(svg_text(left - 10, y + 5, str(tick), size=11, anchor="end"))
        out.append(f'<line x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}" stroke="#222"/>')

    clasp = manifest["tuned_validation"]["clasp_best_completed_by_task"]
    foarl = manifest["tuned_validation"]["foarl_same_settings_jobs"]
    for index, task in enumerate(TASKS):
        x_center = left + group_width * (index + 0.5)
        out.append(svg_text(x_center, baseline + 30, task, size=14, anchor="middle"))
        for offset, method, source in ((-22, "CLASP", clasp[task.lower()]), (22, "FOARL", foarl[task.lower()])):
            feasibility, pending = result_value(source, "feasible")
            if feasibility is not None:
                height_px = feasibility / 96 * 170
                out.append(bar(x_center + offset - 12, baseline - height_px, 24, height_px, COLORS[method], False))
                out.append(svg_text(x_center + offset, baseline - height_px - 6, f"{feasibility:.0f}/96", size=10, anchor="middle"))
            else:
                out.append(bar(x_center + offset - 12, baseline - 14, 24, 14, COLORS[method], True))
                out.append(svg_text(x_center + offset, baseline - 22, "pending", size=9, anchor="middle"))
            gap, pending = result_value(source, "gap_percent")
            if gap is not None:
                gap_height = gap / 8 * 170
                out.append(bar(x_center + offset - 12, gap_baseline - gap_height, 24, gap_height, COLORS[method], False))
                out.append(svg_text(x_center + offset, gap_baseline - gap_height - 6, f"{gap:.2f}", size=10, anchor="middle"))
            else:
                out.append(bar(x_center + offset - 12, gap_baseline - 14, 24, 14, COLORS[method], True))
                out.append(svg_text(x_center + offset, gap_baseline - 22, "pending", size=9, anchor="middle"))

    out.extend(
        [
            '<rect x="760" y="70" width="18" height="18" fill="#2166ac"/>',
            svg_text(785, 85, "CLASP", size=13),
            '<rect x="860" y="70" width="18" height="18" fill="#b2182b"/>',
            svg_text(885, 85, "FOARL", size=13),
            '<rect x="960" y="70" width="18" height="18" fill="#dddddd" stroke="#666" stroke-dasharray="5,4"/>',
            svg_text(985, 85, "pending", size=13),
            "</svg>",
        ]
    )
    return "\n".join(out)


def build_pdf(manifest: dict) -> bytes:
    """Write a small dependency-free vector PDF for paper inclusion."""
    width, height = 792, 468
    left, right = 64, 748
    plot_width = right - left
    baseline, gap_baseline = 205, 405
    group_width = plot_width / len(TASKS)
    commands = ["1 1 1 rg 0 0 792 468 re f", "0 0 0 rg"]

    def text(x: float, y: float, value: str, size: int = 10, align: str = "left") -> None:
        escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if align == "center":
            # The labels are short; an approximate width keeps this PDF dependency-free.
            x -= len(value) * size * 0.25
        commands.append(f"/F1 {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ({escaped}) Tj")

    def rect(x: float, y: float, w: float, h: float, color: tuple[float, float, float]) -> None:
        commands.append(f"{color[0]} {color[1]} {color[2]} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")

    text(width / 2, 445, "Interim tuned validation results (eval[100:196])", 16, "center")
    text(width / 2, 428, "Complete values are shown; pending runs are intentionally grey and unfilled.", 9, "center")
    clasp = manifest["tuned_validation"]["clasp_best_completed_by_task"]
    foarl = manifest["tuned_validation"]["foarl_same_settings_jobs"]
    colors = {"CLASP": (0.13, 0.40, 0.67), "FOARL": (0.70, 0.09, 0.17)}
    for axis_y, title, maximum, key in (
        (baseline, "Instance feasibility (%)", 100, "feasible"),
        (gap_baseline, "Mean best-feasible reference gap (%)", 8, "gap_percent"),
    ):
        text(left, axis_y + 184, title, 11)
        for tick in (0, 25, 50, 75, 100) if key == "feasible" else (0, 2, 4, 6, 8):
            y = axis_y + (tick / maximum) * 150
            commands.append(f"0.86 0.86 0.86 RG 0.5 w {left} {y:.1f} m {right} {y:.1f} l S")
            text(left - 8, y - 3, str(tick), 8, "right")
        commands.append(f"0 0 0 RG 0.7 w {left} {axis_y} m {right} {axis_y} l S")
    for index, task in enumerate(TASKS):
        x_center = left + group_width * (index + 0.5)
        text(x_center, 168, task, 10, "center")
        for offset, method, source in ((-14, "CLASP", clasp[task.lower()]), (14, "FOARL", foarl[task.lower()])):
            if key == "feasible":
                value, pending = result_value(source, "feasible")
                scale = 150 / 96
                label = f"{value:.0f}/96" if value is not None else "pending"
            else:
                value, pending = result_value(source, "gap_percent")
                scale = 150 / 8
                label = f"{value:.2f}" if value is not None else "pending"
            x = x_center + offset - 8
            h = (value or 0) * scale if value is not None else 10
            if value is not None:
                rect(x, axis_y, 16, h, colors[method])
            else:
                commands.append(f"0.86 0.86 0.86 rg {x:.1f} {axis_y:.1f} 16 10 re f")
                commands.append(f"0.35 0.35 0.35 RG 0.7 w {x:.1f} {axis_y:.1f} 16 10 re S")
            text(x_center + offset, axis_y + h + 5, label, 7, "center")
    # Legend.
    rect(555, 420, 12, 12, colors["CLASP"])
    text(572, 422, "CLASP", 9)
    rect(615, 420, 12, 12, colors["FOARL"])
    text(632, 422, "FOARL", 9)
    commands.append("0.86 0.86 0.86 rg 680 420 12 12 re f 0.35 0.35 0.35 RG 0.7 w 680 420 12 12 re S")
    text(697, 422, "pending", 9)
    stream = "BT\n" + "\n".join(commands) + "\nET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 468] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode())} >>\nstream\n{stream}endstream".encode(),
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    pdf.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode())
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(pdf)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("final_clasp/revised/paper_results_source_of_truth.json"))
    parser.add_argument("--output", type=Path, default=Path("final_clasp/revised/figs/current_validation_results.svg"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(manifest), encoding="utf-8")
    pdf_output = args.output.with_suffix(".pdf")
    pdf_output.write_bytes(build_pdf(manifest))
    print(args.output)
    print(pdf_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
