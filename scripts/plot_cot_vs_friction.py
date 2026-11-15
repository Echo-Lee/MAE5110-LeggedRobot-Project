"""Plot matched-speed cost of transport against friction for one or more policies.

    uv run --frozen scripts/plot_cot_vs_friction.py logs/eval/cot_go2.json \\
        logs/eval/cot_lowfric.json --output logs/eval/cot_vs_friction.png

One panel per actual target speed: CoT (interpolated at exactly that speed,
with the replica spread as error bars) vs friction, one line per report. A
gap means the policy never reached that speed on that surface. The last panel
shows the top forward speed each policy reached on each surface, which
explains every gap.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
MARKERS = ("o", "s", "^", "D", "v", "P")
INK = "#3d3d3a"
MUTED_INK = "#73726c"
GRID = "#e4e3dd"


def by_target(report: dict) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for case in report["cases"]:
        grouped.setdefault(case["target_speed_m_s"], []).append(case)
    return {t: sorted(c, key=lambda x: x["friction"]) for t, c in grouped.items()}


def style_axis(ax, frictions, title):
    ax.set_xscale("log")
    ax.set_xticks(frictions, [f"{f:g}" for f in frictions], fontsize=7)
    ax.minorticks_off()
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Contact friction coefficient μ", color=MUTED_INK, fontsize=9)


def series_style(i):
    return dict(
        color=SERIES_COLORS[i],
        marker=MARKERS[i],
        markersize=6,
        linewidth=2,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )


def plot(reports, labels, output: Path, key: str, energy_name: str):
    targets = sorted({c["target_speed_m_s"] for r in reports for c in r["cases"]})
    frictions = sorted({c["friction"] for r in reports for c in r["cases"]})
    panels = len(targets) + 1
    cols = 2 if panels == 4 else panels
    rows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6.2 * cols, 4.3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = list(axes.flat)
    cot_max = 0.0
    for ax, target in zip(flat, targets):
        for i, (report, label) in enumerate(zip(reports, labels)):
            shown = [
                (c["friction"], c[key], c[key + "_std"])
                for c in by_target(report)[target]
                if c["status"] == "ok"
            ]
            if not shown:
                continue
            xs, ys, es = zip(*shown)
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                capsize=3,
                elinewidth=1,
                label=label,
                **series_style(i),
            )
            cot_max = max(cot_max, max(y + e for y, e in zip(ys, es)))
        style_axis(ax, frictions, f"At {target:g} m/s actual speed")
        ax.set_ylabel(energy_name, color=MUTED_INK, fontsize=9)
    for ax in flat[: len(targets)]:
        ax.set_ylim(0, cot_max * 1.1 if cot_max else 1)

    top_ax = flat[len(targets)]
    for i, (report, label) in enumerate(zip(reports, labels)):
        tops = sorted(
            {
                c["friction"]: c["top_speed_m_s"]
                for c in report["cases"]
                if c["top_speed_m_s"] is not None
            }.items()
        )
        if tops:
            xs, ys = zip(*tops)
            top_ax.plot(xs, ys, label=label, **series_style(i))
    for target in targets:
        top_ax.axhline(target, color=MUTED_INK, linewidth=0.8, linestyle="--")
    style_axis(top_ax, frictions, "Top forward speed reached")
    top_ax.set_ylabel("Speed (m/s); dashed: target speeds", color=MUTED_INK, fontsize=9)
    top_ax.set_ylim(bottom=0)
    for ax in flat[panels:]:
        ax.set_visible(False)

    handles, names = [], []
    for ax in flat:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    fig.legend(handles, names, loc="outside lower center", frameon=False, ncols=6)
    fig.suptitle(
        "Cost of transport vs. friction at matched actual speed "
        "(gap: speed never reached on that surface)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def markdown_table(reports, labels, key: str) -> str:
    lines = []
    targets = sorted({c["target_speed_m_s"] for r in reports for c in r["cases"]})
    for target in targets:
        lines.append(f"\n**At {target:g} m/s actual speed**: CoT ± replica std\n")
        lines.append("| μ | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        tables = [{c["friction"]: c for c in by_target(r)[target]} for r in reports]
        for friction in sorted(tables[0]):
            cells = []
            for table in tables:
                case = table[friction]
                if case["status"] == "ok":
                    cells.append(f"{case[key]:.2f} ± {case[key + '_std']:.2f}")
                elif case["status"] == "unreachable":
                    top = case["top_speed_m_s"]
                    cells.append("– (fell)" if top is None else f"– (top {top:.2f})")
                else:
                    cells.append("– (gap in sweep)")
            lines.append(f"| {friction:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument(
        "--positive_work",
        action="store_true",
        help="plot CoT from positive joint work only instead of |work|",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reports) > len(SERIES_COLORS):
        parser.error(f"compare at most {len(SERIES_COLORS)} reports per figure")
    reports = [json.loads(path.read_text()) for path in args.reports]
    labels = args.labels or [report["label"] for report in reports]
    if len(labels) != len(reports):
        parser.error("give one label per report")
    if args.positive_work:
        key, name = "cot_positive_work", "Cost of transport (positive work)"
    else:
        key, name = "cot", "Cost of transport (|work|)"
    plot(reports, labels, args.output, key, name)
    print(markdown_table(reports, labels, key))
    print(f"\nfigure: {args.output}")


if __name__ == "__main__":
    main()
