"""
Compare evaluate_go2_friction.py reports as metric-vs-friction curves.

    uv run --frozen scripts/plot_friction_eval.py \\
        logs/eval/go2.json logs/eval/go2_lowfric.json \\
        --labels baseline lowfric --output logs/eval/friction.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gym.envs.go2.go2_lowfric_config import GROUND_FRICTION, ICE_FRICTION  # noqa: E402

# (key, title, commands averaged). Distance-normalized metrics exclude the
# stationary and turn-in-place commands, for which they are undefined.
MOVING = ("forward_slow", "forward", "forward_fast", "left")
ALL = ("stopped", *MOVING, "turn_left")
METRICS = (
    ("fall_rate", "Fall rate", ALL),
    ("linear_rmse_m_s", "Linear tracking RMSE (m/s)", ALL),
    ("yaw_rmse_rad_s", "Yaw tracking RMSE (rad/s)", ALL),
    ("slip_ratio", "Slip ratio (foot slip / body distance)", MOVING),
    ("cost_of_transport", "Cost of transport", MOVING),
    ("touchdowns_per_m", "Touchdowns per meter", MOVING),
)
# Validated categorical slots (dataviz reference palette, light mode).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
MARKERS = ("o", "s", "^", "D", "v", "P")
INK = "#3d3d3a"
MUTED_INK = "#73726c"
GRID = "#e4e3dd"
BAND = "#f1f0ec"


def curve(report: dict, metric: str, commands: tuple) -> dict[float, float | None]:
    by_friction: dict[float, list[float]] = {}
    for case in report["cases"].values():
        if case["command_name"] not in commands:
            continue
        values = by_friction.setdefault(case["friction"], [])
        if case[metric] is not None:
            values.append(case[metric])
    return {
        friction: (sum(values) / len(values) if values else None)
        for friction, values in sorted(by_friction.items())
    }


def plot(reports: list[dict], labels: list[str], output: Path) -> None:
    frictions = sorted({c["friction"] for r in reports for c in r["cases"].values()})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (metric, title, commands) in zip(axes.flat, METRICS):
        drawn = False
        for low, high in (ICE_FRICTION, GROUND_FRICTION):
            ax.axvspan(low, high, color=BAND, zorder=0, linewidth=0)
        for i, (report, label) in enumerate(zip(reports, labels)):
            points = [(f, v) for f, v in curve(report, metric, commands).items()]
            defined = [(f, v) for f, v in points if v is not None]
            if not defined:
                continue
            x, y = zip(*defined)
            ax.plot(
                x,
                y,
                color=SERIES_COLORS[i],
                marker=MARKERS[i],
                markersize=6,
                linewidth=2,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=label,
                zorder=3,
            )
            drawn = True
        if not drawn:
            ax.text(
                0.5,
                0.5,
                "undefined: robot did not travel",
                transform=ax.transAxes,
                ha="center",
                color=MUTED_INK,
                fontsize=9,
            )
        ax.set_xscale("log")
        ax.set_xticks(frictions, [f"{f:g}" for f in frictions])
        ax.minorticks_off()
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Contact friction coefficient μ", color=MUTED_INK, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED_INK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylim(bottom=0)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper right", frameon=False, ncols=6)
    fig.suptitle(
        "Go2 policies vs. contact friction (shaded: training friction bins)",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def markdown_table(reports: list[dict], labels: list[str]) -> str:
    frictions = sorted({c["friction"] for r in reports for c in r["cases"].values()})
    lines = []
    for metric, title, commands in METRICS:
        lines.append(f"\n**{title}** (mean over {', '.join(commands)})\n")
        lines.append("| μ | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        curves = [curve(report, metric, commands) for report in reports]
        for friction in frictions:
            cells = []
            for values in curves:
                value = values.get(friction)
                cells.append("–" if value is None else f"{value:.3f}")
            lines.append(f"| {friction:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.reports) > len(SERIES_COLORS):
        parser.error(f"compare at most {len(SERIES_COLORS)} reports per figure")
    labels = args.labels or [path.stem for path in args.reports]
    if len(labels) != len(args.reports):
        parser.error("give one label per report")
    reports = [json.loads(path.read_text()) for path in args.reports]
    plot(reports, labels, args.output)
    print(markdown_table(reports, labels))
    print(f"\nfigure: {args.output}")


if __name__ == "__main__":
    main()
