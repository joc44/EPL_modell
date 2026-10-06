"""Save calibration figures as PNG files without opening a graphical window."""

from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
except ImportError as error:
    raise SystemExit(
        "Matplotlib is required for the figures. Run this in the project virtual environment: "
        "python -m pip install -r requirements.txt"
    ) from error

import pandas as pd

from calibration_analysis import EVENTS, MODEL_LABELS, MODEL_NAMES


COLORS = {"league_average": "#0072B2", "raw": "#D55E00",
          "smoothed": "#009E73", "time_weighted": "#CC79A7"}


def plot_calibration(
    bins: pd.DataFrame, events: tuple[str, ...], output_path: Path,
    title: str, season: str, min_bin_matches: int, columns: int,
) -> None:
    """x is the mean probability; y is the observed rate. Empty bins have no point."""
    rows = (len(events) + columns - 1) // columns
    height = 6.3 if rows == 1 else 9.5
    width = 15.0 if columns == 3 else 12.0
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False)
    try:
        for event, ax in zip(events, axes.flat):
            ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1.1, zorder=1)
            for name in MODEL_NAMES:
                selected = bins.loc[(bins["event"] == event) & (bins["model"] == name)].sort_values("bin")
                # Missing values break the line at empty bins.
                ax.plot(selected["mean_probability"], selected["observed_rate"],
                        color=COLORS[name], linewidth=1.3, alpha=0.85, zorder=2)
                for sparse, filled in ((False, True), (True, False)):
                    points = selected.loc[(~selected["empty"]) & (selected["sparse"] == sparse)]
                    ax.scatter(points["mean_probability"], points["observed_rate"], s=42,
                               edgecolors=COLORS[name], facecolors=COLORS[name] if filled else "white",
                               linewidths=1.3, zorder=3)
            ax.set_title(EVENTS[event][0], fontsize=12, pad=10)
            ax.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_xlabel("Mean forecast probability", fontsize=10)
            ax.set_ylabel("Observed event rate", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(events):]:
            ax.set_visible(False)
        handles = [Line2D([0], [0], color=COLORS[name], marker="o", linewidth=1.3,
                          markersize=5, label=MODEL_LABELS[name]) for name in MODEL_NAMES]
        handles.append(Line2D([0], [0], color="#6B7280", linestyle="--", label="Reference: y = x"))
        fig.suptitle(f"{title} · {season}", fontsize=17, fontweight="bold", y=0.975)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                   ncol=5, frameon=False, fontsize=10)
        fig.text(0.5, 0.045, f"Hollow circle: fewer than {min_bin_matches} matches. "
                 "Each point is a bin average; empty bins have no point.",
                 ha="center", fontsize=10, color="#374151")
        fig.text(0.5, 0.016, "Development backtest; bin-level deviations also contain sampling variation.",
                 ha="center", fontsize=9, color="#6B7280")
        bottom = 0.14 if rows == 1 else 0.09
        fig.tight_layout(rect=[0.015, bottom, 0.985, 0.865])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
