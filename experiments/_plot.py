from __future__ import annotations

import shutil

import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


if shutil.which("latex") is not None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times", "TeX Gyre Termes"],
        }
    )
else:
    import warnings

    warnings.warn("LaTeX not found. Using non-LaTeX fonts for plotting.")
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],
        }
    )


def plot_results(
    results: list[dict[str, np.ndarray]],
    ylabel: str,
    *,
    figsize: tuple[float, float] = (6, 4),
    xlabels: list[str | None] | None = None,
    xs: list[np.ndarray | dict[str, np.ndarray] | None] | None = None,
    titles: list[str | None] | None = None,
    axhlines: list[float] | None = None,
    axhlines_labels: list[str] | None = None,
    axhlines_colors: list[str] | None = None,
    share_yaxis: bool = True,
    logx: bool = False,
    logy: bool = False,
    x_range: tuple[float, float] | None = None,
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    markersizes: dict[str, float] | None = None,
    markevery: int = 1,
    linestyles: dict[str, str | tuple[float, tuple[float, ...]]] | None = None,
    linewidths: dict[str, float] | None = None,
    zorders: dict[str, int] | None = None,
    legend_loc: str = "upper left",
    legend_column: int | None = None,
    title_fontsize: int | float = 16,
    legend_fontsize: int | float = 12,
    xlabel_fontsize: int | float = 14,
    ylabel_fontsize: int | float = 14,
    tick_fontsize: int | float = 15,
    use_tight_layout: bool = True,
) -> Figure:
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(results),
        figsize=figsize,
        sharey=share_yaxis,
    )
    for i, result in enumerate(results):
        ax_i = ax if len(results) == 1 else ax[i]
        x = xs[i] if xs is not None else None
        title = titles[i] if titles is not None else None
        xlabel = xlabels[i] if xlabels is not None else None
        ah_y = axhlines[i] if axhlines is not None else None
        ah_label = axhlines_labels[i] if axhlines_labels is not None else None
        ah_color = axhlines_colors[i] if axhlines_colors is not None else None
        for name, values in result.items():
            mean = np.mean(values, axis=0)
            stderr = np.std(values, axis=0) / np.sqrt(values.shape[0]) if values.shape[0] else None
            xj = (
                x[name]
                if isinstance(x, dict)
                else x
                if x is not None
                else np.arange(mean.shape[0])
            )
            if x_range is not None:
                xj, mean = (xj[mask := (xj >= x_range[0]) & (xj <= x_range[1])], mean[mask])
                if stderr is not None:
                    stderr = stderr[mask]
            ax_i.plot(
                xj,
                mean,
                label=name,
                color=(colors or {}).get(name, None),
                marker=(markers or {}).get(name, None),
                markersize=(markersizes or {}).get(name, None),
                linestyle=(linestyles or {}).get(name, None),
                linewidth=(linewidths or {}).get(name, None),
                zorder=(zorders or {}).get(name, None),
                markevery=markevery,
            )
            if stderr is not None:
                ax_i.fill_between(
                    xj,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.3,
                    color=(colors or {}).get(name, None),
                )
            ax_i.set_xlabel(xlabel, fontsize=xlabel_fontsize)
        if ah_y is not None:
            ax_i.axhline(ah_y, color=ah_color, linestyle="--", linewidth=2, label=ah_label)

        if not i:
            ax_i.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        if share_yaxis and i:
            ax_i.tick_params(axis="x", labelsize=tick_fontsize)
            ax_i.tick_params(axis="y", labelleft=False)
        else:
            ax_i.tick_params(axis="both", labelsize=tick_fontsize)

        if i == (len(results) - 1 if legend_column is None else legend_column):
            ax_i.legend(
                fontsize=legend_fontsize,
                labelspacing=0.05,
                handletextpad=0.4,
                borderpad=0.3,
                loc=legend_loc,
            )
        if title:
            ax_i.set_title(title, fontsize=title_fontsize)
        ax_i.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        if logx:
            ax_i.set_xscale("log")
        if logy:
            ax_i.set_yscale("log")
    if use_tight_layout:
        fig.tight_layout()
    return fig
