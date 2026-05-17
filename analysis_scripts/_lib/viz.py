"""Shared matplotlib/seaborn helpers for analysis plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_PUBLICATION_RCPARAMS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
}


def set_publication_rcparams() -> None:
    """Apply the project's publication-quality matplotlib defaults."""
    plt.rcParams.update(_PUBLICATION_RCPARAMS)


def similarity_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    *,
    ax=None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    annot: bool = True,
    title: str | None = None,
):
    """Render a square similarity heatmap with the project's standard styling.

    Returns the matplotlib Axes so callers can add titles/save as needed.
    """
    ax = sns.heatmap(
        matrix,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
    )
    if title:
        ax.set_title(title)
    return ax
