"""
Figure 1: Assembled overview figure.

Loads the already-saved sub-figure PNGs (1a, 1b, 1c, 1d) and combines
them into a single PDF.

Expected inputs
---------------
  1a  figures_out/figure_1/1a/<scope>_<mode>/paper_phenotype_grid.png
  1b  figures_out/figure_1/1b/schematic.png          (user-provided, fixed)
  1c  figures_out/figure_1/1c/<scope>_<mode>/protocol_heatmap.png
  1d  figures_out/figure_1/1d/<scope>_<mode>/protocol_boxplots.png

If a PNG is missing a grey placeholder panel is shown instead.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as mgridspec
import numpy as np


def _load_or_placeholder(path: Path) -> np.ndarray:
    if path and Path(path).exists():
        return mpimg.imread(str(path))
    h, w = 600, 800
    arr = np.full((h, w, 3), 0.88)
    return arr


def assemble_figure1(
    path_1a: Path,
    path_1b: Path,
    path_1c: Path,
    path_1d: Path,
    figsize: tuple = (34, 22),
    row1_ratios: tuple = (4, 3),   # A wider than B
    row2_ratios: tuple = (2, 4),   # D wider than C
    row2_shift: float = 0.5,       # invisible left spacer to shift C and D right
) -> plt.Figure:
    """
    Assemble panels A-D from PNG files into one combined Figure 1.

    Layout (A and D are wider)
    ------
      ┌──────────────┬───────┐
      │      A       │   B   │
      ├───────┬──────────────┤
      │   C   │      D       │
      └───────┴──────────────┘
    """
    imgs = {
        "A": _load_or_placeholder(path_1a),
        "B": _load_or_placeholder(path_1b),
        "C": _load_or_placeholder(path_1c),
        "D": _load_or_placeholder(path_1d),
    }

    fig = plt.figure(figsize=figsize)
    outer = mgridspec.GridSpec(2, 1, figure=fig, hspace=0.12,
                               left=0.06, right=0.98, top=0.97, bottom=0.02)

    gs_top = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], width_ratios=list(row1_ratios), wspace=0.04)
    gs_bot = mgridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        width_ratios=[row2_shift] + list(row2_ratios), wspace=0.04)

    panels = [
        (fig.add_subplot(gs_top[0]), "A", imgs["A"]),
        (fig.add_subplot(gs_top[1]), "B", imgs["B"]),
        (fig.add_subplot(gs_bot[1]), "C", imgs["C"]),
        (fig.add_subplot(gs_bot[2]), "D", imgs["D"]),
    ]

    for ax, letter, img in panels:
        ax.imshow(img, aspect="auto", interpolation="none")
        ax.set_axis_off()
        ax.text(0.01, 0.99, letter, transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="top", zorder=10)

    return fig
