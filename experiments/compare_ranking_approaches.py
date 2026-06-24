"""
Compare ranking normalization approaches for the distribution investigation.

Three modes, each saved to its own directory under figures_out/:

  regular  → distribution_approach         : original order, positional tie-breaking
  ordered  → distribution_approach_ordered : fully-ordered datasets shuffled before ranking
  average  → distribution_approach_average : average rank for ties (order-invariant)

Usage
-----
Run all three:
    python experiments/compare_ranking_approaches.py

Run a specific mode:
    python experiments/compare_ranking_approaches.py regular
    python experiments/compare_ranking_approaches.py ordered
    python experiments/compare_ranking_approaches.py average
"""

import sys
from pathlib import Path

from experiments.run_pipeline import phenotypes_pipeline, CONFIG
from experiments.investigate_distribution_approach import (
    run_distribution_investigation,
    print_auc_summary_table,
)

FIGURES_OUT = Path("figures_out")

_common = dict(
    phenotypes=phenotypes_pipeline,
    plot_only=False,
    stability_percentile_metagenomics=CONFIG["stability_percentile_global_metagenomics"],
    stability_percentile_amplicon=CONFIG["stability_percentile_global_amplicon"],
    stability_percentile_global_combined=CONFIG["stability_percentile_global_combined"],
    min_size=CONFIG["min_samples_per_dataset"],
    taxonomy_level_metagenomics=CONFIG["taxonomy_level_metagenomics"],
    taxonomy_level_amplicon=CONFIG["taxonomy_level_amplicon"],
    taxonomy_level_combined=CONFIG["taxonomy_level_combined"],
)

MODES = {
    "regular": dict(
        output_dir=FIGURES_OUT / "distribution_approach",
        shuffle_ordered=False,
        rank_tie_method="first",
    ),
    "ordered": dict(
        output_dir=FIGURES_OUT / "distribution_approach_ordered",
        shuffle_ordered=True,
        rank_tie_method="first",
    ),
    "average": dict(
        output_dir=FIGURES_OUT / "distribution_approach_average",
        shuffle_ordered=False,
        rank_tie_method="average",
    ),
}


def run(mode: str):
    cfg = MODES[mode]
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}  →  {cfg['output_dir']}")
    print(f"{'='*60}")
    run_distribution_investigation(**cfg, **_common)
    print_auc_summary_table(cfg["output_dir"])


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODES.keys())
    for mode in requested:
        if mode not in MODES:
            print(f"Unknown mode '{mode}'. Choose from: {list(MODES.keys())}")
            sys.exit(1)
        run(mode)
