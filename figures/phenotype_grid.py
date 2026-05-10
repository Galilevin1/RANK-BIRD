import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def plot_paper_phenotype_grid_circles(data_shotgun, data_16s,
                                      primary_phenos=('CRC', 'IBD', 'CD', 'UC'),
                                      primary_color='#4363d8',  # blue
                                      secondary_color='#999999',  # gray
                                      figsize=(24, 20),
                                      title='Paper–Phenotype Grid: Dataset Usage',
                                      ax=None):
    """
    Grid visualization where:
    - Rows = Papers (Shotgun top, 16S bottom)
    - Columns = Phenotypes
    - Circles = Presence of phenotype in paper
    - Circle size = proportional to number of UNIQUE datasets using that phenotype
    - Color = Blue for CRC/IBD/CD/UC, Gray for others

    Sizing logic: For each phenotype, count how many unique papers/datasets use it.
    If a phenotype appears in BOTH Shotgun AND 16S for the same paper, count it only ONCE.

    Example:
    - p1 in [d1-16S, d2-16S, d1-Shotgun] → count = 2 (d1 and d2)
    - p2 in [d1-16S, d2-16S, d3-16S, d4-16S] → count = 4 (d1, d2, d3, d4)
    - p3 in [d1-16S, d2-Shotgun, d3-Shotgun, d3-16S] → count = 3 (d1, d2, d3)
    """

    # ---- Collect all papers and phenotypes ----
    all_papers = sorted(set(data_shotgun.keys()) | set(data_16s.keys()))
    all_phenotypes = set()

    # Collect phenotypes from both datasets
    for dataset_dict in [data_shotgun, data_16s]:
        for paper, pheno_dict in dataset_dict.items():
            for phenotype in pheno_dict.keys():
                all_phenotypes.add(phenotype)

    # Count unique datasets (papers) for each phenotype
    # If a phenotype appears in both Shotgun AND 16S for the same paper, count it only ONCE
    phenotype_dataset_count = {}
    for phenotype in all_phenotypes:
        unique_papers = set()

        # Check Shotgun section
        for paper in all_papers:
            if paper in data_shotgun and phenotype in data_shotgun[paper]:
                if len(data_shotgun[paper][phenotype]) > 0:
                    unique_papers.add(paper)

        # Check 16S section
        for paper in all_papers:
            if paper in data_16s and phenotype in data_16s[paper]:
                if len(data_16s[paper][phenotype]) > 0:
                    unique_papers.add(paper)  # set automatically handles duplicates

        phenotype_dataset_count[phenotype] = len(unique_papers)

    # Sort phenotypes: primary first (by count), then others (by count)
    primary_phenos_sorted = sorted(
        [p for p in all_phenotypes if p in primary_phenos],
        key=lambda x: phenotype_dataset_count[x],
        reverse=True
    )
    secondary_phenos_sorted = sorted(
        [p for p in all_phenotypes if p not in primary_phenos],
        key=lambda x: phenotype_dataset_count[x],
        reverse=True
    )
    all_phenotypes = primary_phenos_sorted + secondary_phenos_sorted

    # Determine max dataset count for scaling circle sizes
    max_dataset_count = max(phenotype_dataset_count.values()) if phenotype_dataset_count else 1
    min_dataset_count = min(phenotype_dataset_count.values()) if phenotype_dataset_count else 1

    print("\n=== Phenotype Unique Dataset Counts ===")
    for phenotype in all_phenotypes:
        count = phenotype_dataset_count[phenotype]
        print(f"{phenotype}: {count} unique datasets")

    # ---- Plot setup ----
    _standalone = ax is None
    if _standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Calculate the offset for 16S section
    offset = len(all_papers)

    # ---- Function to draw circles WITHOUT NUMBERS ----
    def draw_circle(x, y, phenotype, color, alpha=1.0):
        """
        Draw a circle sized by UNIQUE dataset count for that phenotype.
        Area is EXACTLY proportional: 5 datasets = 5× area of 1 dataset.
        """
        # Get unique dataset count for this phenotype (determines size)
        total_dataset_count = phenotype_dataset_count[phenotype]

        if total_dataset_count == 0:
            return

        # TRULY PROPORTIONAL: Area proportional to count
        # If phenotype has 5 datasets and another has 1, the first has 5× the area
        # Since Area = π × r², we need: r = sqrt(count) × base_radius

        base_radius = 0.20  # Reduced from 0.48 to make circles smaller
        # This base_radius is for 1 dataset

        # For proportional area: radius scales with sqrt(count)
        radius = base_radius * np.sqrt(total_dataset_count)

        circle = plt.Circle(
            (x, y),
            radius,
            color=color,
            alpha=alpha,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(circle)

        # NO TEXT LABEL - circles only, no numbers

    # ---- Draw Shotgun section ----
    for paper_idx, paper in enumerate(all_papers):
        for pheno_idx, phenotype in enumerate(all_phenotypes):
            # Check if this paper has data for this phenotype
            has_data = False
            if paper in data_shotgun and phenotype in data_shotgun[paper]:
                if len(data_shotgun[paper][phenotype]) > 0:
                    has_data = True

            if has_data:
                # Determine color
                color = primary_color if phenotype in primary_phenos else secondary_color
                # Draw circle (size based on total grid count for this phenotype)
                draw_circle(pheno_idx, paper_idx, phenotype, color, alpha=0.85)

    # ---- Draw 16S section with hatching background ----
    for paper_idx, paper in enumerate(all_papers):
        for pheno_idx, phenotype in enumerate(all_phenotypes):
            # Draw hatched background rectangle for 16S section
            rect = plt.Rectangle(
                (pheno_idx - 0.5, paper_idx + offset - 0.5),
                1, 1,
                edgecolor='lightgray',
                facecolor='white',
                hatch='///',
                linewidth=0.5,
                alpha=0.3
            )
            ax.add_patch(rect)

            # Check if this paper has data for this phenotype
            has_data = False
            if paper in data_16s and phenotype in data_16s[paper]:
                if len(data_16s[paper][phenotype]) > 0:
                    has_data = True

            if has_data:
                # Determine color
                color = primary_color if phenotype in primary_phenos else secondary_color
                # Draw circle (size based on total grid count for this phenotype)
                draw_circle(pheno_idx, paper_idx + offset, phenotype, color, alpha=0.85)

    # ---- Draw grid lines ----
    for i in range(len(all_phenotypes) + 1):
        ax.axvline(x=i - 0.5, color='black', linewidth=0.7, alpha=0.3)

    for i in range(2 * len(all_papers) + 1):
        ax.axhline(y=i - 0.5, color='black', linewidth=0.7, alpha=0.3)

    # ---- Section divider (bold line between Shotgun and 16S) ----
    ax.axhline(y=offset - 0.5, color='black', linewidth=3)

    # ---- Axis setup ----
    ax.set_xlim(-0.5, len(all_phenotypes) - 0.5)
    ax.set_ylim(-0.5, 2 * len(all_papers) - 0.5)
    ax.invert_yaxis()

    # X-axis: Phenotypes
    ax.set_xticks(range(len(all_phenotypes)))
    ax.set_xticklabels(all_phenotypes, rotation=45, ha='right', fontsize=32)

    # Y-axis: Papers (Shotgun then 16S)
    yticks = list(range(len(all_papers))) + [offset + i for i in range(len(all_papers))]
    ylabels = all_papers + all_papers
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=32)

    # ---- Section labels ----
    y_mid_shotgun = len(all_papers) / 2 - 0.5
    y_mid_16s = offset + len(all_papers) / 2 - 0.5

    bbox_props_shotgun = dict(boxstyle='round,pad=1.0', facecolor='lightblue',
                              edgecolor='black', linewidth=2.5, alpha=0.8)
    bbox_props_16s = dict(boxstyle='round,pad=1.0', facecolor='lightcoral',
                          edgecolor='black', linewidth=2.5, alpha=0.8)

    ax.text(-5.5, y_mid_shotgun, "Metagenomics\n(Shotgun)",
            fontsize=32, fontweight='bold',
            rotation=0, va='center', ha='center',
            bbox=bbox_props_shotgun, clip_on=False)

    ax.text(-5.5, y_mid_16s, "Amplicon\n(16S rRNA)",
            fontsize=32, fontweight='bold',
            rotation=0, va='center', ha='center',
            bbox=bbox_props_16s, clip_on=False)

    # ---- Axis labels (no title) ----
    ax.set_xlabel('Phenotype', fontsize=36, fontweight='bold')
    ax.set_ylabel('Paper', fontsize=36, fontweight='bold')

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(facecolor=primary_color, edgecolor='black',
                       label='CRC / IBD / CD / UC\n(Primary phenotypes)'),
        mpatches.Patch(facecolor=secondary_color, edgecolor='black',
                       label='Other phenotypes'),
        mpatches.Patch(facecolor='white', edgecolor='lightgray', hatch='///',
                       label='16S section\n(hatched background)'),
    ]

    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=20, frameon=True, shadow=True,
              title='Legend', title_fontsize=22)

    # ---- Circle size annotation box ----
    _y_yellow = 2 * len(all_papers) - 0.3
    if _standalone:
        fig.text(0.04, 0.06,
                 'Circle size: Area proportional to dataset count\n',
                 fontsize=22, style='italic',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 ha='left', va='bottom')
    else:
        ax.text(-5.5, _y_yellow, 'Circle size\n∝ dataset count',
                fontsize=20, style='italic', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                          edgecolor='goldenrod', linewidth=1.5, alpha=0.7),
                ha='center', va='center', clip_on=False)

    ax.grid(False)
    ax.set_facecolor('white')

    if _standalone:
        plt.subplots_adjust(left=0.15, right=0.88, top=0.97, bottom=0.22)

    return fig, ax


def plot_figure_1a(ax=None):
    """Figure 1a: Paper–Phenotype grid showing dataset usage across papers and sequencing types."""
    data_shotgun = {
        "SIAMCAT": {"CRC": ["CRC"], "CD": ["CD"], "UC": ["UC"]},
        "Li-et-al": {"CRC": ["CRC"], "Adenoma": ["Adenoma"], "RA": ["RA"], "ASD": ["ASD"], "CD": ["CD"],
                     "UC": ["UC"]},
        "melody": {"CRC": ["CRC"]},
        "vänni-et-al": {},
        "GCN": {"CRC": ["CRC"]},
        "Debias-M": {"CRC": ["CRC"], "HIV": ["HIV"], "Carcinoma": ["Carcinoma"], "CIN": ["CIN"]},
    }

    data_16s = {
        "SIAMCAT": {},
        "Li-et-al": {"AD": ["AD"], "PD": ["PD"], "T2D": ["T2D"], "ASD": ["ASD"], "CD": ["CD"], "IBS": ["IBS"],
                     "UC": ["UC"], "NAFLD": ["NAFLD"]},
        "melody": {},
        "vänni-et-al": {"Delivery mode": ["Delivery mode"]},
        "GCN": {},
        "Debias-M": {"Carcinoma": ["Carcinoma"], "CIN": ["CIN"], "HIV": ["HIV"]},
    }

    fig, ax = plot_paper_phenotype_grid_circles(
        data_shotgun, data_16s,
        title='Paper–Phenotype Grid: Dataset Usage Frequency',
        ax=ax,
    )
    return fig, ax
