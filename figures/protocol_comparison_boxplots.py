import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_comprehensive_boxplot_comparison(
        df_papers,
        df_LightGBM,
        base_folder,
        phenotypes_list,
        selected_combinations=None,
        figsize=(16, 9),
        show_points=True,
        box_width=0.65
):
    """
    Create boxplot comparison of all four result types:
    1. Papers LODO
    2. LightGBM LODO
    3. LightGBM Internal Validation
    4. LightGBM Within-Dataset

    Parameters:
    -----------
    csv_path : str
        Path to CSV with paper results
    base_folder : str
        Path to folder containing phenotype data
    phenotypes_list : list of tuples
        List of (phenotype, type) combinations
    selected_combinations : list of tuples, optional
        Subset of phenotypes to plot
    figsize : tuple
        Figure size (width, height)
    show_points : bool
        Whether to show individual data points
    box_width : float
        Width of boxplots

    Returns:
    --------
    fig, ax : matplotlib figure and axis
    combined_data : pd.DataFrame
        All data used for plotting
    """

    # ============================================
    # 1. Aggregate paper results
    # ============================================

    df_papers = df_papers.dropna(subset=["Phenotype", "Type"], how="all").reset_index(drop=True)
    df_papers["Phenotype"] = df_papers["Phenotype"].astype(str)
    df_papers["Type"] = df_papers["Type"].astype(str)
    df_papers["Group"] = df_papers["Phenotype"] + " " + df_papers["Type"]

    if selected_combinations:
        groups_to_plot = [f"{p} {t}" for p, t in selected_combinations]
        df_papers = df_papers[df_papers["Group"].isin(groups_to_plot)]

    ignore = {"Phenotype full name", "Phenotype", "Type", "Metadata Mapping",
              "Dataset", "Notes", "Group"}
    paper_cols = [c for c in df_papers.columns if c not in ignore
                  and pd.api.types.is_numeric_dtype(df_papers[c])]

    # Collect all paper AUC values
    paper_aucs = []
    for group_name in df_papers["Group"].unique():
        group_df = df_papers[df_papers["Group"] == group_name]
        auc_values = []
        for paper in paper_cols:
            vals = pd.to_numeric(group_df[paper], errors="coerce").dropna().values
            auc_values.extend(vals)

        if len(auc_values) == 0:
            continue

        group_mean_auc = np.mean(auc_values)
        paper_aucs.append({
            "Group": group_name,
            "Method": "Papers LODO",
            "AUC": group_mean_auc
        })

    # ============================================
    # 2. COMPUTE LIGHTGBM RESULTS (ALL THREE TYPES)
    # ============================================

    lightgbm_data = []
    for _, row in df_LightGBM.iterrows():
        group_name = f"{row['Phenotype']}"
        LODO_auc = f"{row['LODO']}"
        Internal_Validation_auc = f"{row['Internal Validation']}"
        Within_Learning_auc = f"{row['Within Learning']}"

        lightgbm_data.append({"Group": group_name, "Method": "LightGBM LODO", "AUC": LODO_auc})
        lightgbm_data.append(
            {"Group": group_name, "Method": "LightGBM Internal Validation", "AUC": Internal_Validation_auc})
        lightgbm_data.append({"Group": group_name, "Method": "LightGBM Within Learning", "AUC": Within_Learning_auc})

    # ============================================
    # 3. COMBINE ALL DATA
    # ============================================
    combined_data = pd.DataFrame(paper_aucs + lightgbm_data)
    combined_data.to_csv(f"{base_folder}/combined_data_papers_lightGBM.csv")

    if combined_data.empty:
        raise ValueError("No data available for plotting")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    for method in combined_data["Method"].unique():
        method_data = pd.to_numeric(combined_data[combined_data["Method"] == method]["AUC"], errors="coerce")
        print(f"\n{method}:")
        print(f"  N = {len(method_data)}")
        print(f"  Mean = {method_data.mean():.3f}")
        print(f"  Median = {method_data.median():.3f}")
        print(f"  Std = {method_data.std():.3f}")
        print(f"  Range = [{method_data.min():.3f}, {method_data.max():.3f}]")
    print("=" * 70)

    # ============================================
    # 4. CREATE ENHANCED BOXPLOT
    # ============================================
    # Set style with cleaner background
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#FAFAFA'
    plt.rcParams['figure.facecolor'] = 'white'

    # Define order
    method_order = [
        "Papers LODO",
        "LightGBM LODO",
        "LightGBM Internal Validation",
        "LightGBM Within Learning"
    ]

    # Enhanced grayscale color scheme with subtle variations
    colors = {
        "Papers LODO": "#E8E8E8",  # Light gray
        "LightGBM LODO": "#D0D0D0",  # Medium-light gray
        "LightGBM Internal Validation": "#B8B8B8",  # Medium gray
        "LightGBM Within Learning": "#A0A0A0"  # Medium-dark gray
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Filter data
    available_methods = [m for m in method_order if m in combined_data["Method"].unique()]
    print(f"Available Methods: {available_methods}")
    plot_data = combined_data[combined_data["Method"].isin(available_methods)].copy()
    plot_data["AUC"] = pd.to_numeric(plot_data["AUC"], errors="coerce")

    # Create enhanced boxplot
    bp = ax.boxplot(
        [plot_data[plot_data["Method"] == method]["AUC"].values for method in available_methods],
        labels=available_methods,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        meanline=False,  # Changed to show mean as marker
        showfliers=False,
        medianprops=dict(color='#DC143C', linewidth=3, solid_capstyle='round'),  # Crimson red
        meanprops=dict(marker='D', markerfacecolor='#1E90FF', markeredgecolor='black',
                       markersize=8, linewidth=1.5, zorder=4),  # Blue diamond
        boxprops=dict(linewidth=2, edgecolor='#2F2F2F'),  # Dark gray edges
        whiskerprops=dict(linewidth=2, color='#2F2F2F', linestyle='-'),
        capprops=dict(linewidth=2, color='#2F2F2F')
    )

    # Color the boxes with gradient effect
    for patch, method in zip(bp['boxes'], available_methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.85)
        patch.set_edgecolor('#2F2F2F')

    # Add individual points with LARGER size and better styling
    if show_points:
        for i, method in enumerate(available_methods, 1):
            method_data = plot_data[plot_data["Method"] == method]["AUC"].values

            # Add jitter to x-coordinates
            np.random.seed(42)  # For reproducibility
            x_jitter = np.random.normal(i, 0.05, size=len(method_data))

            # Plot points with enhanced styling
            ax.scatter(x_jitter, method_data,
                       alpha=0.6,  # More visible
                       s=80,  # LARGER size (increased from 30)
                       color='#404040',  # Dark gray
                       edgecolors='black',  # Black edge
                       linewidths=1,  # Thicker edge
                       zorder=3)

    # Customize plot
    ax.set_ylabel("AUC", fontsize=26, fontweight='bold', labelpad=10)
    ax.set_xlabel("Method", fontsize=26, fontweight='bold', labelpad=10)

    # Enhanced x-axis labels
    ax.set_xticklabels(
        ["Papers\nLODO", "LGBM\nLODO", "LGBM\nInternal\nValidation", "LGBM\nWithin\nLearning"],
        rotation=0, ha='center', fontsize=24
        #, fontweight='medium'
    )
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)

    # Enhanced gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=1, color='gray')
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim(0.4, 1.05)

    # Add reference line at AUC = 0.5 with enhanced styling
    ax.axhline(y=0.5, color='#696969', linestyle=':', linewidth=2.5,
               alpha=0.6, label='Random Chance', zorder=1)

    # Enhanced legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='#DC143C', linewidth=3, label='Median',
               solid_capstyle='round'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#1E90FF',
               markeredgecolor='black', markersize=9, label='Mean', linewidth=0),
        Line2D([0], [0], color='#696969', linewidth=2.5, linestyle=':',
               label='Random Chance (0.5)'),
    ]

    if show_points:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#404040',
                   markeredgecolor='black', markersize=9, label='Individual Values',
                   alpha=0.6, linewidth=1)
        )

    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=True, shadow=True, fontsize=12, framealpha=0.95,
              edgecolor='gray')


    # Add subtle border around plot
    for spine in ax.spines.values():
        spine.set_edgecolor('#2F2F2F')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    return fig, ax, combined_data




def plot_protocol_boxplots(
    results_df_left: pd.DataFrame,
    results_df_right: pd.DataFrame = None,
    label_left: str = "Original",
    label_right: str = "Normalized",
):
    """
    Plot protocol boxplots. If results_df_right is provided, draws two
    side-by-side subplots sharing the same y-axis scale.
    """
    has_two = results_df_right is not None

    fig, axes = plt.subplots(
        1, 2 if has_two else 1,
        figsize=(14 if has_two else 7, 6),
        sharey=True,
    )
    if not has_two:
        axes = [axes]

    def _draw(ax, df, title):
        sns.boxplot(data=df, x="protocol", y="auc", ax=ax,
                    order=sorted(df["protocol"].unique()))
        sns.stripplot(data=df, x="protocol", y="auc", color="black",
                      alpha=0.4, ax=ax,
                      order=sorted(df["protocol"].unique()))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("AUC" if ax is axes[0] else "", fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    _draw(axes[0], results_df_left, label_left)
    if has_two:
        _draw(axes[1], results_df_right, label_right)

    # Shared y-axis scale: use combined min/max
    all_auc = results_df_left["auc"].dropna()
    if has_two:
        all_auc = pd.concat([all_auc, results_df_right["auc"].dropna()])
    y_min = max(0.0, all_auc.min() - 0.05)
    y_max = min(1.0, all_auc.max() + 0.05)
    axes[0].set_ylim(y_min, y_max)

    plt.tight_layout()
    return fig, axes
