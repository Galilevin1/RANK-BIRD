import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


def load_microbiome_datasets_with_targets(data_folder: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[str]]:
    """Load microbiome datasets and their corresponding target files from folder structure."""
    data_path = Path(data_folder)
    if not data_path.exists():
        raise ValueError(f"Data folder {data_folder} does not exist")

    microbiome_dataframes = []
    target_dataframes = []
    dataset_names = []

    for subdir in data_path.iterdir():
        if subdir.is_dir():
            microbiome_file = subdir / "microbiome.csv"
            target_file = subdir / "target.csv"

            if microbiome_file.exists() and target_file.exists():
                try:
                    microbiome_df = pd.read_csv(microbiome_file, index_col=0)
                    microbiome_df = microbiome_df[(microbiome_df != 0).any(axis=1)]
                    microbiome_df.columns = (
                        microbiome_df.columns
                        .str.replace('|', ';', regex=False)
                        .str.replace('[{}"\\[\\],:]', '_', regex=True)
                        .str.replace(r'\s+', '_', regex=True)
                    )
                    target_df = pd.read_csv(target_file, index_col=0)
                    target_df = target_df.loc[microbiome_df.index]
                    target_df = process_target_labels(target_df)

                    microbiome_dataframes.append(microbiome_df)
                    target_dataframes.append(target_df)
                    dataset_names.append(subdir.name)

                    print(f"Loaded {subdir.name}:")
                    print(f"  Microbiome: {microbiome_df.shape[0]} samples, {microbiome_df.shape[1]} microbes")
                    print(f"  Targets: {len(target_df)} samples, {target_df.value_counts().to_dict()}")

                except Exception as e:
                    print(f"Error loading {subdir}: {e}")

    if not microbiome_dataframes:
        raise ValueError(f"No valid dataset pairs found in {data_folder}")

    return microbiome_dataframes, target_dataframes, dataset_names


def process_target_labels(target_df: pd.DataFrame) -> pd.DataFrame:
    """Process target labels, converting string labels to binary if needed."""
    target_df = target_df.copy()
    unique_tags = target_df['Tag'].unique()
    print(f"  Found unique tags: {unique_tags}")

    label_mapping = {"Study Control": 0, "Control": 0, "CTR": 0, "Case": 1, "HC": 0, "PD": 1}

    if any(isinstance(tag, str) for tag in unique_tags):
        print(f"  Converting string labels using mapping: {label_mapping}")
        target_df['Tag'] = target_df['Tag'].map(lambda x: label_mapping.get(x, x))
        unmapped = target_df['Tag'].isnull().sum()
        if unmapped > 0:
            print(f"  Warning: {unmapped} unmapped labels found, dropping these samples")
            target_df = target_df.dropna(subset=['Tag'])

    target_df['Tag'] = target_df['Tag'].astype(int)

    unique_final = sorted(target_df['Tag'].unique())
    if not all(label in [0, 1] for label in unique_final):
        print(f"  Warning: Non-binary labels found: {unique_final}")
        print("  Mapping to binary: min value -> 0, max value -> 1")
        min_val, max_val = target_df['Tag'].min(), target_df['Tag'].max()
        target_df['Tag'] = target_df['Tag'].map({min_val: 0, max_val: 1})

    return target_df[['Tag']]


def combine_datasets(microbiome_dfs: List[pd.DataFrame], target_dfs: List[pd.DataFrame],
                     dataset_indices: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine multiple datasets into one."""
    combined_microbiome = pd.concat([microbiome_dfs[i] for i in dataset_indices], axis=0)
    combined_targets = pd.concat([target_dfs[i] for i in dataset_indices], axis=0)

    # Align features across datasets (intersection of columns)
    common_features = combined_microbiome.columns
    for i in dataset_indices:
        common_features = common_features.intersection(microbiome_dfs[i].columns)

    combined_microbiome = combined_microbiome[common_features]
    return combined_microbiome, combined_targets


def build_papers_auc_df(csv_path_papers):
    return pd.read_csv(csv_path_papers)



