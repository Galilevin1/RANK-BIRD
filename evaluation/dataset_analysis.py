import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import glob
import re

import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple, Optional, Union




class MicrobiomeAnalysisPipeline:
    """
    A pipeline for analyzing microbiome datasets across different phenotypes.

    This pipeline processes microbiome data along with confounders and targets,
    and generates summary statistics for each project within each phenotype.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the pipeline with the directory containing the data.

        Args:
            data_dir: Path to the directory containing the datasets
        """
        self.data_dir = data_dir
        self.phenotypes = self._get_phenotypes()
        self.results = {}

    def _get_phenotypes(self) -> List[str]:
        """Get list of phenotype directories in the data directory."""
        return [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))]

    def load_data(self) -> Dict:
        """
        Load all data for all phenotypes.

        Returns:
            Dict containing the loaded data organized by phenotype
        """
        data = {}

        for phenotype in self.phenotypes:
            phenotype_dir = os.path.join(self.data_dir, phenotype)

            # Get all project directories
            project_dirs = [d for d in os.listdir(phenotype_dir)
                            if os.path.isdir(os.path.join(phenotype_dir, d))]

            microbiome_dfs = []
            confounder_dfs = []
            target_dfs = []

            for project in project_dirs:
                project_dir = os.path.join(phenotype_dir, project)

                # Load microbiome data
                microbiome_path = os.path.join(project_dir, "microbiome.csv")
                if os.path.exists(microbiome_path):
                    microbiome_df = pd.read_csv(microbiome_path)
                    microbiome_df.columns = [re.sub(r'[^\w\s]', '_', col) for col in microbiome_df.columns]

                    # Ensure "ID" column exists, if not, rename the first column
                    if "ID" not in microbiome_df.columns and microbiome_df.shape[1] > 0:
                        microbiome_df = microbiome_df.rename(columns={microbiome_df.columns[0]: "ID"})

                    microbiome_dfs.append(microbiome_df)

                # Load confounder data
                confounder_path = os.path.join(project_dir, "confounders.csv")
                if os.path.exists(confounder_path):
                    confounder_df = pd.read_csv(confounder_path)

                    # Ensure "ID" column exists, if not, rename the first column
                    if "ID" not in confounder_df.columns and confounder_df.shape[1] > 0:
                        confounder_df = confounder_df.rename(columns={confounder_df.columns[0]: "ID"})

                    confounder_dfs.append(confounder_df)

                # Load target data
                target_path = os.path.join(project_dir, "target.csv")
                if os.path.exists(target_path):
                    target_df = pd.read_csv(target_path)

                    # Ensure "ID" column exists, if not, rename the first column
                    if "ID" not in target_df.columns and target_df.shape[1] > 0:
                        target_df = target_df.rename(columns={target_df.columns[0]: "ID"})

                    # Ensure "Tag" column exists for targets
                    if target_df.shape[1] > 1 and "Tag" not in target_df.columns:
                        # If there's a second column, rename it to "Tag"
                        second_col = target_df.columns[1]
                        target_df = target_df.rename(columns={second_col: "Tag"})

                    target_dfs.append(target_df)

            data[phenotype] = {
                'projects': project_dirs,
                'microbiome_dfs': microbiome_dfs,
                'confounder_dfs': confounder_dfs,
                'target_dfs': target_dfs
            }

        return data

    def analyze_microbiome_data(self, phenotype_data: Dict) -> List[Dict]:
        """
        Analyze microbiome data for a phenotype.

        Args:
            phenotype_data: Dict containing data for a specific phenotype

        Returns:
            List of dictionaries with analysis results for each project
        """
        projects = phenotype_data['projects']
        microbiome_dfs = phenotype_data['microbiome_dfs']
        confounder_dfs = phenotype_data['confounder_dfs']
        target_dfs = phenotype_data['target_dfs']

        results = []

        for i, project in enumerate(projects):
            result = {'project_name': project}

            # Get current project's data
            current_microbiome = microbiome_dfs[i]
            current_confounder = confounder_dfs[i] if i < len(confounder_dfs) else None
            current_target = target_dfs[i] if i < len(target_dfs) else None

            # Analyze microbiome data - pass all target dataframes
            result.update(self._analyze_microbiome(
                current_microbiome,
                microbiome_dfs,
                i,
                current_target,
                target_dfs  # Pass all target dataframes for control/test comparisons
            ))

            # Analyze confounder data if available
            if current_confounder is not None:
                result.update(self._analyze_confounders(current_confounder))

            results.append(result)

        return results

    def _analyze_microbiome(self, current_df: pd.DataFrame, all_dfs: List[pd.DataFrame],
                            current_idx: int, target_df: Optional[pd.DataFrame] = None,
                            all_target_dfs: Optional[List[pd.DataFrame]] = None) -> Dict:
        """
        Analyze microbiome data for a project.

        Args:
            current_df: Microbiome data for the current project
            all_dfs: List of all microbiome data frames for the phenotype
            current_idx: Index of the current project
            target_df: Target data frame for the current project
            all_target_dfs: List of all target data frames for the phenotype

        Returns:
            Dict with analysis results
        """
        features = current_df.shape[1]
        samples = current_df.shape[0]

        # Create a combined dataframe of all other projects
        other_dfs = [df for i, df in enumerate(all_dfs) if i != current_idx]

        current_microbes = set(current_df.columns)

        # If there are no other projects, set comparison metrics to N/A
        if not other_dfs:
            return {
                'sample_count': samples,
                'feature_count': features,
                'microbes_in_current_not_in_others_pct': "N/A",
                'microbes_in_others_not_in_current_pct': "N/A",
                'similarity_score_all': "N/A",
                'similarity_score_control': "N/A",
                'similarity_score_test': "N/A"
            }

        # Combine all other dataframes
        combined_others = pd.concat(other_dfs, ignore_index=True)
        other_microbes = set(combined_others.columns)

        # Calculate percentages of microbes in current not in others and vice versa
        in_current_not_in_others = len(current_microbes - other_microbes)
        in_current_not_in_others_pct = (in_current_not_in_others / len(
            current_microbes)) * 100 if current_microbes else 0

        in_others_not_in_current = len(other_microbes - current_microbes)
        in_others_not_in_current_pct = (in_others_not_in_current / len(other_microbes)) * 100 if other_microbes else 0

        # Similarity / dissimilarity scores
        similarity_all   = self._calculate_similarity(current_df, combined_others)
        bray_curtis_all  = self._calculate_bray_curtis(current_df, combined_others)

        # Similarity scores for control and test samples if target data is available
        similarity_control = "N/A"
        similarity_test = "N/A"

        if target_df is not None and all_target_dfs is not None:
            try:
                # Get other target dataframes
                other_target_dfs = [all_target_dfs[i] for i in range(len(all_target_dfs)) if i != current_idx]

                # For current dataset: Filter by control and test samples from its target file
                current_control_samples = target_df[target_df['Tag'] == 0].index.tolist()
                current_test_samples = target_df[target_df['Tag'] == 1].index.tolist()

                current_control = current_df.loc[current_df.index.isin(current_control_samples)]
                current_test = current_df.loc[current_df.index.isin(current_test_samples)]

                # For other datasets: We need to combine all control samples and all test samples
                other_control_dfs = []
                other_test_dfs = []

                # For each other project with a target file
                for other_df, other_target in zip(other_dfs, other_target_dfs):
                    if other_target is None:
                        continue

                    # Get control and test sample IDs for this other project
                    try:
                        other_control_samples = other_target[other_target['Tag'] == 0].index.tolist()
                        other_test_samples = other_target[other_target['Tag'] == 1].index.tolist()

                        # Filter this other project's samples and add to the list
                        if other_control_samples:
                            other_control_df = other_df.loc[other_df.index.isin(other_control_samples)]
                            if not other_control_df.empty:
                                other_control_dfs.append(other_control_df)

                        if other_test_samples:
                            other_test_df = other_df.loc[other_df.index.isin(other_test_samples)]
                            if not other_test_df.empty:
                                other_test_dfs.append(other_test_df)
                    except Exception as e:
                        print(f"Error processing other project target data: {str(e)}")
                        continue

                # Combine all other control and test dataframes
                if other_control_dfs and not current_control.empty:
                    other_control_combined = pd.concat(other_control_dfs, ignore_index=True)
                    print(
                        f"Control samples - Current: {len(current_control)}, Other combined: {len(other_control_combined)}")

                    # Calculate similarity
                    similarity_control = self._calculate_similarity(current_control, other_control_combined)
                    print(f"Control similarity: {similarity_control}")

                if other_test_dfs and not current_test.empty:
                    other_test_combined = pd.concat(other_test_dfs, ignore_index=True)
                    print(f"Test samples - Current: {len(current_test)}, Other combined: {len(other_test_combined)}")

                    # Calculate similarity
                    similarity_test = self._calculate_similarity(current_test, other_test_combined)
                    print(f"Test similarity: {similarity_test}")

            except Exception as e:
                print(f"Error calculating control/test similarity: {str(e)}")
                similarity_control = "N/A"
                similarity_test = "N/A"

        return {
            'sample_count': samples,
            'feature_count': features,
            'microbes_unique_test':  round(in_current_not_in_others_pct, 2) if in_current_not_in_others_pct > 0 else np.nan,
            'microbes_unique_train': round(in_others_not_in_current_pct, 2) if in_others_not_in_current_pct > 0 else np.nan,
            'jaccard_index':         round(similarity_all, 4)    if similarity_all    != "N/A" else "N/A",
            'bray_curtis_lodo':      round(bray_curtis_all, 4)   if bray_curtis_all   != "N/A" else "N/A",
            'jaccard_index_control': round(similarity_control, 4) if similarity_control != "N/A" else "N/A",
            'jaccard_index_case':    round(similarity_test, 4)   if similarity_test   != "N/A" else "N/A",
        }

    def _calculate_bray_curtis(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Bray-Curtis dissimilarity between the mean abundance profiles of two datasets.
        Uses only common features. Returns dissimilarity in [0, 1].
        """
        common_features = list(set(df1.columns) & set(df2.columns))
        if not common_features:
            return "N/A"
        mean1 = df1[common_features].mean(axis=0).values.astype(float)
        mean2 = df2[common_features].mean(axis=0).values.astype(float)
        denominator = mean1.sum() + mean2.sum()
        if denominator == 0:
            return "N/A"
        return float(2.0 * np.minimum(mean1, mean2).sum() / denominator)

    def _calculate_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calculate Jaccard similarity between two dataframes.

        Jaccard similarity is defined as the size of the intersection divided by the size of the union
        of the sample sets. It's especially useful for microbiome data as it focuses on presence/absence
        rather than abundance values.

        Args:
            df1: First dataframe
            df2: Second dataframe

        Returns:
            Jaccard similarity score
        """
        # Ensure we have common features (microbes) to compare
        common_features = set(df1.columns) & set(df2.columns)

        if not common_features:
            print("No common features between train and test")
            return "N/A"

        # Use only common features for comparison
        df1_subset = df1[list(common_features)]
        df2_subset = df2[list(common_features)]

        # For each dataset, calculate the presence (1) or absence (0) of each microbe
        # A microbe is considered present if its abundance is above 0 in any sample
        presence_df1 = (df1_subset > 0).any(axis=0).astype(int)
        presence_df2 = (df2_subset > 0).any(axis=0).astype(int)

        # Calculate intersection and union
        intersection = sum(presence_df1 & presence_df2)
        union = sum(presence_df1 | presence_df2)

        # Calculate Jaccard similarity
        if union > 0:  # Avoid division by zero
            jaccard = intersection / union
        else:
            jaccard = 0.0

        return jaccard


    def _analyze_confounders(self, confounder_df: pd.DataFrame) -> Dict:
        """
        Analyze confounder data for a project, handling non-numeric values.

        Args:
            confounder_df: Confounder data for the project

        Returns:
            Dict with analysis results
        """
        result = {}

        # Process 'Bases' column if it exists
        if 'Bases' in confounder_df.columns:
            try:
                # Convert to numeric, coercing errors to NaN
                bases = pd.to_numeric(confounder_df['Bases'], errors='coerce')
                bases = bases.dropna()

                result['bases_median'] = round(bases.median(), 2) if not bases.empty else "N/A"
                result['bases_min'] = round(bases.min(), 2) if not bases.empty else "N/A"
                result['bases_max'] = round(bases.max(), 2) if not bases.empty else "N/A"
            except Exception as e:
                print(f"Error processing 'Bases' column: {str(e)}")
                result['bases_median'] = "N/A"
                result['bases_min'] = "N/A"
                result['bases_max'] = "N/A"

        # Process 'country' column if it exists
        if 'country' in confounder_df.columns:
            try:
                countries = confounder_df['country'].dropna()
                result['countries'] = ', '.join(countries.unique()) if not countries.empty else "N/A"
            except Exception as e:
                print(f"Error processing 'country' column: {str(e)}")
                result['countries'] = "N/A"

        # Process 'age' column if it exists
        if 'age' in confounder_df.columns:
            try:
                # Convert to numeric, coercing errors (like 'not provided') to NaN
                age = pd.to_numeric(confounder_df['age'], errors='coerce')
                age = age.dropna()

                result['age_median'] = round(age.median(), 2) if not age.empty else "N/A"
                result['age_min'] = round(age.min(), 2) if not age.empty else "N/A"
                result['age_max'] = round(age.max(), 2) if not age.empty else "N/A"
            except Exception as e:
                print(f"Error processing 'age' column: {str(e)}")
                result['age_median'] = "N/A"
                result['age_min'] = "N/A"
                result['age_max'] = "N/A"

        # Process 'sex' column if it exists
        if 'sex' in confounder_df.columns:
            try:
                # Filter out NaN values
                sex_data = confounder_df['sex'].dropna()

                if not sex_data.empty:
                    # Convert to lowercase for consistency and check for variations of male representation
                    sex_lower = sex_data.str.lower()
                    male_count = sex_lower.isin(['m', 'male', '1']).sum()
                    total_count = len(sex_lower)

                    result['male_percentage'] = round((male_count / total_count) * 100, 2) if total_count > 0 else "N/A"
                else:
                    result['male_percentage'] = "N/A"
            except Exception as e:
                print(f"Error processing 'sex' column: {str(e)}")
                result['male_percentage'] = "N/A"

        # Process 'BMI' column if it exists (checking for different capitalizations)
        bmi_col = next((col for col in confounder_df.columns if col.lower() == 'bmi'), None)
        if bmi_col:
            try:
                # Convert to numeric, coercing errors to NaN
                bmi = pd.to_numeric(confounder_df[bmi_col], errors='coerce')
                bmi = bmi.dropna()

                result['bmi_median'] = round(bmi.median(), 2) if not bmi.empty else "N/A"
                result['bmi_min'] = round(bmi.min(), 2) if not bmi.empty else "N/A"
                result['bmi_max'] = round(bmi.max(), 2) if not bmi.empty else "N/A"
            except Exception as e:
                print(f"Error processing '{bmi_col}' column: {str(e)}")
                result['bmi_median'] = "N/A"
                result['bmi_min'] = "N/A"
                result['bmi_max'] = "N/A"

        return result

    def run_analysis(self) -> pd.DataFrame:
        """
        Run analysis for all phenotypes and generate a summary dataframe.

        Returns:
            DataFrame with analysis results
        """
        all_data = self.load_data()
        all_results = []

        for phenotype, phenotype_data in all_data.items():
            phenotype_results = self.analyze_microbiome_data(phenotype_data)

            # Add phenotype to each result
            for result in phenotype_results:
                result['phenotype'] = phenotype
                all_results.append(result)

        # Convert to dataframe
        results_df = pd.DataFrame(all_results)

        # Reorder columns to put phenotype and project_name first
        cols = ['phenotype', 'project_name'] + [col for col in results_df.columns
                                                if col not in ['phenotype', 'project_name']]
        results_df = results_df[cols]

        return results_df

    def save_results(self, output_path: str) -> None:
        """
        Save analysis results to a CSV file.

        Args:
            output_path: Path to save the CSV file
        """
        results_df = self.run_analysis()
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


class MicrobiomeLearningModel:
    """
    A class for training and evaluating LightGBM models on microbiome data.
    Supports leave-one-dataset-out cross-validation and single dataset evaluation.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the model with a random seed for reproducibility.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        # Modified LightGBM parameters to be compatible with basic training
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': random_state,  # Changed from random_state to seed
        }

    def prepare_data(self, microbiome_dfs: List[pd.DataFrame],
                     target_dfs: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Prepare data for modeling by extracting features and labels.

        Args:
            microbiome_dfs: List of microbiome dataframes
            target_dfs: List of target dataframes

        Returns:
            Tuple of (feature_dfs, label_dfs)
        """
        feature_dfs = []
        label_dfs = []

        for i, (microbiome_df, target_df) in enumerate(zip(microbiome_dfs, target_dfs)):
            # Join microbiome data with target data on index
            merged_df = microbiome_df.join(target_df, how='inner')

            # Extract features (all columns except Tag)
            features = merged_df.drop(columns=['Tag'])

            # Extract labels
            labels = merged_df['Tag']

            # Store in lists
            feature_dfs.append(features)
            label_dfs.append(labels)

        return feature_dfs, label_dfs

    def leave_one_dataset_out_cv(self, microbiome_dfs: List[pd.DataFrame],
                                 target_dfs: List[pd.DataFrame],
                                 project_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform leave-one-dataset-out cross-validation.

        Args:
            microbiome_dfs: List of microbiome dataframes
            target_dfs: List of target dataframes
            project_names: List of project names

        Returns:
            Dictionary with AUC scores for each project
        """
        # Prepare data - this will also clean feature names
        feature_dfs, label_dfs = self.prepare_data(microbiome_dfs, target_dfs)

        # Results dictionary
        results = {}

        # Perform leave-one-dataset-out CV
        for i in range(len(feature_dfs)):
            test_project = project_names[i]
            test_features = feature_dfs[i]
            test_labels = label_dfs[i]

            # Skip if test dataset is too small
            if len(test_labels) < 10:
                results[test_project] = {'leave_one_out_auc': np.nan}
                continue

            # Combine all other datasets for training
            train_features_dfs = [feature_dfs[j] for j in range(len(feature_dfs)) if j != i]
            train_labels_dfs = [label_dfs[j] for j in range(len(label_dfs)) if j != i]

            # Skip if no training data available
            if not train_features_dfs:
                results[test_project] = {'leave_one_out_auc': np.nan}
                continue

            try:
                # Find common features between all training datasets and test dataset
                common_features = list(set.intersection(
                    *[set(df.columns) for df in train_features_dfs + [test_features]]
                ))

                # Skip if no common features
                if not common_features:
                    results[test_project] = {'leave_one_out_auc': np.nan}
                    continue

                # Use only common features for all datasets
                train_features_filtered = [df[common_features] for df in train_features_dfs]

                # Concatenate training data
                train_features = pd.concat(train_features_filtered, axis=0)
                train_labels = pd.concat(train_labels_dfs, axis=0)

                # Use only common features for test data
                test_features = test_features[common_features]

                # Verify there are both positive and negative samples in training data
                if len(np.unique(train_labels)) < 2:
                    results[test_project] = {'leave_one_out_auc': np.nan}
                    continue

                # Train model - FIXED: removed early_stopping_rounds parameter
                lgb_train = lgb.Dataset(train_features, train_labels)
                model = lgb.train(self.lgb_params, lgb_train)

                # Make predictions
                y_pred = model.predict(test_features)

                # Calculate AUC
                if len(np.unique(
                        test_labels)) > 1:  # Only calculate AUC if there are both positive and negative samples
                    auc = roc_auc_score(test_labels, y_pred)
                    results[test_project] = {'leave_one_out_auc': auc}
                else:
                    results[test_project] = {'leave_one_out_auc': np.nan}

            except Exception as e:
                print(f"Error in leave-one-dataset-out CV for {test_project}: {str(e)}")
                results[test_project] = {'leave_one_out_auc': np.nan}

        return results

    def single_dataset_cv(self, microbiome_dfs: List[pd.DataFrame],
                          target_dfs: List[pd.DataFrame],
                          project_names: List[str],
                          test_size: float = 0.2,
                          val_size: float = 0.1,
                          n_folds: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on each individual dataset.

        Args:
            microbiome_dfs: List of microbiome dataframes
            target_dfs: List of target dataframes
            project_names: List of project names
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            n_folds: Number of cross-validation folds

        Returns:
            Dictionary with AUC scores for each project
        """
        # Prepare data - this will also clean feature names
        feature_dfs, label_dfs = self.prepare_data(microbiome_dfs, target_dfs)

        # Results dictionary
        results = {}

        # Perform CV on each dataset
        for i, (features, labels, project) in enumerate(zip(feature_dfs, label_dfs, project_names)):
            # Skip if dataset is too small
            if len(labels) < 20:  # Need enough samples for train/val/test split
                results[project] = {'single_dataset_auc': np.nan}
                continue

            # Check if there are both positive and negative samples
            if len(np.unique(labels)) < 2:
                results[project] = {'single_dataset_auc': np.nan}
                continue

            try:
                # Split data into train+val and test
                X_trainval, X_test, y_trainval, y_test = train_test_split(
                    features, labels, test_size=test_size, random_state=self.random_state, stratify=labels
                )

                # Further split train+val into train and val
                effective_val_size = val_size / (1 - test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, test_size=effective_val_size,
                    random_state=self.random_state, stratify=y_trainval
                )

                # Initialize arrays for storing predictions
                cv_aucs = []

                # Perform k-fold CV on training data
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

                for train_idx, val_idx in skf.split(X_train, y_train):
                    # Get train and validation data for this fold
                    fold_X_train = X_train.iloc[train_idx]
                    fold_y_train = y_train.iloc[train_idx]
                    fold_X_val = X_train.iloc[val_idx]
                    fold_y_val = y_train.iloc[val_idx]

                    # Skip if not enough samples of each class
                    if len(np.unique(fold_y_train)) < 2 or len(np.unique(fold_y_val)) < 2:
                        continue

                    # Train model
                    lgb_train = lgb.Dataset(fold_X_train, fold_y_train)
                    lgb_val = lgb.Dataset(fold_X_val, fold_y_val)

                    # Removed both early_stopping_rounds and verbose_eval parameters
                    model = lgb.train(
                        self.lgb_params,
                        lgb_train,
                        valid_sets=[lgb_val]
                    )

                    # Make predictions on validation set
                    val_preds = model.predict(fold_X_val)

                    # Calculate AUC
                    if len(np.unique(fold_y_val)) > 1:
                        fold_auc = roc_auc_score(fold_y_val, val_preds)
                        cv_aucs.append(fold_auc)

                # Calculate mean AUC across folds
                mean_cv_auc = np.mean(cv_aucs) if cv_aucs else np.nan

                # Train final model on all training data
                lgb_train_final = lgb.Dataset(X_trainval, y_trainval)
                final_model = lgb.train(self.lgb_params, lgb_train_final)

                # Make predictions on test set
                test_preds = final_model.predict(X_test)

                # Calculate test AUC
                if len(np.unique(y_test)) > 1:
                    test_auc = roc_auc_score(y_test, test_preds)
                else:
                    test_auc = np.nan

                # Store results
                results[project] = {
                    'single_dataset_auc': test_auc,
                    'cv_auc': mean_cv_auc
                }

            except Exception as e:
                print(f"Error in single dataset CV for {project}: {str(e)}")
                results[project] = {
                    'single_dataset_auc': np.nan,
                    'cv_auc': np.nan
                }

        return results

    def internal_validation_cv(self, microbiome_dfs: List[pd.DataFrame],
                               target_dfs: List[pd.DataFrame],
                               project_names: List[str],
                               test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Internal validation: for each dataset context, train on all other datasets combined
        and evaluate on a held-out portion of that combined training set.
        """
        feature_dfs, label_dfs = self.prepare_data(microbiome_dfs, target_dfs)
        results = {}

        for i, project in enumerate(project_names):
            train_feature_dfs = [feature_dfs[j] for j in range(len(feature_dfs)) if j != i]
            train_label_dfs   = [label_dfs[j]   for j in range(len(label_dfs))   if j != i]

            if not train_feature_dfs:
                results[project] = {'internal_validation_auc': np.nan}
                continue

            try:
                common_features = list(set.intersection(*[set(df.columns) for df in train_feature_dfs]))
                if not common_features:
                    results[project] = {'internal_validation_auc': np.nan}
                    continue

                X_train_all = pd.concat([df[common_features] for df in train_feature_dfs], axis=0)
                y_train_all = pd.concat(train_label_dfs, axis=0)

                if len(np.unique(y_train_all)) < 2:
                    results[project] = {'internal_validation_auc': np.nan}
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X_train_all, y_train_all,
                    test_size=test_size, random_state=self.random_state, stratify=y_train_all
                )

                lgb_train = lgb.Dataset(X_train, y_train)
                model = lgb.train(self.lgb_params, lgb_train)
                y_pred = model.predict(X_test)

                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, y_pred)
                    results[project] = {'internal_validation_auc': auc}
                else:
                    results[project] = {'internal_validation_auc': np.nan}

            except Exception as e:
                print(f"Error in internal validation CV for {project}: {str(e)}")
                results[project] = {'internal_validation_auc': np.nan}

        return results

    def run_all_evaluations(self, phenotype_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Run both leave-one-dataset-out and single dataset evaluations.

        Args:
            phenotype_data: Dictionary containing data for a phenotype

        Returns:
            Dictionary with results for each project
        """
        projects = phenotype_data['projects']
        microbiome_dfs = phenotype_data['microbiome_dfs']
        target_dfs = phenotype_data['target_dfs']

        # Check if we have enough datasets
        if len(projects) < 2 or len(microbiome_dfs) < 2 or len(target_dfs) < 2:
            return {project: {'leave_one_out_auc': np.nan, 'internal_validation_auc': np.nan,
                              'within_dataset_auc': np.nan, 'auc_difference': np.nan}
                    for project in projects}

        leave_one_out_results      = self.leave_one_dataset_out_cv(microbiome_dfs, target_dfs, projects)
        internal_validation_results = self.internal_validation_cv(microbiome_dfs, target_dfs, projects)
        single_dataset_results     = self.single_dataset_cv(microbiome_dfs, target_dfs, projects)

        combined_results = {}
        for project in projects:
            lodo_auc     = leave_one_out_results.get(project, {}).get('leave_one_out_auc', np.nan)
            int_val_auc  = internal_validation_results.get(project, {}).get('internal_validation_auc', np.nan)
            within_auc   = single_dataset_results.get(project, {}).get('single_dataset_auc', np.nan)

            if not np.isnan(within_auc) and not np.isnan(lodo_auc):
                auc_difference = within_auc - lodo_auc
            else:
                auc_difference = np.nan

            combined_results[project] = {
                'leave_one_out_auc':       lodo_auc,
                'internal_validation_auc': int_val_auc,
                'within_dataset_auc':      within_auc,
                'auc_difference':          auc_difference,
            }

        return combined_results


def integrate_with_pipeline(pipeline):
    """
    Integrate the learning model with the microbiome analysis pipeline.

    Args:
        pipeline: Instance of MicrobiomeAnalysisPipeline

    Returns:
        Modified pipeline with learning capabilities
    """
    # Store the original analyze_microbiome_data method
    original_analyze_method = pipeline.analyze_microbiome_data

    # Define a new method that adds learning results
    def analyze_with_learning(phenotype_data):
        # Call the original method
        results = original_analyze_method(phenotype_data)

        # Run learning evaluations
        model = MicrobiomeLearningModel()
        learning_results = model.run_all_evaluations(phenotype_data)

        # Integrate learning results into the original results
        for i, result in enumerate(results):
            project_name = result['project_name']
            if project_name in learning_results:
                # Add learning metrics
                result['leave_one_out_auc'] = round(learning_results[project_name]['leave_one_out_auc'], 4) \
                    if not np.isnan(learning_results[project_name]['leave_one_out_auc']) else "N/A"
                result['single_dataset_auc'] = round(learning_results[project_name]['single_dataset_auc'], 4) \
                    if not np.isnan(learning_results[project_name]['single_dataset_auc']) else "N/A"
                result['auc_difference'] = round(learning_results[project_name]['auc_difference'], 4) \
                    if not np.isnan(learning_results[project_name]['auc_difference']) else "N/A"
            else:
                # Set to N/A if no learning results available
                result['leave_one_out_auc'] = "N/A"
                result['single_dataset_auc'] = "N/A"
                result['auc_difference'] = "N/A"

        return results

    # Replace the original method with the enhanced one
    pipeline.analyze_microbiome_data = analyze_with_learning

    return pipeline

def run_dataset_analysis(
    microbiome_dfs: list,
    target_dfs: list,
    dataset_names: list,
    dataset_to_phenotype: dict,
    output_path: str,
    confounder_dfs: list = None,
) -> pd.DataFrame:
    """
    Compute dataset-level statistics and AUC metrics from already-loaded data.
    confounder_dfs: per-dataset list (or None) as returned by
                   load_microbiome_datasets_with_targets(..., include_confounders=True).
    Saves results to output_path and returns the DataFrame.
    """
    if confounder_dfs is None:
        confounder_dfs = [None] * len(dataset_names)

    # Group indices by phenotype
    pheno_groups: dict = {}
    for i, name in enumerate(dataset_names):
        pheno = dataset_to_phenotype[name]
        pheno_groups.setdefault(pheno, []).append(i)

    pipe  = MicrobiomeAnalysisPipeline.__new__(MicrobiomeAnalysisPipeline)
    model = MicrobiomeLearningModel()
    all_results = []

    for pheno, idxs in pheno_groups.items():
        mb_dfs   = [microbiome_dfs[i]  for i in idxs]
        tgt_dfs  = [target_dfs[i]      for i in idxs]
        names    = [dataset_names[i]   for i in idxs]
        conf_dfs = [confounder_dfs[i]  for i in idxs]

        # Per-dataset microbiome + confounder analysis
        for i, (mb, tgt, name) in enumerate(zip(mb_dfs, tgt_dfs, names)):
            result = {'phenotype': pheno, 'project_name': name}
            result.update(pipe._analyze_microbiome(mb, mb_dfs, i, tgt, tgt_dfs))
            if conf_dfs[i] is not None:
                result.update(pipe._analyze_confounders(conf_dfs[i]))
            all_results.append(result)

        # Learning evaluation (LODO + single-dataset CV)
        learning_results = model.run_all_evaluations({
            'projects':       names,
            'microbiome_dfs': mb_dfs,
            'target_dfs':     tgt_dfs,
        })

        for result in all_results:
            if result['phenotype'] != pheno:
                continue
            lr = learning_results.get(result['project_name'], {})
            for key in ['leave_one_out_auc', 'internal_validation_auc', 'within_dataset_auc', 'auc_difference']:
                val = lr.get(key, np.nan)
                try:
                    result[key] = round(float(val), 4) if not np.isnan(float(val)) else "N/A"
                except (TypeError, ValueError):
                    result[key] = "N/A"

    results_df = pd.DataFrame(all_results)
    cols = ['phenotype', 'project_name'] + [c for c in results_df.columns
                                             if c not in ('phenotype', 'project_name')]
    results_df = results_df[cols]
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


# Example usage
if __name__ == "__main__":
    # Directory structure should be:
    # data_dir/
    #   phenotype1/
    #     project1/
    #       microbiome.csv
    #       confounders.csv
    #       target.csv
    #     project2/
    #       ...
    #   phenotype2/
    #     ...

    data_dir = "Phenotype_Data"
    pipeline = MicrobiomeAnalysisPipeline(data_dir)

    # Integrate learning capabilities
    pipeline = integrate_with_pipeline(pipeline)

    # Run analysis and save results
    pipeline.save_results("microbiome_analysis_results.csv")