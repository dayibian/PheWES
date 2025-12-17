import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline

from plotting import (
    plot_feature_importances, 
    compute_permutation_importance, 
    plot_permutation_importance, 
    plot_CM, 
    plot_ROC, 
    plot_precision_recall
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is not installed. Please install xgboost to use XG model.")

import matplotlib.pyplot as plt

import joblib
import gzip
from tqdm import tqdm
import yaml

import logging, sys, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True

try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Try to open config.yaml in the current directory
    with open("config.yaml") as f:
        config = yaml.safe_load(f)


def setup_log(fn_log: Union[str, Path], mode: str = 'w') -> None:
    '''
    Print log message to console and write to a log file.
    Will overwrite existing log file by default
    Params:
    - fn_log: name of the log file
    - mode: writing mode. Change mode='a' for appending
    '''
    # Remove any existing handlers to avoid duplicate logs
    logging.root.handlers = [] # Remove potential handler set up by others (especially in google colab)
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(filename=fn_log, mode=mode),
                                  logging.StreamHandler()], format='%(message)s')

def process_args() -> argparse.Namespace:
    '''
    Process command-line arguments and set up logging.
    Returns:
        argparse.Namespace: Parsed arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='High-level folder that contains data for each trait', type=str,
                        default='../data/')
    parser.add_argument('--output_folder', help='High-level folder that contains output for each trait', type=str,
                        default='../results/')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='als')
    parser.add_argument('--output_prefix', type=str, default='output')
    parser.add_argument('--model_type', type=str, default='CART')
    parser.add_argument('--matched_controls_for_ML', type=int, default=1)
    parser.add_argument('--n_controls_per_case', type=int, default=5, 
                        help='Number of controls to use per case')
    parser.add_argument('--use_smoten', type=int, default=0,
                        help='Whether to use SMOTEN for class balancing')
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.output_folder) / f'_PheML_{args.trait}.log'
    setup_log(fn_log, mode='a')

    # Record script used
    cmd_used = 'python ' + ' '.join(sys.argv)

    logging.info('\n# Call used:')
    logging.info(cmd_used+'\n')
    
    logging.info('# Arguments used:')
    for arg in vars(args):
        cmd_used += f' --{arg} {getattr(args, arg)}'
        msg = f'# - {arg}: {getattr(args, arg)}'
        logging.info(msg)

    return args

def get_phecode_features(
    output_path: Path,
    trait: str,
    prefix: str,
    number_of_cases: int,
    phecode_map: pd.DataFrame,
    cutoff_frequency: float = 0.02
) -> List[str]:
    '''
    Get enriched phecodes, drop those used for phenotyping.
    Args:
        output_path (Path): Path to output directory, including the enriched phecode file
        trait (str): Trait of interest
        prefix (str): Prefix for the output file
        number_of_cases (int): Number of cases for the enrichment analysis
        phecode_map (pd.DataFrame): Phecode map with columns including 'ICD' and 'Phecode'
        cutoff_frequency (float): Cutoff frequency for the enriched phecodes, default is 0.02
    Returns:
        List[str]: List of phecode features (excluding those used for phenotyping)
    '''
    
    # Get excluded codes from config
    excluded_code = []
    
    if 'excluded_codes' in config:
        excluded_codes_config = config['excluded_codes']
        
        # Get ICD codes from config and convert to Phecodes using phecode_map
        if 'ICD' in excluded_codes_config and excluded_codes_config['ICD']:
            icd_codes_to_exclude = [str(code).strip() for code in excluded_codes_config['ICD']]
            # Filter phecode_map to find matching Phecodes for the ICD codes
            # Convert ICD column to string for matching
            phecode_map_icd_str = phecode_map['ICD'].astype(str).str.strip()
            icd_matches = phecode_map[phecode_map_icd_str.isin(icd_codes_to_exclude)]
            # Get unique Phecodes and strip whitespace
            phecodes_from_icd = icd_matches['Phecode'].astype(str).str.strip().unique().tolist()
            excluded_code.extend(phecodes_from_icd)
        
        # Get Phecode list directly from config
        if 'Phecode' in excluded_codes_config and excluded_codes_config['Phecode']:
            phecodes_to_exclude = [str(code).strip() for code in excluded_codes_config['Phecode']]
            excluded_code.extend(phecodes_to_exclude)
    
    # Get enriched phecode
    enrich_results = pd.read_csv(output_path / f'{trait}_{prefix}_enriched_phecode.csv', sep='\t', dtype={'Phecode':str})
    enrich_results = enrich_results[enrich_results.Count > number_of_cases * cutoff_frequency] # Remove those phecodes that has counts less than the cutoff frequency of case number, regardless of significance
    phecode_features = enrich_results.Phecode.astype(str).unique().tolist()
    
    excluded_set = set(excluded_code)
    phecode_features_ = [code for code in phecode_features if code not in excluded_set] if excluded_set else phecode_features
    return phecode_features_


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'RF',
    use_smoten: bool = False,
    random_state: int = 42,
    verbose: int = 2,
    n_jobs: int = -1
) -> Any:
    '''
    Train a machine learning model with hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_type (str): 'CART', 'RF', 'XG', or 'NN'/'MLP'
        use_smoten (bool): Whether to use SMOTEN for class balancing
        random_state (int): Random seed
        verbose (int): Verbosity level
        n_jobs (int): Number of parallel jobs
    Returns:
        Any: The best trained model
    '''
    if model_type.upper() == 'CART':
        m = X_train.shape[1]
        param_dist = {
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': randint(max(1, m // 2), m)
        }
        base_model = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
        n_iter = 10
    elif model_type.upper() == 'RF':
        param_dist = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        base_model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        n_iter = 50
    elif model_type.upper() == 'XG':
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'subsample': [0.7, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        base_model = XGBClassifier(eval_metric='logloss', random_state=random_state, use_label_encoder=False)
        n_iter = 20
    elif model_type.upper() in ['NN', 'MLP']:
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        }
        base_model = MLPClassifier(max_iter=500, random_state=random_state)
        n_iter = 20
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from 'CART', 'RF', 'XG', or 'NN'/'MLP'.")

    if use_smoten:
        pipeline = Pipeline([
            ('smote', SMOTEN(random_state=random_state)),
            ('model', base_model)
        ])
        param_dist = {'model__' + k: v for k, v in param_dist.items()}
    else:
        pipeline = base_model
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs
    )

    random_search.fit(X_train, y_train)
    if use_smoten:
        best_model = random_search.best_estimator_['model']
    else:
        best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)
    return best_model

def get_cases_and_controls(
    pair_file: Union[str, Path],
    potential_controls: list,
    n_controls_per_case: int = 5,
    use_matched_controls: bool = True
) -> Tuple[List[Any], List[Any]]:
    """
    Reads a case-control pair file and returns lists of case and control IDs.
    Args:
        pair_file (str or Path): Path to the case-control pairs file.
        potential_controls (list): List of unmatched control IDs. Required if use_matched_controls is False.
        n_controls_per_case (int): Number of controls to use per case (max is the number of control columns in the file or number to sample from potential_controls).
        use_matched_controls (bool): Whether to use the controls from the matched control set. If False, use potential_controls.
    Returns:
        cases (list): List of case IDs.
        controls (list): List of unique control IDs (from matched controls or randomly sampled from potential_controls).
    """
    import random

    df = pd.read_csv(pair_file, sep='\t')
    cases = df['case'].dropna().tolist()
    if use_matched_controls:
        # Get only the first n_controls_per_case control columns
        control_cols = [col for col in df.columns if col.startswith('Control')][:n_controls_per_case]
        controls = pd.unique(df[control_cols].values.ravel('K'))
        controls = [c for c in controls if pd.notnull(c)]
    else:
        if potential_controls is None or not isinstance(potential_controls, list) or len(potential_controls) == 0:
            raise ValueError("unmatched_controls must be provided as a non-empty list when use_matched_controls is False.")
        # Sample n_controls_per_case * number of cases, or the max available if not enough
        unmatched_controls = list(set(potential_controls) - set(cases))
        n_controls_total = min(n_controls_per_case * len(cases), len(unmatched_controls))
        controls = random.sample(unmatched_controls, n_controls_total)
    return cases, controls

def main() -> None:
    '''
    Main function to orchestrate data preparation, model training, evaluation, and saving.
    '''
    args = process_args()
    trait = args.trait
    output_path = Path(args.output_folder)
    prefix = args.output_prefix
    model_type = args.model_type
    use_matched_controls = args.matched_controls_for_ML

    # Import case control and corresponding phecodes
    logging.info('Preparing data for model development...')

    # Use Polars for efficient loading (lazy)
    logging.info('Loading phecode data with Polars (lazy mode)...')
    sd_phecode_lazy = pl.scan_ipc(config['phecode_binary_feather_file'])
    all_sd_grids = sd_phecode_lazy.select('grid').collect()['grid'].to_list()
    
    case_grid, control_grid = get_cases_and_controls(output_path / f'case_control_pairs_{prefix}.txt', 
                                                     potential_controls=all_sd_grids, 
                                                     n_controls_per_case=args.n_controls_per_case,
                                                     use_matched_controls=use_matched_controls
                                                     )
    number_of_cases = len(case_grid)

    # Optimization: Load only relevant samples (cases + controls) into memory
    all_relevant_ids = list(set(case_grid) | set(control_grid))
    logging.info(f'Loading {len(all_relevant_ids)} relevant samples (cases + controls)...')
    sd_phecode_subset = (sd_phecode_lazy
                         .filter(pl.col('grid').is_in(all_relevant_ids))
                         .collect()
                         .to_pandas())  # Convert to pandas for sklearn compatibility

    # Generate dataframe for case and control, add labels, and merge them
    case_df = sd_phecode_subset[sd_phecode_subset.grid.isin(case_grid)].copy()
    case_df['label'] = 1
    control_df = sd_phecode_subset[sd_phecode_subset.grid.isin(control_grid)].copy()
    control_df['label'] = 0
    data = pd.concat([case_df, control_df], ignore_index=True)
    # Ensure all feature columns are strings
    # feature_cols = [col for col in data.columns if col not in ['grid', 'label']]
    # data[feature_cols] = data[feature_cols].astype(str)
    # print(data.head())

    phecode_map = pd.read_csv(config['phecode_map_file'], dtype={'Phecode':str})
    phecode_features_ = get_phecode_features(output_path, trait, prefix, number_of_cases*0.8, phecode_map)
    
    if not phecode_features_:
        logging.info("No enriched phecode found. Model training will be skipped.")
        # Create a fake model file to satisfy Snakemake
        with open(output_path / f'PheML_{model_type}_{prefix}.model', 'w') as f:
            f.write("No enriched phecode found. Model training skipped.")
        return

    data[['grid']+phecode_features_+['label']].to_csv(output_path / f'{prefix}_data_for_ML.csv', index=False)
    # print(phecode_features_)

    def get_all_grids(case_control_file):
        df = pd.read_csv(case_control_file, sep='\t')
        case_grids = set(df['case'].dropna().tolist())
        control_cols = [col for col in df.columns if col.startswith('Control')]
        control_grids = set([g for g in pd.unique(df[control_cols].values.ravel()) if pd.notna(g)])
        all_grids = case_grids | control_grids
        return list(all_grids)

    train_grids = get_all_grids(output_path / f'case_control_pairs_{prefix}_train.txt')
    test_grids = get_all_grids(output_path / f'case_control_pairs_{prefix}_test.txt')

    # X_train, X_test, y_train, y_test = train_test_split(data[phecode_features_], data.label, train_size=0.8,
    #                                                     random_state=2024, stratify=data.label)
    train_data = data[data.grid.isin(train_grids)]
    test_data = data[data.grid.isin(test_grids)]
    X_train, y_train = train_data[phecode_features_], train_data.label
    X_test, y_test = test_data[phecode_features_], test_data.label

    logging.info('Training the model...')
    final_model = train_model(X_train, y_train, model_type=model_type, use_smoten=args.use_smoten)

    logging.info('Reading phecode map...')
    
    phecode_map = phecode_map[['Phecode', 'PhecodeString']].drop_duplicates(ignore_index=True)
    phecode_map.Phecode = phecode_map.Phecode.apply(lambda x: x.strip())
    phecode_map.index = phecode_map.Phecode
    phecode_map.drop(columns=['Phecode'], inplace=True)
    phecode_map = phecode_map.to_dict()
    phecode_map = phecode_map['PhecodeString']

    logging.info('Plotting model results...')
    # Call the feature importance plotting function (for models that support it)

    plot_feature_importances(final_model, X_train, output_path, prefix, n_top=10, phecode_map=phecode_map)
    
    # Compute and plot permutation importance for all model types
    perm_results = compute_permutation_importance(
        final_model, X_test, y_test, output_path, prefix, model_type, 
        n_repeats=10, random_state=42, n_jobs=-1
    )
    plot_permutation_importance(perm_results, output_path, prefix, model_type, 
                               n_top=15, phecode_map=phecode_map)
    
    precision = plot_CM(final_model, X_test, y_test, output_path, model_type, trait, prefix)
    auc = plot_ROC(final_model, X_test, y_test, output_path, trait, model_type, prefix)
    _ = plot_precision_recall(final_model, X_test, y_test, output_path, trait, model_type, prefix)
    logging.info(f'Precision is: {precision:.2f}')
    logging.info(f'AUC is: {auc:.2f}')

    logging.info('Saving model...')
    joblib.dump(final_model, output_path / f'PheML_{model_type}_{prefix}.model')
    logging.info('Done. Model building completed.')

if __name__ == '__main__':
    main()