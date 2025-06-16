import pandas as pd
import numpy as np
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import joblib
import gzip
from tqdm import tqdm

import logging, sys, argparse
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True

full_name = {
    'als': 'ALS',
    'ftld': 'FTLD',
    'vasc_dementia': 'Vascular Dementia',
    'lewy_body': 'Lewy Body Dementia'
}

def setup_log(fn_log, mode='w'):
    '''
    Print log message to console and write to a log file.
    Will overwrite existing log file by default
    Params:
    - fn_log: name of the log file
    - mode: writing mode. Change mode='a' for appending
    '''
    # f string is not fully compatible with logging, so use %s for string formatting
    logging.root.handlers = [] # Remove potential handler set up by others (especially in google colab)
    logging.basicConfig(level=logging.DEBUG,
                        handlers=[logging.FileHandler(filename=fn_log, mode=mode),
                                  logging.StreamHandler()], format='%(message)s')

def process_args():
    '''
    Process arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='High-level folder that contains data for each trait', type=str,
                        default='/data100t1/home/biand/Projects/Comorbidity_analysis/data/')
    parser.add_argument('--output_folder', help='High-level folder that contains output for each trait', type=str,
                        default='/data100t1/home/biand/Projects/Comorbidity_analysis/output/')
    parser.add_argument('--trait', help='Trait of interest', type=str, default='als')
    parser.add_argument('--output_prefix', type=str, default='output')
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.data_folder) / f'{args.trait}/{args.trait}.log'
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

def get_phecode_features(data_path, output_path, trait):
    '''
    Get enriched phecodes, drop those used for phenotyping.
    '''
    with gzip.open(data_path / f'{trait}/{trait}_codes_and_dates.csv.gz') as f:
        case_codes = pd.read_csv(f, dtype={'phecode':str})
    
    icd_codes = {
        'als': ['G12.21', 'G12.20', 'G12.24', 'G12.29', '335.20', '335.21', '335.29'],
        'ftld': ['G31.01', 'G31.09', 'G21.1', 'G31.85', '331.11', '331.19', '331.6'],
        'vasc_dementia': ['F01.50', 'F01.51', 'F01.511', 'F01.518', '290.40', '290.41'],
        'lewy_body': ['G31.83', 'G20', 'F02.80', '331.82', '332.0', '294.10']
    }
    case_codes_ = case_codes[case_codes.concept_code.isin(icd_codes[trait])]
    excluded_code = case_codes_.phecode.unique()
    
    # Get enriched phecode
    enrich_results = pd.read_csv(output_path / f'{trait}/{trait}_enriched_phecode.csv', sep='\t', dtype={'Phecode':str})
    phecode_features = list(enrich_results.Phecode)
    
    phecode_features_ = phecode_features[:]
    for code in excluded_code:
        phecode_features_.remove(code)
    return phecode_features_

def train_CART_model(X_train, y_train):
    '''
    Train a decision tree model. Use 5-fold cross validation to choose optimal hyper-parameters.
    '''
    m = X_train.shape[1]
    param_dist = {
        'max_depth': randint(1, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': randint(m // 2, m)
    }

    # Create a decision tree classifier
    dt = DecisionTreeClassifier()

    # Create a RandomizedSearchCV object
    random_search = RandomizedSearchCV(dt, param_dist, n_iter=10, cv=5, scoring='accuracy')

    # Fit the random search to the data
    random_search.fit(X_train, y_train)

    # Get the best parameters
    # best_params = random_search.best_params_

    # Get the best estimator
    final_model = random_search.best_estimator_
    _ = final_model.fit(X_train, y_train)
    return final_model

def train_RF_model(X_train, y_train):
    '''
    Train a random forest model.
    '''
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30, 40],  # Depth of the trees
        'min_samples_split': [2, 5, 10],      # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],        # Minimum samples required at a leaf node
        'bootstrap': [True, False]           # Whether bootstrap samples are used when building trees
    }

    # Use RandomizedSearchCV to find the best parameters
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,         # Number of parameter settings sampled
        cv=5,              # Number of cross-validation folds
        verbose=2,         # Show the progress
        random_state=42,
        n_jobs=-1          # Use all available cores
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Retrain the model with optimal hyper-parameters on the whole training data.
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)
    return best_rf


def train_XG_model(X_train, y_train):
    # Hyperparameter grid
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # RandomizedSearchCV
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric='logloss', random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        random_state=42
    )

    xgb_search.fit(X_train, y_train)

    # Best model
    best_xgb = xgb_search.best_estimator_
    best_xgb.fit(X_train, y_train)
    return best_xgb


def plot_CM(model, X, y, output_path, trait, prefix):
    '''
    Plot confusion matrix for testing data.
    '''
    class_names = ['Control', full_name[trait]]
    disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X,
            y,
            display_labels=class_names,
            cmap=plt.cm.Blues,
        )
    p = precision_score(y, model.predict(X))
    disp.ax_.set_title(f'Confusion Matrix of {full_name[trait]} Prediction Model (Precision: {p:.2f}')
    plt.savefig(output_path / f'{trait}/{trait}_CM_{prefix}.png', bbox_inches='tight')
    return p

def plot_ROC(final_model, X_test, y_test, output_path, trait, prefix):
    '''
    Plot ROC curve for testing data.
    '''
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Calculate the AUC (Area Under the Curve)
    auc = roc_auc_score(y_test, y_pred_prob)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'Decision Tree (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {full_name[trait]} prediction model')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(output_path / f'{trait}/{trait}_ROC_curve_{prefix}.png', bbox_inches='tight')
    return auc


if __name__ == '__main__':
    args = process_args()
    trait = args.trait
    data_path = Path(args.data_folder)
    output_path = Path(args.output_folder)
    prefix = args.output_prefix

    # Import case control and corresponding phecodes
    logging.info('Preparing data for model development...')
    # phecode_table = pd.read_csv(data_path / f'{trait}/phecode_table.txt', sep='\t')
    cases = pd.read_csv(data_path / f'{trait}/case_for_ML.txt', sep='\t')
    controls_clean = pd.read_csv(data_path / f'{trait}/controls_age_corrected.csv')

    phecode_fn = data_path / 'sd_phecode.feathter'
    sd_phecode = pd.read_feather(phecode_fn) 

    case_grid = list(set(cases.GRID))
    control_grid = list(controls_clean.sample(n=len(case_grid)*30, random_state=2024).GRID)

    # Generate dataframe for case and control, add labels, and merge them
    case_df = sd_phecode[sd_phecode.grid.isin(case_grid)]
    case_df['label'] = 1
    control_df = sd_phecode[sd_phecode.grid.isin(control_grid)]
    control_df['label'] = 0
    data = pd.concat([case_df, control_df], ignore_index=True)

    phecode_features_ = get_phecode_features(data_path, output_path, trait)
    X_train, X_test, y_train, y_test = train_test_split(data[phecode_features_], data.label, train_size=0.8,
                                                        random_state=2024, stratify=data.label)

    logging.info('Training the model...')
    # final_model = train_CART_model(X_train, y_train)
    final_model = train_RF_model(X_train, y_train)
    # final_model = train_XG_model(X_train, y_train)

    logging.info('Plotting model results...')
    precision = plot_CM(final_model, X_test, y_test, output_path, trait, prefix)
    auc = plot_ROC(final_model, X_test, y_test, output_path, trait, prefix)
    logging.info(f'Precision is: {precision:.2f}')
    logging.info(f'AUC is: {auc:.2f}')

    logging.info('Saving model...')
    joblib.dump(final_model, output_path / f'{trait}/{trait}_PheML_{prefix}.model')
    logging.info('Done. Model building completed.')