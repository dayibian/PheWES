# This script is used to find matched controls for each case
# It is used to find the best, non-repeated N controls for each case.
# Note this script doesn't use any genetic information for matching.
# The matching criteria are:
# - Sex: Exact match
# - Race: Exact match
# - Ethnicity: Exact match
# - Age: Within +/- 5 years
# - Visit count: Within +/- 5 visits


import argparse
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import yaml

# Load config.yaml for default paths
try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Try to open config.yaml in the current directory
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

# Set random seed for reproducibility
random.seed(2025)


def setup_log(fn_log: str, mode: str = 'w') -> None:
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


def process_args() -> argparse.Namespace:
    '''
    Process arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--icd_count', help='Number of ICD code needed to count as case', type=int, default=0)
    parser.add_argument('--result_path', help='Path to save the results', type=str, default='../results')
    parser.add_argument('--result_filename', help='Name of the result file', type=str, default='case_control_pairs')
    parser.add_argument('--case_path', help='Path to case data file', type=str, default=config['case_path'])
    parser.add_argument('--demographics_file', help='Path to demographics file', type=str, default=config['demographics_file'])
    parser.add_argument('--depth_of_record_path', help='Path to depth of record file', type=str, default=config['depth_of_record_path'])
    parser.add_argument('--control_exclusion_list', help='Path to control exclusion list file', type=str, default=None)
    parser.add_argument('--train_split_ratio', help='Proportion of matched pairs for training split (0-1)', type=float,
                        default=config.get('train_split_ratio', 0.8))

    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(f'{args.result_path}/get_matched_controls.log')
    setup_log(fn_log, mode='w')

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


def import_data(icd_count: int = -1, case_path: Path = None, demographics_file: Path = None, 
                depth_of_record_path: Path = None, control_exclusion_list: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Import and preprocess case and control data for matching.
    
    Args:
        icd_count: Minimum number of ICD codes required to classify as a case. If icd_count=0, all cases will be included.
        case_path: Path to case data file
        demographics_file: Path to demographics file
        depth_of_record_path: Path to depth of record data
        control_exclusion_list: List of control GRIDs to exclude from matching
    Returns:
        Tuple containing:
        - DataFrame with case information (demographics + depth of record)
        - DataFrame with control information (demographics + depth of record)
    """
    # Read case data and filter by ICD code count threshold
    case_df = pd.read_csv(case_path)
    # Ensure the case_df has a 'grid' column; if not, try to rename the first column to 'grid'
    if 'grid' not in case_df.columns:
        # Try to rename the first column to 'grid'
        case_df = case_df.rename(columns={case_df.columns[0]: 'grid'})
    if icd_count > 0:
        case_df = case_df[case_df.icd_code_count>=icd_count]
    
    # Read demographic and depth of record data
    demo_df = pd.read_csv(demographics_file)
    depth_df = pd.read_csv(depth_of_record_path)

    # Merge demographic and depth of record data
    demo_info_df = demo_df.merge(depth_df, on='grid')
    
    # Calculate age in days for matching
    demo_info_df['birthday'] = pd.to_datetime(demo_info_df['birth_datetime'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    demo_info_df['age_in_days'] = (now - demo_info_df['birthday']).dt.days

    # Split into case and control groups
    case_with_info_df = demo_info_df[demo_info_df.grid.isin(case_df.grid)]
    control_with_info_df = demo_info_df[~demo_info_df.grid.isin(case_df.grid)]
    if control_exclusion_list is not None:
        control_with_info_df = control_with_info_df[~control_with_info_df.grid.isin(control_exclusion_list)]
    return case_with_info_df, control_with_info_df


def find_match_controls(case_of_interest: str, cases_df: pd.DataFrame, controls_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    '''
    Find matching controls for a specific case based on demographic and visit criteria.
    
    Matching criteria:
    - Sex: Exact match
    - Race: Exact match
    - Ethnicity: Exact match
    - Age: Within +/- 5 years
    - Visit count: Within +/- 5 visits
    
    Args:
        case_of_interest: Grid ID of the case to find matches for
        cases_df: DataFrame containing case information
        controls_df: DataFrame containing potential control information
        
    Returns:
        Tuple containing:
        - List of matching control grid IDs
        - DataFrame of matching controls with their information
    '''
    # Extract case characteristics for matching
    case = cases_df[cases_df.grid==case_of_interest]
    sex = case.gender_source_value.iloc[0]
    age_in_days = case.age_in_days.iloc[0]
    eth = case.ethnicity_source_value.iloc[0]
    race = case.race_source_value.iloc[0]
    depth_of_record = case.depth_of_record.iloc[0]

    # Apply matching criteria
    mask = ((controls_df.gender_source_value == sex) &
            (controls_df.race_source_value == race) &
            (controls_df.ethnicity_source_value == eth) &
            (controls_df.age_in_days < age_in_days + 365 * 5) &
            (controls_df.age_in_days > age_in_days - 365 * 5) &
            (controls_df.depth_of_record < depth_of_record + 5) &
            (controls_df.depth_of_record > depth_of_record - 5)
           )
    controls_matched = list(controls_df[mask].grid)
    return controls_matched, controls_df[mask]


def main():
    """
    Main execution function that:
    1. Processes command line arguments
    2. Imports and prepares case/control data
    3. Finds matching controls for each case
    4. Writes results to output file
    """
    args = process_args()
    start_time = time.time()
    result_fp = Path(args.result_path) / (args.result_filename + '.txt')

    logging.info('Importing data...\n')

    if args.control_exclusion_list is not None and args.control_exclusion_list != 'None':
        control_exclusion_list = pd.read_csv(args.control_exclusion_list, header=None, names=['grid']).grid.tolist()
    else:
        control_exclusion_list = None
    cases_df, controls_df = import_data(args.icd_count, args.case_path, args.demographics_file, args.depth_of_record_path, control_exclusion_list)

    logging.info('Finding matches...\n')
    found_controls = set()  # Track used controls to ensure no reuse
    header = ['case'] + [f'Control{i}' for i in range(1, 11)]
    with open(result_fp, 'w') as fh:
        fh.write('\t'.join(header))
        fh.write('\n')
        for case in tqdm(cases_df.grid):
            # Find potential matches and filter out already used controls
            potential_matches, _ = find_match_controls(case, cases_df, controls_df)
            exclusive_matches = [x for x in potential_matches if x not in found_controls]
            random.shuffle(exclusive_matches)  # Randomize to avoid bias
            exclusive_matches_10 = exclusive_matches[:10]  # Take top 10 matches
            found_controls.update(exclusive_matches_10)
            line = '\t'.join([case] + exclusive_matches_10) + '\n'
            fh.write(line)

    # Save a portion of the case-control pairs as a training file according to train_split_ratio

    # 1. Read the pairs file we just wrote
    pairs_df = pd.read_csv(result_fp, sep='\t')

    # 2. Calculate the number of training cases
    n_cases = len(pairs_df)
    n_train = int(args.train_split_ratio * n_cases)

    # 3. Shuffle and split
    pairs_df = pairs_df.sample(frac=1, random_state=2025).reset_index(drop=True)
    train_df = pairs_df.iloc[:n_train]
    test_df = pairs_df.iloc[n_train:]

    # 4. Output files
    train_fp = Path(args.result_path) / (args.result_filename + '_train.txt')
    test_fp = Path(args.result_path) / (args.result_filename + '_test.txt')

    train_df.to_csv(train_fp, sep='\t', index=False)
    test_df.to_csv(test_fp, sep='\t', index=False)
    logging.info(f"Saved {len(train_df)} training pairs to {train_fp}")
    logging.info(f"Saved {len(test_df)} test pairs to {test_fp}")

    # Log execution time
    time_elapsed = time.time() - start_time
    if time_elapsed < 60:
        logging.info('\n# Done. Finished in %.2f seconds\n\n' % time_elapsed)
    else:
        logging.info('\n# Done. Finished in %.2f minutes\n\n' % (time_elapsed/60))


if __name__ == '__main__':
    main()