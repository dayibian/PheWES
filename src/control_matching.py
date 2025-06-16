import argparse
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(2025)

# Define paths to input data files
CASE_PATH = Path(
    '/data100t1/home/wanying/BioVU/202505_hypophosphatasia/data/hpp_icd_code_counts_in_sd.case_only.csv')
SD_DEMO_PATH = Path(
    '/data100t1/share/synthetic-deriv/demographic-releases/2025-mar-19/sd_demographics.csv.gz')
DEPTH_OF_RECORD_PATH = Path(
    '/data100t1/home/wanying/BioVU/202505_hypophosphatasia/data/depth_of_record_in_sd.csv')

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
    parser.add_argument('--icd_count', help='Number of ICD code needed to count as case', type=int, default=1)
    parser.add_argument('--result_path', help='Path to save the results', type=str, default='../results')
    parser.add_argument('--filename', help='Name of the result file', type=str, default='case_control_pairs')

    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(f'{args.result_path}/get_controls_count_{args.icd_count}.log')
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


def import_data(icd_count: int = 1, case_path: Path = CASE_PATH, sd_demo_path: Path = SD_DEMO_PATH,
                depth_of_record_path: Path = DEPTH_OF_RECORD_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Import and preprocess case and control data for matching.
    
    Args:
        icd_count: Minimum number of ICD codes required to classify as a case
        case_path: Path to case data file
        sd_demo_path: Path to synthetic derivative demographics file
        depth_of_record_path: Path to depth of record data
        
    Returns:
        Tuple containing:
        - DataFrame with case information (demographics + depth of record)
        - DataFrame with control information (demographics + depth of record)
    """
    # Read case data and filter by ICD code count threshold
    case_df = pd.read_csv(case_path)
    case_df = case_df[case_df.count_hpp_icd_code>=icd_count]
    
    # Read demographic and depth of record data
    sd_demo_df = pd.read_csv(sd_demo_path)
    depth_df = pd.read_csv(depth_of_record_path)

    # Merge demographic and depth of record data
    sd_info_df = sd_demo_df.merge(depth_df, on='grid')
    
    # Calculate age in days for matching
    sd_info_df['birthday'] = pd.to_datetime(sd_info_df['birth_datetime'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    sd_info_df['age_in_days'] = (now - sd_info_df['birthday']).dt.days

    # Split into case and control groups
    case_with_info_df = sd_info_df[sd_info_df.grid.isin(case_df.grid)]
    control_with_info_df = sd_info_df[~sd_info_df.grid.isin(case_df.grid)]
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


def find_exclusive_matches(cases_df: pd.DataFrame, controls_df: pd.DataFrame, n_matches_per_case: int = 10) -> dict[str, list[str]]:
    """
    Finds the best, non-repeated N controls for each case. (Corrected Version)

    This function implements a "best fit" greedy algorithm:
    1. Finds all possible case-control pairs based on exact and range criteria.
    2. Scores each pair based on age difference (smaller is better).
    3. Iterates through pairs from best to worst score, assigning unique controls
       to cases until each case has its desired number of matches.
    """
    # 1. Prepare for merge
    exact_match_cols = ['gender_source_value', 'race_source_value', 'ethnicity_source_value']
    
    # Add a suffix to case columns to distinguish them after the merge.
    cases_renamed = cases_df.add_suffix('_case')
    # Create the list of renamed key columns for the left DataFrame.
    left_on_cols = [col + '_case' for col in exact_match_cols]

    # CORRECTED: Use left_on and right_on for different key column names.
    merged_df = pd.merge(
        cases_renamed,
        controls_df,
        left_on=left_on_cols,
        right_on=exact_match_cols,
        how='inner'
    )

    # 2. Apply the range-based filters (this part remains the same)
    age_tolerance_days = 365.25 * 5
    visit_tolerance = 5
    mask = (
        (merged_df['age_in_days'] < merged_df['age_in_days_case'] + age_tolerance_days) &
        (merged_df['age_in_days'] > merged_df['age_in_days_case'] - age_tolerance_days) &
        (merged_df['depth_of_record'] < merged_df['depth_of_record_case'] + visit_tolerance) &
        (merged_df['depth_of_record'] > merged_df['depth_of_record_case'] - visit_tolerance)
    )
    potential_matches = merged_df[mask].copy()

    # 3. Calculate match quality score and sort (this part remains the same)
    potential_matches['match_score'] = abs(
        potential_matches['age_in_days_case'] - potential_matches['age_in_days']
    )
    potential_matches = potential_matches.sort_values('match_score', ascending=True)

    # 4. Iteratively assign unique controls to cases (this part remains the same)
    used_controls = set()
    final_matches = {case_id: [] for case_id in cases_df['grid']}
    
    for _, row in potential_matches.iterrows():
        case_id = row['grid_case']
        control_id = row['grid']

        if len(final_matches.get(case_id, [])) >= n_matches_per_case:
            continue

        if control_id in used_controls:
            continue
            
        final_matches[case_id].append(control_id)
        used_controls.add(control_id)

    return final_matches


def write_matches_to_file(matches_dict: dict[str, list[str]], output_path: str, n_matches: int = 10) -> None:
    """
    Writes the case-control matches to a tab-separated text file.

    Args:
        matches_dict (dict): A dictionary where keys are case IDs and
                             values are lists of matched control IDs.
        output_path (str): The path to the output .txt file.
        n_matches (int): The maximum number of controls per case, used
                         to generate the header and pad rows.
    """
    # 1. Create the header string
    header_cols = ['case'] + [f'control{i+1}' for i in range(n_matches)]
    header = '\t'.join(header_cols)

    # 2. Open the file and write the content
    with open(output_path, 'w') as f:
        # Write the header first
        f.write(header + '\n')

        # Iterate through each case and its matched controls
        for case_id, controls_list in matches_dict.items():
            # Start the row with the case ID
            row_data = [str(case_id)]

            # Add the controls to the row data
            row_data.extend(map(str, controls_list))

            # Pad the row with empty strings if it has fewer than n_matches
            padding = [''] * (n_matches - len(controls_list))
            row_data.extend(padding)

            # Join all parts of the row with a tab and write to the file
            f.write('\t'.join(row_data) + '\n')
            
    logging.info(f"Successfully wrote matches to {output_path}")


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
    result_fp = Path(args.result_path) / (args.filename + '_' + str(args.icd_count) + '.txt')

    logging.info('Importing data...\n')
    cases_df, controls_df = import_data(args.icd_count)

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

    # Log execution time
    time_elapsed = time.time() - start_time
    if time_elapsed < 60:
        logging.info('\n# Done. Finished in %.2f seconds\n\n' % time_elapsed)
    else:
        logging.info('\n# Done. Finished in %.2f minutes\n\n' % (time_elapsed/60))


if __name__ == '__main__':
    main()