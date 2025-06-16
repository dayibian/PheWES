import pandas as pd
import numpy as np

import gzip
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from tqdm import tqdm
import random

import warnings
import os
import argparse
import sys
import logging
import time
warnings.filterwarnings('ignore')

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
    parser.add_argument('--trait', help='Trait of interest', type=str, default='als')
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.data_folder) / f'{args.trait}/{args.trait}.log'
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

def correct_age(controls, controls_clean, case_demo):
    '''
    "age" is defined as the number of days between the birth date and current day
    Calculate the day difference between control pull time and current time, so we can calculate the "age" for cases accordingly
    '''
    control_case_overlap = set(controls.GRID) & set(case_demo.grid)
    control_case_overlap = list(control_case_overlap)

    ID = control_case_overlap[0]

    now = datetime.now(ZoneInfo('America/Chicago'))
    d = now - pd.to_datetime(case_demo[case_demo.grid==ID]['birth_datetime'])

    days_diff = d.iloc[0].days - controls[controls.GRID==ID]['AGE'].iloc[0]

    # Add the age offset for controls to match the cases
    controls_age_corrected = controls_clean.copy()
    controls_age_corrected['AGE'] = controls_age_corrected.AGE + days_diff
    controls_age_corrected.drop_duplicates(inplace=True)

    # Add age to the cases
    def calc_age(dob):
        now = datetime.now(ZoneInfo('America/Chicago'))
        try:
            age = now - pd.to_datetime(dob)
            age = int(age.days)
        except:
            print(dob)
        return age

    case_demo.loc[:, 'AGE'] = case_demo.birth_datetime.apply(calc_age)
    return controls_age_corrected, case_demo

def get_dementia_phecode(data_path):
    icd_codes = {
        'als': ['G12.21', 'G12.20', 'G12.24', 'G12.29', '335.20', '335.21', '335.29'],
        'ftld': ['G31.01', 'G31.09', 'G21.1', 'G31.85', '331.11', '331.19', '331.6'],
        'vasc_dementia': ['F01.50', 'F01.51', 'F01.511', 'F01.518', '290.40', '290.41'],
        'lewy_body': ['G31.83', 'G20', 'F02.80', '331.82', '332.0', '294.10']
    }
    excluded_code = []
    for trait in icd_codes.keys():
        with gzip.open(data_path / f'{trait}/{trait}_codes_and_dates.csv.gz') as f:
                case_codes = pd.read_csv(f, dtype={'phecode':str})
        case_codes_ = case_codes[case_codes.concept_code.isin(icd_codes[trait])]
        excluded_code_ = list(case_codes_.phecode.unique())
        excluded_code.extend(excluded_code_)
    return excluded_code

def get_dementia_grids(data_path):
    traits = ['als', 'ftld', 'vasc_dementia', 'lewy_body']
    dementia_grids = []
    for trait in traits:
        with gzip.open(data_path / f'{trait}/{trait}_demographics.csv.gz') as f:
                case_demo = pd.read_csv(f, dtype={'phecode':str})
        case_grid = list(case_demo.grid)
        dementia_grids.extend(case_grid)
    return list(set(dementia_grids))

def find_match_controls(case_of_interest, cases_df, controls_df):
    '''
    Find matching controls for the cases
    SEX, RACE, ETH matched
    AGE: +/- 5 year-old
    VISIT_COUNT: +/- 5 visits
    '''
    case = cases_df[cases_df.GRID==case_of_interest]
    visit = case.VISIT_COUNT.iloc[0]
    sex = case.SEX.iloc[0]
    age_of_days = case.AGE.iloc[0]
    eth = case.ETH.iloc[0]
    race = case.RACE.iloc[0]

    mask = ((controls_df.SEX == sex) &
            (controls_df.RACE == race) &
            (controls_df.ETH == eth) &
            (controls_df.AGE < age_of_days + 365 * 5) &
            (controls_df.AGE > age_of_days - 365 * 5) &
            (controls_df.VISIT_COUNT < visit + 5) &
            (controls_df.VISIT_COUNT > visit - 5)
           )
    controls_matched = list(controls_df[mask].GRID)
    return controls_matched, controls_df[mask]

if __name__ == '__main__':
    args = process_args()
    start_time = time.time()
    control_fn = '/data100t1/share/stuttering/SexSpecificStutteringML/control.pheno'
    data_path = Path(args.data_folder)
    phecode_fn = data_path / 'sd_phecode.feathter'
    TRAIT = args.trait
    bioVU_grids_fn = data_path / 'biovu_grids.csv.gz'

    bioVU_grids = []
    logging.info('\nImporting BioVU GRIDs...')
    with gzip.open(bioVU_grids_fn) as f:
        header = f.readline()
        for line in f:
            line_ = line.decode("utf-8")
            bioVU_grids.append(line_.strip())

    logging.info('Importing all the phecodes from SD...')
    sd_phecode = pd.read_feather(phecode_fn)
    logging.info(f'Shape of SD phecode is: {sd_phecode.shape}')

    logging.info('Importing all the control data used in Stuttering study...')
    controls = pd.read_csv(control_fn, sep='\t')

    logging.info('Importing demographic, visit count, and codes for the trait...')
    with gzip.open(data_path / f'{TRAIT}/{TRAIT}_demographics.csv.gz') as f:
        case_demo = pd.read_csv(f)
    with gzip.open(data_path / f'{TRAIT}/{TRAIT}_visit_count.csv.gz') as f:
        case_vc = pd.read_csv(f)
    with gzip.open(data_path / f'{TRAIT}/{TRAIT}_codes_and_dates.csv.gz') as f:
        case_codes = pd.read_csv(f, dtype={'phecode':str})

    # Exclude GRIDs in controls that are also in all dementia cases and BioVU
    # Normally we only want to exclude those specific case grids
    dementia_grids = get_dementia_grids(data_path)
    controls_clean = controls.loc[~controls.GRID.isin(dementia_grids)]
    controls_clean = controls_clean.loc[~controls_clean.GRID.isin(bioVU_grids)]
    # controls_clean.to_csv(data_path / f'{TRAIT}/controls_clean.csv', index=False)

    controls_age_corrected, case_demo = correct_age(controls, controls_clean, case_demo)
    controls_age_corrected.to_csv(data_path / f'{TRAIT}/controls_age_corrected.csv', index=False)

    # Add visits to the case demographic data and change the column names to be consistent with controls data
    case_demo_with_visits = case_demo.merge(case_vc, how='left', on='grid')
    case_demo_with_visits.drop(columns=['birth_datetime'], inplace=True)
    case_demo_with_visits.rename(columns={'grid':'GRID', 'ethnicity_source_value':'ETH', 'race_source_value': 'RACE', 'gender_source_value':'SEX', 'count(1)':'VISIT_COUNT'}, inplace=True)
    case_demo_with_visits.dropna(inplace=True, ignore_index=True)
    case_demo_with_visits.to_csv(data_path / f'{TRAIT}/case_demo_with_visits.csv', index=False)

    case_demo_with_visits_noBV = case_demo_with_visits[~case_demo_with_visits.GRID.isin(bioVU_grids)]
    case_biovu = case_demo_with_visits[case_demo_with_visits.GRID.isin(bioVU_grids)].GRID
    case_for_EA = case_demo_with_visits_noBV.sample(frac=.5, random_state=2024)
    case_for_ML = case_demo_with_visits_noBV[~case_demo_with_visits_noBV.GRID.isin(case_for_EA.GRID)]

    cols = ['Case'] + ['Control'+str(i) for i in range(1, 11)]
    case_controls_matched = pd.DataFrame(columns=cols)
    case_controls_matched.loc[:, 'Case'] = list(case_for_EA.GRID)
    # case_controls_matched.loc[:, 'Case'] = list(case_demo_with_visits.GRID) # This is to generate all the case controls
    logging.info('Generating case-control pairs...')
    for i in tqdm(range(case_controls_matched.shape[0])):
        case_grid = case_controls_matched.loc[i, 'Case']
        controls_, _ = find_match_controls(case_grid, case_demo_with_visits, controls_age_corrected)
        random.shuffle(controls_)
        for j in range(1, min(10, len(controls_))+1):
            case_controls_matched.loc[i, 'Control'+str(j)] = controls_[j-1]

    # Remove cases where there is no matching controls
    case_controls_matched_cleaned = case_controls_matched.dropna(
        subset=['Control'+str(i) for i in range(1, 11)], how='all', ignore_index=True)
    logging.info(f'The number of {TRAIT} case used in enrichment analysis is: {case_controls_matched_cleaned.shape[0]}')
    logging.info(f'THe number of case in BioVU is: {len(case_biovu)}')
    case_AFR = list(case_demo[case_demo.race_source_value=='B'].grid)
    case_EUR = list(case_demo[case_demo.race_source_value=='W'].grid)
    case_controls_matched_cleaned_AFR = case_controls_matched_cleaned[case_controls_matched_cleaned.Case.isin(case_AFR)]
    case_controls_matched_cleaned_EUR = case_controls_matched_cleaned[case_controls_matched_cleaned.Case.isin(case_EUR)]

    # Save files for WZ's code
    case_biovu.to_csv(data_path / f'{TRAIT}/case_biovu.txt', header=False, index=False)
    case_controls_matched_cleaned.Case.to_csv(data_path / f'{TRAIT}/case.txt', header=False, index=False)
    case_controls_matched_cleaned.to_csv(data_path / f'{TRAIT}/matched_controls.txt', sep='\t', index=False)
    case_controls_matched_cleaned_AFR.to_csv(data_path / f'{TRAIT}/{TRAIT.upper()}_matched_controls_AFR.txt', sep='\t', index=False)
    case_controls_matched_cleaned_EUR.to_csv(data_path / f'{TRAIT}/{TRAIT.upper()}_matched_controls_EUR.txt', sep='\t', index=False)
    case_for_ML.to_csv(data_path / f'{TRAIT}/case_for_ML.txt', sep='\t', index=False)

    # Get phecode table for permutation
    case_grids = set(case_controls_matched_cleaned.Case)
    control_grids = set()
    for i in range(1, 11):
        control_grids = control_grids | set(case_controls_matched_cleaned['Control'+str(i)])
    case_code = sd_phecode[sd_phecode.grid.isin(case_grids)]
    control_code = sd_phecode[sd_phecode.grid.isin(control_grids)]
    
    # Select phecode that has at least 20 appearences
    phecode_count_case = case_code.sum(numeric_only=True)
    min_count = 0.05 * case_code.shape[0]
    phecode_of_interest = case_code.columns[1:][phecode_count_case > min_count]
    case_code_pruned = case_code[phecode_of_interest.append(pd.Index(['grid']))]
    control_code_pruned = control_code[phecode_of_interest.append(pd.Index(['grid']))]

    phecode_table = pd.concat((case_code_pruned, control_code_pruned))
    phecode_table.to_csv(data_path / f'{TRAIT}/phecode_table.txt', sep='\t', index=False)
    logging.info(f'The number of phecode of interest for {TRAIT} is: {phecode_table.shape[1]-1}')

    time_elapsed = time.time() - start_time
    if time_elapsed<60:
        logging.info('\n# Done. Finished in %.2f seconds\n\n' % time_elapsed)
    else:
        logging.info('\n# Done. Finished in %.2f minutes\n\n' % (time_elapsed/60))
