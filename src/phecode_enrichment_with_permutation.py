# Find enriched phecode with permutation
'''
Example call:

python /data100t1/home/wanying/BioVU/202409_celiac_project/code/utils/phecode_enrichment_with_permutation.py \
--case_fn ./sample_data/case_list.txt \
--control_fn ./sample_data/control_list.txt \
--output_path ./ \
--output_prefix output \
--phecode_table ./sample_data/phecode_table.binary.txt.gz \
--phecode_delimiter tab \
--control_delimiter tab \
--n_permute 1000 
'''

import pandas as pd
import polars as pl
import numpy as np
import os
import argparse
import sys
import logging
import time
from pathlib import Path
import yaml

# Load config.yaml for default paths
try:
    with open("../config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Try to open config.yaml in the current directory
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

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
    parser.add_argument('--case_fn', help='List of cases. One ID per line without column header', type=str,
                        default='./sample_data/case_list.txt')
    parser.add_argument('--control_fn', help='Table of matched controls of each case. Missing values are labeled as NA. See sample data', type=str,
                        default='./sample_data/control_list.txt')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--output_prefix', type=str, default='output')
    parser.add_argument('--phecode_table', type=str, default=config['phecode_binary_file'],
                        help='Binary counts of phecode for each sample. Must have column names, with the first column being sample ID')
    parser.add_argument('--phecode_delimiter', default='tab', choices=[',', 'tab', 'space', 'whitespace'],
                        help='Delimiter of the phecode table file')
    parser.add_argument('--control_delimiter', default='tab', choices=[',', 'tab', 'space', 'whitespace'],
                        help='Delimiter of the control file')
    parser.add_argument('--n_permute', help='Number of permutations', type=int,
                        default=10000)
    parser.add_argument('--phecode_binary_feather_file', type=str, default=config['phecode_binary_feather_file'],
                        help='Path to the feather file with binary phecode data')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_path): # Create output folder if not exists
        print('# Create output path: ' + args.output_path)
        os.makedirs(args.output_path)

    # Record arguments used
    fn_log = os.path.join(args.output_path, args.output_prefix+'_permutation.log')
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

    # Get delimiter
    dict_delimiter = {',':',', 'tab':'\t', 'space':' ', 'whitespace':'\s+'}
    if not args.phecode_delimiter:
        # Infer delimiter from phecode table suffix
        if args.phecode_table.endswith('.csv'):
            args.phecode_delimiter = ','
        else:
            args.phecode_delimiter = '\t'
    else:
        args.phecode_delimiter = dict_delimiter[args.phecode_delimiter]

    if not args.control_delimiter:
        if args.control_fn.endswith('.csv'):
            args.control_delimiter = ','
        else:
            args.control_delimiter = '\t'
    else:
        args.control_delimiter = dict_delimiter[args.control_delimiter]
    return args
    
def get_lst_ids(in_fn):
    '''
    Get a list of sample ids from file (case_fn)
    Params:
    - in_fn: input file name. One ID per row without column header
    Return:
    - A list of sample IDs
    '''
    lst_ids = []
    with open(in_fn) as fh:
        for line in fh:
            line = line.strip()
            if line != '':
                lst_ids.append(line)
    return lst_ids

def get_case_ids(in_fn, delimiter='\t'):
    df = pd.read_csv(in_fn, sep=delimiter)
    return df['case'].to_list()

def get_control_dict(in_fn, delimiter, header):
    '''
    Get control file into a dictionary
    Params:
    - in_fn
    - delimiter:
    - header: False = no header, True = Skip the first row
    Return:
    - A dictionary. Keys are IDs of cases, values are matched controls in lists.
    Eg:
    {'sample1': ['sample100', 'sample150'],
    'sample2': ['sample7', 'sample10','sample70']}
    '''
    dict_controls = {}
    with open(in_fn) as fh:
        if header: fh.readline()   
        for line in fh:
            line = line.strip()
            if line != '':
                line_lst = line.split(delimiter)
                controls = np.array([val for val in line_lst[1:] if val!='NA']) # Skip missing values NA
                if len(controls) != 0: dict_controls[line_lst[0]] = controls
    return dict_controls
    
def get_lst_controls(lst_case, dict_control, rng):
    '''
    Given a list of cases, get a list of controls to calcualte phecode frequencies based on the dict_controls
    Params:
    - lst_case, dict_control
    - rng: random number generator
    '''
    lst_control = []
    for case in lst_case:
        # rng = np.random.default_rng(seed=2024)
        lst_control.append(rng.choice(dict_control[case], 1)[0])
    return lst_control

def get_frequencies(lst_ids, df_phecode_lazy, phecode_cols=None):
    '''
    Given a list of ids (lst_ids) and a phecode table (polars LazyFrame),
    return count and frequencies of phecodes in a pandas Series
    
    Params:
    - lst_ids: list of sample IDs to filter
    - df_phecode_lazy: polars LazyFrame with phecode data
    - phecode_cols: pre-computed list of phecode column names (excluding 'grid')
                    to avoid repeated schema lookups
    '''
    # Get phecode columns once if not provided (caching optimization)
    if phecode_cols is None:
        phecode_cols = [col for col in df_phecode_lazy.collect_schema().names() if col != 'grid']
    
    n_samples = len(lst_ids)
    
    # Optimize: select only needed columns during filtering, avoiding drop operation
    # Use sum directly in the lazy evaluation chain before collecting
    counts_pl = (df_phecode_lazy
                 .filter(pl.col('grid').is_in(lst_ids))
                 .select(phecode_cols)
                 .sum()
                 .collect())
    
    # More efficient conversion: use squeeze() to convert single-row DataFrame to Series
    counts = counts_pl.to_pandas().squeeze()
    frequencies = counts / n_samples
    
    return counts, frequencies  # Return counts and frequency as pandas Series

def main():
    args = process_args()
    start_time = time.time()
    logging.info('\n# Load list of cases and controls')
    # lst_case = get_lst_ids(args.case_fn)
    lst_case = get_case_ids(in_fn=args.control_fn, delimiter=args.control_delimiter) # The original case list
    dict_control = get_control_dict(in_fn=args.control_fn, delimiter=args.control_delimiter, header=True)
    logging.info('# - N cases=%s' % len(lst_case))
    logging.info('# - Control file: %s' % len(dict_control))
    lst_case = list(dict_control.keys()) # Update the case list, remove cases that have no matched controls
    
    logging.info('\n# Load binary phecode table (lazy loading with polars)')
    df_phecode = pl.scan_ipc(args.phecode_binary_feather_file)
    # Get schema info for logging (lazy - no data loaded yet)
    schema_names = df_phecode.collect_schema().names()
    phecode_cols = [col for col in schema_names if col != 'grid']
    n_cols = len(phecode_cols)
    logging.info('# - Binary phecode table loaded (lazy). N phecodes: %s' % n_cols)

    logging.info('\n# Calcualte frequency of each phecode in cases')
    df_case_count, _ = get_frequencies(lst_case, df_phecode, phecode_cols)

    logging.info('\n# Calcualte frequency of each phecode in controls with permutation')
    all_control_count = [] # Store counts of controls
    rng = np.random.default_rng(seed=2024)
    for i in range(args.n_permute):
        lst_control_count, _ = get_frequencies(get_lst_controls(lst_case, dict_control, rng), df_phecode, phecode_cols)
        all_control_count.append(lst_control_count)
        if i%10==0: print(f'\r - Permutation {i+1}   ', end='', flush=True)
    print(f'\r - Permutation {i+1}   ', end='\n')
    df_all_count = pd.concat([df_case_count]+all_control_count, axis=1).reset_index()
    df_all_count.columns = ['phecode', 'case_count'] + [f'control_count_{x+1}' for x in range(args.n_permute)]
    
    logging.info('\n# Calcualte p values')
    df_pvals = df_all_count.iloc[:, 2:].ge(df_all_count['case_count'], axis=0).sum(axis=1)/args.n_permute
    df_all = pd.concat([df_all_count, df_pvals], axis=1).rename(columns={0:'pval'})
    # Reorder columns
    new_cols = ['phecode', 'pval', 'case_count'] + [f'control_count_{x+1}' for x in range(args.n_permute)]

    # Save permutation and pvalues to file
    output_fn = f"{os.path.join(args.output_path, args.output_prefix+'.counts_and_pval.txt')}"
    df_all[new_cols].to_csv(output_fn, index=False, sep='\t')

    time_elapsed = time.time() - start_time
    if time_elapsed<60:
        logging.info('\n# Done. Finished in %.2f seconds' % time_elapsed)
    else:
        logging.info('\n# Done. Finished in %.2f minutes' % (time_elapsed/60))
    
    
if __name__ == '__main__':
    main()


    
