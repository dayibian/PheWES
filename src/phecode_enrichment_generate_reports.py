import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import argparse
import sys
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def setup_log(fn_log, mode='a'):
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
    parser.add_argument('--input_prefix', help='The prefix for the input file.', type=str, default='output')
    
    args = parser.parse_args()

    # Record arguments used
    fn_log = Path(args.output_folder) / f'{args.trait}_{args.input_prefix}_report.log'
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


def main():
    args = process_args()
    TRAIT = args.trait
    data_path = Path(args.data_folder)
    output_path = Path(args.output_folder)
    prefix = args.input_prefix

    logging.info('\nReading enrichment analysis results...')
    results = pd.read_csv(output_path / f'{prefix}.frequencies_and_pval.txt', sep='\t', dtype={'phecode':str})
    results_sig = results[results.pval<1e-5]
    logging.info(f'The number of enriched phecode is: {results_sig.shape[0]}')

    logging.info('Reading phecode map...')
    phecode_map = pd.read_csv(data_path / 'Phecode_map12_filtered.csv', dtype={'Phecode':str})
    phecode_map = phecode_map[['Phecode', 'PhecodeString']].drop_duplicates(ignore_index=True)
    phecode_map.Phecode = phecode_map.Phecode.apply(lambda x: x.strip())
    phecode_map.index = phecode_map.Phecode
    phecode_map.drop(columns=['Phecode'], inplace=True)
    phecode_map = phecode_map.to_dict()
    phecode_map = phecode_map['PhecodeString']

    def find_phecode_string(x):
        x = str(x)
        try:
            s = phecode_map[x]
        except:
            s = 'NA'
        return s

    results_sig.loc[:, 'PhecodeString'] = results_sig.phecode.apply(find_phecode_string)
    results_sig = results_sig.sort_values(by='phecode', ignore_index=True)
    # results_sig.head()

    enriched_phecode = pd.DataFrame(columns=['Phecode', 'Description', 'Count', 'p.value',
                                            'p01', 'p05', 'p10', 'p50', 'p90',
                                            'p95', 'p99', 'max', 'case_to_control_ratio'])

    logging.info('Generating statistics for enrichment analysis...')
    for i in tqdm(range(len(results_sig))):
        control_count = pd.to_numeric(results_sig.iloc[i, 3:-1]).to_list()
        case_count = results_sig.loc[i, 'case_freq']
        code, pval, desc = results_sig.loc[i, 'phecode'], results_sig.loc[i, 'pval'], results_sig.loc[i, 'PhecodeString']
        max_count = int(max(control_count))
        p01 = np.percentile(control_count, 1)
        p05 = np.percentile(control_count, 5)
        p10 = np.percentile(control_count, 10)
        p50 = np.percentile(control_count, 50)
        p90 = np.percentile(control_count, 90)
        p95 = np.percentile(control_count, 95)
        p99 = np.percentile(control_count, 99)
        enriched_phecode.loc[i, 'Phecode'] = code
        enriched_phecode.loc[i, 'Description'] = desc
        enriched_phecode.loc[i, 'Count'] = int(case_count)
        enriched_phecode.loc[i, 'p.value'] = pval
        enriched_phecode.loc[i, 'p01'] = int(p01)
        enriched_phecode.loc[i, 'p05'] = int(p05)
        enriched_phecode.loc[i, 'p10'] = int(p10)
        enriched_phecode.loc[i, 'p50'] = int(p50)
        enriched_phecode.loc[i, 'p90'] = int(p90)
        enriched_phecode.loc[i, 'p95'] = int(p95)
        enriched_phecode.loc[i, 'p99'] = int(p99)
        enriched_phecode.loc[i, 'max'] = max_count
        if max_count > 0:
            enriched_phecode.loc[i, 'case_to_control_ratio'] = round(int(case_count) / max_count, 2)
        else:
            enriched_phecode.loc[i, 'case_to_control_ratio'] = 1000

    enriched_phecode = enriched_phecode[enriched_phecode.Description!='NA']
    enriched_phecode.sort_values(by='case_to_control_ratio', ascending=False, inplace=True)
    enriched_phecode.to_csv(output_path / f'{TRAIT}_{prefix}_enriched_phecode.csv', sep='\t', index=False)

    logging.info('\nDone. Report generated.\n\n')

if __name__ == '__main__':
    main()