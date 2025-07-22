# Snakefile

# Raw data files required for this workflow (must be present before running):
# - case list (case_path)
# - SD demographics file (sd_demographics_file)
# - depth of record file (depth_of_record_path)

configfile: "config.yaml"

import os

CASE_PATH = config["case_path"]
SD_DEMO_PATH = config["sd_demographics_file"]
DEPTH_OF_RECORD_PATH = config["depth_of_record_path"]

# Configurable parameters
TRAIT = config.get("trait", "als")
ICD_COUNT = config.get("icd_count", 0)
RESULTS_DIR = config.get("results_dir", "results")
DATA_DIR = config.get("data_dir", "data")
OUTPUT_PREFIX = config.get("output_prefix", "output")
N_PERMUTE = config.get("n_permute", 10000)

rule all:
    input:
        f"{RESULTS_DIR}/case_control_pairs.txt",
        f"{RESULTS_DIR}/{OUTPUT_PREFIX}.counts_and_pval.txt",
        f"{RESULTS_DIR}/{TRAIT}_{OUTPUT_PREFIX}_enriched_phecode.csv",
        f"{RESULTS_DIR}/PheML_{OUTPUT_PREFIX}.model"
    conda:
        "environment.yaml"

rule find_matched_controls:
    input:
        case=CASE_PATH,
        demographics=SD_DEMO_PATH,
        depth=DEPTH_OF_RECORD_PATH
    output:
        f"{RESULTS_DIR}/case_control_pairs.txt"
    params:
        icd_count=ICD_COUNT,
        result_path=RESULTS_DIR,
        result_filename="case_control_pairs"
    conda:
        "environment.yaml"
    shell:
        """
        python src/find_matched_controls.py \
            --icd_count {params.icd_count} \
            --result_path {params.result_path} \
            --result_filename {params.result_filename}
        """

rule phecode_enrichment_with_permutation:
    input:
        case_control_pairs=f"{RESULTS_DIR}/case_control_pairs.txt"
    output:
        f"{RESULTS_DIR}/{OUTPUT_PREFIX}.counts_and_pval.txt"
    params:
        output_path=RESULTS_DIR,
        output_prefix=OUTPUT_PREFIX,
        control_fn=f"{RESULTS_DIR}/case_control_pairs.txt",
        n_permute=N_PERMUTE
    conda:
        "environment.yaml"
    shell:
        """
        python src/phecode_enrichment_with_permutation.py \
            --control_fn {params.control_fn} \
            --output_path {params.output_path} \
            --output_prefix {params.output_prefix} \
            --n_permute {params.n_permute}
        """

rule phecode_enrichment_generate_reports:
    input:
        enrichment=f"{RESULTS_DIR}/{OUTPUT_PREFIX}.counts_and_pval.txt"
    output:
        f"{RESULTS_DIR}/{TRAIT}_{OUTPUT_PREFIX}_enriched_phecode.csv"
    params:
        data_folder=DATA_DIR,
        output_folder=RESULTS_DIR,
        trait=TRAIT,
        input_prefix=OUTPUT_PREFIX
    conda:
        "environment.yaml"
    shell:
        """
        python src/phecode_enrichment_generate_reports.py \
            --data_folder {params.data_folder} \
            --output_folder {params.output_folder} \
            --trait {params.trait} \
            --input_prefix {params.input_prefix}
        """

rule pheML_develop:
    input:
        enriched_phecode=f"{RESULTS_DIR}/{TRAIT}_{OUTPUT_PREFIX}_enriched_phecode.csv"
    output:
        f"{RESULTS_DIR}/PheML_{OUTPUT_PREFIX}.model"
    params:
        data_folder=DATA_DIR,
        output_folder=RESULTS_DIR,
        trait=TRAIT,
        output_prefix=OUTPUT_PREFIX
    conda:
        "environment.yaml"
    shell:
        """
        python src/pheML_develop.py \
            --data_folder {params.data_folder} \
            --output_folder {params.output_folder} \
            --trait {params.trait} \
            --output_prefix {params.output_prefix}
        """