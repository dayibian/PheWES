# Snakefile

# Raw data files required for this workflow (must be present before running):
# - case list (case_path)
# - demographics file (demographics_file)
# - depth of record file (depth_of_record_path)

configfile: "config.yaml"

import os

CASE_PATH = config["case_path"]
CONTROL_EXCLUSION_LIST = config["control_exclusion_list"]
DEMO_PATH = config["demographics_file"]
DEPTH_OF_RECORD_PATH = config["depth_of_record_path"]

# Configurable parameters
TRAIT = config.get("trait", "als")
ICD_COUNT = config.get("icd_count", 0)
RESULTS_DIR = config.get("results_dir", "results")
DATA_DIR = config.get("data_dir", "data")
OUTPUT_PREFIX = config.get("output_prefix", "output")
N_PERMUTE = config.get("n_permute", 10000)
MODEL_TYPE = config.get("model_type", 'RF')
USE_MATCHED_CONTROLS = config.get("use_matched_controls", 0)
N_CONTROLS_PER_CASE = config.get("n_controls_per_case", 5)
USE_SMOTEN = config.get("use_smoten", 0)

rule all:
    input:
        f"{RESULTS_DIR}/case_control_pairs_{OUTPUT_PREFIX}_train.txt",
        f"{RESULTS_DIR}/{OUTPUT_PREFIX}.counts_and_pval.txt",
        f"{RESULTS_DIR}/{TRAIT}_{OUTPUT_PREFIX}_enriched_phecode.csv",
        f"{RESULTS_DIR}/PheML_{MODEL_TYPE}_{OUTPUT_PREFIX}.model"
    conda:
        "environment.yaml"

rule find_matched_controls:
    input:
        case=CASE_PATH,
        demographics=DEMO_PATH,
        depth=DEPTH_OF_RECORD_PATH
    output:
        f"{RESULTS_DIR}/case_control_pairs_{OUTPUT_PREFIX}_train.txt"
    params:
        icd_count=ICD_COUNT,
        result_path=RESULTS_DIR,
        result_filename=f"case_control_pairs_{OUTPUT_PREFIX}",
        control_exclusion_list=CONTROL_EXCLUSION_LIST
    conda:
        "environment.yaml"
    shell:
        """
        python src/find_matched_controls.py \
            --icd_count {params.icd_count} \
            --result_path {params.result_path} \
            --result_filename {params.result_filename} \
            --control_exclusion_list {params.control_exclusion_list}
        """

rule phecode_enrichment_with_permutation:
    input:
        case_control_pairs=f"{RESULTS_DIR}/case_control_pairs_{OUTPUT_PREFIX}_train.txt"
    output:
        f"{RESULTS_DIR}/{OUTPUT_PREFIX}.counts_and_pval.txt"
    params:
        output_path=RESULTS_DIR,
        output_prefix=OUTPUT_PREFIX,
        control_fn=f"{RESULTS_DIR}/case_control_pairs_{OUTPUT_PREFIX}_train.txt",
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
        f"{RESULTS_DIR}/PheML_{MODEL_TYPE}_{OUTPUT_PREFIX}.model"
    params:
        data_folder=DATA_DIR,
        output_folder=RESULTS_DIR,
        trait=TRAIT,
        output_prefix=OUTPUT_PREFIX,
        model_type=MODEL_TYPE,
        use_matched_controls=USE_MATCHED_CONTROLS,
        n_controls_per_case=N_CONTROLS_PER_CASE,
        use_smoten=USE_SMOTEN
    conda:
        "environment.yaml"
    shell:
        """
        python src/pheML_develop.py \
            --data_folder {params.data_folder} \
            --output_folder {params.output_folder} \
            --trait {params.trait} \
            --output_prefix {params.output_prefix} \
            --model_type {params.model_type} \
            --matched_controls_for_ML {params.use_matched_controls} \
            --n_controls_per_case {params.n_controls_per_case} \
            --use_smoten {params.use_smoten}
        """