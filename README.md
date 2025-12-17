# PheWES & PheML
PheWES is a fully automated, modular pipeline for conducting phenome-wide enrichment studies and machine-learning–based phenotype prediction. Built with Snakemake for workflow orchestration and Python for data processing and modeling, the system provides a reproducible and scalable framework for exploring clinical phenotypes derived from ICD and phecode mappings.

The pipeline includes robust modules for cohort construction, permutation-based phecode enrichment, and supervised learning models for disease prediction. Users configure workflows through a clean YAML-based interface, ensuring portability across systems and datasets. The project emphasizes reproducibility, software engineering best practices, and extensibility—supporting researchers who aim to transition from raw EHR data to statistically validated insights and deployable machine-learning outputs.

## Project Structure
```
.
├── data/           # Data files
├── notebooks/      # Jupyter notebooks
├── results/        # Output results
└── src/           # Source code
```

## Data Requirements
The pipeline requires the following data files. Paths should be configured in `config.yaml`.

- **Demographics File** (`sd_demographics.csv`):
  - Columns: `grid`, `birth_datetime` (YYYY-MM-DD), `gender_source_value`, `race_source_value`, `ethnicity_source_value`
- **Depth of Record File** (`depth_of_record.csv`):
  - Columns: `grid`, `depth_of_record` (integer, number of visits in unique days)
- **Case File**:
  - A CSV or text file containing at least a `grid` column defining the cases.
- **Phecode Binary File** (`phecode_binary.feather`):
  - A Feather format file containing a `grid` column and binary columns for each phecode (1 = present, 0 = absent).
- **Control Exclusion List** (Optional):
  - A file containing GRIDs to exclude from control selection.
- **ICD to Phecode Mapping** (`ICD_phecode_mapping.csv`):
  - Columns: `ICD`, `Flag`, `ICDString`, `Phecode`, `PhecodeString`, `PhecodeCategory` (used for mapping raw diagnosis codes to phenotype groupings)


## Setup
1. Create and activate the conda environment using the provided environment.yaml:
```bash
conda env create -f environment.yaml
conda activate PheWES  # On Linux/Mac
```

*This will install all required dependencies as specified in environment.yaml.*

## Usage
### Running the Pipeline

This project uses [Snakemake](https://snakemake.readthedocs.io/) to manage the analysis workflow. The main steps are:

1. **Configure your analysis**  
   Edit `config.yaml` to set your trait of interest, minimum ICD code count, input/output directories, and output prefix.

2. **Prepare your input data**  
   Place your input data files in the directory specified by `data_dir` in `config.yaml`.

3. **Run the pipeline**  
   From the project root, execute:
   ```bash
   snakemake --cores 4
   ```
   Replace `4` with the number of CPU cores you wish to use.

   This will:
   - Find matched controls for your cases
   - Perform phecode enrichment analysis with permutation
   - Generate enrichment reports
   - Train a PheML model

4. **View results**  
   Output files will be saved in the directory specified by `results_dir` in `config.yaml`. Key outputs include:
   - `case_control_pairs.txt`: Matched case-control pairs
   - `{output_prefix}.counts_and_pval.txt`: Phecode enrichment results
   - `{trait}_{output_prefix}_enriched_phecode.csv`: Enriched phecodes
   - `PheML_{output_prefix}.model`: Trained model

### Example

To run the pipeline with the default configuration:


## Contributing
[Add contribution guidelines here]

## License
This project is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
You are free to use, share, and adapt this work for non-commercial purposes, provided you give appropriate credit.  
For more details, see [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

If you wish to use this project for commercial purposes, please contact the authors for permission.