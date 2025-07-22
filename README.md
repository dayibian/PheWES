# PheWES (Phenome-wide enrichment analysis)

## Project Structure
```
.
├── data/           # Data files
├── notebooks/      # Jupyter notebooks
├── results/        # Output results
└── src/           # Source code
```

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
   snakemake --use-conda --cores 4
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
[Add license information here] 