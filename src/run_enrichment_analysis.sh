# python phecode_enrichment_with_permutation.py \
# --control_fn ../results/case_control_pairs_1.txt \
# --output_path ../results \
# --output_prefix icd_count_1 \
# --phecode_table /data100t1/share/synthetic-deriv/phecodes/all-sd-phecodes-mar-2025/sd_samples_phecode.binary.txt.gz \
# --phecode_delimiter tab \
# --control_delimiter tab \
# --n_permute 1000 

python phecode_enrichment_generate_reports.py \
--output_folder ../results/ \
--trait hpp \
--input_prefix icd_count_2