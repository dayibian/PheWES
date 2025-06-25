python find_matched_controls.py \
--icd_count 5 \
--result_path ../results \
--result_filename case_control_pairs

python phecode_enrichment_with_permutation.py \
--control_fn ../results/case_control_pairs_icd_count_5.txt \
--output_path ../results \
--output_prefix icd_count_5 \
--phecode_table /data100t1/share/synthetic-deriv/phecodes/all-sd-phecodes-mar-2025/sd_samples_phecode.binary.txt.gz \
--phecode_delimiter tab \
--control_delimiter tab \
--n_permute 10000

python phecode_enrichment_generate_reports.py \
--output_folder ../results/ \
--trait hpp \
--input_prefix icd_count_5