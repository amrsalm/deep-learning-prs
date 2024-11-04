import os
import subprocess
import numpy as np
import pandas as pd
import hail as hl
# Set the reference genome
hl.default_reference("GRCh38")

# get the bucket name
my_bucket = os.getenv('WORKSPACE_BUCKET')

# copy csv file from the bucket to the current working space
os.system(f"gsutil cp '{my_bucket}/data/{name_of_file_in_bucket}' .")

print(f'[INFO] {name_of_file_in_bucket} is successfully downloaded into your working space')
# save dataframe in a csv file in the same workspace as the notebook
gwas_df = pd.read_csv(name_of_file_in_bucket)


# Set filtering criteria
p_value_threshold = 5e-13
maf_threshold = 0.001

# Filter SNPs based on criteria
important_snps = gwas_df[
    (gwas_df['p_value'] < p_value_threshold) &
    (gwas_df['effect_allele_frequency'] >= maf_threshold)
]
# Import VAT table
vat_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/vat/vat_complete_v7.1.bgz.tsv.gz"
vat_table = hl.import_table(vat_path, force=True, quote='"', delimiter="\t", force_bgz=True)

# Step 1: Filter VAT table to contain only important SNPs by 'rsid'
important_rsids_set = hl.literal(important_snps['rsid'].tolist())
vat_filtered = vat_table.filter(important_rsids_set.contains(vat_table['dbsnp_rsid']))

# Annotate `vat_filtered` with `locus`
vat_filtered = vat_filtered.annotate(
    locus=hl.locus(vat_filtered['contig'], hl.int(vat_filtered['position']))
).key_by('locus')

# Path to the VDS file
vds_srwgs_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/vds/hail.vds"
vds = hl.vds.read_vds(vds_srwgs_path)

# Step 2: Filter VDS rows to only include loci in `vat_filtered`
vds_filtered = vds.variant_data.semi_join_rows(vat_filtered)

# Step 3: Import phenotypes and filter to only relevant samples
phenotype_filename = "gs://fc-secure-2a9e0d8c-68f8-4084-b07d-564875de91a1/data/case_control.tsv"
phenotypes = hl.import_table(
    phenotype_filename,
    types={'person_id': hl.tstr, 'has_N20': hl.tint32, 'gender': hl.tstr, 'ethnicity': hl.tstr, 'sex_at_birth': hl.tstr},
    key='person_id'
)

# Filter VDS columns to include only samples in the phenotype file
relevant_person_ids = hl.set(phenotypes.aggregate(hl.agg.collect(phenotypes['person_id'])))
vds_filtered = vds_filtered.filter_cols(relevant_person_ids.contains(vds_filtered.s))
# Step 4: Annotate the VDS with phenotype information
vds_filtered = vds_filtered.annotate_cols(pheno=phenotypes[vds_filtered.s])
tall_table = vds_filtered.select_entries(vds_filtered.GT).entries().key_by()  # Unkey to avoid issues
bucket_path = os.getenv('WORKSPACE_BUCKET')
destination_path = f"{bucket_path}/data/genotype_data_tall.tsv"
tall_table.export(destination_path)
print(f"Genotype data exported to {destination_path}")
