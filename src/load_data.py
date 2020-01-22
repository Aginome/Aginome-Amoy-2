import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import sys
import os

## Sub-challenge 1
sc1_phenotype = pd.read_csv('./data/sc1_Phase1_GE_Phenotype.tsv',index_col=0,sep='\t')
sc1_phenotype['SEX'] = sc1_phenotype['SEX'].replace(' ',np.NaN)
sc1_phenotype = pd.get_dummies(sc1_phenotype[['SEX','WHO_GRADING','CANCER_TYPE']])
sc1_outcome = pd.read_csv('./data/sc1_Phase1_GE_Outcome.tsv',index_col=0,sep='\t')
sc1_rna_data = pd.read_csv('./data/sc1_Phase1_GE_FeatureMatrix.zip',index_col=0,sep='\t')

## Sub-challenge 2
sc2_phenotype = pd.read_csv('./data/sc2_Phase1_CN_Phenotype.tsv',index_col=0,sep='\t')
sc2_phenotype['SEX'] = sc2_phenotype['SEX'].replace(' ',np.NaN)
sc2_phenotype = pd.get_dummies(sc2_phenotype[['SEX','WHO_GRADING','CANCER_TYPE']])
sc2_outcome = pd.read_csv('./data/sc2_Phase1_CN_Outcome.tsv',index_col=0,sep='\t')
sc2_cnv_data = pd.read_csv('./data/sc2_Phase1_CN_FeatureMatrix.zip',index_col=0,sep='\t')

## Sub-challenge 3
sc3_phenotype = pd.read_csv('./data/sc3_Phase1_CN_GE_Phenotype.tsv',index_col=0,sep='\t')
sc3_phenotype['SEX'] = sc3_phenotype['SEX'].replace(' ',np.NaN)
sc3_phenotype = pd.get_dummies(sc3_phenotype[['SEX','WHO_GRADING','CANCER_TYPE']])
sc3_outcome = pd.read_csv('./data/sc3_Phase1_CN_GE_Outcome.tsv',index_col=0,sep='\t')
sc3_rna_cnv = pd.read_csv('./data/sc3_Phase1_CN_GE_FeatureMatrix.zip',index_col=0,sep='\t')

with open('./data/gene_feature.txt','r') as f:
    gene_list = list(set([i for i in f.read().split('\n') if i])&set(sc1_rna_data.columns))

# Selection gene's variance in top maximum 100  
gene_list = list(sc1_rna_data[gene_list].var().sort_values(ascending=False)[:100].index)