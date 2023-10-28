import scanpy as sc
import numpy as np
import pandas as pd

def load_data(data_dir, slice_name):
    #load ST data
    slice = sc.read_visium(path=data_dir+str(slice_name)+'/', count_file=str(slice_name)+'_filtered_feature_bc_matrix.h5')
    #load ground true
    coldata = pd.read_csv(data_dir+str(slice_name)+'/'+'col_data_'+str(slice_name)+'.csv')
    spatial_coor = (coldata.loc[:, ['row', 'col']]).values
    spatialLIBD_cluster = (coldata.loc[:,['Cluster']]).values
    real_label = (coldata.loc[:, ['layer_guess_reordered_short']]).values
    slice.obsm['spatial_coor'] = spatial_coor
    slice.obs['Ground Truth'] = real_label.reshape(-1)
    slice.obs['spatialLIBD_cluster'] = spatialLIBD_cluster.reshape(-1)
    
    #QC
    slice.var_names_make_unique()
    sc.pp.filter_genes(slice, min_counts = 5)
    sc.pp.highly_variable_genes(slice, n_top_genes=3000 ,flavor='seurat_v3')
    slice = slice[:, slice.var.highly_variable]
    sc.pp.normalize_total(slice, target_sum=1e4)
    sc.pp.log1p(slice)
    return slice