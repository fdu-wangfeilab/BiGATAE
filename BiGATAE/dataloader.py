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

def load_slices_fourST(data_dir, slice_names):
    slices = []  
    for slice_name in slice_names:
        print(data_dir + slice_name + ".csv")
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial_coor'] = slice_i_coor
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts = 15)
        sc.pp.filter_cells(slice_i, min_counts = 100)
        slices.append(slice_i)
    adata_layer_1, adata_layer_2, adata_layer_3, adata_layer_4 = slices
    common_genes = intersect(adata_layer_1.var.index, adata_layer_2.var.index)  
    common_genes = intersect(common_genes, adata_layer_3.var.index)
    common_genes = intersect(common_genes, adata_layer_4.var.index)
    adata_layer_1 = adata_layer_1[:, common_genes]
    adata_layer_2 = adata_layer_2[:, common_genes]
    adata_layer_3 = adata_layer_3[:, common_genes]
    adata_layer_4 = adata_layer_4[:, common_genes]
    slices = [adata_layer_1, adata_layer_2, adata_layer_3, adata_layer_4]
    return slices

def load_cSCC_46(data_dir, P_name, slice_num):
    slice = sc.read_10x_mtx(data_dir+P_name+'/'+P_name+'_rep'+slice_num)
    position = sc.read_csv(data_dir+P_name+'/'+P_name+'_rep'+slice_num+'/spatial/'+P_name+'_rep'+slice_num+'_tissue_positions_list.csv')
    position = position[slice.obs.index]
    spatial_coor = np.array([row[1:3] for row in position.X.tolist()]).astype(int)
    spatial = np.array([row[3:5] for row in position.X.tolist()]).astype(int)
    slice.obsm['spatial_coor']=spatial_coor
    slice.obsm['spatial'] = spatial
    sc.pp.filter_cells(slice, min_counts = 3)
    print(slice)
    sc.pp.highly_variable_genes(slice, n_top_genes=3000 )
    slice = slice[:, slice.var.highly_variable]
    sc.pp.normalize_total(slice, target_sum=1e4)
    sc.pp.log1p(slice)
    return slice

def load_cSCC_25910(data_dir, P_name, slice_num):
    slice = sc.read_text(data_dir+P_name+'/'+P_name+'_ST_rep'+slice_num+'_stdata.tsv')
    position = sc.read_text(data_dir+P_name+'/spot_data-selection-'+P_name+'_ST_rep'+slice_num+'.tsv')
    spatial_coor = np.array([row[:2] for row in position.X.tolist()]).astype(int)
    index = [f'{x}x{y}' for x, y in spatial_coor]
    position.obs.index = index
    index = np.intersect1d(index, slice.obs.index)
    slice = slice[index]
    position = position[index]
    spatial_coor = []
    spatial_coor = np.array([row[:2] for row in position.X.tolist()]).astype(int)
    spatial = np.array([row[4:6] for row in position.X.tolist()]).astype(int)
    slice.obsm['spatial_coor'] = spatial_coor
    slice.obsm['spatial'] = spatial
    sc.pp.filter_cells(slice, min_counts = 3)
    print(slice)
    sc.pp.highly_variable_genes(slice, n_top_genes=3000 )
    slice = slice[:, slice.var.highly_variable]
    sc.pp.normalize_total(slice, target_sum=1e4)
    sc.pp.log1p(slice)
    return slice
