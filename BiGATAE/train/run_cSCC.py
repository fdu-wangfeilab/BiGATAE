import model
import torch
import scanpy as sc
import anndata as ad
import dataloader as dl
from utils.get_edge_index import get_edge_index
from paste_ import get_pi
import numpy as np


# def intersect(lst1, lst2): 
#     temp = set(lst2)
#     print(len(temp))
#     lst3 = [value for value in lst1 if value in temp]
#     return lst3

# def load_slices_fourST(data_dir='/home/sxa/Datasets/breast_cancer_data/', slice_names=["slice1", "slice2", "slice3", "slice4"]):
#     slices = []  
#     for slice_name in slice_names:
#         print(data_dir + slice_name + ".csv")
#         slice_i = sc.read_csv(data_dir + slice_name + ".csv")
#         slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
#         slice_i.obsm['spatial_coor'] = slice_i_coor
#         # Preprocess slices
#         sc.pp.filter_genes(slice_i, min_counts = 15)
#         sc.pp.filter_cells(slice_i, min_counts = 100)
#         slices.append(slice_i)
#     adata_layer_1, adata_layer_2, adata_layer_3, adata_layer_4 = slices
#     common_genes = intersect(adata_layer_1.var.index, adata_layer_2.var.index)  
#     common_genes = intersect(common_genes, adata_layer_3.var.index)
#     common_genes = intersect(common_genes, adata_layer_4.var.index)
#     adata_layer_1 = adata_layer_1[:, common_genes]
#     adata_layer_2 = adata_layer_2[:, common_genes]
#     adata_layer_3 = adata_layer_3[:, common_genes]
#     adata_layer_4 = adata_layer_4[:, common_genes]
#     slices = [adata_layer_1, adata_layer_2, adata_layer_3, adata_layer_4]
#     return slices

# slice1, slice2, slice3, slice4 = load_slices_fourST()

# slice1_slice2_pi = get_pi(slice1, slice2)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice1(2).pt')

# slice1_slice2_pi = get_pi(slice2, slice1)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice2(1).pt')

# slice1_slice2_pi = get_pi(slice2, slice3)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice2(3).pt')

# slice1_slice2_pi = get_pi(slice3, slice2)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice3(2).pt')

# slice1_slice2_pi = get_pi(slice3, slice4)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice3(4).pt')

# slice1_slice2_pi = get_pi(slice4, slice3)
# slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
# torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/BC_edges/slice4(3).pt')

def load_cSCC_46(data_dir='/home/sxa/Datasets/cSCC/', P_name='P4', slice_num='1'):
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

def load_cSCC_25910(data_dir='/home/sxa/Datasets/cSCC/', P_name='P2', slice_num='1'):
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

#p2
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_2.h5ad')
slice3 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_3.h5ad')

slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P2_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P2_slice2(1).pt')

slice1_slice2_pi = get_pi(slice2, slice3)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P2_slice2(3).pt')

slice1_slice2_pi = get_pi(slice3, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P2_slice3(2).pt')

#p4
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P4_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P4_2.h5ad')


slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P4_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P4_slice2(1).pt')

#p5
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_2.h5ad')
slice3 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_3.h5ad')

slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P5_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P5_slice2(1).pt')

slice1_slice2_pi = get_pi(slice2, slice3)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P5_slice2(3).pt')

slice1_slice2_pi = get_pi(slice3, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P5_slice3(2).pt')

#p6
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P6_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P6_2.h5ad')

slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P6_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P6_slice2(1).pt')

#p9
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_2.h5ad')
slice3 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_3.h5ad')

slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P9_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P9_slice2(1).pt')

slice1_slice2_pi = get_pi(slice2, slice3)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P9_slice2(3).pt')

slice1_slice2_pi = get_pi(slice3, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P9_slice3(2).pt')

#p10
slice1 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_1.h5ad')
slice2 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_2.h5ad')
slice3 = sc.read_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_3.h5ad')

slice1_slice2_pi = get_pi(slice1, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P10_slice1(2).pt')

slice1_slice2_pi = get_pi(slice2, slice1)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P10_slice2(1).pt')

slice1_slice2_pi = get_pi(slice2, slice3)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P10_slice2(3).pt')

slice1_slice2_pi = get_pi(slice3, slice2)
slice1_slice2_pi_edge_index = get_edge_index(slice1_slice2_pi)
torch.save(slice1_slice2_pi_edge_index, '/home/tyh/new_work/BiGATAE/tmp_data/cSCC_edges/P10_slice3(2).pt')