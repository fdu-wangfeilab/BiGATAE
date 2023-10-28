import model
import torch
import scanpy as sc
import anndata as ad
import dataloader as dl
from utils.get_edge_index import get_edge_index
from paste_ import get_pi
import numpy as np

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
    sc.pp.highly_variable_genes(slice, n_top_genes=3000 ,flavor='seurat_v3')
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
    sc.pp.highly_variable_genes(slice, n_top_genes=3000,flavor='seurat_v3')
    slice = slice[:, slice.var.highly_variable]
    sc.pp.normalize_total(slice, target_sum=1e4)
    sc.pp.log1p(slice)
    return slice

slice1=load_cSCC_25910(P_name='P2',slice_num='1')
slice2=load_cSCC_25910(P_name='P2',slice_num='2')
slice3=load_cSCC_25910(P_name='P2',slice_num='3')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_2.h5ad')
slice3.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P2_3.h5ad')



slice1=load_cSCC_25910(P_name='P5',slice_num='1')
slice2=load_cSCC_25910(P_name='P5',slice_num='2')
slice3=load_cSCC_25910(P_name='P5',slice_num='3')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_2.h5ad')
slice3.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P5_3.h5ad')


slice1=load_cSCC_25910(P_name='P9',slice_num='1')
slice2=load_cSCC_25910(P_name='P9',slice_num='2')
slice3=load_cSCC_25910(P_name='P9',slice_num='3')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_2.h5ad')
slice3.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P9_3.h5ad')

slice1=load_cSCC_25910(P_name='P10',slice_num='1')
slice2=load_cSCC_25910(P_name='P10',slice_num='2')
slice3=load_cSCC_25910(P_name='P10',slice_num='3')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_2.h5ad')
slice3.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P10_3.h5ad')



slice1=load_cSCC_46(P_name='P4',slice_num='1')
slice2=load_cSCC_46(P_name='P4',slice_num='2')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P4_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P4_2.h5ad')

slice1=load_cSCC_46(P_name='P6',slice_num='1')
slice2=load_cSCC_46(P_name='P6',slice_num='2')

slice1.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P6_1.h5ad')
slice2.write_h5ad('/home/tyh/new_work/BiGATAE/tmp_data/cSCC_data/P6_2.h5ad')