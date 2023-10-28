import scanpy as sc
import numpy as np
import ot

# def create_similar_network0(slice):
def create_similar_network0(slice):
    sc.pp.highly_variable_genes(slice, n_top_genes=2000, flavor='seurat_v3')
    sc.pp.pca(slice)
    # sc.pp.neighbors(slice, n_pcs=50)
    sc.pp.neighbors(slice, use_rep='X')
    M = slice.obsp['connectivities'].todense()
    # print(slice1.obsp['connectivities'].todense()) 
    # print(slice1.obsp['distances'])
    coordinates = slice.obsm['spatial'].copy()
    D = ot.dist(coordinates, coordinates, metric='euclidean')
    row, col = np.diag_indices_from(D)
    D[row,col] = 1
    S = M/D
    max = S.max()
    S[S==0]=1
    min = S.min()
    S[S==1]=0
    S_ = (S-min)/(max-min)
    S_[S_==-min/(max-min)]=0
    return S_

