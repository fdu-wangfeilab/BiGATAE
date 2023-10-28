import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import copy
from igraph import Graph as IGraph

from model import model
from utils.get_edge_index import get_edge_index
from paste_ import get_pi
from tist import create_similar_network0


def intersect(lst1, lst2): 

    temp = set(lst2)
    print(len(temp))
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def load_slices_fourST(data_dir='/home/sxa/Datasets/breast_cancer_data/', slice_names=["slice1", "slice2", "slice3", "slice4"]):
    slices = []  
    for slice_name in slice_names:
        print(slice_name)
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

slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

# slice1
slice1_spots_num = slice1.X.shape[0]
slice2_spots_num = slice2.X.shape[0]
slice1_genes_num = slice1.X.shape[1]
slice2_genes_num = slice2.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice1(2).pt')

device = torch.device('cuda:0')
model_ = model(slice1_spots_num, slice2_spots_num, slice1_genes_num, slice2_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice1_X = torch.tensor(slice1.X).to(device)
slice2_X = torch.tensor(slice2.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice1_X, slice2_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice1_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())


slice1.write_h5ad('./tmp_data/BC_data/slice1.h5ad')
model_.eval()
slice1.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice1_X, slice2_X).detach().cpu().numpy()
slice1.write_h5ad('./tmp_data/BC_enhanced_data/slice1(2).h5ad')



# slice2(1)
slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

slice1_spots_num = slice1.X.shape[0]
slice2_spots_num = slice2.X.shape[0]
slice1_genes_num = slice1.X.shape[1]
slice2_genes_num = slice2.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice2(1).pt')

device = torch.device('cuda:0')
model_ = model(slice2_spots_num, slice1_spots_num, slice2_genes_num, slice1_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice1_X = torch.tensor(slice1.X).to(device)
slice2_X = torch.tensor(slice2.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice2_X, slice1_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice2_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())

slice2.write_h5ad('./tmp_data/BC_data/slice2.h5ad')
model_.eval()
slice2.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice2_X, slice1_X).detach().cpu().numpy()
slice2.write_h5ad('./tmp_data/BC_enhanced_data/slice2(1).h5ad')


# slice2(3)
slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

slice3_spots_num = slice3.X.shape[0]
slice2_spots_num = slice2.X.shape[0]
slice3_genes_num = slice3.X.shape[1]
slice2_genes_num = slice2.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice2(3).pt')

device = torch.device('cuda:0')
model_ = model(slice2_spots_num, slice3_spots_num, slice2_genes_num, slice3_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice3_X = torch.tensor(slice3.X).to(device)
slice2_X = torch.tensor(slice2.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice2_X, slice3_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice2_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())

model_.eval()
slice2.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice2_X, slice3_X).detach().cpu().numpy()
slice2.write_h5ad('./tmp_data/BC_enhanced_data/slice2(3).h5ad')

# slice3(2)
slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

slice3_spots_num = slice3.X.shape[0]
slice2_spots_num = slice2.X.shape[0]
slice3_genes_num = slice3.X.shape[1]
slice2_genes_num = slice2.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice3(2).pt')

device = torch.device('cuda:0')
model_ = model(slice3_spots_num, slice2_spots_num, slice3_genes_num, slice2_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice3_X = torch.tensor(slice3.X).to(device)
slice2_X = torch.tensor(slice2.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice3_X, slice2_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice3_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())

slice3.write_h5ad('./tmp_data/BC_data/slice3.h5ad')
model_.eval()
slice3.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice3_X, slice2_X).detach().cpu().numpy()
slice3.write_h5ad('./tmp_data/BC_enhanced_data/slice3(2).h5ad')

# slice3(4)
slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

slice3_spots_num = slice3.X.shape[0]
slice4_spots_num = slice4.X.shape[0]
slice3_genes_num = slice3.X.shape[1]
slice4_genes_num = slice4.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice3(4).pt')

device = torch.device('cuda:0')
model_ = model(slice3_spots_num, slice4_spots_num, slice3_genes_num, slice4_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice3_X = torch.tensor(slice3.X).to(device)
slice4_X = torch.tensor(slice4.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice3_X, slice4_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice3_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())

model_.eval()
slice3.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice3_X, slice4_X).detach().cpu().numpy()
slice3.write_h5ad('./tmp_data/BC_enhanced_data/slice3(4).h5ad')

# slice4(3)
slices = load_slices_fourST()
slice1, slice2, slice3, slice4 = slices

slice3_spots_num = slice3.X.shape[0]
slice4_spots_num = slice4.X.shape[0]
slice3_genes_num = slice3.X.shape[1]
slice4_genes_num = slice4.X.shape[1]
hidden_dim = 1000
slice1_slice2_pi_edge_index = torch.load('./tmp_data/BC_edges/slice4(3).pt')

device = torch.device('cuda:0')
model_ = model(slice4_spots_num, slice3_spots_num, slice4_genes_num, slice3_genes_num, hidden_dim).to(device)

epochs = 1000
mse_loss = nn.MSELoss()
ae_optim = optim.Adam(model_.parameters(), lr=0.0001)
slice3_X = torch.tensor(slice3.X).to(device)
slice4_X = torch.tensor(slice4.X).to(device)

for epoch in range(epochs):
    _, recreated_slice1_X = model_(slice1_slice2_pi_edge_index.to(device), slice4_X, slice3_X) # slice1_edge_index.to(device)
    loss = mse_loss(recreated_slice1_X, slice4_X)
    ae_optim.zero_grad()
    loss.backward()
    ae_optim.step()
    #print("epoch=", epoch, " loss=", loss.data.float())

slice4.write_h5ad('./tmp_data/BC_data/slice4.h5ad')
model_.eval()
slice4.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice4_X, slice3_X).detach().cpu().numpy()
slice4.write_h5ad('./tmp_data/BC_enhanced_data/slice4(3).h5ad')