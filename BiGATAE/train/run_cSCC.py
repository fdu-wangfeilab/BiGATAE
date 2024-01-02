import model
import torch
import scanpy as sc
import anndata as ad
from dataloader import load_cSCC_25910,load_cSCC_46
from utils.get_edge_index import get_edge_index
from paste_ import get_pi
import numpy as np

# If you change the format of the original data to h5ad, please use the following code，
#It should be noted that patients 4 and 6 are inconsistent with others
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

#get bipartite graph

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

# The training process for the cSCC dataset follows the same procedure as that for the BC dataset and the Human DLPFC dataset.

slice1_spots_num = slice1.X.shape[0]
slice2_spots_num = slice2.X.shape[0]
slice1_genes_num = slice1.X.shape[1]
slice2_genes_num = slice2.X.shape[1]

# model init
hidden_dim = 1000
device = torch.device('cuda:1')
model_ = model(slice1_spots_num, slice2_spots_num, slice1_genes_num, slice2_genes_num, hidden_dim).to(device)
# train
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
    print("epoch=", epoch, " loss=", loss.data.float()) 

model_.eval()
slice1.X=model_.encoder(slice1_slice2_pi_edge_index.to(device), slice4_X, slice3_X).detach().cpu().numpy()
slice1.write_h5ad('./tmp_data/cSS_enhanced_data/P10_slice1(2).h5ad')

