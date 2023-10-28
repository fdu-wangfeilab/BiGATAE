import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv

from layer.bipartite_gat import BipartiteGATEncoder

class Encoder(torch.nn.Module):
    def __init__(self, slice1_spots_num, slice2_spots_num,
                 slice1_genes_num, slice2_genes_num, hidden_dim):
        super(Encoder, self).__init__()
        self.slice1_spots_num = slice1_spots_num
        self.slice2_spots_num = slice2_spots_num
        self.slice2_to_slice1 = BipartiteGATEncoder(slice2_genes_num, slice1_genes_num, hidden_dim, dropout=0.1, flow='source_to_target')
        self.act = nn.ReLU()

    def forward(self, pi_edge_index, slice1_X, slice2_X):
        '''
        pi_edge_index：paste得到的映射矩阵pi---->edge
        slice1_edge_index：tist得到的邻接矩阵---->edge
        slice1_edge_weight: GCN使用
        '''
        slice1_feature_from_slice2 = self.act(self.slice2_to_slice1((slice2_X, slice1_X), pi_edge_index, size=(self.slice2_spots_num, self.slice1_spots_num)))
        
        return slice1_feature_from_slice2