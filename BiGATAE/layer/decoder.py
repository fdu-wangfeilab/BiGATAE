import torch
from torch import nn
from torch_geometric.nn import GATConv
from layer.bipartite_gat import BipartiteGATDecoder

class Decoder(torch.nn.Module):
    """
    decoder
    return recreate slice.X
    """

    def __init__(self, slice1_spots_num, slice2_spots_num,
                 slice1_genes_num, slice2_genes_num, hidden_dim):
        super(Decoder, self).__init__()
        self.slice1_spots_num = slice1_spots_num
        self.slice2_spots_num = slice2_spots_num
        self.slice2_to_slice1 = BipartiteGATDecoder(slice2_genes_num, slice1_genes_num, hidden_dim, dropout=0.1, flow='source_to_target')
        self.act = nn.ReLU()

    def forward(self, pi_edge_index, slice1_feature, slice2_X):
        # slice1_feature = self.act(self.slice1_gat(slice1_feature, slice1_edge_index))
        slice1_X0 = self.act(self.slice2_to_slice1((slice2_X, slice1_feature), pi_edge_index, size=(self.slice2_spots_num, self.slice1_spots_num)))
        return slice1_X0