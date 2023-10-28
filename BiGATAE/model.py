from layer.decoder import *
from layer.encoder import *

class model(torch.nn.Module):
    def __init__(self, slice1_spots_num, slice2_spots_num, slice1_genes_num, slice2_genes_num, hidden_dim):
        super(model, self).__init__()
        self.encoder = Encoder(slice1_spots_num, slice2_spots_num, slice1_genes_num, slice2_genes_num, hidden_dim)
        self.decoder = Decoder(slice1_spots_num, slice2_spots_num, slice1_genes_num, slice2_genes_num, hidden_dim)

    def forward(self, pi_edge_index, slice1_X, slice2_X): # slice1_edge_index
        slice1_embedding = self.encoder(pi_edge_index, slice1_X, slice2_X) # slice1_edge_index

        recreated_slice1_X = self.decoder(pi_edge_index, slice1_embedding, slice2_X)

        return slice1_embedding, recreated_slice1_X