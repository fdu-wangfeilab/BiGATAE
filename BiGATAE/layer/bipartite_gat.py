import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from utils.weight_inits import glorot, zeros

class BipartiteGATEncoder(MessagePassing):

    def __init__(self, in_channels_i, in_channels_j, out_channels, negative_slope=0.2, dropout=0.0, bias=True, **kwargs):
        """
        :param in_channels: Size of each input sample.
        :param out_channels: Size of each output sample.
        :param negative_slope: LeakyReLU angle of the negative.
        :param dropout: Dropout probability of the normalized attention coefficients which exposes each node to a
                        stochastically sampled neighborhood during training.
        :param bias: If set to False, the layer will not learn an additive bias.
        :param kwargs: Additional arguments of `torch_geometric.nn.conv.MessagePassing.
        """
        super(BipartiteGATEncoder, self).__init__(aggr='add', **kwargs)

        self.in_channels_i = in_channels_i
        self.in_channels_j = in_channels_j
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.att = Parameter(torch.Tensor(1, in_channels_i)) # 1000

        # self.mlp_i = Linear(in_channels_i, out_channels)
        # self.mlp_j = Linear(in_channels_j, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels_i)) # 1000
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        # feature dimension alignment
        # x = (self.mlp_i(x[0]), self.mlp_j(x[1]))
        propagate_result = self.propagate(edge_index, size=size, x=x)
        # print("propagate_result:", propagate_result.shape, propagate_result)
        index = 1 if self.flow == "source_to_target" else 0
        final_result = propagate_result + x[index]
        # print("final_result:", final_result.shape, final_result)
        # print("x[index]:", x[index].shape, x[index]) torch.Size([3639, 3000])
        return final_result

    def message(self, edge_index_i, x_j, size_i):
        # Compute attention coefficients.
        # print("x_j:", x_j.shape, x_j) torch.Size([7311, 3000])
        alpha = (x_j * self.att).sum(dim=-1) # torch.Size([7311])
        # print("alpha:", alpha.shape, alpha)

        # self.negative_slope=0.2
        alpha = F.leaky_relu(alpha, self.negative_slope) # torch.Size([7311])
        # print("alpha:", alpha.shape, alpha)
        # alpha: torch.Size([503])
        alpha = softmax(alpha, edge_index_i, num_nodes = size_i) # torch.Size([7311])
        # print("alpha:", alpha.shape, alpha)
        # alpha: torch.Size([503])
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) # torch.Size([7311])
        # print("alpha:", alpha.shape, alpha)
        
        out = x_j * alpha.view(-1, 1)
        # print("alpha.view(-1, 1):", alpha.view(-1, 1).shape, alpha.view(-1, 1)) torch.Size([7311, 1])
        # print("out:", out.shape, out) torch.Size([7311, 3000])
        return out

    def update(self, aggr_out):
        '''
        aggr_out: torch.Size([254, 1000])
        self.bias: # alpha: torch.Size([1000]
        '''
        if self.bias is not None:
            aggr_out = aggr_out + self.bias.unsqueeze(dim=0).repeat(aggr_out.shape[0], 1)
        # print("aggr_out:", aggr_out.shape, aggr_out)
        return aggr_out

class BipartiteGATDecoder(MessagePassing):

    def __init__(self, in_channels_i, in_channels_j, out_channels, negative_slope=0.2, dropout=0.0, bias=True, **kwargs):
        """
        :param in_channels: Size of each input sample.
        :param out_channels: Size of each output sample.
        :param negative_slope: LeakyReLU angle of the negative.
        :param dropout: Dropout probability of the normalized attention coefficients which exposes each node to a
                        stochastically sampled neighborhood during training.
        :param bias: If set to False, the layer will not learn an additive bias.
        :param kwargs: Additional arguments of `torch_geometric.nn.conv.MessagePassing.
        """
        super(BipartiteGATDecoder, self).__init__(aggr='add', **kwargs)

        self.in_channels_i = in_channels_i
        self.in_channels_j = in_channels_j
        # self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.att = Parameter(torch.Tensor(1, in_channels_i))

        # self.mlp_i = Linear(in_channels_i, out_channels)
        # self.mlp_j = Linear(in_channels_j, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels_i))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        # feature dimension alignment
        # x = (self.mlp_i(x[0]), self.mlp_j(x[1]))
        '''
        x[0]: slice2 250*1000
        x[1]: slice1 250*1000
        edge_index: 2*503
        size: (250, 254)
        ''' 
        propagate_result = self.propagate(edge_index, size=size, x=x)
        # print(propagate_result.shape)
        index = 1 if self.flow == "source_to_target" else 0
        final_result = x[index] - propagate_result
        return final_result

    def message(self, edge_index_i, x_j, size_i):
        # Compute attention coefficients.
        '''
        edge_index_i: 503
        x_i: 503*1000
        x_j: 503*1000
        size_i: 254
        ''' 
        alpha = (x_j * self.att).sum(dim=-1)
        '''
        self.att: torch.Size([1000])
        (x_j * self.att): torch.Size([503, 1000])
        alpha: torch.Size([503, 1])
        '''
        # self.negative_slope=0.2
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # alpha: torch.Size([503])
        alpha = softmax(alpha, edge_index_i, num_nodes = size_i)
        # alpha: torch.Size([503])
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        '''
        alpha: torch.Size([503])----->503*1
        out: torch.Size([503, 1000])
        '''
        out = x_j * alpha.view(-1, 1)
        return out

    def update(self, aggr_out):
        '''
        aggr_out: torch.Size([254, 1000])
        self.bias: # alpha: torch.Size([1000]
        '''
        # print(aggr_out.shape)
        if self.bias is not None:
            # print(aggr_out.shape, self.bias.unsqueeze(dim=0).repeat(aggr_out.shape[0], 1).shape)
            aggr_out = aggr_out + self.bias.unsqueeze(dim=0).repeat(aggr_out.shape[0], 1)
        return aggr_out