import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, PairNorm, GCNConv, GATConv, SAGEConv
import math

__all__ = ['GNNLayer', 'NormLayer', 'ActivationLayer']

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr, att_type, dropout=0, bias=True, **kwargs):
        super(GNNLayer, self).__init__(aggr=aggr, **kwargs)
        self.heads = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_type = att_type
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if self.att_type in ['gat', 'sym-gat', 'cos']:
            self.att = Parameter(torch.Tensor(1, out_channels * 2))
            glorot(self.att)
        elif self.att_type == 'linear':
            self.att_l = Parameter(torch.Tensor(1, out_channels))
            self.att_r = Parameter(torch.Tensor(1, out_channels))
            glorot(self.att_l)
            glorot(self.att_r)

        if self.att_type == 'gene-linear':
            self.gene_layer = nn.Linear(out_channels, 1, bias=False)
            glorot(self.gene_layer.weight)
        if self.att_type == 'gcn':
            self.gcn_weight = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        glorot(self.weight)
        zeros(self.bias)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def message(self, edge_index, index, x_i, x_j, ptr, num_nodes):
        if self.att_type == 'const':
            # if self.dropout:
            #     x_j = F.dropout(x_j, p=self.dropout, training=self.training)
            alpha = torch.ones((x_j.shape[0], 1))
        
        if self.att_type == 'gcn':
            if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):
                _, norm = self.norm(edge_index, num_nodes, None)
                self.gcn_weight = norm
            return self.gcn_weight.view(-1, 1, 1) * x_j

        elif self.att_type == 'gat':
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=num_nodes)
            # if self.dropout:
            #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        elif self.att_type == 'sym-gat':
            alpha_l = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha_l = F.leaky_relu(alpha_l, 0.2)
            alpha_r = (torch.cat([x_j, x_i], dim=-1) * self.att).sum(dim=-1)
            alpha_r = F.leaky_relu(alpha_r, 0.2)
            alpha = alpha_l + alpha_r
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=num_nodes)
        elif self.att_type == 'cos':
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=num_nodes)
        elif self.att_type == 'linear':
            alpha = (x_i * self.att_l).sum(dim=-1) + (x_j * self.att_r).sum(dim=-1)
            alpha = torch.tanh(alpha)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=num_nodes)
        elif self.att_type == 'gene-linear':
            alpha = torch.tanh(x_i + x_j)
            alpha = self.gene_layer(alpha)
            alpha = softmax(alpha, index=index, ptr=ptr, num_nodes=num_nodes)
        
        return x_j * alpha.unsqueeze(-1)



    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        x = torch.mm(x, self.weight)
        out = self.propagate(edge_index, x=x, num_nodes=num_nodes)

        if self.bias is not None:
            out += self.bias
        
        return out

class NoNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NoNorm, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class NormLayer(nn.ModuleList):
    def __init__(self, norm_type, in_channels):
        super(NormLayer, self).__init__()
        if norm_type == 'bn':
            self.norm = BatchNorm(in_channels)
        elif norm_type == 'ln':
            self.norm = LayerNorm(in_channels)
        elif norm_type == 'in':
            self.norm = InstanceNorm(in_channels)
        else:
            self.norm = NoNorm(in_channels)
    
    def forward(self, x):
        return self.norm(x)

class ActivationLayer(nn.ModuleList):
    def __init__(self, act_type):
        super(ActivationLayer, self).__init__()
        if act_type == 'tan_h':
            self.act = torch.nn.Tanh()
        elif act_type == 'relu':
            self.act = torch.nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = torch.nn.LeakyReLU()
        elif act_type == 'prelu':
            self.act = torch.nn.PReLU()
        elif act_type == 'elu':
            self.act = torch.nn.ELU()
        elif act_type == 'identity':
            self.act = lambda x:x
    
    def forward(self, x):
        return self.act(x)
        
