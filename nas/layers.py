import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_scatter import scatter_add
from torch_geometric.nn import BatchNorm, LayerNorm, InstanceNorm, PairNorm
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, TAGConv, ARMAConv, GINConv, SplineConv
import math

__all__ = ['GNNLayer', 'NormLayer', 'ActivationLayer']

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, aggr_type, conv_type):
        super(GNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_type = conv_type

        if self.conv_type.startswith('gat'):
            heads = int(self.conv_type[4:])
            self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False)
        elif self.conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif self.conv_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif self.conv_type == 'cheb':
            self.conv = ChebConv(in_channels, out_channels, K=2)
        elif self.conv_type == 'tag':
            self.conv = TAGConv(in_channels, out_channels)
        elif self.conv_type == 'arma':
            self.conv = ARMAConv(in_channels, out_channels)
        elif self.conv_type == 'gin':
            self.conv = GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)))
        self.conv.aggr = self.aggr_type

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)

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
        if act_type == 'tanh':
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
        
