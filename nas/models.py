import torch
import torch.nn as nn
from .layers import GNNLayer, NormLayer, ActivationLayer
import torch.nn.functional as F
from torch_geometric.nn import APPNP
__all__ = 'GNNModel'

class GNNModel(nn.Module):
    def __init__(self, args, in_channels, num_class):
        super(GNNModel, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.in_channels = in_channels
        self.num_class = num_class
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.layer_aggr_type = args.layer_aggr_type
        self.conv_type = args.conv_type
        self.norm_type = args.norm_type
        self.dropout = args.dropout
        self.act_first = args.act_first
        self.alpha = 0.1
        self.beta = 0.5

        self.generate_model(args)

    def generate_model(self, args):
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()

        if self.gnn_layers == 1:
            self.gnn_layers.append(GNNLayer(self.in_channels, self.num_class, args.aggr_type, args.conv_type))
        else:
            self.gnn_layers.append(GNNLayer(self.in_channels, args.hidden_size, args.aggr_type, args.conv_type))
            self.norm_layers.append(NormLayer(args.norm_type, args.hidden_size))
            self.act_layers.append(ActivationLayer(args.act_type))

            if self.layer_aggr_type != 'dense':
                for i in range(1, self.num_layers):
                    self.gnn_layers.append(GNNLayer(args.hidden_size, args.hidden_size, args.aggr_type, args.conv_type))
                    self.norm_layers.append(NormLayer(args.norm_type, args.hidden_size))
                    self.act_layers.append(ActivationLayer(args.act_type))
            else:
                for i in range(1, self.num_layers):
                    self.gnn_layers.append(GNNLayer(i * args.hidden_size, args.hidden_size, args.aggr_type, args.conv_type))
                    self.norm_layers.append(NormLayer(args.norm_type, args.hidden_size))
                    self.act_layers.append(ActivationLayer(args.act_type))
        
        if self.layer_aggr_type == 'jk':
            self.out_lin = nn.Linear(self.num_layers * args.hidden_size, self.num_class)
        else:
            self.out_lin = nn.Linear(args.hidden_size, self.num_class)

        if self.conv_type == 'appnp':
            self.prop = APPNP(10, 0.1)
        

    def forward(self, x, edge_index):
        hs = []
        gnn_layer = self.gnn_layers[0]
        norm_layer = self.norm_layers[0]
        act_layer = self.act_layers[0]
        h = gnn_layer(x, edge_index)
        if self.act_first:
            h = act_layer(h)
            h = norm_layer(h)
        else:
            h = act_layer(h)
            h = norm_layer(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        hs.append(h)
    
        for i in range(1, self.num_layers):
            gnn_layer = self.gnn_layers[i]
            norm_layer = self.norm_layers[i]
            act_layer = self.act_layers[i]
            
            if self.layer_aggr_type == 'dense':
                h_in = torch.cat(hs, dim=1)
            else:
                h_in = hs[-1]

            h = gnn_layer(h_in, edge_index)

            if self.layer_aggr_type == 'res':
                h = h + hs[-1]
            if self.act_first:
                h = act_layer(h)
                h = norm_layer(h)
            else:
                h = act_layer(h)
                h = norm_layer(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            hs.append(h)

        if self.layer_aggr_type == 'jk':
            out = self.out_lin(torch.cat(hs, dim=-1))
        else:
            out = self.out_lin(hs[-1])
        
        if self.conv_type == 'appnp':
            out = self.prop(out, edge_index)
        
        return F.log_softmax(out, dim=1)
