import torch

def dropout_x(data, p):
    data.x = torch.nn.functional.dropout(data.x, p)

def dropout_nodes(data, p):
    num_nodes = data.x.shape[0]
    mask = torch.full((num_nodes, 1), 1-p)
    mask = torch.bernoulli(mask).expand(data.x.shape)
    data.x = data.x.mul(mask) / (1-p)
