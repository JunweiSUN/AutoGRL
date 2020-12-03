import pickle
from nas.models import GNNModel
import argparse
import random
import torch

class Sampler:
    def __init__(self, name, search_space):
        self.name = name
        self.ss = search_space
        self.hashs = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def sample(self, n):
        '''
        randomly sample n structures,
        return their tabular representations
        '''
        archs = []
        i = 0
        while i < n:
            arch = []
            for k, v in self.ss.items():
                arch.append(random.choice(v))
            combine_str = '_'.join([self.name] + [str(e) for e in arch])
            if combine_str not in self.hashs:
                self.hashs[combine_str] = None
                archs.append(arch)
                i += 1
        
        return archs

    def update_search_space(self, search_space):
        self.ss = search_space

    def load_data(self, arch):
        '''
        load a pre-computed data according to the sampled arch
        '''
        combine_str = '_'.join([self.name] + [str(e) for e in arch[:6]])
        data = pickle.load(open(f'cache/data/{combine_str}.pt', 'rb'))
        
        return data.to(self.device)
    
    def build_model(self, arch, in_channels, num_class):
        '''
        build a model according to the sampled arch
        '''
        args = argparse.Namespace()
        args.dropout = arch[8]
        args.norm_type = arch[9]
        args.act_type = arch[10]
        args.hidden_size = arch[11]
        args.act_first = arch[12]
        args.conv_type = arch[13]
        args.aggr_type = arch[14]
        args.layer_aggr_type = arch[15]
        args.num_layers = arch[16]
        model = GNNModel(args, in_channels, num_class).to(self.device)
        return model