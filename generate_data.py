seed = 0

import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

from torch_geometric.transforms import NormalizeFeatures
from search_space import pruning_search_space_by_eda, precompute_cache
from data_prepare import load_data

# for name in ['cora', 'usa-airports', 'photo', 'wikics']:
for name in ['photo']:
    data = load_data(name, seed, transform=NormalizeFeatures())
    ss, (data_aug, fe, hpo, nas) = pruning_search_space_by_eda(data)
    precompute_cache(name, data, data_aug, fe)
