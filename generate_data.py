from torch_geometric.transforms import NormalizeFeatures
from search_space import pruning_search_space_by_eda, precompute_cache
from data_prepare import load_data

for name in ['cora', 'usa-airports', 'photo', 'wikics']:
    data = load_data(name, transform=NormalizeFeatures())
    ss, (data_aug, fe, hpo, nas) = pruning_search_space_by_eda(data)
    precompute_cache(name, data, data_aug, fe)