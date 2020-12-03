from search_space import precompute_cache, pruning_search_space
from data_prepare import load_data
from torch_geometric.transforms import NormalizeFeatures

def main():
    # for name in ['cora', 'usa-airports', 'photo', 'wikics']:
    for name in ['photo']:
        data = load_data(name, transform=NormalizeFeatures())
        ss, (data_aug, fe, hpo, nas) = pruning_search_space(data)
        precompute_cache(name, data, data_aug, fe)





if __name__ == '__main__':
    main()