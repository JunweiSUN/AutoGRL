import argparse

parser = argparse.ArgumentParser(description='AutoGRL')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--iterations', type=int, default=3, help='GBDT iteration rounds')
parser.add_argument('--n', type=int, default=500, help='number of initial archs')
parser.add_argument('--m', type=int, default=10000, help='number of archs to predict in each round')
parser.add_argument('--k', type=int, default=500, help='number of top archs to evaluate in each round')
parser.add_argument('--p', type=int, default=5, help='pruning features with lowest p shap values')
parser.add_argument('--k_test', type=int, default=10, help='number of archs that will be evaluated on test set')
parser.add_argument('--gbdt_lr', type=float, default=0.05, help='GBDT argument')
args = parser.parse_args()

import random
random.seed(args.seed)
import numpy as np
np.random.seed(args.seed)
import torch
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

from search_space import pruning_search_space_by_eda, pruning_search_space_by_shap
from data_prepare import load_data
from utils import Sampler
from utils import TransductiveTrainer, InductiveTrainer
import pandas as pd
import lightgbm as lgb
import pickle
from catboost import CatBoostRegressor, Pool

def main(args):

    # build search space
    data = load_data(args.dataset, args.seed)
    ss, _ = pruning_search_space_by_eda(data)

    if data.setting == 'inductive':
        trainer = InductiveTrainer()
    else:
        trainer = TransductiveTrainer()

    sampler = Sampler(args.dataset, ss)

    archs = []
    val_scores = []
    test_scores = []

    # init training data for GBDT
    sampled_archs = sampler.sample(3000)

    i = 0
    while i < len(sampled_archs):
        arch = sampled_archs[i]
        data = sampler.load_data(arch)
        try:
            model = sampler.build_model(arch, data.x.shape[1], int(max(data.y)) + 1)
            trainer.init_trainer(model, arch[7], arch[6])
            val_score = trainer.train(data)
            test_score = trainer.test(data)
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):     # CUDA OOM, sample another arch
                print(e)
                sampled_archs += sampler.sample(1)
                i += 1
                continue
            else:
                raise e

        archs.append(arch)
        val_scores.append(val_score)
        test_scores.append(test_score)
        print(arch, f'real val score: {val_score} | real test score: {test_score}')
        print(f'Number of evaluated archs: {len(archs)}')

        i += 1
        if i % 500 == 0:
            print(f'Round {i // 500} | best test score: {max(test_scores)}')
        
        if i >= 2000:
            break

main(args)
