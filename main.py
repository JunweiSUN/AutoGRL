import argparse
from search_space import pruning_search_space_by_eda, pruning_search_space_by_shap
from data_prepare import load_data
from utils import Sampler
from utils import TransductiveTrainer, InductiveTrainer
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import torch
import random
from catboost import CatBoostRegressor, Pool

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main(args):
    setup_seed(args.seed)
    gbdt_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': args.gbdt_leaves,
        'learning_rate': args.gbdt_lr,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # build search space
    data = load_data(args.dataset)
    ss, _ = pruning_search_space_by_eda(data)

    if data.setting == 'inductivate':
        trainer = InductiveTrainer()
    else:
        trainer = TransductiveTrainer()
    sampler = Sampler(args.dataset, ss)

    archs = []
    val_scores = []

    top_archs = []
    top_val_scores = []
    top_test_scores = []

    # init training data for GBDT
    sampled_archs = sampler.sample(args.n)
    for arch in sampled_archs:
        data = sampler.load_data(arch)
        model = sampler.build_model(arch, data.x.shape[1], int(max(data.y)) + 1)
        trainer.init_trainer(model, arch[7], arch[6])
        val_score = trainer.train(data)
        archs.append(arch)
        val_scores.append(val_score)
        print(f'Number of evaluated archs: {len(archs)}')

    # train GBDT predictor
    for iter_round in range(1, args.iterations + 1):
        print(f'Iteration round {iter_round}')
        # train GBDT
        X = [[str(e) for e in row] for row in archs]
        y = np.array(val_scores)
        train_pool = Pool(X, y, cat_features=[i for i in range(len(X[0]))])
        # X = lgb.Dataset(pd.DataFrame(X, columns=ss.keys()), label=np.array(val_scores))
        # gbdt_model = lgb.train(gbdt_params, X, args.gbdt_num_boost_round, categorical_feature=ss.keys())
        gbdt_model = CatBoostRegressor(
            learning_rate=args.gbdt_lr,
            verbose=False
        )
        gbdt_model.fit(train_pool)
        # pruning search space
        ss = pruning_search_space_by_shap(archs, gbdt_model, ss, args.p)
        sampler.update_search_space(ss)
        

        # predict some archs
        sampled_archs = sampler.sample(args.m)
        X = [[str(e) for e in row] for row in sampled_archs]
        test_pool = Pool(X, cat_features=[i for i in range(len(X[0]))])
        predicted_val_scores = gbdt_model.predict(test_pool)

        # sort the archs according to the predicted value
        zipped = zip(sampled_archs, predicted_val_scores)
        zipped = sorted(zipped, key=lambda e: e[1], reverse=True) # sort in decreaing order
        sampled_archs, predicted_val_scores = zip(*zipped)
        sampled_archs, predicted_val_scores = list(sampled_archs), list(predicted_val_scores)

        # evaluate top k archs
        for i, arch in enumerate(sampled_archs):
            data = sampler.load_data(arch)
            model = sampler.build_model(arch, data.x.shape[1], int(max(data.y)) + 1)
            trainer.init_trainer(model, arch[7], arch[6])
            val_score = trainer.train(data)
            
            archs.append(arch)
            val_scores.append(val_score)
            print(f'Number of evaluated archs: {len(archs)}')
            if i + 1 >= args.k:
                break
        
        # sort all the evaluated archs
        zipped = zip(archs, val_scores)
        zipped = sorted(zipped, key=lambda e: e[1], reverse=True)
        archs, val_scores = zip(*zipped)
        archs, val_scores = list(archs), list(val_scores)

        # evaluate top k_test archs on test set
        for i, arch in enumerate(archs):
            data = sampler.load_data(arch)
            model = sampler.build_model(arch, data.x.shape[1], int(max(data.y)) + 1)
            trainer.init_trainer(model, arch[7], arch[6])
            val_score = trainer.train(data)
            test_score = trainer.test(data)
            
            top_archs.append(arch)
            top_val_scores.append(val_score)
            top_test_scores.append(test_score)
            
            print(f'Testing... arch {i + 1} | val score {val_score} | test score {test_score}')

            if i + 1 >= args.k_test:
                break
        
        zipped = zip(top_val_scores, top_test_scores)
        zipped = sorted(zipped, key=lambda e: e[0], reverse=True)
        best_val_score, best_test_score = zipped[0][0], zipped[0][1]

        # logging
        print(f'iteration {iter_round} | best val score {best_val_score} | corresponding test score {best_test_score}')

        pickle.dump((ss, sampler, trainer, archs, val_scores, gbdt_model, sampled_archs, predicted_val_scores, top_val_scores, top_test_scores), open(f'cache/gbdt/{args.dataset}_seed{args.seed}_round{iter_round}.pt', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoGRL')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--iterations', type=int, default=3, help='GBDT iteration rounds')
    parser.add_argument('--n', type=int, default=500, help='number of initial archs')
    parser.add_argument('--m', type=int, default=10000, help='number of archs to predict in each round')
    parser.add_argument('--k', type=int, default=500, help='number of top archs to evaluate in each round')
    parser.add_argument('--p', type=int, default=5, help='pruning features with lowest p shap values')
    parser.add_argument('--k_random', type=int, default=200, help='number of random archs to evaluate in each round')
    parser.add_argument('--k_test', type=int, default=10, help='number of archs that will be evaluated on test set')
    parser.add_argument('--gbdt_num_boost_round', type=int, default=100, help='GBDT argument')
    parser.add_argument('--gbdt_lr', type=float, default=0.05, help='GBDT argument')
    parser.add_argument('--gbdt_leaves', type=int, default=31, help='GBDT argument')
    args = parser.parse_args()
    main(args)
