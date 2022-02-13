1. install torch, torch-geometric and other packages (sparsesvd needs vc++ 14 or higher if you are using windows)
2. install julia and julia packages
3. set num_workers=0 in pytorch's dataloader if you are using windows, since pytorch doesn't support multiprocessing on windows
4. create some temp dirs (logs, figures, embeddings, data, cache(/{data, edge, feature, gbdt, label, node}))