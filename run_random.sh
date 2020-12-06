CUDA_VISIBLE_DEVICES=0 nohup python -u AutoGRL_random.py --dataset cora --seed 0 > cora_random_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u AutoGRL_random.py --dataset usa-airports --seed 0 > usa_airports_random_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u AutoGRL_random.py --dataset photo --seed 0 > photo_random_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u AutoGRL_random.py --dataset wikics --seed 0 > wikics_random_0.log 2>&1 &
