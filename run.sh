CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset cora --seed 0 > cora_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset usa-airports --seed 0 > usa_airports_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset photo --seed 0 > photo_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset wikics --seed 0 > wikics_0.log 2>&1 &
