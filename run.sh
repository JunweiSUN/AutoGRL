CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --dataset cora --seed 1 > cora_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --dataset usa-airports --seed 1 > usa_airports_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --dataset photo --seed 1 > photo_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --dataset wikics --seed 1 > wikics_1.log 2>&1 &
