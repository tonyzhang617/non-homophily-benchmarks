#!/bin/bash

dataset_lst=("yelp-chi")

hidden_channels_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    hidden_channels_lst=(4 8 12)
else 
    hidden_channels_lst=(4 8 12 32)
fi 

for dataset in "${dataset_lst[@]}"; do
	for hidden_channels in "${hidden_channels_lst[@]}"; do
		CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset $dataset \
			--method gcn --num_layers 2 --hidden_channels $hidden_channels \
			--lr 0.01 --display_step 25 --runs 5 > logs/${dataset}_gcn_${hidden_channels}.out
done            
done
