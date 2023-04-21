#!/bin/bash

dataset_lst=("yelp-chi")

hidden_channels_lst=()
gat_heads_lst=()

if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "pokec" ]; then 
    gat_heads_lst=(2 4)
    hidden_channels_lst=(4 8 12)
else 
    gat_heads_lst=(2 4 8)
    hidden_channels_lst=(4 8 12 32)
fi 

for dataset in "${dataset_lst[@]}"; do
	for hidden_channels in "${hidden_channels_lst[@]}"; do
		for gat_heads in "${gat_heads_lst[@]}"; do
			CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset $dataset \
				--method gat --num_layers 2 --hidden_channels $hidden_channels \
				--lr 0.01 --gat_heads $gat_heads --display_step 25 --runs 3 > logs/${dataset}_gat_${hidden_channels}_${gat_heads}.out
done            
done
done
