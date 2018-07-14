#!/bin/bash
python3 ../main.py --images_idx '{"01":["002","005","021"],"02":["006","007","014"]}' --load_dataset_to_ram 1 --num_workers 0 --dataset_len_multiplier 2 --batch_size 2 --num_epochs 100 --save_model_frequency 10 --aug_elastic_params "[[50,5,5],[-1,-1,1]]" --resolution_levels "[-2,-1]" --structure "[[3,16,3],[2,8,3]]" --dt_bound 3 --validation_percentage 0.17 --mode train --dataset_root DIC-C2DH-HeLa_training --output_dir results