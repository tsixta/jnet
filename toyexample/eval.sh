#!/bin/bash
python3 ../main.py  --resolution_levels '[-2,-1]' --dt_bound 3 --images_idx '{"01":["002","005","021"],"02":["006","007","014"]}' --mode eval --dataset_root DIC-C2DH-HeLa_training --model_file results/model_best_train_train --output_dir results
