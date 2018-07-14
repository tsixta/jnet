#!/bin/bash
python3 ../main.py  --resolution_levels '[-2,-1]' --dt_bound 3 --images_idx '{"01":[],"02":[]}' --mode vis --dataset_root DIC-C2DH-HeLa_test --model_file results/model_best_train_train --output_dir results
