#!/bin/bash

python models/slim/eval_image_classifier.py \
    --gpu 1 \
    --dataset_split_name validation \
    --checkpoint_path /runs/demo_train \
    --eval_dir /runs/demo_eval \
    --dataset_name vinci_demo \
    --dataset_dir /data/tf_record \
    --model_name inception_v1 \
    --batch_size 16 \
    --eval_time 120
