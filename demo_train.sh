#!/bin/bash

python models/slim/train_image_classifier.py \
    --gpu 2 \
    --save_summaries_secs 120 \
    --save_interval_secs 120 \
    --train_dir /runs/demo_train \
    --dataset_name vinci_demo \
    --dataset_dir /data/tf_record \
    --model_name inception_v1 \
    --batch_size 16 \
    --num_epochs_per_decay 2
