#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py infer \
        --img_path /root/datasets/COCO/images/test2017/000000038717.jpg \
        --checkpoint 90000 \
        --backbone fpn50 \
        --num_classes 80 \
        --num_features 256 \
        --cuda \
        --resize 400 \
        --max_size 600 \
        --dec_threshold 0.05 \
        --nms_threshold 0.5 \
        --topn 1000 \
        --ndetections 100 \
        --checkpoint_dir experiments/checkpoints \
        --result_dir experiments/results