#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python piqa.py --model-type bert-base-uncased --output_dir bert-base-piqa-9595 --seed 9595 --percentage 20

CUDA_VISIBLE_DEVICES=2 python piqa.py --model-type bert-base-uncased --output_dir bert-base-piqa-0 --seed 0 --percentage 20

CUDA_VISIBLE_DEVICES=3 python piqa.py --model-type bert-base-uncased --output_dir bert-base-piqa-42 --seed 42 --percentage 20
