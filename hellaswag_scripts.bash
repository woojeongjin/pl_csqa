#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python hellaswag.py --model-type bert-base-uncased --output_dir bert-base-hellaswag-9595 --seed 9595 --percentage 20

CUDA_VISIBLE_DEVICES=2 python hellaswag.py --model-type bert-base-uncased --output_dir bert-base-hellaswag-0 --seed 0 --percentage 20

CUDA_VISIBLE_DEVICES=2 python hellaswag.py --model-type bert-base-uncased --output_dir bert-base-hellaswag-42 --seed 42 --percentage 20
