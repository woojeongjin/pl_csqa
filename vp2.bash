#!/usr/bin/env bash


srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_9595 --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_0 --seed 0 --learning-rate 3e-4
srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_42 --seed 42 --learning-rate 3e-4


srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_9595_5 --seed 9595 --percentage 5 --learning-rate 3e-4
srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_0_5 --seed 0 --percentage 5 --learning-rate 3e-4
srun --gres=gpu:1080:1 --qos general python vp.py --output_dir vp_42_5 --seed 42 --percentage 5 --learning-rate 3e-4
