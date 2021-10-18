#!/usr/bin/env bash

srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_9595_6000-3e4 --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_0_6000-3e4 --seed 0 --learning-rate 3e-4
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_42_6000-3e4 --seed 42 --learning-rate 3e-4

srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_9595_6000-1e5 --seed 9595 --learning-rate 1e-5
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_0_6000-1e5 --seed 0 --learning-rate 1e-5
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_42_6000-1e5 --seed 42 --learning-rate 1e-5

srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_9595_6000-3e5 --seed 9595 --learning-rate 3e-5
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_0_6000-3e5 --seed 0 --learning-rate 3e-5
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_42_6000-3e5 --seed 42 --learning-rate 3e-5

srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_9595_6000-5e4 --seed 9595 --learning-rate 5e-4
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_0_6000-5e4 --seed 0 --learning-rate 5e-4
srun --gres=gpu:1 --qos general python hellaswag.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire/Step6000/pytorch_model.bin --output_dir hellaswag_42_6000-5e4 --seed 42 --learning-rate 5e-4

