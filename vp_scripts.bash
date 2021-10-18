#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/0 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-0 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/64 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-64 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/128 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-128 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/192 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-192 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/256 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-256 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/320 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-320 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/384 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-384 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/448 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-448 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/512 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-512 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/1024 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-1024 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/2048 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-2048 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/4096 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-4096 \
--max-nb-epochs 15 \
--gpus 1

CUDA_VISIBLE_DEVICES=2 \
python csqa_inhouse.py \
--model checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve/9152 \
--output-dir checkpoints/kgcl-bert-base-avg-spearman-negation-ddp-mlm-curve-csqa-9152 \
--max-nb-epochs 15 \
--gpus 1
