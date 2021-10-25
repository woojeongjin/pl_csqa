# pytorch-lightning-transformers [Medium article](https://towardsdatascience.com/how-to-fine-tune-bert-with-pytorch-lightning-ba3ad2f928d2)


Original code is from https://github.com/sobamchan/pytorch-lightning-transformers.git

## Setup
```bash
$ git clone https://github.com/woojeongjin/pl_csqa.git
$ cd pytorch-lightning-transformers
$ pipenv install
$ pipenv shell
```

## Usage

### Fine-tune for [CommonsenseQA](https://www.tau-nlp.org/commonsenseqa) with 2 gpus.
```bash
CUDA_VISIBLE_DEVICES=0,1 python csqa.py --gpus 2
```

### Fine-tune for [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
```bash
$ python mrpc.py
```
This will load pre-trained BERT and fine-tune it with putting classification layer on top on MRPC task (paraphrase identification).



srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --output_dir com2sense-9595 --seed 9595 --learning-rate 3e-4 --model-type bert-large-uncased
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --output_dir com2sense-42 --seed 42 --learning-rate 3e-4 --model-type bert-large-uncased
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --output_dir com2sense-0 --seed 0 --learning-rate 3e-4 --model-type bert-large-uncased

srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614/Step4000/pytorch_model.bin --output_dir com2sense-9595-negative --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614/Step4000/pytorch_model.bin --output_dir com2sense-42-negative --seed 42 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614/Step4000/pytorch_model.bin --output_dir com2sense-0-negative --seed 0 --learning-rate 3e-4

srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_withpos/Step4000/pytorch_model.bin --output_dir com2sense-9595-positive --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_withpos/Step4000/pytorch_model.bin --output_dir com2sense-42-positive --seed 42 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_withpos/Step4000/pytorch_model.bin --output_dir com2sense-0-positive --seed 0 --learning-rate 3e-4

srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire_epoch5/Step6000/pytorch_model.bin --output_dir com2sense-9595-positive --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire_epoch5/Step6000/pytorch_model.bin --output_dir com2sense-42-positive --seed 42 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_one_neg_entire_epoch5/Step6000/pytorch_model.bin --output_dir com2sense-0-positive --seed 0 --learning-rate 3e-4

srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_3words_again/Step6000/pytorch_model.bin --output_dir com2sense-9595-negative --seed 9595 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_3words_again/Step6000/pytorch_model.bin --output_dir com2sense-42-negative --seed 42 --learning-rate 3e-4
srun --gres=gpu:1080:1 --nodelist ink-ron -t 1000 python com2sense.py --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_3words_again/Step6000/pytorch_model.bin --output_dir com2sense-0-negative --seed 0 --learning-rate 3e-4

