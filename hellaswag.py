from os import stat
from typing import Dict
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser
import random

import numpy as np

import lineflow.datasets as lfds
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import json
from functools import lru_cache

from models import Model
from params import parse_args

MAX_LEN = 64
NUM_LABELS = 4
label_map = {"0": 0, "1": 1, "2": 2, "3": 3}


tokenizer_dict = {
        # "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-large-uncased": BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-base-cased": BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True, return_token_type_ids=True),
        "roberta-base": RobertaTokenizer.from_pretrained("roberta-base"),
        "roberta-large": RobertaTokenizer.from_pretrained("roberta-large")
        }

class HellaswagDataset:
    def __init__(self, path: str, tokenizer, args ):
        """
        :param split: train, valid, test
        :param sources: The data sources to be loaded, separated by comma.
                       from: mscoco, cc, vg, vqa, gqa, visual7w
                             'vg' stands for visual genome captions
                             'cc' stands for conceptual captions.
                       example: 'mscoco, vg'
        """
        self.tokenizer = tokenizer
        self.data = []
        with open(path, 'r') as f:
            data_pre = [json.loads(line) for line in f.readlines()]
        for x in data_pre:
            ctx = x["ctx"]
            answer_key = x["label"]
            options = {str(i): choice for i, choice in enumerate(x["endings"])}
            self.data.append({
                "id": x["ind"],
                "answer_key": str(answer_key),
                "options": options,
                "stem": ctx
            })
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        choices_features = []

        if self.args.cl:
            text_a = x["stem"]
            inputs = self.tokenizer.encode_plus(
                    text_a,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    truncation=True,
                    )

            input_ids = inputs["input_ids"]
            if "roberta" not in self.args.model_type:
                token_type_ids = inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            pad_token_id = self.tokenizer.pad_token_id
            padding_length = MAX_LEN - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            if "roberta" not in self.args.model_type:
                token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

            assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
            assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
            if "roberta" not in self.args.model_type:
                assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)
            if "roberta" not in self.args.model_type:
                choices_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    })
            else:
                choices_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    })       

        for key, option in x["options"].items():
            text_a = x["stem"]
            text_b = option
            if self.args.cl:
                inputs = self.tokenizer.encode_plus(
                    text_b,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    truncation=True,
                    )
            else:
                inputs = self.tokenizer.encode_plus(
                        text_a,
                        text_b,
                        add_special_tokens=True,
                        max_length=MAX_LEN,
                        truncation=True,
                        )
            

            input_ids = inputs["input_ids"]
            if "roberta" not in self.args.model_type:
                token_type_ids = inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            pad_token_id = self.tokenizer.pad_token_id
            padding_length = MAX_LEN - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            if "roberta" not in self.args.model_type:
                token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

            assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
            assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
            if "roberta" not in self.args.model_type:
                assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)
            if "roberta" not in self.args.model_type:
                choices_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    })
            else:
                choices_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    })

        label = label_map.get(x["answer_key"], -1)
        label = torch.tensor(label).long()
        if "roberta" not in self.args.model_type:
            return {
                    "id": x["id"],
                    "label": label,
                    "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
                    "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
                    "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
                    }
        else:
            return {
                    "id": x["id"],
                    "label": label,
                    "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
                    "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
                    }

def get_dataloader(model_type, batch_size, args):
    tokenizer = tokenizer_dict[model_type]
    train = HellaswagDataset('hellaswag/train.jsonl', tokenizer, args)
    val = HellaswagDataset('hellaswag/dev.jsonl', tokenizer, args)
    test = HellaswagDataset('hellaswag/test.jsonl', tokenizer, args)


    train_dataloader = DataLoader(
            # train.map(preprocessor),
            train,
            sampler=RandomSampler(train),
            batch_size=batch_size
            )
    val_dataloader = DataLoader(
            # val.map(preprocessor),
            val,
            sampler=SequentialSampler(val),
            batch_size=batch_size
            )

    test_dataloader = DataLoader(
            # test.map(preprocessor),
            test,
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader


class Model_Hellaswag(Model):

    def test_epoch_end(self, outputs):
        if self.trainer.use_dp:
            test_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.mean(out["batch_size"].float()) for out in outputs)
            test_loss = sum([torch.mean(out["test_loss"].float()) for out in outputs]) / len(outputs)

            results = []
            for out in outputs:
                for i, idd in enumerate(out['ids']):
                    results.append({'id': idd, 'pred': int(out['predict'][i]), 'label': int(out['labels'][i])})

            with open('pred_roberta_large_hellaswag.jsonl' ,'r')  as f:
                roberta = []
                for d in f:
                    roberta.append(json.loads(d))
            easy = []
            hard = []
            for res, rob in zip(results, roberta):
                assert res['id'] == rob['id']
                if rob['pred'] == rob['label']:
                    easy.append(res['pred'] == res['label'])
                else:
                    hard.append(res['pred'] == res['label']) 

            print("easy: ", np.mean(easy), 'hard:', np.mean(hard))


        else:
            test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
            test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                }
            
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "test_loss": test_loss}


    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader



if __name__ == "__main__":
    # srun --gres=gpu:8000:1 --nodelist ink-nova python csqa.py --gpus 1 --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined0_100k/BEST.pth_lang --output_dir combine0 --seed 42
    # /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined03_100k_5e-5/best/pytorch_model.bin
    # CUDA_VISIBLE_DEVICES=8 python csqa.py --gpus 1 --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined0_100k/BEST.pth_lang --output_dir combine0 --seed 42
    # srun --gres=gpu:8000:1 python csqa.py --gpus 1 --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined09_wiki_book_100k/BEST.pth_lang --output_dir res7 --seed 42
    # sleep 7h && srun --gres=gpu:8000:1 --nodelist ink-ruby python csqa.py --gpus 1 --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined03_wiki_book_200k/BEST.pth_lang --output_dir wiki_book_03 --seed 42
    # srun --gres=gpu:8000:1 python csqa.py --gpus 1 --load /home/woojeong2/VidLanKD/snap/bert/wiki_bert_base_wiki_book_10/checkpoint-epoch0009/pytorch_model.bin --output_dir res6 --seed 42 
    args = parse_args()

    # seed = args.seed if args.seed else random.randint(1, 100)
    # args.seed = seed


    # early_stop_callback = EarlyStopping(
    #         monitor="val_loss",
    #         min_delta=0.0,
    #         patience=args.patience,
    #         verbose=True,
    #         mode="min",
    #         )


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir, 
        filename= "{epoch}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=20,
        verbose=True,
    )

    if args.test_run:
        trainer = pl.Trainer(
                gpus=args.gpus,
                callbacks=[checkpoint_callback],
                val_check_interval=args.val_check_interval,
                distributed_backend=args.distributed_backend,
                default_save_path="./test_run_logs",
                train_percent_check=0.001,
                # val_percent_check=0.001,
                check_val_every_n_epoch=1,
                max_epochs=1,
                )
    else:
        trainer = pl.Trainer(
                gpus=args.gpus,
                callbacks=[checkpoint_callback],
                # val_check_interval=0.5,
                check_val_every_n_epoch=1,
                distributed_backend=args.distributed_backend,
                max_epochs=args.max_nb_epochs,
                )

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args.model_type, args.batch_size, args)
    model = Model_Hellaswag(args, train_dataloader, val_dataloader, test_dataloader)

    trainer.fit(model)
    trainer.test()
