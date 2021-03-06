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

MAX_LEN = 50

NUM_LABELS = 4
label_map = {"0": 0, "1": 1, "2": 2, "3": 3}


tokenizer_dict = {
        # "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-large-uncased": BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, return_token_type_ids=True),
        "roberta-base": RobertaTokenizer.from_pretrained("roberta-base"),
        "roberta-large": RobertaTokenizer.from_pretrained("roberta-large")
        }

class SWAGDataset:
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

        for i, x in enumerate(data_pre):
            answer_key = x['label']
            options = {'0': x['options']['0'], '1': x['options']['1'], '2': x['options']['2'], '3': x['options']['3']}
            stem = x['sent']
            self.data.append({
                "id": i,
                "answer_key": answer_key,
                "options": options,
                "stem": stem
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

        label = label_map.get(str(x["answer_key"]), -1)
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

    train = SWAGDataset('swag/train_ih.jsonl', tokenizer, args)

    cutoff = int(len(train.data) * (args.percentage/100))
    random.seed(args.dataseed)
    random.shuffle(train.data)
    if args.shots != 0:
        train.data = train.data[:args.shots]
    else:
        train.data = train.data[:cutoff]

    
    val = SWAGDataset('swag/valid.jsonl', tokenizer, args)
    test = SWAGDataset('swag/valid.jsonl', tokenizer, args)
    # test = SWAGDataset('swag/test_ih.jsonl', tokenizer, args)

    # train = SWAGDataset('swag/train.jsonl', 'swag/train-labels.lst', tokenizer, args)
    # val = SWAGDataset('swag/valid.jsonl', 'swag/valid-labels.lst', tokenizer, args)
    # test = SWAGDataset('swag/valid.jsonl', 'swag/valid-labels.lst',tokenizer, args)

    
    # preprocessor = partial(preprocess, tokenizer)

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
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_tset,
    #     batch_size=(args.batch_size // args.world_size),
    #     shuffle=False,          # Will be shuffled in the sampler.
    #     num_workers=max(args.num_workers // args.world_size, 1),
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True
    # )
    test_dataloader = DataLoader(
            # test.map(preprocessor),
            test,
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader


class Model_SWAG(Model):



    def test_epoch_end(self, outputs):
        # print(outputs)
        if self.trainer.use_dp:
            test_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.mean(out["batch_size"].float()) for out in outputs)
            test_loss = sum([torch.mean(out["test_loss"].float()) for out in outputs]) / len(outputs)

            results = []
            index = 0
            for out in outputs:
                # print(out['ids'])
                for i, idd in enumerate(out['predict']):
                    results.append({'id': index, 'pred': int(out['predict'][i]), 'label': int(out['labels'][i])})
                    index+=1


            correct = 0
            total = 0
            for res in results:
                print(res['pred'], res['label'])
                if res['pred'] == res['label']:
                    correct += 1
                total += 1

            print("acc: ", correct / total)

            # with open('pred_robertalarge_swag.jsonl' ,'r')  as f:
            #     roberta = []
            #     for d in f:
            #         roberta.append(json.loads(d))
            # easy = []
            # hard = []
            # for res, rob in zip(results, roberta):
            #     assert res['id'] == rob['id']
            #     if int(rob['pred']) == int(rob['label']):
            #         easy.append(int(res['pred']) == int(res['label']))
            #     else:
            #         hard.append(int(res['pred']) == int(res['label'])) 
            # print("second one easy: ", np.mean(easy), 'hard:', np.mean(hard))


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
                        
    # srun --gres=gpu:8000:1 python swag.py --gpus 1  --output_dir swag_test --seed 42 --model-type roberta-large
    args = parse_args()



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
        save_top_k=1,
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
    model = Model_SWAG(args, train_dataloader, val_dataloader, test_dataloader)
    if args.test_only:
        # model = model.load_from_checkpoint(args.ckpt)
        trainer.test(model)
    else:
        trainer.fit(model)
        trainer.test()
