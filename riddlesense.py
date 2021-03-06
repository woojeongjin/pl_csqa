from typing import Dict
from collections import OrderedDict
from argparse import ArgumentParser

from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl

from models import Model
from params import parse_args

import json

MAX_LEN = 50
NUM_LABELS = 5
label_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


class RiddleSenseDataset:
    def __init__(self, path: str, label_path, tokenizer, args):
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
        with open(label_path, 'r') as f:
            labels = f.read().splitlines()

        for i, (x, label) in enumerate(zip(data_pre, labels)):
            answer_key = label
            options = {'1': x['answer_options'][0], '2': x['answer_options'][1], '3': x['answer_options'][2], '4': x['answer_options'][3], '5': x['answer_options'][4]}
            stem = x['question']
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
            if "roberta" not in self.args.model:
                token_type_ids = inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            pad_token_id = self.tokenizer.pad_token_id
            padding_length = MAX_LEN - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            if "roberta" not in self.args.model:
                token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

            assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
            assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                             MAX_LEN)
            if "roberta" not in self.args.model:
                assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                 MAX_LEN)
            if "roberta" not in self.args.model:
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
            if "roberta" not in self.args.model:
                token_type_ids = inputs["token_type_ids"]
            attention_mask = [1] * len(input_ids)

            pad_token_id = self.tokenizer.pad_token_id
            padding_length = MAX_LEN - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            if "roberta" not in self.args.model:
                token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

            assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
            assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                             MAX_LEN)
            if "roberta" not in self.args.model:
                assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                 MAX_LEN)
            if "roberta" not in self.args.model:
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
        if "roberta" not in self.args.model:
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

from transformers import AutoModel, AutoTokenizer



def get_dataloader(model, batch_size, args):
    tokenizer = AutoTokenizer.from_pretrained(model)
    train = RiddleSenseDataset('riddle_sense/train.jsonl', 'riddle_sense/train-labels.lst', tokenizer, args)
    val = RiddleSenseDataset('riddle_sense/dev.jsonl', 'riddle_sense/dev-labels.lst', tokenizer, args)
    test = RiddleSenseDataset('riddle_sense/test.jsonl', 'riddle_sense/test-labels.lst', tokenizer, args)

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


class Model_riddlesense(Model):




    def test_epoch_end(self, outputs):
        if self.trainer.use_dp:
            test_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float() \
                       / \
                       sum(torch.mean(out["batch_size"].float()) for out in outputs)
            test_loss = sum([torch.mean(out["test_loss"].float()) for out in outputs]) / len(outputs)

            results = []
            index = 0
            for out in outputs:
                for i, idd in enumerate(out['predict']):
                    results.append({'id': index, 'pred': int(out['predict'][i]), 'label': int(out['labels'][i])})
                    index += 1

            correct = 0
            total = 0
            for res in results:
                if res['pred'] == res['label']:
                    correct += 1
                total += 1

            print("acc: ", correct/total)
        else:
            test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(
                out["batch_size"] for out in outputs)
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

    args = parse_args()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="{epoch}-{val_loss:.6f}",
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
            check_val_every_n_epoch=1,
            max_epochs=1,
        )
    else:
        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
            distributed_backend=args.distributed_backend,
            max_epochs=args.max_nb_epochs,
        )

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(args.model_type, args.batch_size, args)
    model = Model_riddlesense(args, train_dataloader, val_dataloader, test_dataloader)
    trainer.fit(model)
    trainer.test()

