from typing import Dict
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser
import random

import numpy as np

import scipy.io

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

MAX_LEN = 128
NUM_LABELS = 4
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

model_class_dict = {
        "bert-base-uncased": BertModel,
        "bert-large-uncased": BertModel,
        "roberta-base": RobertaModel,
        "roberta-large": RobertaModel,
        }
tokenizer_dict = {
        # "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-large-uncased": BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, return_token_type_ids=True),
        "roberta-base": RobertaTokenizer.from_pretrained("roberta-base"),
        "roberta-large": RobertaTokenizer.from_pretrained("roberta-large")
        }

class FITBDataset:
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
            answer_key = x["label"] 
            options = x['options']
            question_a = x['question_a']
            question_b = x['question_b']
            self.data.append({
                "id": x["id"],
                "answer_key": answer_key,
                "options": options,
                "question_a" : question_a,
                "question_b" : question_b,
            })
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        choices_features = []

        for key, option in x["options"].items():
            
            if len(x['question_a']) == 0:
                text = option + x['question_b']
            elif len(x['question_b']) == 0:
                text = x['question_a'] + option
            else:
                text = x['question_a'] + option + x['question_b']

        

            inputs = self.tokenizer.encode_plus(
                    text,
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
    # def shuffle(self):
    #     random.seed(9595)
    #     random.shuffle(self.data)


class BERT(nn.Module):

    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RoBERTa(nn.Module):

    def __init__(self, roberta):
        super().__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def preprocess(tokenizer: BertTokenizer, x: Dict) -> Dict:

    choices_features = []

    for key, option in x["options"].items():
        text_a = x["stem"]
        text_b = option
        inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=MAX_LEN,
                )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        pad_token_id = tokenizer.pad_token_id
        padding_length = MAX_LEN - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_id] * padding_length)

        assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(input_ids), MAX_LEN)
        assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(len(attention_mask), MAX_LEN)
        assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(len(token_type_ids), MAX_LEN)

        choices_features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            })

    label = label_map.get(x["answer_key"], -1)
    label = torch.tensor(label).long()

    return {
            "id": x["id"],
            "label": label,
            "input_ids": torch.tensor([cf["input_ids"] for cf in choices_features]),
            "attention_mask": torch.tensor([cf["attention_mask"] for cf in choices_features]),
            "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choices_features]),
            }


def get_dataloader(model_type, batch_size, args):
    tokenizer = tokenizer_dict[model_type]
    train = FITBDataset('fitb/train_ih.jsonl', tokenizer, args)
    val = FITBDataset('fitb/valid_ih.jsonl', tokenizer, args)
    test = FITBDataset('fitb/test.jsonl', tokenizer, args)
    # with open('csqa/dev_rand_split.jsonl', 'r') as f:
    #     val_pre = [json.loads(line) for line in f.readlines()]
    
    # with open('csqa/train_ih.jsonl', 'r') as f:
    #     train_pre = [json.loads(line) for line in f.readlines()]

    # with open('csqa/test_ih.jsonl', 'r') as f:
    #     test_pre = [json.loads(line) for line in f.readlines()]

    # def read_data(data):
    #     temp = []
    #     for x in data:
    #         answer_key = x["answerKey"] 
    #         options = {choice["label"]: choice["text"] for choice in x["question"]["choices"]}
    #         stem = x["question"]["stem"]
    #         temp.append({
    #             "id": x["id"],
    #             "answer_key": answer_key,
    #             "options": options,
    #             "stem": stem
    #         })
    #     return temp
    # train = lru_cache()(read_data(train_pre))
    # val = lru_cache()(read_data(val_pre))
    # test = lru_cache()(read_data(test_pre))
    # train = lfds.CommonsenseQA("train")
    # val = lfds.CommonsenseQA("dev")

    
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


class Model(pl.LightningModule):

    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args

        # self.device = torch.device("cuda", gpu)

        model = model_class_dict[args.model_type].from_pretrained(args.model_type, num_labels=NUM_LABELS)
        # model = model_class_dict[args.model_type].from_pretrained(args.load, num_labels=NUM_LABELS)
        if "roberta" not in args.model_type:
            bert = BERT(model)
        else:
            bert = RoBERTa(model)

        self.model = bert.to("cuda:0")
        if args.load is not None:
            print('loaded')
            
            if args.test_only:
                state_dict = torch.load(args.load, map_location="cuda:0")['state_dict']
            else:
                state_dict = torch.load(args.load, map_location="cuda:0")

            new_state_dict = {}
            for key, value in state_dict.items():        # If the ddp state_dict is saved
                if 'num_batches_tracked' not in key:
                    if key.startswith("module.vlbert"):     # for VLBERT
                        new_state_dict[key[len("module.vl"):]] = state_dict[key]    
                    elif key.startswith("module."):
                        new_state_dict[key[len("module."):]] = state_dict[key]
                    elif key.startswith("model."):
                        new_state_dict[key[len("model."):]] = state_dict[key]
                    elif key.startswith("encoder."):
                        new_state_dict["bert."+key] = state_dict[key]
                    elif key.startswith("embeddings."):
                        new_state_dict["bert."+key] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]

            model_keys = set(self.model.state_dict().keys())
            load_keys = set(new_state_dict.keys())
            
            # print(model_keys)
            # print(load_keys)
            # with open('keys.txt', 'w') as f:
            #     for d in model_keys:
            #         f.write(d+'\t')
            #     f.write('\n')
            #     for d in load_keys:
            #         f.write(d+'\t')
            print("Keys in model but not in load:") # model: bert, load: new one - keys in existing one
            for key in sorted(model_keys - load_keys):
                print(key)
                new_state_dict[key] = self.model.state_dict()[key]
            print("Keys in load but not in model:")
            for key in sorted(load_keys - model_keys):
                print(key)
                del new_state_dict[key]
            self.model.load_state_dict(new_state_dict)
            # print(load_keys)

        train_dataloader, val_dataloader, test_dataloader = get_dataloader(args.model_type, args.batch_size, args)
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay
                },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_eps)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warmup_steps, -1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "roberta" not in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "roberta" not in self.hparams.model_type  else None,
                )
        num_choices = input_ids.shape[1]
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
                })

        return output

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "roberta" not in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "roberta" not in self.hparams.model_type else None,
                )
        num_choices = input_ids.shape[1]
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        labels_hat = torch.argmax(reshaped_logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)
            batch_size = torch.tensor(len(labels)).cuda(loss.device.index)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct_count = correct_count.unsqueeze(0)
            batch_size = batch_size.unsqueeze(0)

        output = OrderedDict({
                "val_loss": loss,
                "correct_count": correct_count,
                "batch_size": batch_size,
                
                })
        self.log('val_loss', loss)

        return output

    def validation_epoch_end(self, outputs):
        if self.trainer.use_dp:
            val_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.mean(out["batch_size"].float()) for out in outputs)
            val_loss = sum([torch.mean(out["val_loss"].float()) for out in outputs]) / len(outputs)
        else:
            val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
            val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "roberta" not in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]
        ids = batch['id']

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "roberta" not in self.hparams.model_type else None,
                )
        num_choices = input_ids.shape[1]
        reshaped_logits = logits.view(-1, num_choices)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)

        labels_hat = torch.argmax(reshaped_logits, dim=1)
        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)
            batch_size = torch.tensor(len(labels)).cuda(loss.device.index)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            correct_count = correct_count.unsqueeze(0)
            batch_size = batch_size.unsqueeze(0)

        output = OrderedDict({
                "test_loss": loss,
                "correct_count": correct_count,
                "batch_size": batch_size,
                "ids": ids.cpu().numpy(),
                "predict": labels_hat.cpu().numpy(),
                "labels": labels.cpu().numpy()
                })

        return output

    def test_epoch_end(self, outputs):
        if self.trainer.use_dp:
            test_acc = sum([torch.mean(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.mean(out["batch_size"].float()) for out in outputs)
            test_loss = sum([torch.mean(out["test_loss"].float()) for out in outputs]) / len(outputs)

            results = []
            for out in outputs:
                # print(out['ids'], out['predict'])
                for i, idd in enumerate(out['ids']):
                    results.append({'id': int(idd), 'pred': int(out['predict'][i]), 'label': int(out['labels'][i])})
            # print(results[0])
            # with open('pred_robertalarge_fitb.jsonl', 'w') as outfile:
            #     for entry in results:
            #         json.dump(entry, outfile)
            #         outfile.write('\n')

            with open('pred_robertalarge_fitb.jsonl' ,'r')  as f:
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--distributed-backend", type=str, default="dp")
    parser.add_argument("--model-type", type=str, default="bert-base-uncased")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-check-interval", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-nb-epochs", type=int, default=10)
    parser.add_argument("--min-nb-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-06)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
                        
    # srun --gres=gpu:8000:1 --nodelist ink-nova python fitb.py --gpus 1 --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined0_100k/BEST.pth_lang --output_dir combine0 --seed 42
 
    args = parser.parse_args()

    # seed = args.seed if args.seed else random.randint(1, 100)
    # args.seed = seed
    set_seed(args.seed)

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

    model = Model(args)

    if args.test_only:
        # model = model.load_from_checkpoint(args.ckpt)
        trainer.test(model)
    else:
        trainer.fit(model)
        trainer.test()
