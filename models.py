from os import stat
from typing import Dict
from functools import partial
from collections import OrderedDict
from argparse import ArgumentParser
import random

import numpy as np

# import lineflow.datasets as lfds
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, CLIPTextModel
from transformers import AdamW, get_linear_schedule_with_warmup


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import json
from functools import lru_cache

model_class_dict = {
        "bert-base-uncased": BertModel,
        "bert-large-uncased": BertModel,
        "bert-base-cased": BertModel,
        "roberta-base": RobertaModel,
        "roberta-large": RobertaModel,
        "clip": CLIPTextModel
        }

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class BERT(nn.Module):

    def __init__(self, bert, args, classes):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert.config.hidden_size, classes)
        self.args = args

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

        if not self.args.cl:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = pooled_output
        return logits


class RoBERTa(nn.Module):

    def __init__(self, roberta, args, classes):
        super().__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(roberta.config.hidden_dropout_prob)
        self.classifier = nn.Linear(roberta.config.hidden_size, classes)
        self.args = args

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
        
        if not self.args.cl:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = pooled_output
        return logits

class CLIP(nn.Module):

    def __init__(self, bert, args, classes):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert.config.hidden_size, classes)
        self.args = args

       

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask)
                            # token_type_ids=token_type_ids,
                            # position_ids=position_ids,
                            # head_mask=head_mask)

        pooled_output = outputs.pooler_output

        if not self.args.cl:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = pooled_output
        return logits




class Model(pl.LightningModule):

    def __init__(self, args, train_dataloader, val_dataloader, test_dataloader, classes=1):
        super(Model, self).__init__()
        self.hparams = args
        if args.cl:
            self.sim = Similarity(temp=args.temp)

        # self.device = torch.device("cuda", gpu)
        if args.model_type == 'clip':
            # model = model_class_dict[args.model_type].from_pretrained("openai/clip-vit-large-patch14")
            model = model_class_dict[args.model_type].from_pretrained("openai/clip-vit-base-patch32")
        else:
         model = model_class_dict[args.model_type].from_pretrained(args.model_type)
    
        if "roberta" in args.model_type:
            bert = RoBERTa(model, args, classes)
        elif "clip" in args.model_type:
            bert = CLIP(model, args, classes)
        else:
            bert = BERT(model, args, classes)


        # bert.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        # bert.classifier.bias.data.zero_()
            

        self.model = bert.to("cuda:0")
        if args.load is not None:
            print('loaded')
            
            state_dict = torch.load(args.load, map_location="cuda:0")
            if 'state_dict' in  state_dict.keys():
                state_dict = state_dict['state_dict']

            new_state_dict = {}
            for key, value in state_dict.items():        # If the ddp state_dict is saved
                if 'num_batches_tracked' not in key:

                    if key.startswith("module.vlbert.word_embeddings.weight"):
                        new_state_dict["bert.embeddings.word_embeddings.weight"] = state_dict[key]
                    elif key.startswith("module.vlbert.embedding_LayerNorm.bias"):
                        new_state_dict["bert.embeddings.LayerNorm.bias"] = state_dict[key]
                    elif key.startswith("module.vlbert.embedding_LayerNorm.weight"):
                        new_state_dict["bert.embeddings.LayerNorm.weight"] = state_dict[key]
                    # if key.startswith("module.vlbert.position_embeddings.weight"):
                    #     new_state_dict["bert.embeddings.position_ids"] = state_dict[key]
                    elif key.startswith("module.vlbert.position_embeddings.weight"):
                        new_state_dict["bert.embeddings.position_embeddings.weight"] = state_dict[key]
                    elif key.startswith("module.vlbert.token_type_embeddings.weight"):
                        new_state_dict["bert.embeddings.token_type_embeddings.weight"] = state_dict[key]
                    elif key.startswith("module.vlbert.mlm_head"):
                        new_state_dict["cls"+key[len("module.vlbert.mlm_head"):]] = state_dict[key]

                    if key.startswith("module.vlbert"):     # for VLBERT
                        new_state_dict[key[len("module.vl"):]] = state_dict[key]    
                    elif key.startswith("module."):
                        new_state_dict[key[len("module."):]] = state_dict[key]
                    elif key.startswith("encoder."):
                        new_state_dict["bert."+key] = state_dict[key]
                    elif key.startswith("embeddings."):
                        new_state_dict["bert."+key] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]

                # if 'num_batches_tracked' not in key:
                #     if key.startswith("module.vlbert.word_embeddings.weight"):
                #         new_state_dict["bert.embeddings.word_embeddings.weight"] = state_dict[key]
                #     elif key.startswith("module.vlbert.embedding_LayerNorm.bias"):
                #         new_state_dict["bert.embeddings.LayerNorm.bias"] = state_dict[key]
                #     elif key.startswith("module.vlbert.embedding_LayerNorm.weight"):
                #         new_state_dict["bert.embeddings.LayerNorm.weight"] = state_dict[key]
                #     # if key.startswith("module.vlbert.position_embeddings.weight"):
                #     #     new_state_dict["bert.embeddings.position_ids"] = state_dict[key]
                #     elif key.startswith("module.vlbert.position_embeddings.weight"):
                #         new_state_dict["bert.embeddings.position_embeddings.weight"] = state_dict[key]
                #     elif key.startswith("module.vlbert.token_type_embeddings.weight"):
                #         new_state_dict["bert.embeddings.token_type_embeddings.weight"] = state_dict[key]
                #     elif key.startswith("module.vlbert.mlm_head"):
                #         new_state_dict["cls"+key[len("module.vlbert.mlm_head"):]] = state_dict[key]

                #     elif key.startswith("module.vlbert"):     # for VLBERT
                #         new_state_dict[key[len("module.vl"):]] = state_dict[key]    
                #     elif key.startswith("module."):
                #         new_state_dict[key[len("module."):]] = state_dict[key]
                #     elif key.startswith("encoder."):
                #         new_state_dict["bert."+key] = state_dict[key]
                #     elif key.startswith("embeddings."):
                #         new_state_dict["bert."+key] = state_dict[key]
                #     else:
                #         new_state_dict[key] = state_dict[key]
                

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
            
            new_state_dict['bert.embeddings.token_type_embeddings.weight'] = new_state_dict['bert.embeddings.token_type_embeddings.weight'][:2]
            self.model.load_state_dict(new_state_dict)
            # print(load_keys)

        
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
        if "bert" in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "bert" in self.hparams.model_type  else None,
                )
        # print(torch.sum(self.model.classifier.weight.data))

        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        

        if self.hparams.cl:
            reshaped_logits = logits.view(batch_size, -1, num_choices)
            z1, z2 = reshaped_logits[:,:, 0], reshaped_logits[:,: ,1:]
            reshaped_logits = self.sim(z1.unsqueeze(2), z2)

        else:
            reshaped_logits = logits.view(-1, num_choices)
        # print(reshaped_logits)
        # print(torch.softmax(reshaped_logits, dim=1))

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
        
        # self.log('loss', loss)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return output

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "bert"  in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "bert" in self.hparams.model_type else None,
                )
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        

        if self.hparams.cl:
            reshaped_logits = logits.view(batch_size, -1, num_choices)
            z1, z2 = reshaped_logits[:,:, 0], reshaped_logits[:,: ,1:]
            reshaped_logits = self.sim(z1.unsqueeze(2), z2)

        else:
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
        if "bert" in self.hparams.model_type:
            token_type_ids = batch["token_type_ids"]
        labels = batch["label"]
        ids = batch['id']
       

        logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if "bert" in self.hparams.model_type else None,
                )
        num_choices = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        if self.hparams.cl:
            reshaped_logits = logits.view(batch_size, -1, num_choices)
            z1, z2 = reshaped_logits[:,:, 0], reshaped_logits[:,: ,1:]
            reshaped_logits = self.sim(z1.unsqueeze(2), z2)

        else:
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
            labels_hat = labels_hat.unsqueeze(0)
            labels = labels.unsqueeze(0)

        # print(labels_hat.shape)
        # print(labels.shape)

        output = OrderedDict({
                "test_loss": loss,
                "correct_count": correct_count,
                "batch_size": batch_size,
                # "ids": ids,
                # "predict": labels_hat.cpu().numpy(),
                # "labels": labels.cpu().numpy()
                })

        return output
        

    def test_epoch_end(self, outputs):
        if self.trainer.use_dp:
            test_acc = sum([torch.sum(out["correct_count"].float()) for out in outputs]).float()\
                    /\
                    sum(torch.sum(out["batch_size"].float()) for out in outputs)

            test_loss = sum([torch.mean(out["test_loss"].float()) for out in outputs]) / len(outputs)

            results = []
            # for out in outputs:
            #     for i, idd in enumerate(out['ids']):
            #         results.append({'id': idd, 'pred': int(out['predict'][i]), 'label': int(out['labels'][i])})
            # with open('pred_bert.jsonl', 'w') as outfile:
            #     for entry in results:
            #         json.dump(entry, outfile)
            #         outfile.write('\n')

            # with open('pred_robertalarge5e-5.jsonl' ,'r')  as f:
            #     roberta = []
            #     for d in f:
            #         roberta.append(json.loads(d))
            # easy = []
            # hard = []
            # for res, rob in zip(results, roberta):
            #     assert res['id'] == rob['id']
            #     if rob['pred'] == rob['label']:
            #         easy.append(res['pred'] == res['label'])
            #     else:
            #         hard.append(res['pred'] == res['label']) 

            # print("easy: ", np.mean(easy), 'hard:', np.mean(hard))

            # with open('pred_robertalarge2.jsonl' ,'r')  as f:
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

            # correct = 0
            # total = 0
            # for res in results:
            #     if res['pred'] == res['label']:
            #         correct += 1
            #     total += 1

            # print("acc: ", correct / total)
        else:
            test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
            test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                }

        print("acc: ", test_acc.item())
            
        return {"progress_bar": tqdm_dict, "log": tqdm_dict, "test_loss": test_loss}


    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader