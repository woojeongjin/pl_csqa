import os
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import pytorch_lightning as pl
import json

from models_binary import Model
from params import parse_args

MAX_LEN = 64
NUM_LABELS = 2
label_map = {"True": 0, "False": 1}


tokenizer_dict = {
        "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-large-uncased": BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=True, return_token_type_ids=True),
        "bert-base-cased": BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True, return_token_type_ids=True),
        "roberta-base": RobertaTokenizer.from_pretrained("roberta-base"),
        "roberta-large": RobertaTokenizer.from_pretrained("roberta-large")
        }

class Com2SenseDataset:
    def __init__(self, path: str, tokenizer, args):
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
            data_pre = json.load(f)

        for x in data_pre:
            sent = x["sent"]
            answer_key = x["label"]
            self.data.append({
                "id": x["id"],
                "answer_key": str(answer_key),
                "stem": sent
            })
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]

        text = x["stem"]
        inputs = self.tokenizer(text=text,
                                padding='max_length',
                                max_length=MAX_LEN,
                                add_special_tokens=False,
                                truncation=True)

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

        label = label_map.get(x["answer_key"])
        label = torch.tensor(label).long()
        if "roberta" not in self.args.model_type:
            return {
                    "id": x["id"],
                    "label": label,
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "token_type_ids": torch.tensor(token_type_ids),
                    }
        else:
            return {
                    "id": x["id"],
                    "label": label,
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    }

def get_dataloader(model_type, batch_size, args):
    tokenizer = tokenizer_dict[model_type]
    train = Com2SenseDataset('com2sense/train_ih.json', tokenizer, args)
    val = Com2SenseDataset('com2sense/dev.json', tokenizer, args)
    test = Com2SenseDataset('com2sense/test_ih.json', tokenizer, args)


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
