import json

train = []
with open('train_ih.jsonl', 'r') as f:
    for line in f:
        train.append(json.loads(line))


with open('train_ih-labels.lst', 'r') as f:
    labels = f.read().splitlines()

import random

idx = list(range(len(train)))
random.shuffle(idx)
random_idx_20 = set(idx[:int(0.2*len(train))])

train_20 = []
train_20_label = []
for i, (d, l) in enumerate(zip(train,labels)):
    if i in random_idx_20:
        train_20.append(d)
        train_20_label.append(l)

random.shuffle(idx)
random_idx_50 = set(idx[:int(0.5*len(train))])

train_50 = []
train_50_label = []
for i, (d, l) in enumerate(zip(train,labels)):
    if i in random_idx_50:
        train_50.append(d)
        train_50_label.append(l)




with open('train_ih_20.jsonl', 'w') as f:
    for entry in train_20:
        json.dump(entry, f)
        f.write('\n')


with open('train_ih_20-labels.lst', 'w') as f:
    for entry in train_20_label:
        f.write(entry)
        f.write('\n')




with open('train_ih_50.jsonl', 'w') as f:
    for entry in train_50:
        json.dump(entry, f)
        f.write('\n')

with open('train_ih_50-labels.lst', 'w') as f:
    for entry in train_50_label:
        f.write(entry)
        f.write('\n')