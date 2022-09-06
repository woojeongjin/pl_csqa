import json

train = []
with open('train.jsonl', 'r') as f:
    for line in f:
        train.append(json.loads(line))

import random

random.shuffle(train)

train_20=train[:int(0.2*len(train))]

random.shuffle(train)
train_50=train[:int(0.5*len(train))]





with open('train_20.jsonl', 'w') as f:
    for entry in train_20:
        json.dump(entry, f)
        f.write('\n')


with open('train_50.jsonl', 'w') as f:
    for entry in train_50:
        json.dump(entry, f)
        f.write('\n')