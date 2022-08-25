import json
import random

with open('train.jsonl', 'r') as f:
    data_pre = [json.loads(line) for line in f.readlines()]
with open('train-labels.lst', 'r') as f:
    labels = f.read().splitlines()


indices = set(random.sample(range(len(data_pre)), 2000))

new_train =[]
new_train_label = []
new_test = []
new_test_label =[]
for i, (d, l) in enumerate(zip(data_pre, labels)):
    if i in indices:
        new_test.append(d)
        new_test_label.append(l)
    else:
        new_train.append(d)
        new_train_label.append(l)



with open('train_ih.jsonl', 'w') as f:
    for entry in new_train:
        json.dump(entry, f)
        f.write('\n')

with open('test_ih.jsonl', 'w') as f:
    for entry in new_test:
        json.dump(entry, f)
        f.write('\n')


with open('train_ih-labels.lst', 'w') as f:
    for entry in new_train_label:
        f.write(entry)
        f.write('\n')

with open('test_ih-labels.lst', 'w') as f:
    for entry in new_test_label:
        f.write(entry)
        f.write('\n')
