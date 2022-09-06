import json
import random

with open('hellaswag_train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f.readlines()]
with open('hellaswag_val.jsonl', 'r') as f:
    dev_data = [json.loads(line) for line in f.readlines()]

random.shuffle(train_data)
print(len(train_data))
print(int(len(dev_data)*0.5))

with open('test.jsonl', 'w') as f:
    for item in train_data[:int(len(dev_data)*0.5)]:
        f.write(json.dumps(item) + "\n")

with open('train.jsonl', 'w') as f:
    for item in train_data[int(len(dev_data)*0.5):]:
        f.write(json.dumps(item) + "\n")

with open('train.jsonl', 'r') as f:
    new_train_data = [json.loads(line) for line in f.readlines()]

print(len(new_train_data))
