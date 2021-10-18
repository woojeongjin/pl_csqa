import json
import random

with open('train.json', 'r') as f:
    train_data = json.load(f)
with open('dev.json', 'r') as f:
    dev_data = json.load(f)

random.shuffle(train_data)
print(len(train_data)) #1608
print(len(dev_data)) #782

with open('test_ih.json', 'w') as f:
    json.dump(train_data[:int(len(train_data)*0.75)], f)

with open('train_ih.json', 'w') as f:
    json.dump(train_data[int(len(train_data)*0.75):], f)

