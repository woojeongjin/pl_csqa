import json
import csv
import random
import copy

inhouse_qids =[]

def read_data(path):
    with open(path, 'r') as f:
        data =[]
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(csv_reader):

            options = {0: row['gold-ending'], 1: row['distractor-0'],2: row['distractor-1'],3: row['distractor-2']}
            if i%4 != 0:
                tem = options[0]
                options[0] = options[i%4]
                options[i%4] = tem


            data.append({'sent': row['startphrase'], 'options': options, 'label': i%4})

    return data


train = read_data('train_full.csv')
val = read_data('val_full.csv')

print(len(train))
print(len(val))

indices = set(random.sample(range(len(train)), 10000))

test_ih = []
train_ih = []
for i, d in enumerate(train):
    if i in indices:
        test_ih.append(d)
    else:
        train_ih.append(d)

print(len(train), len(train_ih))
print(len(test_ih))
with open('train_ih.jsonl', 'w') as f:
    for entry in train_ih:
        json.dump(entry, f)
        f.write('\n')

with open('valid.jsonl', 'w') as f:
    for entry in val:
        json.dump(entry, f)
        f.write('\n')

with open('test_ih.jsonl', 'w') as f:
    for entry in test_ih:
        json.dump(entry, f)
        f.write('\n')

# train = []
# with jsonlines.open('train_rand_split.jsonl') as reader:
#     for ann in reader:
#         train.append(ann)



# train_ih = []
# test_ih = []
# for data in train:
#     if data['id'] in inhouse_qids:
#         train_ih.append(data)
#     else:
#         test_ih.append(data)

# print(len(train_ih))
# print(len(test_ih))


# with jsonlines.open('train_ih.jsonl', 'w') as writer:
#     writer.write_all(train_ih)

# with jsonlines.open('test_ih.jsonl', 'w') as writer:
#     writer.write_all(test_ih)