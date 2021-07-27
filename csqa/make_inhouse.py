import json
import jsonlines

inhouse_qids =[]
with open('inhouse_split_qids.txt', 'r') as f:
    # inhouse_qids = f.readlines()
    for line in f:
        inhouse_qids.append(line.strip('\n'))


train = []
with jsonlines.open('train_rand_split.jsonl') as reader:
    for ann in reader:
        train.append(ann)



train_ih = []
test_ih = []
for data in train:
    if data['id'] in inhouse_qids:
        train_ih.append(data)
    else:
        test_ih.append(data)

print(len(train_ih))
print(len(test_ih))


with jsonlines.open('train_ih.jsonl', 'w') as writer:
    writer.write_all(train_ih)

with jsonlines.open('test_ih.jsonl', 'w') as writer:
    writer.write_all(test_ih)