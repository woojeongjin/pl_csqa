import scipy.io
import random
import json


data = scipy.io.loadmat('dataset.mat')
split = scipy.io.loadmat('split.mat')

sentences1 = data['sentences_1']
sentences2 = data['sentences_2']
labels = data['gt']



train_ind =set()
test_ind = set()
for d in split['trainind']:
    train_ind.add(int(d[0])-1)
for d in split['testind']:
    test_ind.add(int(d[0]-1))


train = []
valid = []
test = []

for i, (sent1, sent2, label) in enumerate(zip(sentences1, sentences2, labels)):
    if len(sent1[0]) == 0:
        continue
    if len(sent2[0]) == 0:
        continue
    sent1 = sent1[0][0]
    s1 = ""
    for d in sent1:
        s1 += d[0]
    s2 = ""
    sent2 = sent2[0][0]
    for d in sent2:
        s2 += d[0]
    label = label[0]
    assert int(label) in [0,1]


    if int(label) == 1:
        new = {'id': i, 'sentence_1': s1, 'sentence_2': s2, 'label': 'same'}
    else:
        new = {'id': i, 'sentence_1': s1, 'sentence_2': s2, 'label': 'different'}
    if i in train_ind:
        train.append(new)
    else:
        test.append(new)


print(len(train), len(test))
# print(len(train))
# print(len(test))

indices = set(random.sample(range(len(train)), 2000))

train_new = []
for i in range(len(train)):
    if i in indices:
        valid.append(train[i])
    else:
        train_new.append(train[i])

print(test[0])

with open('train_ih.jsonl', 'w') as f:
    for entry in train_new:
        json.dump(entry, f)
        f.write('\n')

with open('valid_ih.jsonl', 'w') as f:
    for entry in valid:
        json.dump(entry, f)
        f.write('\n')

with open('test.jsonl', 'w') as f:
    for entry in test:
        json.dump(entry, f)
        f.write('\n')