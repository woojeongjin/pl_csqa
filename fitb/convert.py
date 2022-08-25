import scipy.io
import random
import json


data = scipy.io.loadmat('dataset.mat')
split = scipy.io.loadmat('split.mat')



questions = data['question_in_sentences']
options = data['option_in_sentences']
blank_inds = data['blank_ind']
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


for i, (question, option, blank_ind, label) in enumerate(zip(questions, options, blank_inds, labels)):
    question = question[0][0]
    blank_ind = blank_ind[0]
    label = label[0]
    assert int(label) in [1,2,3,4]

    if int(label) == 1:
        label = 'A'
    elif int(label) == 2:
        label = 'B'
    elif int(label) == 3:
        label = 'C'
    else:
        label = 'D'
    question_len = len(question)

    
    question = [q[0].strip().strip('.')+'.' for q in question]
    


    assert int(blank_ind) in [1,2,3,4,5,6,7]
    
    if int(blank_ind) == 1:
        # print(question)
        a = ''
        b = " ".join(question[1:])
    elif int(blank_ind) == 2:
        
        a = question[0]
        b = " ".join(question[2:])
    elif int(blank_ind) == 3:
        
        a = " ".join(question[:2])
        b = " ".join(question[3:])
    elif int(blank_ind) == 4:
        
        a = " ".join(question[:3])
        b = " ".join(question[4:])
    elif int(blank_ind) == 5:
        
        a = " ".join(question[:4])
        b = " ".join(question[5:])
        
    elif int(blank_ind) == 6:
        a = " ".join(question[:5])
        b = " ".join(question[6:])
        print(question)
        print(a)
        print(len(b))
        
    elif int(blank_ind) == 7:
        a = " ".join(question[:6])
        b = " ".join(question[7:])
        
    # if question_len > 3:
    #     print(question)
    #     print(a)
    #     print(b)
    # if len(a) == 0:
    #     print("a",question)
    # if len(b) == 0:
    #     print("b",question)
    options = { 'A': option[0][0].strip().strip(".") + ".", 'B': option[1][0].strip().strip(".") + ".", 'C': option[2][0].strip().strip(".") + ".", 'D': option[3][0].strip().strip(".") + "."}

    new = {'id': i, 'question_a': a, 'question_b': b, 'label': label, 'options':options}
    if i in train_ind:
        train.append(new)
    else:
        test.append(new)


# print(len(train))
# print(len(test))

indices = set(random.sample(range(len(train)), 1000))

train_new = []
for i in range(len(train)):
    if i in indices:
        valid.append(train[i])
    else:
        train_new.append(train[i])

# print(test[0])

# with open('train_ih.jsonl', 'w') as f:
#     for entry in train_new:
#         json.dump(entry, f)
#         f.write('\n')

# with open('valid_ih.jsonl', 'w') as f:
#     for entry in valid:
#         json.dump(entry, f)
#         f.write('\n')

# with open('test.jsonl', 'w') as f:
#     for entry in test:
#         json.dump(entry, f)
#         f.write('\n')