# import json
# import random
#
# with open('train.jsonl', 'r') as f:
#     train_data = [json.loads(line) for line in f.readlines()]
# with open('dev.jsonl', 'r') as f:
#     dev_data = [json.loads(line) for line in f.readlines()]
#
# random.shuffle(train_data)
# print(len(train_data))
# print(len(dev_data))

# with open('test.jsonl', 'w') as f:
#     for item in train_data[:len(dev_data)]:
#         f.write(json.dumps(item) + "\n")
#
# with open('train.jsonl', 'w') as f:
#     for item in train_data[len(dev_data):]:
#         f.write(json.dumps(item) + "\n")
