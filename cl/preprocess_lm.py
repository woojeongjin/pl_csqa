from tqdm import tqdm
import json
import argparse
from generator.concept.lm_generator import *
import random

generator = LMGenerator()

def match_sents(sent_path):
    result = []
    with open(sent_path) as f:
        sents = f.read().split("\n")

    for index, sent in enumerate(tqdm(sents)):
        if sent == "":
            continue

        if generator.check_availability(sent):
            generated_sentence = generator.lm_generate(sent, 20)
            randint = random.randint(0,1)
            if randint == 0:
                instance = {"answerKey": "A",
                            "id": None,
                            "question": {
                                "choices": [
                                    {"label": "A", "text": sent},
                                    {"label": "B", "text": generated_sentence}
                                ]
                            },
                            "stem": "which one is more plausible ?"
                            }
            elif randint == 1:
                instance = {"answerKey": "B",
                            "id": None,
                            "question": {
                                "choices": [
                                    {"label": "A", "text": generated_sentence},
                                    {"label": "B", "text": sent}
                                ]
                            },
                            "stem": "which one is more plausible ?"
                            }
            result.append(instance)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='generics-100k-valid.txt')
args = parser.parse_args()

batch_result = match_sents(sent_path=args.input_path)
save_path = args.input_path.split(".")[0] + "-lm.json"

with open(save_path, 'w') as outfile:
    json.dump(batch_result, outfile, indent=4)
