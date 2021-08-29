from tqdm import tqdm
import csv
import argparse
from generator.concept.concept_generator import *
import random

generator = ConceptGenerator()

def match_sents(sent_path):
    result = []
    with open(sent_path) as f:
        sents = f.read().split("\n")

    for index, sent in enumerate(tqdm(sents)):
        if sent == "":
            continue

        if generator.check_availability(sent):
            generated_sentence = generator.cor_generate(sent)
            result.append([sent, generated_sentence])
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='generics-100k-valid.txt')
args = parser.parse_args()

batch_result = match_sents(sent_path=args.input_path)
save_path = args.input_path.split(".")[0] + ".csv"

fields = ["original", "negation"]

with open(save_path, 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(fields)
    for instance in batch_result:
        csvwriter.writerow(
            [instance[0], instance[1]]
        )
