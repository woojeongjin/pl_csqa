import os
import re
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, required=True, help='Finetuning file')
parser.add_argument('--percentage', type=int, default=0, help='Percentage')
parser.add_argument('--lr', type=float, default=1e-4, help='Percentage')
parser.add_argument('--load', type=str, default=None, help='load checkpoint')
parser.add_argument('--shots', type=int, default=0, help='n-shots')
parser.add_argument('--gpu', type=str, default="0", help='GPU ids separated by "," to use for computation')

args = parser.parse_known_args()[0]

seeds = ['9595', '0', '42']

log_files = []
model_folders = []

for seed in seeds:
    if args.load is None:
        log_file = args.train_file.split(".")[0] + "_" + "BERT-base" + "_" + str(args.percentage) + "_" + str(args.shots) + "_" + seed +"_" + str(args.lr) + ".txt"
        predict_cmd = \
            "python3 " + args.train_file + \
            " --learning-rate " +  str(args.lr) + \
            " --shots " +  str(args.shots) + \
            " --percentage " + str(args.percentage) + \
            " --dataseed " + seed + "  --model-type bert-large-uncased " + " > " + log_file
    else:
        log_file = args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_" + str(args.shots) + "_" + seed +"_" + str(args.lr)+ ".txt"

        predict_cmd = \
            "python3 " + args.train_file + \
            " --load " + args.load + \
            " --learning-rate " +  str(args.lr) + \
            " --shots " +  str(args.shots) + \
            " --percentage " + str(args.percentage) + \
            " --dataseed " + seed + "  --model-type bert-large-uncased " + " > " + log_file
    log_files.append(log_file)

    print(predict_cmd, "\n")
    os.system(predict_cmd)

acc_scores = []
for file in log_files:
    lines = []
    with open(file, 'r') as reader:
        for line in reader:
            if len(line.split("acc:")) > 1:
                print(line.split("acc:"))
                acc_scores.append(float(line.split("acc:")[1].strip()))
            lines.append(line)

arr = numpy.array(acc_scores)
mean = numpy.mean(arr, axis=0)
std = numpy.std(arr, axis=0)
print("average: ", numpy.mean(arr, axis=0))
print("std: ", numpy.std(arr, axis=0))

with open(args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_" + str(args.shots) + ".txt", 'w') as file:
    file.write("average: " + str(mean))
    file.write("std: " + str(std))




# --model-type


# srun --gres=gpu:1  --qos general -t 1000 python sampling_run_large.py   --lr 3e-4  --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_norandomneg_bert_large/Step12000/pytorch_model.bin  --train_file piqa.py --shots 64

