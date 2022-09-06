import os
import re
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, required=True, help='Finetuning file')
parser.add_argument('--percentage', type=int, default=0, help='Percentage')
parser.add_argument('--lr', type=float, default=3e-4, help='Percentage')
parser.add_argument('--load', type=str, default=None, help='load checkpoint')
parser.add_argument('--shots', type=int, default=0, help='n-shots')
parser.add_argument('--gpu', type=str, default="0", help='GPU ids separated by "," to use for computation')

args = parser.parse_known_args()[0]

seeds = ['9595', '0', '42']

seed = '9595'
sizes = ['64', '128', '256', '512', '1024', '2048']

log_files = []
model_folders = []

for shot in sizes:
    if args.load is None:
        log_file = args.train_file.split(".")[0] + "_" + "BERT-base" + "_" + str(args.percentage) + "_" + shot + "_" + seed + ".txt"
        predict_cmd = \
            "python3 " + args.train_file + \
            " --learning-rate " +  str(args.lr) + \
            " --shots " +  shot + \
            " --percentage " + str(args.percentage) + \
            " --dataseed " + seed + " --max-nb-epochs 30 " + " > " + log_file
    else:
        log_file = args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_" + shot + "_" + seed + ".txt"

        predict_cmd = \
            "python3 " + args.train_file + \
            " --load " + args.load + \
            " --learning-rate " +  str(args.lr) + \
            " --shots " +  shot + \
            " --percentage " + str(args.percentage) + \
            " --dataseed " + seed + " --max-nb-epochs 30 " + " > " + log_file
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
print(arr)

with open(args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_trainsize" + ".txt", 'w') as file:
    for d in arr:
        file.write(str(d))





# srun --gres=gpu:1  --qos general -t 1000 python sampling_run_slurm_size.py   --lr 3e-4   --train_file piqa.py --shots 64 --load  /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_withpos_norandomneg_seed42/Step20000/pytorch_model.bin

