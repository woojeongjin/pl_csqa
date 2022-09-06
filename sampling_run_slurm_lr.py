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

# lrs = ['9595', '0', '42']
# lrs = ['5e-5', '1e-4', '3e-4']
lrs = ['4e-4', '5e-4', '6e-4']

log_files = []
model_folders = []

for lr in lrs:
    if args.load is None:
        log_file = args.train_file.split(".")[0] + "_" + "BERT-base" + "_" + str(args.percentage) + "_" + str(args.shots) + "_" + lr + ".txt"
        predict_cmd = \
            "python3 " + args.train_file + \
            " --learning-rate " +  str(lr) + \
            " --shots " +  str(args.shots) + \
            " --dataseed 9595 "  + " > " + log_file
    else:
        log_file = args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_" + str(args.shots) + "_" + lr  + ".txt"
        predict_cmd = \
            "python3 " + args.train_file + \
            " --learning-rate " +  str(lr) + \
            " --shots " +  str(args.shots) + \
            " --dataseed 9595 "  + " > " + log_file

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


max_id = acc_scores.index(max(acc_scores))
print("BEST_lr is",lrs[max_id])
print(acc_scores)

# arr = numpy.array(acc_scores)
# mean = numpy.mean(arr, axis=0)
# std = numpy.std(arr, axis=0)
# print("average: ", numpy.mean(arr, axis=0))
# print("std: ", numpy.std(arr, axis=0))



# with open(args.train_file.split(".")[0] + "_" + args.load.split('/')[-3] + "_" + str(args.percentage) + "_" + str(args.shots) + ".txt", 'w') as file:
#     file.write("average: " + str(mean))
#     file.write("std: " + str(std))



# srun --gres=gpu:1  --qos general -t 1000 python sampling_run_slurm_lr.py --train_file piqa.py --shots 64 --lr 3e-4  --load /home/woojeong2/vok_pretraining/snap/vlpretrain/bert_resnext_combined10_lmperturb_cl_multi_neg_synsets_1614_withpos_norandomneg_large/Step20000/pytorch_model.bin