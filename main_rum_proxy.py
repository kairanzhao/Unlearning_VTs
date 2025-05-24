import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
import unlearn
import utils
import numpy as np


import subprocess

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    args = arg_parser.parse_args()

    # Common parameters
    dataset = args.dataset
    arch = args.arch
    data = '/data/image_data/svhn'
    epochs = args.epochs
    lr = args.lr
    decreasing_lr = args.decreasing_lr
    batch_size = args.batch_size
    class_to_replace = -1
    mem_proxy = args.mem_proxy

    seed = args.seed
    unlearn = args.unlearn
    group_index = None
    num_indexes_to_replace = 200     
    unlearn_step = args.unlearn_step
    uname = args.uname
    
    # Define commands with dynamic mask parameters, below is an example for NegGrad+ --> NegGrad+ --> NegGrad+ RUM experiment (unlearning step 1)
    
    print(f'----------- mem_proxy: {mem_proxy} -----------')
    runs = [  
        # RUM: NegGrad+ --> NegGrad+ --> NegGrad+ (low-medium-high memorization order or the corresponding proxy order)
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {mem_proxy} --mem high --unlearn {unlearn} --unlearn_step {unlearn_step} --unlearn_epochs 5 --alpha 0.99 --unlearn_lr 0.0001 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {mem_proxy} --mem mid --unlearn {unlearn} --unlearn_step {unlearn_step} --unlearn_epochs 5 --alpha 0.99 --unlearn_lr 0.0001 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {mem_proxy} --mem low --unlearn {unlearn} --unlearn_step {unlearn_step} --unlearn_epochs 5 --alpha 0.99 --unlearn_lr 0.0001 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        # Evaluation run
        f"python main_forget.py --seed {seed} --no_aug --unlearn seq_mix --mem_proxy {mem_proxy} --mem mix --uname {uname} --num_indexes_to_replace 3000 --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${final_model_path}'"

    ]


    for command in runs:
        run_command(command)


if __name__ == "__main__":
    main()
