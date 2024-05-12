#!/usr/bin/env python3

import os
import sys
import pandas as pd


input_dir = os.environ["inputDataset"]
output_dir = os.environ["outputDir"]


def fprint(text):
    with open(output_dir+'/out.txt', 'w') as sys.stdout:
        print(text)

def get_task_sets(folder):
    base = input_dir+"/"+folder
    dfs = {}
    for dir in os.listdir(base):
        if dir.endswith(".tsv"):
            dfs[dir] = pd.read_csv(base+"/"+dir, sep="\t")
    return dfs

ori_sets = get_task_dfs("orientation")
pow_sets = get_task_dfs("power")

