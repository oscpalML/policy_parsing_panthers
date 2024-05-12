#!/usr/bin/env python3

import os
import sys
import pandas as pd
from datasets import Dataset

input_dir = os.environ["inputDataset"]
output_dir = os.environ["outputDir"]

field = "text_en"

def fprint(text):
    with open(output_dir+'/out.txt', 'a') as sys.stdout:
        print(text)

country_codes = {
    "at": "Austria",
    "ba": "Bosnia and Herzegovina",
    "be": "Belgium",
    "bg": "Bulgaria",
    "cz": "Czechia",
    "dk": "Denmark",
    "ee": "Estonia",
    "es": "Spain",
    "es-ct": "Catalonia",
    "es-ga": "Galicia",
    "es-pv": "Basque Country",
    "fi": "Finland",
    "fr": "France",
    "gb": "Great Britain",
    "gr": "Greece",
    "hr": "Croatia",
    "hu": "Hungary",
    "is": "Iceland",
    "it": "Italy",
    "lv": "Latvia",
    "nl": "The Netherlands",
    "no": "Norway",
    "pl": "Poland",
    "pt": "Portugal",
    "rs": "Serbia",
    "se": "Sweden",
    "si": "Slovenia",
    "tr": "Turkey",
    "ua": "Ukraine",
    "id": "TEMP4DEBUGGING"
}

sex_codes = {"F":"Female", "M":"Male", "U":"Unknown"}


def prepend_data(row):
    country = "".join(filter(lambda x: not x.isdigit(), str(row["id"])))
    sex = str(row["sex"])
    return country_codes[country]+", "+sex_codes[sex]+". "+row[field]


def get_ds(path):
    df = pd.read_csv(path, sep="\t", usecols=[field, "sex", "id"])
    df[field] = df.apply(prepend_data, axis=1)
    df.drop(columns=["sex"], inplace=True)
    return Dataset.from_pandas(df, preserve_index=False)



def get_task_sets(folder):
    base = input_dir+"/"+folder
    sets = {}
    for dir in os.listdir(base):
        if dir.endswith(".tsv"):
            sets[dir] = get_ds(base+"/"+dir)
    return sets

ori_sets = get_task_sets("orientation")
pow_sets = get_task_sets("power")

fprint(ori_sets)

