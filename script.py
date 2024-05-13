#!/usr/bin/env python3

import os
import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

input_dir = os.environ["inputDataset"]
output_dir = os.environ["outputDir"]

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


def prepend_data_orig(row):
    country = "".join(filter(lambda x: not x.isdigit(), str(row["id"])))
    sex = str(row["sex"])
    return country_codes[country]+", "+sex_codes[sex]+". "+row["text"]

def prepend_data_trans(row):
    country = "".join(filter(lambda x: not x.isdigit(), str(row["id"])))
    sex = str(row["sex"])
    return country_codes[country]+", "+sex_codes[sex]+". "+row["text_en"]


def get_ds(path):
    df = pd.read_csv(path, sep="\t", usecols=["text", "text_en", "sex", "id"])
    df["text"] = df.apply(prepend_data_orig, axis=1)
    df["text_en"] = df.apply(prepend_data_trans, axis=1)
    df.drop(columns=["sex"], inplace=True)
    return Dataset.from_pandas(df, preserve_index=False)


def get_task_sets(folder):
    base = input_dir+"/"+folder
    sets = {}
    for dir in os.listdir(base):
        if dir.endswith(".tsv"):
            sets[dir] = get_ds(base+"/"+dir)
    return DatasetDict(sets)

def get_dsd_logits(task, dsd, field, model_name, seq_len=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return  tokenizer(examples[field], truncation=True, max_length=seq_len)
    dsd = dsd.map(tokenize_function, batched=True)

    args = TrainingArguments(output_dir="/temp_out",
                              use_cpu=True,
                              per_device_eval_batch_size = 2
                              )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="multi_label_classification")
    trainer = Trainer(model = model, args = args)

    for key, ds in dsd.items():
        logits = trainer.predict(test_dataset=ds)
        fprint(logits)

    index = model.config.label2id[task]


    


ori_dsd = get_task_sets("orientation")
pow_dsd = get_task_sets("power")

xlmr_name = "oscpalML/XLM-RoBERTa-political-classification"

get_dsd_logits("orientation", ori_dsd, "text", xlmr_name, seq_len=512)


deb_name = "oscpalML/DeBERTa-political-classification"
