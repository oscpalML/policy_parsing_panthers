#!/usr/bin/env python3

import os
import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import torch

runname = "deberta(long)_xlmr_ensemble"

input_dir = os.environ["inputDataset"]
output_dir = os.environ["outputDir"]

batch_size = int(sys.argv[1])

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
    return country_codes[country]+", "+sex_codes[sex]+". "+str(row["text"])

def prepend_data_trans(row):
    country = "".join(filter(lambda x: not x.isdigit(), str(row["id"])))
    sex = str(row["sex"])
    return country_codes[country]+", "+sex_codes[sex]+". "+str(row["text_en"])


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
            name = dir.removesuffix("-test.tsv")
            name = name.removeprefix(folder+"-")
            sets[name] = get_ds(base+"/"+dir)
    return DatasetDict(sets)

def get_dsd_logits(task, dsd, field, model_name, seq_len=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(examples):
        return  tokenizer(examples[field], truncation=True, max_length=seq_len)
    dsd = dsd.map(tokenize_function, batched=True)    

    args = TrainingArguments(output_dir="/temp_out",
                              use_cpu=not torch.cuda.is_available(),
                              per_device_eval_batch_size = batch_size,
                              group_by_length = True
                              )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="multi_label_classification")
    trainer = Trainer(model = model, 
                      args = args,
                      tokenizer=tokenizer,
                      data_collator=data_collator
                      )

    logits_dict = {}
    for key, ds in dsd.items():
        logits = torch.Tensor(trainer.predict(test_dataset=ds)[0][:, model.config.label2id[task]])
        logits_dict[key] = logits

    return logits_dict

def get_dsd_ids(dsd):
    return {key:ds["id"] for key, ds in dsd.items()}

def logits_to_preds(logits_dict):
    return {key:torch.round(torch.sigmoid(logits)) for key, logits in logits_dict.items()}

def write_preds_dict(preds_dict, task):
    if task=="orientation":
        ids_dict = ori_ids
    elif task=="power":
        ids_dict = pow_ids

    for key, preds in preds_dict.items():
        df = pd.DataFrame(data=preds, index=ids_dict[key])
        df.to_csv(output_dir+"/policyparsingpanthers-"+task+"-"+key+"-"+runname+".tsv", header=False, sep="\t", index=True)

def ensem_logits(first_logits, second_logits):
    res = {}
    for key, _ in first_logits.items():
        res[key] = torch.add(first_logits[key], second_logits[key])/2.0
    return res



ori_dsd = get_task_sets("orientation")
pow_dsd = get_task_sets("power")

ori_ids = get_dsd_ids(ori_dsd)
pow_ids = get_dsd_ids(pow_dsd)

xlmr_name = "oscpalML/XLM-RoBERTa-political-classification"

xlmr_ori_logits = get_dsd_logits("orientation", ori_dsd, "text", xlmr_name, seq_len=512)
xlmr_pow_logits = get_dsd_logits("power", pow_dsd, "text", xlmr_name, seq_len=512)

deberta_name = "oscpalML/DeBERTa-political-classification"

deb_ori_logits = get_dsd_logits("orientation", ori_dsd, "text_en", deberta_name, seq_len=2048)
deb_pow_logits = get_dsd_logits("power", pow_dsd, "text_en", deberta_name, seq_len=2048)

ensem_ori_logits = ensem_logits(xlmr_ori_logits, deb_ori_logits)
ensem_pow_logits = ensem_logits(xlmr_pow_logits, deb_pow_logits)

ensem_ori_preds = logits_to_preds(ensem_ori_logits)
ensem_pow_preds = logits_to_preds(ensem_pow_logits)

write_preds_dict(ensem_ori_preds, "orientation")
write_preds_dict(ensem_pow_preds, "power")