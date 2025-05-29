import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets

def load_data(dir, splits=["train", "dev"]):
    data = {}

    for split in splits:
        path = os.path.join(dir, f"{split}.jsonl")

        samples = []
        with open(path) as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append((sample["prompt"], sample["output"]))

        data[split] = samples

    return data


def pad_data(fill, pad, size, fill_idx, pad_idx=None):
    padded = torch.ones(size, dtype=torch.long) * pad
    padded[fill_idx[0] : fill_idx[1]] = fill
    if pad_idx is not None:
        padded[pad_idx[0] : pad_idx[1]] = pad
    return padded.unsqueeze(0)


def tokenize_data(data, tokenizer, prefix, mask_token_id=-100, max_prompt_length=512, max_input_length=1024):
    prompt_kwargs = {
        "add_special_tokens" : False,
        "truncation" : True, 
        "max_length" : max_prompt_length,
        "return_token_type_ids" : False
    }

    paddings = {
        "input_ids" : tokenizer.eos_token_id,
        "attention_mask" : 0.0,
        "position_ids" : 0.0,
        "label" : mask_token_id,
        "loss_mask" : 0.0
    }

    data_tokenized = {}
    for (split, samples) in data.items():
        split_tokenized = defaultdict(list)

        for (prompt, response) in tqdm(samples):
            prompt_fields = tokenizer.encode_plus(prompt, return_tensors='pt', **prompt_kwargs)
            response_fields = tokenizer.encode_plus(response, return_tensors='pt')

            for (k, v) in prompt_fields.items():
                input = torch.cat((prompt_fields[k][0], response_fields[k][0]), dim=-1)[:max_input_length]
                input_padded = pad_data(input[:-1], paddings[k], max_input_length, fill_idx=(0, len(input)-1))
                split_tokenized[f"{prefix}_{k}"].append(input_padded)

                if k == "input_ids":
                    fill_idx = (0, len(input)-1)
                    pad_idx = (0, len(v[0])-1)
                    label = pad_data(input[1:], paddings["label"], max_input_length, fill_idx=fill_idx, pad_idx=pad_idx)
                    loss_mask = pad_data(1.0, paddings["loss_mask"], max_input_length, fill_idx=fill_idx, pad_idx=pad_idx)

                    gen_fill_idx = (-len(v[0]), max_input_length)
                    gen_input_ids = pad_data(v[0], paddings["input_ids"], max_input_length, fill_idx=gen_fill_idx)
                    gen_attention_mask = pad_data(1.0, paddings["attention_mask"], max_input_length, fill_idx=gen_fill_idx)

                    split_tokenized[f"{prefix}_label"].append(label)
                    split_tokenized[f"{prefix}_loss_mask"].append(loss_mask)
                    split_tokenized[f"{prefix}_gen_input_ids"].append(gen_input_ids) 
                    split_tokenized[f"{prefix}_gen_attention_mask"].append(gen_attention_mask)

        split_tokenized_collated = {}
        for (field, tensors) in split_tokenized.items():
            split_tokenized_collated[field] = torch.cat(tensors, dim=0)

        data_tokenized[split] = split_tokenized_collated

    return data_tokenized

def build_student_teacher_dataset(distiller, data_dir, data_splits=["train", "dev"]):
    data = load_data(data_dir, data_splits)
    s_data_tokenized = tokenize_data(data, distiller.s_tokenizer, "s", distiller.mask_token_id, distiller.max_prompt_length, distiller.max_input_length)
    t_data_tokenized = tokenize_data(data, distiller.t_tokenizer, "t", distiller.mask_token_id,distiller. max_prompt_length, distiller.max_input_length)
    
    datasets = {}
    for split in data_splits:
        comb_data_tokenized = s_data_tokenized[split] | t_data_tokenized[split]
        datasets[split] = Dataset.from_dict(comb_data_tokenized)

    dataset_dict = DatasetDict(datasets)
    dataset_dict.set_format(type="torch")
    return dataset_dict