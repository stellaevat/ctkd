import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DatasetDict

def load_data(dir, splits=["train", "dev"]):
    data = {}

    for split in splits:
        path = os.path.join(dir, f"{split}.jsonl")

        samples = []
        with open(path) as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append((sample["prompt"], sample["output"]))
                # samples.append({
                #     "prompt" : sample["prompt"],
                #     "response" : sample["output"]
                # })

        data["split"] = samples

    return data

def tokenize_data(data, tokenizer, mask_token_id=-100, max_prompt_length=512, max_input_length=1024):
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

        for (prompt, response) in samples:
            prompt_fields = tokenizer(prompt, return_tensors='pt', **prompt_kwargs)
            response_fields = tokenizer(response, return_tensors='pt')

            for (k, v) in prompt_fields.items():
                input = torch.cat((prompt_fields[k], response_fields[k]), dim=-1)[..., :max_input_length]
                input_padded = torch.ones(max_input_length, dtype=torch.long) * paddings[k]
                input_padded[:len(input)-1] = input[:-1]
                split_tokenized[k].append(input_padded)

                if k == "input_ids":
                    label = torch.ones(max_input_length, dtype=torch.long) * paddings["label"]
                    label[:len(input)-1] = input[1:]
                    label[:len(v)-1] = paddings["label"]
                    split_tokenized["label"].append(label)
            
                    loss_mask = torch.ones(max_input_length, dtype=torch.long) * paddings["loss_mask"]
                    loss_mask[:len(input)-1] = 1.0
                    loss_mask[:len(v)-1] = paddings["loss_mask"]
                    split_tokenized["loss_mask"].append(loss_mask)

                    gen_input_ids = torch.ones(max_input_length, dtype=torch.long) * paddings["input_ids"]
                    gen_input_ids[-len(v):] = v
                    split_tokenized["gen_input_ids"].append(gen_input_ids)

                    gen_attention_mask = torch.ones(max_input_length, dtype=torch.long) * paddings["attention_mask"]
                    gen_attention_mask[-len(v):] = 1.0
                    split_tokenized["gen_attention_mask"].append(gen_attention_mask)

        for (field, tensors) in split_tokenized.items():
            split_tokenized[field] = torch.cat(tensors, dim=-1)

        data_tokenized[split] = split_tokenized

    return DatasetDict(data_tokenized)








        
        