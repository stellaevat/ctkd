import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from dskd_loss import dskd_loss_fn


class DSKD(nn.Module):
    def __init__(self, args, device):
        super(DSKD, self).__init__()
        self.device = device

        self.s_model, self.s_tokenizer, self.s_hidden_size = self._load_pretrained(args.s_path, args.s_type, args.s_dtype)
        self.t_model, self.t_tokenizer, self.t_hidden_size = self._load_pretrained(args.t_path, args.t_type, args.t_dtype)
        self.projectors = self._load_projectors(args.proj_path)

        self.kl_temp = args.kl_temperature
        self.kd_loss_fn = dskd_loss_fn
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.weighted_loss_fn = lambda ce_loss, kd_loss: (1 - args.kd_weight) * ce_loss + args.kd_weight * kd_loss
        

    def _load_pretrained(self, model_path, model_type, model_dtype):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hidden_size = config.n_embed if hasattr(config, "n_embed") else config.hidden_size
        
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=model_dtype, trust_remote_code=True)
        model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.eos_token_id = 151643 if model_type == "qwen" else tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer, hidden_size
    

    def _load_projectors(self, proj_path):
        projectors = nn.ModuleDict()
        projectors["s2t"] = nn.Linear(self.s_hidden_size, self.t_hidden_size)
        projectors["t2s"] = nn.Linear(self.t_hidden_size, self.s_hidden_size)
        projectors["query"] = nn.Linear(2 * self.s_hidden_size, 2 * self.t_hidden_size)
        
        if os.path.exists(proj_path):
            params = torch.load(proj_path, map_location=f"cuda:{self.device}")
            for proj_name in projectors:
                state_dict = {k.split('.', 1)[1] : v for (k, v) in params.items() if k.startswith(proj_name)}
                projectors[proj_name].load_state_dict(state_dict)
        
        return projectors
    

    def _get_dskd_args(self, model_prefix, batch, output):
        lm_weights = vars(self)[f"{model_prefix}_model"].lm_head.weight.detach()
        hiddens = output.hidden_states[-1]
        logits = output.logits
        
        input_ids = batch[f"{model_prefix}_input_ids"]
        target_ids = batch[f"{model_prefix}_labels"]

        pad_token_id = vars(self)[f"{model_prefix}_tokenizer"].pad_token_id
        mask = target_ids.ne(pad_token_id)

        token_embeds = vars(self)[f"{model_prefix}_model"].model.embed_tokens
        input_embeds = token_embeds[input_ids * mask] # line 97
        target_embeds = token_embeds[target_ids * mask]

        dskd_args = {
            "lm_weights" : lm_weights,
            "targets" : target_ids,
            "hiddens" : hiddens,
            "logits" : logits,
            "input_embeds" : input_embeds,
            "target_embeds" : target_embeds,
            "mask" : mask
        }
        
        return dskd_args
    

    def forward(self, batch):
        s_output = self.s_model(
            batch["s_input_ids"],
            attention_mask=batch.get("s_attention_mask", None),
            position_ids=batch.get("s_position_ids", None),
            output_hidden_states=True
        )

        with torch.no_grad():
            self.t_model.eval()
            t_output = self.t_model(
                batch["t_input_ids"],
                attention_mask=batch.get("t_attention_mask", None),
                position_ids=batch.get("t_position_ids", None),
                output_hidden_states=True
            )

        s_dskd_args = self._get_dskd_args("s", batch, s_output)
        t_dskd_args = self._get_dskd_args("t", batch, t_output)

        s_targets = batch["s_labels"]
        s_logits = s_output.logits
        s_logits_mask = 1 - (s_logits.isnan() | s_logits.isinf())
        s_pad_mask = s_dskd_args["mask"]

        ce_loss = (s_pad_mask * self.ce_loss_fn(s_logits * s_logits_mask, s_targets * s_pad_mask)).sum()
        kd_loss = self.kd_loss_fn(s_dskd_args, t_dskd_args, self.projectors, self.kl_temp)

        n_batch_tokens = s_targets.ne(self.s_tokenizer.pad_token_id).sum()
        token_level_ce_loss = ce_loss / n_batch_tokens
        token_level_kd_loss = kd_loss / n_batch_tokens
        token_level_loss = self.weighted_loss_fn(token_level_ce_loss, token_level_kd_loss)

        return token_level_loss, token_level_ce_loss, token_level_kd_loss



            

        


    






