import os
import json
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput

from dskd_loss import dskd_loss_fn


class DSKD(nn.Module):
    def __init__(self, args, device):
        super(DSKD, self).__init__()
        self.device = device

        self.s_model, self.s_tokenizer, self.s_hidden_size = self._load_pretrained(args.s_path, args.s_type, args.s_dtype)
        self.t_model, self.t_tokenizer, self.t_hidden_size = self._load_pretrained(args.t_path, args.t_type, args.t_dtype)
        self.projectors = self._load_projectors(args.proj_path)

        self.kd_loss_fn = dskd_loss_fn
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.kl_div_fn = nn.KLDivLoss(reduction='none')
        self.kl_temp = args.kl_temperature
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
    

    def forward(self, inputs, outputs):
        s_output = self.s_model(
            inputs["s_input_ids"],
            attention_mask=inputs.get("s_attention_mask", None),
            position_ids=inputs.get("s_position_ids", None),
            output_hidden_states=True
        )

        with torch.no_grad():
            self.t_model.eval()
            t_output = self.t_model(
                inputs["t_input_ids"],
                attention_mask=inputs.get("t_attention_mask", None),
                position_ids=inputs.get("t_position_ids", None),
                output_hidden_states=True
            )


        student_dict = {
            "input_ids" : inputs["s_input_ids"],
            "target_ids" : outputs["s_labels"],
            "token_embeddings" : self.s_model.model.embed_tokens,
            "hidden_states" : s_output.hidden_states[-1],
            "lm_weights" : self.s_model.lm_head.weight.detach(),
            "pad_token_id" : self.s_tokenizer.pad_token_id
        }

        teacher_dict = {
            "input_ids" : inputs["t_input_ids"],
            "target_ids" : outputs["t_labels"],
            "token_embeddings" : self.t_model.model.embed_tokens,
            "hidden_states" : t_output.hidden_states[-1],
            "lm_weights" : self.t_model.lm_head.weight.detach(),
            "pad_token_id" : self.t_tokenizer.pad_token_id
        }


        ce_loss = self.ce_loss_fn(s_output.logits, outputs["s_labels"])
        kd_loss = self.kd_loss_fn(student_dict, teacher_dict, self.projectors, self.ce_loss_fn, self.kl_div_fn, self.kl_temp)
        loss = self.weighted_loss_fn(ce_loss, kd_loss)

        return CausalLMOutput(loss=loss, logits=s_output.logits)



            

        


    






