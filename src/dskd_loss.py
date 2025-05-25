import torch
import torch.nn as nn


def preprocess_model_data(model_dict):
    inputs = model_dict["input_ids"]
    targets = model_dict["target_ids"]
    mask = targets.ne(model_dict["pad_token_id"])

    hiddens = model_dict["hidden_states"]

    token_embeds = model_dict["token_embeddings"]
    input_embeds = token_embeds[inputs * mask] # line 97
    target_embeds = token_embeds[targets * mask]
    embeds = torch.cat([input_embeds, target_embeds])

    lm_weights = model_dict["lm_weights"]

    return lm_weights, targets, hiddens, embeds, target_embeds, mask

def norm_std(t):
    return t / t.std

def compute_attention(q, k, d, mask, masked_val=1e5):
    attention = q @ k.transpose(-1, -2) / torch.sqrt(2 * d)
    attention = torch.where(mask, attention, attention - masked_val)
    attention = torch.softmax(attention, -1)
    return attention

def compute_projection_logits(lm_weights, projector, input, attention):
    proj_values = projector(input).float()
    proj_hiddens = (attention @ proj_values).to(input)
    proj_logits = lm_weights @ proj_hiddens # TODO: does proj_hiddens @ lm_weights.transpose(-1, -2) make a difference?
    return proj_logits


def dskd_loss_fn(student, teacher, projectors, ce_loss, kl_div):
    # Preprocessing
    s_lm_weights, s_targets, s_hiddens, s_embeds, s_target_embeds, s_mask = preprocess_model_data(student)
    t_lm_weights, t_targets, t_hiddens, t_embeds, t_target_embeds, t_mask = preprocess_model_data(teacher)

    t_hiddens_norm = norm_std(t_hiddens)
    t_embeds_norm = norm_std(t_embeds)
    t_target_embeds_norm = norm_std(t_target_embeds)

    # Projection
    attn_mask = s_mask.unsqueeze(-1).float() * t_mask.unsqueeze(-1).float()
    queries = projectors["query"](s_embeds).float()
    keys = t_embeds_norm.float()
    
    t2s_attention = compute_attention(queries, keys, t_hiddens.shape[-1], attn_mask)
    t2s_logits = compute_projection_logits(s_lm_weights, projectors["t2s"], t_hiddens_norm + t_target_embeds_norm, t2s_attention)
    t2s_mask = s_mask * t2s_logits.argmax(-1).eq(s_targets)

    s2t_attention = compute_attention(keys.transpose(-1, -2), queries.transpose(-1, -2), t_hiddens.shape[-1], attn_mask)
    s2t_logits = compute_projection_logits(t_lm_weights, projectors["s2t"], s_hiddens, s2t_attention)
    s2t_mask = t_mask

    # Loss computation
    t2s_ce_loss = ce_loss(t2s_logits, s_targets)

    t2s_kl_div = kl_div(student["logits"], t2s_logits.detach()) # TODO: incorporate temperature
    t2s_kd_loss = (t2s_kl_div * t2s_mask).sum()

    s2t_kl_div = kl_div(s2t_logits, teacher["logits"])
    s2t_kd_loss = (s2t_kl_div * s2t_mask).sum()

    loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

    return loss












    
