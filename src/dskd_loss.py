import torch
import torch.nn as nn


def preprocess_model_data(model_dict):
    model = model_dict["model"]
    hiddens = model_dict["hidden_states"]
    token_embeds = model_dict["token_embeddings"]

    sequences = model_dict["sequences"]
    inputs = sequences[..., :sequences.shape[1] - 1]
    targets = sequences[..., 1:]
    mask = targets.ne(model_dict["pad_id"])

    input_embeds = token_embeds[inputs * mask] # line 97
    target_embeds = token_embeds[targets * mask]
    embeds = torch.cat([input_embeds, target_embeds])

    return model, targets, hiddens, embeds, target_embeds, mask

def norm_std(t):
    return t / t.std

def compute_attention(q, k, d, mask, masked_val=1e5):
    attention = q @ k.transpose(-1, -2) / torch.sqrt(2 * d)
    attention = torch.where(mask, attention, attention - masked_val)
    attention = torch.softmax(attention, -1)
    return attention

def compute_projection_logits(model, projector, input, attention):
    proj_values = projector(input).float()
    proj_hiddens = (attention @ proj_values).to(input)
    proj_logits = model.lm_head(proj_hiddens)
    return proj_logits


def dskd_loss(student, teacher, projectors):
    ce_loss = nn.CrossEntropyLoss()
    kl_div_loss = nn.KLDivLoss(reduction='none')

    # Preprocessing
    s_model, s_targets, s_hiddens, s_embeds, s_target_embeds, s_mask = preprocess_model_data(student)
    t_model, t_targets, t_hiddens, t_embeds, t_target_embeds, t_mask = preprocess_model_data(teacher)

    t_hiddens_norm = norm_std(t_hiddens)
    t_embeds_norm = norm_std(t_embeds)
    t_target_embeds_norm = norm_std(t_target_embeds)

    # Projection
    attn_mask = s_mask.unsqueeze(-1).float() * t_mask.unsqueeze(-1).float()
    queries = projectors["query"](s_embeds).float()
    keys = t_embeds_norm.float()
    
    t2s_attention = compute_attention(queries, keys, t_hiddens.shape[-1], attn_mask)
    t2s_logits = compute_projection_logits(s_model, projectors["t2s"], t_hiddens_norm + t_target_embeds_norm, t2s_attention)
    t2s_mask = s_mask * t2s_logits.argmax(-1).eq(s_targets)

    s2t_attention = compute_attention(keys.transpose(-1, -2), queries.transpose(-1, -2), t_hiddens.shape[-1], attn_mask)
    s2t_logits = compute_projection_logits(t_model, projectors["s2t"], s_hiddens, s2t_attention)
    s2t_mask = t_mask

    # Loss computation
    t2s_ce_loss = ce_loss(t2s_logits, s_targets)

    t2s_kl_div = kl_div_loss(student["logits"], t2s_logits.detach()) # TODO: incorporate temperature
    t2s_kd_loss = (t2s_kl_div * t2s_mask).sum()

    s2t_kl_div = kl_div_loss(s2t_logits, teacher["logits"])
    s2t_kd_loss = (s2t_kl_div * s2t_mask).sum()

    loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

    return loss












    
