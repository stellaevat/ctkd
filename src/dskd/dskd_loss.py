import torch
import torch.nn as nn
from operator import itemgetter


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

def dskd_loss_fn(s_args, t_args, projectors, kl_temp):
    s_lm_weights, s_targets, s_hiddens, s_logits, s_input_embeds, s_target_embeds, s_pad_mask = itemgetter(
        "lm_weights", "targets", "hiddens", "logits", "input_embeds", "target_embeds", "mask"
    )(s_args)

    t_lm_weights, t_targets, t_hiddens, t_logits, t_input_embeds, t_target_embeds, t_pad_mask = itemgetter(
        "lm_weights", "targets", "hiddens", "logits", "input_embeds", "target_embeds", "mask"
    )(t_args)

    s_embeds = torch.cat([s_input_embeds, s_target_embeds])
    t_embeds = torch.cat([t_input_embeds, t_target_embeds])

    t_hiddens_norm = norm_std(t_hiddens)
    t_embeds_norm = norm_std(t_embeds)
    t_target_embeds_norm = norm_std(t_target_embeds)

    # Projection
    attn_mask = s_pad_mask.unsqueeze(-1).float() * t_pad_mask.unsqueeze(-1).float()
    queries = projectors["query"](s_embeds).float()
    keys = t_embeds_norm.float()
    
    t2s_attention = compute_attention(queries, keys, t_hiddens.shape[-1], attn_mask)
    t2s_logits = compute_projection_logits(s_lm_weights, projectors["t2s"], t_hiddens_norm + t_target_embeds_norm, t2s_attention)
    t2s_true_mask = s_pad_mask * t2s_logits.argmax(-1).eq(s_targets)

    s2t_attention = compute_attention(keys.transpose(-1, -2), queries.transpose(-1, -2), t_hiddens.shape[-1], attn_mask)
    s2t_logits = compute_projection_logits(t_lm_weights, projectors["s2t"], s_hiddens, s2t_attention)
    s2t_true_mask = t_pad_mask

    # Loss computation
    ce_loss_fn = nn.CrossEntropyLoss(reduction='none')
    kl_div_fn = nn.KLDivLoss(reduction='none')

    t2s_logits_mask = 1 - (t2s_logits.isnan() | t2s_logits.isinf())
    s2t_logits_mask = 1 - (s2t_logits.isnan() | s2t_logits.isinf())

    t2s_ce_loss = (s_pad_mask * ce_loss_fn(t2s_logits * t2s_logits_mask, s_targets * s_pad_mask)).sum()

    t2s_kl_div = kl_div_fn((s_logits / kl_temp).log(), t2s_logits/kl_temp)
    t2s_kd_loss = (t2s_kl_div * t2s_true_mask * t2s_logits_mask).sum()

    s2t_kl_div = kl_div_fn(s2t_logits.log(), t_logits)
    s2t_kd_loss = (s2t_kl_div * s2t_true_mask * s2t_logits_mask).sum()

    loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss

    return loss












    
