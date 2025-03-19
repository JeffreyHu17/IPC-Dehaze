# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Henrique Morimitsu
# - Adapt code from JAX to PyTorch

"""Fast decoding routines for non-autoregressive generation."""

from typing import Callable
from einops import rearrange
import torch
import torch.nn.functional as F
import math
from basicsr.utils import mask_schedule

# Confidence score for known tokens to avoid masking or repredicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possiblity larger than 1 due to the noise addition.
def log(t, eps = 1e-10):
    return torch.log(t + eps)
def exists(val):
    return val is not None
def gumbel_noise(probs):
    noise = torch.zeros_like(probs).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(logits, temperature = 1., dim = -1):
    return ((logits / max(temperature, 1e-10)) + gumbel_noise(logits)).argmax(dim = dim)

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def sample_top_p(probs, p=0.75):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return probs_sort
    # next_token = torch.multinomial(probs_sort, num_samples=1)
    # next_token = torch.gather(probs_idx, -1, next_token)
    # return next_token

def mask_by_random_topk(
    mask_len: int,
    confidence: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Modifies from jax.random.choice without replacement.

    JAX's original implementation is as below:
        g = -gumbel(key, (n_inputs,)) - jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]
    We adds temperature annealing on top of it, which is:
        g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]

    Args:
        mask_len: the number to mask.
        probs: the probabilities associated with each entry.
        temperature: when temperature = 1.0, it's identical to jax's implementation.
        The larger this value is, the more random the masking is picked.

    Returns:
        A binary masking map [batch_size, seq_len].
    """

    g = torch.distributions.gumbel.Gumbel(0, 1)
    # confidence = torch.log(probs) + temperature * g.sample(probs.shape).to(probs.device)
    sorted_confidence = torch.sort(confidence, dim=-1,descending=True)[0]
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.gather(sorted_confidence, -1, mask_len)
    # Masks tokens with lower confidence.
    masking = (confidence >= cut_off)
    return masking

def mask_by_random_topk_origin(
    mask_len: int,
    probs: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Modifies from jax.random.choice without replacement.

    JAX's original implementation is as below:
        g = -gumbel(key, (n_inputs,)) - jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]
    We adds temperature annealing on top of it, which is:
        g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
        ind = jnp.argsort(g)[:n_draws]

    Args:
        mask_len: the number to mask.
        probs: the probabilities associated with each entry.
        temperature: when temperature = 1.0, it's identical to jax's implementation.
        The larger this value is, the more random the masking is picked.

    Returns:
        A binary masking map [batch_size, seq_len].
    """
    g = torch.distributions.gumbel.Gumbel(0, 1)
    confidence = torch.log(probs) + temperature * g.sample(probs.shape).to(probs.device)
    sorted_confidence = torch.sort(confidence, dim=-1)[0]
    # Obtains cut off threshold given the mask lengths.
    cut_off = torch.gather(sorted_confidence, -1, mask_len)
    # Masks tokens with lower confidence.
    masking = (confidence < cut_off)
    return masking

class State:
    """Holds decoding state data."""
    def __init__(
        self,
        cur_index: int,  # scalar int32: current decoded length index
        cur_seqs: torch.Tensor,  # int32 [batch, seq_len]
        final_seqs: torch.Tensor,  # int32 [batch, num_iter, seq_len],
        final_masks:torch.Tensor,  # int32 [batch, seq_len]
        final_probs:torch.Tensor

    ) -> None:
        self.cur_index = cur_index
        self.cur_seqs = cur_seqs
        self.final_seqs = final_seqs
        self.final_masks=final_masks
        self.final_probs=final_probs


def state_init(
    init_indices: torch.Tensor,
    num_iter: int,
    start_iter: int = 0,
    h:int = 0,
    w:int = 0
) -> State:
    """Initializes the decoding state data structure."""
    final_seqs0 = init_indices.unsqueeze(1)
    final_seqs0 = final_seqs0.repeat(1, num_iter, 1)
    final_probs=torch.zeros(1,num_iter,h*w).to(final_seqs0.device)

    init_indices = torch.ones(h*w).to(final_seqs0.device)
    return State(
        cur_index=start_iter, cur_seqs=init_indices, final_seqs=final_seqs0,final_masks=final_seqs0.clone(),final_probs=final_probs)

def state_initOrigin(
    init_indices: torch.Tensor,
    num_iter: int,
    start_iter: int = 0,
    h:int = 0,
    w:int = 0
) -> State:
    """Initializes the decoding state data structure."""
    final_seqs0 = init_indices.unsqueeze(1)
    final_seqs0 = final_seqs0.repeat(1, num_iter, 1)
    final_probs=torch.zeros(1,num_iter,h*w).to(final_seqs0.device)

    return State(
        cur_index=start_iter, cur_seqs=init_indices, final_seqs=final_seqs0,final_masks=final_seqs0.clone(),final_probs=final_probs)

def decode_feats(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    t_map:torch.Tensor,
    ic_map:torch.Tensor,
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine"
) -> torch.Tensor:
    b,c,h,w=lq_feats.shape

    hq_feats=(lq_feats).clone()
    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_init(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None


    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        hq_feats=hq_feats*~mask+lq_feats*mask
        if t_map==None:
            input_feats = torch.cat((lq_feats,hq_feats),dim=1)
        elif ic_map==None:
            input_feats = torch.cat((lq_feats,hq_feats,t_map),dim=1)
        else:
            input_feats = torch.cat((lq_feats,hq_feats,t_map,ic_map),dim=1)

        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        state.final_probs[:, step] = probs
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]

        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
        # sampled_ids = probs.argmax(2)
        
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
        selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = torch.where(unknown_map, selected_probs,
                                    torch.zeros_like(selected_probs) + torch.inf)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,mask_scheduling_method)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        # print(step,mask_len-torch.unsqueeze(
        #     torch.floor(unknown_number_in_the_beginning * mask_ratio), 1))

        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)

        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        # print()
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

        # Adds noise for randomness
        masking = mask_by_random_topk(mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
        # Masks tokens with lower confidence.
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        
        state.cur_seqs = sampled_ids
        

    return state.final_seqs,state.final_masks,state.final_probs
def decode_mask(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine"
) -> torch.Tensor:
    """Fast decoding for iterative generation.

    Args:
        mask_tokens:[b,transformer_block]
        inputs: int32 array: [batch_size, seq_length] input sequence of masked
        tokens, where the masking tokens is defined by mask_token_id.
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        mask_token_id: int: [Mask] token id.
        num_iter: int: default is 12.
        start_iter: int: default is 0.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

    Returns:
        [batch_size, num_iter, seq_length] output sequence of tokens in all
        iterations.
    """
    b,c,h,w=lq_feats.shape

    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_initOrigin(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None


    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        input_feats=hq_feats*~mask+lq_feats*mask
        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        state.final_probs[:, step] = probs
        
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]
        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        # sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        # # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
        # selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        # # Ignores the tokens given in the input by overwriting their confidence.
        # selected_probs = torch.where(unknown_map, selected_probs,
        #                             torch.zeros_like(selected_probs) + torch.inf)
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                            mask_scheduling_method)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()
        hq_feats=tokens_to_feats(sampled_ids.reshape(b,1,h,w))
        selected_probs = tokens_to_logits(hq_feats,critic=True)
        # Adds noise for randomness
        masking = mask_by_random_topk(mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
        # # Masks tokens with lower confidence.
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        # state.cur_seqs = torch.cat(sampled_ids,lq_tokens,dim=1)
        state.cur_seqs = sampled_ids
        # T=sampled_ids.reshape(1,32,16)
        # print("!1")

    return state.final_seqs,state.final_masks
# def decode_mask(
#     mask_tokens:torch.Tensor,
#     lq_feats: torch.Tensor,
#     tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
#     tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
#     mask_token_id: int = -1,
#     num_iter: int = 12,
#     start_iter: int = 0,
#     choice_temperature: float = 1.0,
#     mask_scheduling_method: str = "cosine",
#     critic_noise_anneal_schedule='decay'
# ) -> torch.Tensor:
    
#     noise_K=1
#     starting_temperature =0.9
    
#     b,c,h,w=lq_feats.shape
#     device = lq_feats.device
#     # mask_feats=torch.zeros_like(lq_feats)
#     # unknown_number_in_the_beginning=h*w
#     unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
#     # Initializes state
#     state = state_init(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

#     # lq_tokens=inputs[...,inputs.shape[-1]//2:]
#     hq_feats=None
#     scores=None
#     masking = torch.ones((b,unknown_number_in_the_beginning), device = device, dtype = torch.bool)
#     sampled_ids=torch.full((b,unknown_number_in_the_beginning), mask_token_id, device = device)
#     for step in range(start_iter, num_iter):
#         """Beam search."""
#         # Current input ids: [batch_size, seq_length].
        
        
#         if exists(scores):
            
#             ratio = 1. * (step ) / num_iter
#             mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
#                                                 mask_scheduling_method)
#             # Gets mask lens for each sample in the batch according to the mask ratio.

#             mask_len = torch.unsqueeze(
#                 torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
#             # Keeps at least one of prediction in this round and also masks out at least
#             # one and for the next iteration
#             mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(masking, dim=-1, keepdim=True) - 1).long()
        

#             # Adds noise for randomness
#             masking = mask_by_random_topk(mask_len, scores,
#                                         choice_temperature * (1. - ratio))
            
            
#             # time = torch.full((1,), step / num_iter, device = device)

#             # num_tokens_mask = (mask_len * torch.cos(time * math.pi * 0.5)).round().long().clamp(min = 1)
#             # _, indices = scores.topk(num_tokens_mask.item(), dim = -1)
#             # masking = torch.zeros((b,mask_len), device = device).scatter(1, indices, 1).bool()

#             # Obtains cut off threshold given the mask lengths.
#             # Masks tokens with lower confidence.
#             state.final_masks[:, step] = masking
        
#         cur_ids = state.cur_seqs

#         hq_feats=tokens_to_feats((cur_ids*~masking).reshape(b,1,h,w))

#         # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
#         mask=masking.reshape(1,1,h,w).repeat(1,256,1,1)

#         input_feats=hq_feats*~mask+lq_feats*mask
       
#         # Calls model on current seqs to get next-iteration seqs.
#         logits = tokens_to_logits(input_feats)
#         # Computes the probabilities of each selected tokens.
#         # probs = F.softmax(logits, -1)
#         temperature = starting_temperature * (1-(step+1) / num_iter)
#         sampled_ids = gumbel_sample(logits, temperature = temperature)
            
#         state.final_probs[:, step] = logits
#         # Samples the ids using categorical sampling: [batch_size, seq_length].
#         # b = probs.shape[0]


#         # sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
#         # sampled_ids = probs.argmax(-1)        
#         if critic_noise_anneal_schedule == 'fixed':
#             noise_multiplier = 1.
#         elif critic_noise_anneal_schedule == 'decay':
#             noise_multiplier = 1 - (step + 1) / num_iter
#         elif critic_noise_anneal_schedule == 'increase':
#             noise_multiplier = (step + 1) / num_iter
#         critic_logits = tokens_to_logits(sampled_ids,h,w,critic=True)
        
#         noise = noise_K * (uniform(critic_logits.shape, device) - 0.5) * noise_multiplier
        
#         scores = critic_logits + noise
        
#         # sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

#         # sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
#         # Updates final seqs with the current sampled_ids.
#         # sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
#         state.final_seqs[:, step] = sampled_ids
        

#         state.cur_index += 1
#         state.cur_seqs = sampled_ids
  
#     return state.final_seqs,state.final_masks
def decode_mask2(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine"
) -> torch.Tensor:
    b,c,h,w=lq_feats.shape

    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_init(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None

    mask = torch.ones((1,h,w),device=lq_feats.device).bool()
    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs
        
        # Just updates the masked tokens.
        # unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats(cur_ids.reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        # mask=unknown_map.reshape(1,1,h,w)

        input_feats=hq_feats*~mask+lq_feats*mask
        if step !=0:
            critic_logits = tokens_to_logits(input_feats,True)
              # Defines the mask ratio for the next round. The number to mask out is determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1. * (step) / num_iter
            mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,mask_scheduling_method)
            # Gets mask lens for each sample in the batch according to the mask ratio.

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)

            # Keeps at least one of prediction in this round and also masks out at least one and for the next iteration
            mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

            # Adds noise for randomness
            mask = mask_by_random_topk(mask_len, critic_logits,
                                        choice_temperature * (ratio))

            hq_feats=tokens_to_feats(cur_ids).reshape(b,1,h,w)

            input_feats=hq_feats*~mask+lq_feats*mask

        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        state.final_probs[:, step] = probs
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]


        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
       
        # sampled_ids = probs.argmax(-1)
        
        # sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        # sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = mask.flatten(1)
        

    #     selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
    # # Ignores the tokens given in the input by overwriting their confidence.
    #     selected_probs = torch.where(unknown_map, selected_probs,
    #                             torch.zeros_like(selected_probs) + torch.inf)
        
  

    # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
    # selected_probs = torch.gather(critic_logits, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
    # Ignores the tokens given in the input by overwriting their confidence.
    # selected_probs = torch.where(unknown_map, selected_probs,
    #                             torch.zeros_like(selected_probs) + torch.inf)
    
        
      
        state.cur_index += 1
        state.cur_seqs = sampled_ids
  
    return state.final_seqs,state.final_masks,state.final_probs

def decode(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine",
    beta:float = 1
) -> torch.Tensor:
    """Fast decoding for iterative generation.

    Args:
        mask_tokens:[b,transformer_block]
        inputs: int32 array: [batch_size, seq_length] input sequence of masked
        tokens, where the masking tokens is defined by mask_token_id.
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        mask_token_id: int: [Mask] token id.
        num_iter: int: default is 12.
        start_iter: int: default is 0.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

    Returns:
        [batch_size, num_iter, seq_length] output sequence of tokens in all
        iterations.
    """
    b,c,h,w=lq_feats.shape
    
    threshold = 0.6

    # mask_feats=torch.zeros_like(lq_feats)


    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    mask_len=unknown_number_in_the_beginning
    # Initializes state
    state = state_initOrigin(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None
    iter_slow=-1
    
    for step in range(start_iter+1, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        input_feats=hq_feats*~mask+lq_feats*mask
        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        # state.final_probs[:, step] = probs
        
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
        selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        # Ignores the tokens given in the input by overwriting their confidence.
        
        selected_probs = torch.where(unknown_map, selected_probs,
                                    torch.zeros_like(selected_probs) + torch.inf)
        if iter_slow<0:
        
            masking = selected_probs<threshold
            uniform_iter_len=torch.floor(unknown_number_in_the_beginning /num_iter)
            # 比规定的小，应该转为cosine慢推理
            if mask_len-torch.sum(masking,dim=-1)<uniform_iter_len:
                iter_slow=step
            
            mask_len = torch.sum(masking,dim=-1)    
            unknown_number_in_the_second = mask_len
        else:
            ratio = 1. * (step + 1-iter_slow) / (num_iter-iter_slow)
            mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_second,
                                                mask_scheduling_method,beta)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_second * mask_ratio), 1)
            mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            masking = mask_by_random_topk_origin(mask_len, selected_probs,
                                        choice_temperature * (1. - ratio))
            
            
            # # ==============top-p===============                      
            # NucleusSampler(0.75, TemperatureSampler(1.))()
            # # sample(model, tokenizer, NucleusSampler(0.95, TemperatureSampler(1.)), 4, 32, 128, prompt)
            # ==============top-k===============
            # probs=top_k(probs)(probs)
            
            # masking = (confidence < cut_off)
            
            
        state.final_probs[:,step]=selected_probs
        # ratio = 1. * (step + 1) / num_iter
        

            
        '''
        
        
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                            mask_scheduling_method)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

        # Adds noise for randomness
        masking = mask_by_random_topk_origin(mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
        # # Masks tokens with lower confidence.
        '''
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        # state.cur_seqs = torch.cat(sampled_ids,lq_tokens,dim=1)
        state.cur_seqs = sampled_ids
        
        # T=sampled_ids.reshape(1,32,16)
        # print("!1")

    return state.final_seqs,state.final_masks,state.final_probs

def decode_nomal(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    tokens_to_logits: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine",
    beta:float=1.0,
) -> torch.Tensor:
    """Fast decoding for iterative generation.

    Args:
        mask_tokens:[b,transformer_block]
        inputs: int32 array: [batch_size, seq_length] input sequence of masked
        tokens, where the masking tokens is defined by mask_token_id.
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        mask_token_id: int: [Mask] token id.
        num_iter: int: default is 12.
        start_iter: int: default is 0.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

    Returns:
        [batch_size, num_iter, seq_length] output sequence of tokens in all
        iterations.
    """
    b,c,h,w=lq_feats.shape

    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_initOrigin(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None


    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        input_feats=hq_feats*~mask+lq_feats*mask
        # Calls model on current seqs to get next-iteration seqs.
        logits = tokens_to_logits(input_feats)
        # Computes the probabilities of each selected tokens.
        probs = F.softmax(logits, -1)
        # state.final_probs[:, step] = probs
        
        # Samples the ids using categorical sampling: [batch_size, seq_length].
        b = probs.shape[0]
        sampled_ids = torch.multinomial(rearrange(probs, 'b n c -> (b n) c'), 1)[..., 0]
        sampled_ids = rearrange(sampled_ids, '(b n) -> b n', b=b)

        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
        selected_probs = torch.gather(probs, -1, sampled_ids.clamp(0, probs.shape[-1]-1).unsqueeze(-1)).squeeze(-1)
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = torch.where(unknown_map, selected_probs,
                                    torch.zeros_like(selected_probs) + torch.inf)
        
        state.final_probs[:,step]=selected_probs
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                            mask_scheduling_method,beta)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

        # Adds noise for randomness
        masking = mask_by_random_topk_origin(mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
        # # Masks tokens with lower confidence.
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        # state.cur_seqs = torch.cat(sampled_ids,lq_tokens,dim=1)
        state.cur_seqs = sampled_ids
        
        # T=sampled_ids.reshape(1,32,16)
        # print("!1")

    return state.final_seqs,state.final_masks,state.final_probs
def decode_codeinstance_nomal(
    mask_tokens:torch.Tensor,
    lq_feats: torch.Tensor,
    get_codeinstance: Callable[[torch.Tensor], torch.Tensor],
    tokens_to_feats:Callable[[torch.Tensor], torch.Tensor],
    mask_token_id: int = -1,
    num_iter: int = 12,
    start_iter: int = 0,
    choice_temperature: float = 1.0,
    mask_scheduling_method: str = "cosine",
    beta:float=1.0,
) -> torch.Tensor:
    """Fast decoding for iterative generation.

    Args:
        mask_tokens:[b,transformer_block]
        inputs: int32 array: [batch_size, seq_length] input sequence of masked
        tokens, where the masking tokens is defined by mask_token_id.
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        mask_token_id: int: [Mask] token id.
        num_iter: int: default is 12.
        start_iter: int: default is 0.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

    Returns:
        [batch_size, num_iter, seq_length] output sequence of tokens in all
        iterations.
    """
    b,c,h,w=lq_feats.shape

    # mask_feats=torch.zeros_like(lq_feats)
    mask_len=h*w

    unknown_number_in_the_beginning = torch.sum(mask_tokens == mask_token_id, dim=-1)
    # Initializes state
    state = state_initOrigin(mask_tokens, num_iter, start_iter=start_iter,h=h,w=w)

    # lq_tokens=inputs[...,inputs.shape[-1]//2:]
    hq_feats=None


    for step in range(start_iter, num_iter):
        """Beam search."""
        # Current input ids: [batch_size, seq_length].
        cur_ids = state.cur_seqs

        # Just updates the masked tokens.
        unknown_map = (cur_ids == mask_token_id)

        hq_feats=tokens_to_feats((cur_ids*~unknown_map).reshape(b,1,h,w))

        # unknown_map=unknown_map.unsqueeze(1).repeat(1,256,1,1)
        mask=unknown_map.reshape(1,1,h,w).repeat(1,256,1,1)

        input_feats=hq_feats*~mask+lq_feats*mask
        # Calls model on current seqs to get next-iteration seqs.
        
        min_values, min_indices=get_codeinstance(input_feats)
        # logits = tokens_to_logits(input_feats)
        # # Computes the probabilities of each selected tokens.
        # probs = F.softmax(logits, -1)
        # # state.final_probs[:, step] = probs
        
        # Samples the ids using categorical sampling: [batch_size, seq_length].
       
        sampled_ids =min_indices.reshape(b,-1)

        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        # Updates final seqs with the current sampled_ids.
        state.final_seqs[:, step] = sampled_ids
        state.final_masks[:, step] = unknown_map
        # Get probs for these id, which is benefit the operation of overrite the low confidence tokens after
        
        selected_d = torch.gather(min_values, -1, sampled_ids.clamp(0, min_values.shape[-1]-1))
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_d = torch.where(unknown_map, selected_d,
                                    torch.zeros_like(selected_d))
        
        state.final_probs[:,step]=selected_d
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                            mask_scheduling_method,beta)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.unsqueeze(
            torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = mask_len.clamp(torch.ones_like(mask_len), torch.sum(unknown_map, dim=-1, keepdim=True) - 1).long()

        # Adds noise for randomness
        masking = mask_by_random_topk(mask_len, selected_d,
                                    choice_temperature * (1. - ratio))
        # # Masks tokens with lower confidence.
        sampled_ids = torch.where(masking, mask_token_id, sampled_ids)
        state.cur_index += 1
        # state.cur_seqs = torch.cat(sampled_ids,lq_tokens,dim=1)
        state.cur_seqs = sampled_ids
        
        # T=sampled_ids.reshape(1,32,16)
        # print("!1")

    return state.final_seqs,state.final_masks,state.final_probs