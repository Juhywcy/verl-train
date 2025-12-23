# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor with Gradient Truncation
"""

import itertools
import logging
import os
import sys
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActorGradTrunc"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TruncatedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, truncation_value, truncation_mask=None, top_k=None):
        # logits: (N, V)
        # labels: (N,)
        # truncation_mask: (N,) boolean tensor, True means apply truncation
        # top_k: int, if not None, only truncate top-k unsampled tokens

        

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        

        ctx.save_for_backward(log_probs, labels, truncation_mask)
        ctx.truncation_value = truncation_value
        ctx.top_k = top_k


        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)


    @staticmethod
    def backward(ctx, grad_output):
        log_probs, labels, truncation_mask = ctx.saved_tensors

        truncation_value = ctx.truncation_value
        top_k = ctx.top_k
        
        probs = torch.exp(log_probs)
        grad_output_expanded = grad_output.unsqueeze(-1) # (N, 1)
        
        # Gradient for all j: - p_j * g
        grad_logits = - probs * grad_output_expanded
        
        if truncation_value is not None:
             # Clamp gradients for unsampled tokens
             grad_logits_clamped = torch.clamp(grad_logits, min=-truncation_value, max=truncation_value)
             
             # Restore the values at labels indices from the original grad_logits (which are -p_y * g)
             grad_logits_clamped.scatter_(-1, labels.unsqueeze(-1), grad_logits.gather(-1, labels.unsqueeze(-1)))
             
             if truncation_mask is not None:
                 # Apply truncation only where mask is True
                 # truncation_mask is (N,), expand to (N, 1)
                 mask = truncation_mask.unsqueeze(-1)
                 
                 if top_k is not None and top_k > 0:
                     # Only truncate top-k unsampled tokens
                     # We need to find top-k indices based on probs (or log_probs)
                     # But we must exclude the label token from being considered "unsampled"
                     
                     # Use probs directly to avoid extra memory allocation
                     # Set label probability to -1 (or very small) so it's not selected as top-k unsampled
                     probs.scatter_(-1, labels.unsqueeze(-1), -1.0)
                     
                     # Get top-k indices
                     _, topk_indices = torch.topk(probs, top_k, dim=-1) # (N, k)
                     
                     # Gather values to scatter
                     values_to_scatter = grad_logits_clamped.gather(-1, topk_indices)
                     original_values = grad_logits.gather(-1, topk_indices)
                     
                     # Apply mask: if mask is False, keep original values
                     values_to_scatter = torch.where(mask, values_to_scatter, original_values)
                     
                     # Scatter back to grad_logits
                     grad_logits.scatter_(-1, topk_indices, values_to_scatter)
                 else:
                     grad_logits = torch.where(mask, grad_logits_clamped, grad_logits)
             else:
                 grad_logits = grad_logits_clamped
        
        # Add g to target (gradient of log_p_y w.r.t z_y is 1 - p_y)
        # We already have -p_y * g. Adding g gives g * (1 - p_y).
        grad_logits.scatter_add_(-1, labels.unsqueeze(-1), grad_output_expanded)
        
        return grad_logits, None, None, None, None


class DataParallelPPOActorGradTrunc(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.grad_trunc_clip_value = self.config.get("grad_trunc_clip_value", None)
        if self.grad_trunc_clip_value is not None:
            if torch.distributed.get_rank() == 0:
                print(f"Using gradient truncation with value {self.grad_trunc_clip_value}")
        
        self.grad_trunc_k = self.config.get("grad_trunc_k", None)
        if self.grad_trunc_k is not None:
            if torch.distributed.get_rank() == 0:
                print(f"Using gradient truncation top-k: {self.grad_trunc_k}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)

                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                use_fused_kernels = self.use_fused_kernels and (self.grad_trunc_clip_value is None)
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                if use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    if self.grad_trunc_clip_value is not None and "truncation_mask" in micro_batch:
                        truncation_mask = micro_batch["truncation_mask"]
                        truncation_mask_rmpad = None
                        if truncation_mask is not None:
                            # Process truncation_mask similar to input_ids
                            # truncation_mask: (bsz, seqlen)
                            truncation_mask_rmpad = index_first_axis(
                                rearrange(truncation_mask.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                            ).transpose(0, 1).squeeze(0).squeeze(-1) # (total_nnz,)

                            if self.use_ulysses_sp:
                                truncation_mask_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                                    truncation_mask_rmpad.unsqueeze(0),
                                    position_ids_rmpad=None,
                                    sp_size=self.ulysses_sequence_parallel_size
                                )
                                truncation_mask_rmpad = truncation_mask_rmpad.squeeze(0)

                        log_probs = TruncatedLogProb.apply(logits_rmpad, input_ids_rmpad_rolled, self.grad_trunc_clip_value, truncation_mask_rmpad, self.grad_trunc_k)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                use_fused_kernels = self.use_fused_kernels and (self.grad_trunc_clip_value is None)
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    if self.grad_trunc_clip_value is not None and "truncation_mask" in micro_batch:
                        # Flatten for TruncatedLogProb
                        B, L, V = logits.shape
                        logits_flat = logits.reshape(-1, V)
                        labels_flat = micro_batch["responses"].reshape(-1)
                        
                        truncation_mask = micro_batch["truncation_mask"]
                        truncation_mask_flat = truncation_mask.reshape(-1)

                        log_probs = TruncatedLogProb.apply(logits_flat, labels_flat, self.grad_trunc_clip_value, truncation_mask_flat, self.grad_trunc_k)
                        log_probs = log_probs.view(B, L)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    response_mask = data["response_mask"]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    # Determine truncation mask based on rewards
                    # Assuming rewards is (B, L) or (B,)
                    # We want to truncate only for positive samples (reward > 0.5)
                    if "token_level_scores" in data:
                        rewards = data["token_level_scores"]
                        # If rewards is (B, L), sum over L to get sequence score
                        if rewards.dim() == 2:
                            seq_score = rewards.sum(dim=-1)
                        else:
                            seq_score = rewards
                        
                        # Create mask: True if positive sample (should truncate)
                        is_positive = seq_score > 0 # (B,)

                        # Construct full-length truncation mask [B, SeqLen] to match input_ids for unpad
                        input_ids = data["input_ids"]
                        B, SeqLen = input_ids.shape
                        ResponseLen = data["responses"].shape[-1]
                        PromptLen = SeqLen - ResponseLen
                        
                        truncation_mask = torch.zeros((B, SeqLen), dtype=torch.bool, device=input_ids.device)
                        # Only apply truncation to the response part of positive samples
                        truncation_mask[:, PromptLen:] = is_positive.unsqueeze(-1).expand(B, ResponseLen)
                        
                        data["truncation_mask"] = truncation_mask

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    # #zhr
                    # if self.config.adjust_adv:
                    #     calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if self.config.policy_loss.loss_mode == "vanilla":
                        # zhr
                        # def adjust_adv(advantages, entropy, response_mask, is_negative, pos_entropy_weight, neg_entropy_weight):
                        #     # entropy advantage
                        #     eps=1e-8
                        #     entropy_mean = torch.mean(entropy, dim=1, keepdim=True)
                        #     entropy_std = torch.std(entropy, dim=1, keepdim=True) + eps 
                        #     entropy = (entropy - entropy_mean) / entropy_std
                        #     advantages = torch.where(
                        #                 advantages > 0,  # 条件：优势>0
                        #                 torch.max(advantages, advantages - pos_entropy_weight * entropy),  # 正优势的调整
                        #                 torch.min(advantages, advantages + neg_entropy_weight * entropy)   # 负优势的调整
                        #             )
                        #     advantages = torch.where(response_mask==0, torch.tensor(0.0, device=advantages.device), advantages)
                        #     valid_length = response_mask.sum(-1)
                        #     B,L = advantages.shape
                        #     adv_device = advantages.device
                        #     pos = torch.arange(L, device=adv_device).unsqueeze(0).repeat(B, 1)
                        #     half_lengths = (valid_length // 2).unsqueeze(1)
                        #     first_half_mask = pos < half_lengths
                        #     safe_half = torch.max(half_lengths, torch.tensor(1, device=adv_device))
                        #     linear_scale = torch.clamp(0.5 + (pos / torch.clamp(safe_half, min=1) ) * 0.5, max=1)
                        #     scale = torch.ones_like(advantages)
                        #     scale = torch.where(
                        #             first_half_mask & (response_mask==1),  # 仅对前1/2有效位置生效
                        #             linear_scale,
                        #             scale
                        #         )
                        #     scale = torch.where(response_mask==1, scale, torch.tensor(0.0, device=adv_device))
                        #     scale = torch.where(
                        #         is_negative.unsqueeze(1),  # 扩展维度到[B,1]以匹配[B,L]
                        #         scale,
                        #         torch.ones_like(scale)
                        #     )
                        #     adjusted_advantages = advantages * scale
                        #     return adjusted_advantages
                        # if self.config.adjust_adv:
                        #     advantages = adjust_adv(advantages=advantages, 
                        #                             entropy=entropy,
                        #                             response_mask=response_mask,
                        #                             is_negative= advantages[:, 0] < 0,
                        #                             pos_entropy_weight = self.config.pos_entropy_weight,
                        #                             neg_entropy_weight = self.config.neg_entropy_weight,
                        #                             )
                        # def add_pr(log_prob, old_log_prob, advantages):
                        #     with torch.no_grad():
                        #         return advantages + torch.exp(log_prob - old_log_prob) - 1
                        # if self.config.add_pr:
                        #     advantages = add_pr(log_prob, old_log_prob, advantages)
                        
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=advantages,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
