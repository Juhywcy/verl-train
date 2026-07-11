# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import functools

import torch

import verl.trainer.ppo.ray_trainer as base_ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.grpo_weighting_ig import compute_grpo_weights, compute_sample_entropy


ResourcePoolManager = base_ray_trainer.ResourcePoolManager
Role = base_ray_trainer.Role


def _apply_grpo_weighting(data, config):
    weighting_cfg = None
    if config is not None:
        weighting_cfg = config.get("grpo_weighting", None)
    if weighting_cfg is None or not weighting_cfg.enable:
        return data

    scores = data.batch["token_level_rewards"].sum(dim=-1)
    sample_entropy = data.batch.get("sample_entropy", None)
    weights = compute_grpo_weights(
        scores=scores,
        sample_entropy=sample_entropy,
        index=data.non_tensor_batch["uid"],
        beta=weighting_cfg.beta,
        entropy_clip=weighting_cfg.entropy_clip,
        eps=weighting_cfg.eps,
        correct_threshold=weighting_cfg.correct_threshold,
    )
    weights = weights.unsqueeze(-1) * data.batch["response_mask"]
    data.batch["advantages"] = data.batch["advantages"] * weights
    data.batch["returns"] = data.batch["returns"] * weights
    return data


if not hasattr(base_ray_trainer, "_verl_ig_original_compute_advantage"):
    base_ray_trainer._verl_ig_original_compute_advantage = base_ray_trainer.compute_advantage

_real_original_compute_advantage = base_ray_trainer._verl_ig_original_compute_advantage

def compute_advantage_wrapper(
    data,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    print("Advantages before GRPO weighting:")
    data = _real_original_compute_advantage(
        data=data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    if adv_estimator == AdvantageEstimator.GRPO:
        data = _apply_grpo_weighting(data, config)
    print("Advantages after GRPO weighting:")
    return data


base_ray_trainer.compute_advantage = compute_advantage_wrapper


class RayPPOTrainer(base_ray_trainer.RayPPOTrainer):
    def init_workers(self):
        super().init_workers()
        original_compute_log_prob = self.actor_rollout_wg.compute_log_prob

        @functools.wraps(original_compute_log_prob)
        def _compute_log_prob_with_entropy(batch):
            old_log_prob = original_compute_log_prob(batch)
            if "entropys" in old_log_prob.batch and "response_mask" in batch.batch:
                sample_entropy = compute_sample_entropy(
                    old_log_prob.batch["entropys"],
                    batch.batch["response_mask"],
                )
                old_log_prob.batch["sample_entropy"] = sample_entropy
            return old_log_prob

        self.actor_rollout_wg.compute_log_prob = _compute_log_prob_with_entropy
