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

from collections import defaultdict
from typing import Optional

import torch


def compute_sample_entropy(entropys: torch.Tensor, response_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Aggregate token entropy into a per-sample scalar."""
    entropys = entropys.float()
    response_mask = response_mask.float()
    denom = response_mask.sum(dim=-1).clamp_min(1.0)
    return (entropys * response_mask).sum(dim=-1) / (denom + eps)


def compute_grpo_weights(
    scores: torch.Tensor,
    sample_entropy: Optional[torch.Tensor],
    index,
    beta: float,
    entropy_clip: float,
    eps: float,
    correct_threshold: float,
) -> torch.Tensor:
    """Compute per-sample weights for GRPO advantage reallocation.

    scores: (bs,) raw outcome rewards
    sample_entropy: (bs,) entropy summary per sample, or None
    index: (bs,) group id per sample
    """
    device = scores.device
    bsz = scores.shape[0]
    correct = (scores > correct_threshold).to(torch.float32)

    id2sum = defaultdict(float)
    id2count = defaultdict(int)
    for i in range(bsz):
        idx = index[i]
        id2sum[idx] += float(correct[i].item())
        id2count[idx] += 1

    group_acc = torch.empty(bsz, device=device)
    for i in range(bsz):
        idx = index[i]
        acc = id2sum[idx] / max(id2count[idx], 1)
        group_acc[i] = acc

    group_factor = torch.where(correct > 0.5, 1.0 - group_acc, group_acc)

    if sample_entropy is None:
        sample_factor = torch.ones_like(group_factor)
    else:
        sample_entropy = sample_entropy.to(device)
        id2mean = defaultdict(float)
        id2var = defaultdict(float)
        for i in range(bsz):
            idx = index[i]
            id2mean[idx] += float(sample_entropy[i].item())
        for idx in id2mean:
            id2mean[idx] /= max(id2count[idx], 1)
        for i in range(bsz):
            idx = index[i]
            diff = float(sample_entropy[i].item()) - id2mean[idx]
            id2var[idx] += diff * diff
        id2std = {idx: (id2var[idx] / max(id2count[idx], 1)) ** 0.5 for idx in id2var}

        sample_factor = torch.empty_like(group_factor)
        for i in range(bsz):
            idx = index[i]
            std = id2std[idx]
            if std < eps:
                z = 0.0
            else:
                z = (float(sample_entropy[i].item()) - id2mean[idx]) / (std + eps)
            z = max(min(z, entropy_clip), -entropy_clip)
            sample_factor[i] = torch.sigmoid(torch.tensor(beta * z, device=device))

    weights = group_factor * sample_factor
    mean_w = weights.mean().clamp_min(eps)
    return weights / mean_w
