from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseBatch:
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


def pairwise_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Eq (4): -E[log σ(r(c,y_w) - r(c,y_l))]
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def pairwise_cost_loss(cost_bad: torch.Tensor, cost_good: torch.Tensor) -> torch.Tensor:
    """
    Eq (5): -E[log σ(c(c,y_l) - c(c,y_w))]  (bad should score higher cost)
    """
    return -F.logsigmoid(cost_bad - cost_good).mean()


class ScalarHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled).squeeze(-1)

