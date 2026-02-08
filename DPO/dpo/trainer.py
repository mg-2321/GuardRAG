from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    from trl import DPOTrainer
except Exception as exc:  # pragma: no cover
    raise ImportError("trl is required for DPO training") from exc


@dataclass(frozen=True)
class GuardRAGDPOConfig:
    """
    Implements the paper's weighted DPO objective:
      L = λ_s * E_{P_s}[L_DPO] + λ_u * E_{P_u}[L_DPO]

    plus optional entropy regularization as the "uncertainty/entropy" hook:
      L_total = L + entropy_coef * (-H)

    Here, entropy_coef > 0 encourages *higher* entropy by subtracting entropy
    from the minimized loss (i.e., maximize entropy).
    """

    lambda_security: float = 1.0
    lambda_utility: float = 0.8
    entropy_coef: float = 0.0


class WeightedEntropyDPOTrainer(DPOTrainer):
    """
    TRL DPOTrainer + per-example weighting + optional entropy regularization.

    Dataset must provide a `pair_type` column with values:
      - "security" (poisoned / adversarial contexts)
      - "utility"   (benign contexts)
    """

    def __init__(self, *args, guardrag_cfg: Optional[GuardRAGDPOConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.guardrag_cfg = guardrag_cfg or GuardRAGDPOConfig()

    def _pair_weights(self, batch: Dict[str, Any]) -> torch.Tensor:
        pair_type = batch.get("pair_type", None)
        if pair_type is None:
            # default: treat everything as security-weighted 1.0
            base = torch.ones(self.args.per_device_train_batch_size, device=self.accelerator.device)
        else:
            # `pair_type` may be a list[str] on CPU
            w_s = self.guardrag_cfg.lambda_security
            w_u = self.guardrag_cfg.lambda_utility
            weights = []
            for t in pair_type:
                if isinstance(t, bytes):
                    t = t.decode("utf-8")
                t = str(t)
                weights.append(w_s if t.lower().startswith("sec") else w_u)
            base = torch.tensor(weights, dtype=torch.float32, device=self.accelerator.device)

        # Optional per-example weight (novelty hook for "ambiguous wins"):
        # example_weight in [0,1] scales loss contribution.
        ex_w = batch.get("example_weight", None)
        if ex_w is None:
            return base
        if isinstance(ex_w, torch.Tensor):
            ex_w_t = ex_w.to(device=self.accelerator.device, dtype=torch.float32)
        else:
            ex_w_t = torch.tensor([float(x) for x in ex_w], device=self.accelerator.device, dtype=torch.float32)
        return base * ex_w_t

    @staticmethod
    def _token_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, T, V]
        returns: [B] mean entropy over T
        """
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(dim=-1)  # [B, T]
        return ent.mean(dim=-1)  # [B]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Let TRL compute the standard DPO loss and outputs
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # Apply GuardRAG weighting
        weights = self._pair_weights(inputs)
        # TRL's DPO loss is already reduced; when reduced, we can't reweight.
        # So we rely on TRL providing per-sample losses when available.
        per_sample = outputs.get("losses", None)
        if per_sample is not None:
            # [B]
            weighted = (per_sample * weights).mean()
        else:
            # Fallback: approximate by scaling the batch loss by mean weight.
            weighted = loss * weights.mean()

        # Optional entropy regularization (on chosen forward pass if available)
        if self.guardrag_cfg.entropy_coef != 0.0:
            chosen_logits = outputs.get("chosen_logits", None)
            if chosen_logits is not None:
                ent = self._token_entropy(chosen_logits)  # [B]
                # maximize entropy => subtract entropy from loss
                weighted = weighted - (self.guardrag_cfg.entropy_coef * ent.mean())

        if return_outputs:
            outputs["guardrag_weighted_loss"] = weighted.detach()
            return weighted, outputs
        return weighted

