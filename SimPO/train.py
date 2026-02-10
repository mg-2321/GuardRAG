"""
GuardRAG SimPO Trainer - Reference-Free Preference Optimization.


- No reference model needed (simpler, faster, less memory)
- Uses length-normalized log probabilities
- Adds a target reward margin (gamma) to encourage larger gaps


- SimPO: L = -log σ(β * (log π(y_w|x)/|y_w| - log π(y_l|x)/|y_l| - γ))

Where:
- β controls preference strength
- γ (gamma) is the target reward margin
- Length normalization prevents bias toward shorter responses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from trl import DPOTrainer
except ImportError:
    raise ImportError("trl is required for SimPO training: pip install trl")


@dataclass(frozen=True)
class GuardRAGSimPOConfig:
    """
    GuardRAG SimPO configuration.
    
    SimPO loss with GuardRAG weighting:
        L = λ_s * E_{P_s}[L_SimPO] + λ_u * E_{P_u}[L_SimPO]
    
    Args:
        lambda_security: Weight for security pairs (poisoned contexts)
        lambda_utility: Weight for utility pairs (clean contexts)
        beta: Temperature parameter (higher = more deterministic)
        gamma: Target reward margin (encourages larger preference gaps)
        gamma_beta_ratio: Alternative way to specify gamma = gamma_beta_ratio * beta
        loss_type: "sigmoid" (default) or "hinge"
        label_smoothing: Label smoothing factor (0 = none)
        sft_weight: Weight for optional SFT loss term (0 = disabled)
    """
    lambda_security: float = 1.0
    lambda_utility: float = 0.8
    beta: float = 2.0
    gamma: Optional[float] = None  # If None, use gamma_beta_ratio * beta
    gamma_beta_ratio: float = 0.5  # gamma = 0.5 * 2.0 = 1.0 by default
    loss_type: str = "sigmoid"  # "sigmoid" or "hinge"
    label_smoothing: float = 0.0
    sft_weight: float = 0.0  # Optional supervised fine-tuning loss


class WeightedSimPOTrainer(DPOTrainer):
    """
    SimPO Trainer with GuardRAG-specific weighting.
    
    Key features:
    - No reference model needed (reference-free)
    - Length-normalized log probabilities
    - Target reward margin (gamma)
    - Per-example weighting by pair_type (security vs utility)
    - Optional SFT loss term
    
    Dataset must provide:
    - prompt: The input prompt
    - chosen: The preferred response
    - rejected: The dispreferred response
    - pair_type: "security" or "utility" (for weighting)
    """
    
    def __init__(
        self,
        *args,
        guardrag_cfg: Optional[GuardRAGSimPOConfig] = None,
        **kwargs
    ):
        # SimPO doesn't need a reference model
        if "ref_model" in kwargs:
            kwargs.pop("ref_model")
        
        super().__init__(*args, ref_model=None, **kwargs)
        
        self.guardrag_cfg = guardrag_cfg or GuardRAGSimPOConfig()
        
        # Compute gamma
        if self.guardrag_cfg.gamma is not None:
            self.gamma = self.guardrag_cfg.gamma
        else:
            self.gamma = self.guardrag_cfg.gamma_beta_ratio * self.guardrag_cfg.beta
        
        self.simpo_beta = self.guardrag_cfg.beta
        self.loss_type = self.guardrag_cfg.loss_type
        self.label_smoothing = self.guardrag_cfg.label_smoothing
        self.sft_weight = self.guardrag_cfg.sft_weight
        
        print(f"SimPO Config: beta={self.simpo_beta}, gamma={self.gamma}, "
              f"loss_type={self.loss_type}, sft_weight={self.sft_weight}")
    
    def _pair_weights(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute per-example weights based on pair_type."""
        pair_type = batch.get("pair_type", None)
        
        # Get batch size
        batch_size = batch.get("chosen_input_ids", batch.get("input_ids_chosen"))
        if isinstance(batch_size, torch.Tensor):
            batch_size = batch_size.shape[0]
        else:
            batch_size = self.args.per_device_train_batch_size
        
        if pair_type is None:
            return torch.ones(batch_size, device=self.accelerator.device)
        
        w_s = self.guardrag_cfg.lambda_security
        w_u = self.guardrag_cfg.lambda_utility
        
        weights = []
        for t in pair_type:
            if isinstance(t, bytes):
                t = t.decode("utf-8")
            t = str(t).lower()
            weights.append(w_s if t.startswith("sec") else w_u)
        
        base = torch.tensor(weights, dtype=torch.float32, device=self.accelerator.device)
        
        # Apply example_weight if provided (from CRS/BSS scoring)
        ex_w = batch.get("example_weight", None)
        if ex_w is not None:
            if isinstance(ex_w, torch.Tensor):
                ex_w_t = ex_w.to(device=self.accelerator.device, dtype=torch.float32)
            else:
                ex_w_t = torch.tensor([float(x) for x in ex_w], 
                                       device=self.accelerator.device, dtype=torch.float32)
            base = base * ex_w_t
        
        return base
    
    def simpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the SimPO loss.
        
        SimPO uses length-normalized log probabilities:
            r(y|x) = β * log π(y|x) / |y|
        
        Loss:
            L = -log σ(β * (log π(y_w|x)/|y_w| - log π(y_l|x)/|y_l| - γ/β))
        
        Args:
            policy_chosen_logps: Log probs for chosen responses [batch_size]
            policy_rejected_logps: Log probs for rejected responses [batch_size]
            chosen_lengths: Number of tokens in chosen responses [batch_size]
            rejected_lengths: Number of tokens in rejected responses [batch_size]
        
        Returns:
            losses: Per-example SimPO losses [batch_size]
            chosen_rewards: Rewards for chosen responses [batch_size]
            rejected_rewards: Rewards for rejected responses [batch_size]
        """
        # Length-normalize log probabilities
        chosen_logps_norm = policy_chosen_logps / chosen_lengths.clamp(min=1)
        rejected_logps_norm = policy_rejected_logps / rejected_lengths.clamp(min=1)
        
        # Compute logit difference with margin
        pi_logratios = chosen_logps_norm - rejected_logps_norm
        gamma_term = self.gamma / self.simpo_beta  # γ/β in the formula
        logits = pi_logratios - gamma_term
        
        # Compute loss
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.simpo_beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.simpo_beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.simpo_beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute rewards for logging
        chosen_rewards = self.simpo_beta * chosen_logps_norm.detach()
        rejected_rewards = self.simpo_beta * rejected_logps_norm.detach()
        
        return losses, chosen_rewards, rejected_rewards
    
    def _get_batch_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities and response lengths for a batch.
        
        Returns:
            logps: Sum of log probs for response tokens [batch_size]
            lengths: Number of response tokens [batch_size]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction
        
        shifted_labels = labels[:, 1:]
        
        # Mask for response tokens (labels != -100)
        response_mask = (shifted_labels != -100).float()
        
        # Clamp labels for gather
        gather_labels = shifted_labels.clamp(min=0)
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply mask
        per_token_logps = per_token_logps * response_mask
        
        # Sum and count
        logps = per_token_logps.sum(dim=-1)
        lengths = response_mask.sum(dim=-1)
        
        return logps, lengths
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted SimPO loss for GuardRAG.
        """
        # Get input tensors (handle different naming conventions)
        chosen_ids = inputs.get("chosen_input_ids", inputs.get("input_ids_chosen"))
        chosen_mask = inputs.get("chosen_attention_mask", inputs.get("attention_mask_chosen"))
        chosen_labels = inputs.get("chosen_labels", inputs.get("labels_chosen"))
        
        rejected_ids = inputs.get("rejected_input_ids", inputs.get("input_ids_rejected"))
        rejected_mask = inputs.get("rejected_attention_mask", inputs.get("attention_mask_rejected"))
        rejected_labels = inputs.get("rejected_labels", inputs.get("labels_rejected"))
        
        # Build labels if not provided
        if chosen_labels is None:
            prompt_mask = inputs.get("prompt_attention_mask")
            if prompt_mask is not None:
                prompt_lens = prompt_mask.sum(dim=-1)
                chosen_labels = self._build_labels(chosen_ids, prompt_lens)
                rejected_labels = self._build_labels(rejected_ids, prompt_lens)
            else:
                # Fallback: use input_ids as labels (will compute loss on all tokens)
                chosen_labels = chosen_ids.clone()
                rejected_labels = rejected_ids.clone()
        
        # Compute log probabilities and lengths
        chosen_logps, chosen_lengths = self._get_batch_logps(
            model, chosen_ids, chosen_mask, chosen_labels
        )
        rejected_logps, rejected_lengths = self._get_batch_logps(
            model, rejected_ids, rejected_mask, rejected_labels
        )
        
        # Compute SimPO loss
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            chosen_logps, rejected_logps, chosen_lengths, rejected_lengths
        )
        
        # Apply GuardRAG weighting
        weights = self._pair_weights(inputs)
        weighted_losses = losses * weights
        
        # Optional SFT loss on chosen responses
        if self.sft_weight > 0:
            sft_loss = -chosen_logps / chosen_lengths.clamp(min=1)
            weighted_losses = weighted_losses + self.sft_weight * sft_loss
        
        loss = weighted_losses.mean()
        
        # Log metrics
        self.log({
            "simpo/loss": loss.item(),
            "simpo/chosen_rewards": chosen_rewards.mean().item(),
            "simpo/rejected_rewards": rejected_rewards.mean().item(),
            "simpo/reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "simpo/chosen_lengths": chosen_lengths.float().mean().item(),
            "simpo/rejected_lengths": rejected_lengths.float().mean().item(),
        })
        
        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards,
            }
        
        return loss
    
    @staticmethod
    def _build_labels(input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        """Build labels with -100 for prompt tokens."""
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, :int(plen)] = -100
        return labels
