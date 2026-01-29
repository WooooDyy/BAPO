"""Custom PPO policy loss for the BAPO algorithm.

The BAPO strategy dynamically adjusts the PPO clipping bounds to satisfy a
target ratio between the aggregated positive- and negative-advantage
contributions within a batch. The policy loss implemented here follows the
standard PPO surrogate but iteratively relaxes the clipping window until the
desired ratio is reached (or the configured maxima are exceeded).
"""

from __future__ import annotations

from typing import Dict

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


def _compute_effective_advantage(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked effective advantage and the positive/negative sums."""

    # NOTE:
    # In the BAPO recipe, clip_low/clip_high are **explicit importance-ratio bounds**
    # (e.g. 0.6 and 1.2), NOT epsilons for (1±eps).
    clip_min = ratio.new_tensor(float(clip_low))
    clip_max = ratio.new_tensor(float(clip_high))

    ratio_pos = torch.minimum(ratio, clip_max)
    ratio_neg = torch.maximum(ratio, clip_min)
    effective_ratio = torch.where(advantages >= 0, ratio_pos, ratio_neg)

    effective_adv = effective_ratio * advantages * response_mask

    pos_contrib = effective_adv.clamp(min=0).sum()
    neg_contrib = (-effective_adv.clamp(max=0)).sum()

    # Make the bound-search decision **globally consistent** across DP ranks.
    # Otherwise each rank may choose different bounds (because each sees different data),
    # and the final all-reduced gradient corresponds to a mixture of objectives.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.stack(
            [pos_contrib.to(torch.float32), neg_contrib.to(torch.float32)],
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        pos_contrib = stats[0].to(pos_contrib.dtype)
        neg_contrib = stats[1].to(neg_contrib.dtype)

    if torch.all(neg_contrib <= 0):
        pos_neg_ratio = ratio.new_tensor(float("inf"))
    else:
        pos_neg_ratio = pos_contrib / (neg_contrib + ratio.new_tensor(eps))

    return effective_adv, pos_neg_ratio


def _append_metric(buffer: list[Dict[str, float]], clip_low: float, clip_high: float, ratio_value: float):
    buffer.append(
        {
            "clip_low": float(clip_low),
            "clip_high": float(clip_high),
            "pos_neg_ratio": float(ratio_value),
        }
    )


@register_policy_loss("bapo")
def compute_policy_loss_bapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: AlgoConfig | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO policy loss with dynamically adjusted clipping bounds."""

    assert config is not None, "Actor config must be provided for BAPO loss."
    assert hasattr(config, "policy_loss"), "Config is expected to carry a policy_loss section."

    pl_config = config.policy_loss

    # Start from actor defaults if recipe overrides are not provided
    # Convert legacy epsilon-style actor defaults (0.2) to explicit bounds (1-0.2, 1+0.2)
    clip_ratio = getattr(config, "clip_ratio", 0.2)
    default_low = 1.0 - (config.clip_ratio_low if getattr(config, "clip_ratio_low", None) is not None else clip_ratio)
    default_high = 1.0 + (
        config.clip_ratio_high if getattr(config, "clip_ratio_high", None) is not None else clip_ratio
    )

    ratio_low_cur = pl_config.ratio_lower_start if pl_config.ratio_lower_start is not None else default_low
    ratio_high_cur = pl_config.ratio_upper_start if pl_config.ratio_upper_start is not None else default_high

    ratio_low_max = pl_config.ratio_lower_max if pl_config.ratio_lower_max is not None else ratio_low_cur
    ratio_high_max = pl_config.ratio_upper_max if pl_config.ratio_upper_max is not None else ratio_high_cur

    ratio_low_step = max(pl_config.ratio_lower_step, 0.0)
    ratio_high_step = max(pl_config.ratio_upper_step, 0.0)

    target_ratio = max(pl_config.adv_ratio_target, 0.0)
    eps = pl_config.eps

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    buffer: list[Dict[str, float]] = []

    _, pos_neg_ratio = _compute_effective_advantage(
        ratio=ratio,
        advantages=advantages,
        response_mask=response_mask,
        clip_low=ratio_low_cur,
        clip_high=ratio_high_cur,
        eps=eps,
    )

    # adjust lower clip range first
    while pos_neg_ratio.item() < target_ratio and ratio_low_cur < ratio_low_max and ratio_low_step > 0:
        ratio_low_cur = min(ratio_low_cur + ratio_low_step, ratio_low_max)
        _, pos_neg_ratio = _compute_effective_advantage(
            ratio=ratio,
            advantages=advantages,
            response_mask=response_mask,
            clip_low=ratio_low_cur,
            clip_high=ratio_high_cur,
            eps=eps,
        )
        if pos_neg_ratio.item() >= target_ratio:
            break

    # if still unmet, increase upper bound
    while pos_neg_ratio.item() < target_ratio and ratio_high_cur < ratio_high_max and ratio_high_step > 0:
        ratio_high_cur = min(ratio_high_cur + ratio_high_step, ratio_high_max)
        _, pos_neg_ratio = _compute_effective_advantage(
            ratio=ratio,
            advantages=advantages,
            response_mask=response_mask,
            clip_low=ratio_low_cur,
            clip_high=ratio_high_cur,
            eps=eps,
        )
        if pos_neg_ratio.item() >= target_ratio:
            break

    # record metrics for the actor to log later
    _append_metric(buffer, ratio_low_cur, ratio_high_cur, pos_neg_ratio.item())

    # attach buffer to policy config for the actor to consume after the update
    if hasattr(pl_config, "_bapo_metrics_buffer"):
        pl_config._bapo_metrics_buffer.extend(buffer)
    else:
        pl_config._bapo_metrics_buffer = buffer

    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, ratio_low_cur, ratio_high_cur)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


