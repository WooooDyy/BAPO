"""Configuration helpers for the BAPO recipe.

These dataclasses extend the core verl configuration objects with the
hyper-parameters that are specific to the BAPO clipping strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verl.workers.config import FSDPActorConfig, PolicyLossConfig


@dataclass
class BAPOPolicyLossConfig(PolicyLossConfig):
    """Policy-loss configuration for the BAPO clipping strategy (explicit bounds).

    All bounds are the true ratio bounds, not 1±epsilon.

    Attributes
    ----------
    adv_ratio_target:
        Minimum ratio between positive and negative advantage contributions
        (positive / negative) that we try to satisfy.
    ratio_lower_start:
        Initial lower bound for the importance ratio clamp. Example: 0.6
    ratio_lower_max:
        Maximum lower bound value allowed during search. Example: 0.8
    ratio_lower_step:
        Increment for the lower bound during search.
    ratio_upper_start:
        Initial upper bound for the importance ratio clamp. Example: 1.2
    ratio_upper_max:
        Maximum upper bound value allowed during search. Example: 2.0
    ratio_upper_step:
        Increment for the upper bound during search.
    eps:
        Numerical stability constant used when computing ratios.
    """

    loss_mode: str = "bapo"
    adv_ratio_target: float = 1.0
    ratio_lower_start: float | None = None
    ratio_lower_max: float | None = None
    ratio_lower_step: float = 0.05
    ratio_upper_start: float | None = None
    ratio_upper_max: float | None = None
    ratio_upper_step: float = 0.05
    eps: float = 1e-8


@dataclass
class BAPOActorConfig(FSDPActorConfig):
    """Actor configuration for the BAPO recipe.

    The only behavioural change compared to the base ``FSDPActorConfig`` is the
    substitution of the default ``PolicyLossConfig`` with
    :class:`BAPOPolicyLossConfig` so that Hydra can populate the additional
    hyper-parameters.
    """

    policy_loss: BAPOPolicyLossConfig = field(default_factory=BAPOPolicyLossConfig)


