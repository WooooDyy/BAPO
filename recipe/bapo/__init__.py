
"""BAPO recipe package.

Importing :mod:`recipe.bapo.policy_loss` registers the custom policy loss with
the core PPO utilities so that it can be referenced via
``policy_loss.loss_mode = "bapo"`` inside configuration files.
"""

from . import policy_loss as _policy_loss  # noqa: F401  (ensures registration)

__all__ = ["_policy_loss"]


