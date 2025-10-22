"""Data parallel actor extension that records BAPO-specific metrics."""

from __future__ import annotations

import logging
import os
from typing import Any

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelBAPOActor(DataParallelPPOActor):
    """Wrap :class:`DataParallelPPOActor` to surface extra BAPO metrics."""

    def update_policy(self, data: DataProto) -> dict[str, Any]:  # type: ignore[override]
        metrics = super().update_policy(data=data)

        policy_loss_cfg = self.config.policy_loss
        buffer = getattr(policy_loss_cfg, "_bapo_metrics_buffer", [])
        if buffer:
            latest = buffer[-1]
            metrics.update(
                {
                    "actor/bapo_clip_low": latest.get("clip_low", 0.0),
                    "actor/bapo_clip_high": latest.get("clip_high", 0.0),
                    "actor/bapo_pos_neg_ratio": latest.get("pos_neg_ratio", 0.0),
                }
            )

        append_to_dict(metrics, {})
        return metrics


