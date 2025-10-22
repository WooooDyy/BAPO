"""Ray trainer wrapper for the BAPO recipe.

It reuses the standard :class:`RayPPOTrainer` implementation but plugs in the
BAPO actor worker so that the custom metrics are surfaced during training.
"""

from __future__ import annotations

from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role, WorkerType


class RayBAPOTrainer(RayPPOTrainer):
    """Thin wrapper around :class:`RayPPOTrainer` for BAPO-specific defaults."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )


