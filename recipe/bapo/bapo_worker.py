"""FSDP worker definition for the BAPO recipe."""

from __future__ import annotations

import logging
import os

from omegaconf import OmegaConf, open_dict

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import ActorRolloutRefWorker


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class BAPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout worker that instantiates the BAPO actor implementation."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):  # type: ignore[override]
        from .dp_actor import DataParallelBAPOActor

        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                )
            )

            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelBAPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            raise NotImplementedError("Reference policy is not enabled in the BAPO recipe.")


