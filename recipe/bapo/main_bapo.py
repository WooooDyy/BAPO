"""Entry point for running the BAPO recipe."""

from __future__ import annotations

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import validate_config
from .bapo_ray_trainer import RayBAPOTrainer
from .bapo_worker import BAPOActorRolloutRefWorker


@hydra.main(config_path="config", config_name="bapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ.setdefault("ENSURE_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import fs as verl_fs
        from verl.utils import hf_processor, hf_tokenizer

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("BAPO recipe currently supports FSDP-based strategies only.")

        role_worker_mapping: dict[Role, Role.value] = {
            Role.ActorRollout: ray.remote(BAPOActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}

        use_ref = Role.RefPolicy in role_worker_mapping
        validate_config(config=config, use_reference_policy=use_ref, use_critic=False)

        local_path = verl_fs.copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, use_fast=True)

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayBAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()


