"""Short integration test of the real Trainer, TD3Agent and Gymnasium environment."""

import json
import math

import keras
import pytest
import yaml

from rl_system.agents.td3.agent import TD3Agent
from rl_system.agents.td3.config import TD3Config
from rl_system.config import load_config
from rl_system.environment import make_environment
from rl_system.training import Trainer


def test_training_pendulum(tmp_path):
    config_path = tmp_path / "td3_pendulum.yaml"
    config_path.write_text(yaml.safe_dump({
        "environment": {
            "id": "Pendulum-v1",
            "kwargs": {"max_episode_steps": 10},
        },
        "algorithm": {
            "name": "td3",
            "params": {
                "replay_capacity": 32,
                "batch_size": 4,
                "gamma": 0.99,
                "tau": 0.005,
                "actor_learning_rate": 0.001,
                "critic_learning_rate": 0.001,
                "policy_delay": 2,
                "exploration_noise_std": 0.1,
                "target_policy_noise_std": 0.2,
                "target_noise_clip": 0.5,
                "actor_output_init_limit": 0.003,
                "actor": {
                    "hidden_units": [8, 8],
                    "hidden_activations": ["relu", "relu"],
                },
                "critic": {
                    "hidden_units": [8, 8],
                    "hidden_activations": ["relu", "relu"],
                },
            },
        },
        "training": {
            "total_steps": 24,
            "seed": 42,
            "warmup_steps": 4,
            "learning_starts": 4,
            "train_frequency": 2,
            "gradient_steps": 2,
            "progress": {
                "step_interval": 0,
                "episode_interval": 0,
                "episode_start": 1,
                "reward_window": 0,
                "show_update_count": False,
                "show_update_metrics": False,
                "show_metric_counts": False,
                "show_throughput": False,
            },
        },
    }), encoding = "utf-8")

    config = load_config(config_path)
    td3_config = TD3Config.model_validate(config.algorithm.params)
    keras.utils.set_random_seed(config.training.seed)

    env, spec = make_environment(config.environment)
    try:
        agent = TD3Agent(spec, td3_config, seed = config.training.seed)
        trainer = Trainer(env, agent, config.training)
        result = trainer.train()

        assert result["environment_steps"] == 24
        assert result["update_count"] == 22
        assert result["episode_count"] == 2

        episodes = result["episode_metrics"]
        assert len(episodes) == 2
        for episode, expected_number in zip(episodes, (1, 2), strict = True):
            assert episode["episode"] == expected_number
            assert episode["global_step"] == expected_number * 10
            assert episode["episode_length"] == 10
            assert math.isfinite(episode["episode_reward"])
            assert episode["terminated"] is False
            assert episode["truncated"] is True

        partial = result["partial_episode"]
        assert partial is not None
        assert partial["episode"] == 3
        assert partial["global_step"] == 24
        assert partial["episode_length"] == 4
        assert math.isfinite(partial["episode_reward"])
        assert partial["terminated"] is False
        assert partial["truncated"] is False
        assert sum(ep["episode_length"] for ep in episodes) + partial["episode_length"] == 24

        updates = result["update_metrics"]
        expected_steps = [step for step in range(4, 25, 2) for _ in range(2)]
        assert len(updates) == 22
        assert [record["global_step"] for record in updates] == expected_steps
        assert [record["update"] for record in updates] == list(range(1, 23))

        for record in updates:
            expected_keys = {"critic_1_loss", "critic_2_loss"}
            if record["update"] % 2 == 0:
                expected_keys.add("actor_loss")
            metrics = record["metrics"]
            assert set(metrics) == expected_keys
            assert all(type(value) is float and math.isfinite(value) for value in metrics.values())

        assert sum("actor_loss" in record["metrics"] for record in updates) == 11

        duration = result["duration_seconds"]
        assert math.isfinite(duration) and duration > 0
        assert result["steps_per_second"] == pytest.approx(24 / duration)
        assert result["updates_per_second"] == pytest.approx(22 / duration)
        assert json.loads(json.dumps(result, allow_nan = False)) == result
    finally:
        env.close()
