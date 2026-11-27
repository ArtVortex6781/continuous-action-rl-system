"""Sequential off-policy training with complete in-memory metric histories."""

import time
from typing import TypedDict

import gymnasium as gym
import numpy as np

from rl_system.agents.base import OffPolicyAgent
from rl_system.config import TrainingConfig
from rl_system.exceptions import TrainingError, TrainingInterruptedError


class EpisodeMetrics(TypedDict):
    """Metrics for one completed or partial episode.

    :param episode: one-based episode number.
    :param global_step: last successfully processed environment step of the episode.
    :param episode_reward: undiscounted sum of observed rewards.
    :param episode_length: number of successfully processed transitions in the episode.
    :param terminated: whether the environment terminated the episode.
    :param truncated: whether the environment truncated the episode.
    """

    episode: int
    global_step: int
    episode_reward: float
    episode_length: int
    terminated: bool
    truncated: bool


class UpdateMetrics(TypedDict):
    """Metrics returned by one successful agent update.

    :param global_step: environment step at which the update was performed.
    :param update: one-based successful update number.
    :param metrics: complete algorithm-specific metrics returned by the agent.
    """

    global_step: int
    update: int
    metrics: dict[str, float]


class TrainingResult(TypedDict):
    """Complete or partial training result with original metric histories.

    :param environment_steps: number of successfully processed transitions.
    :param update_count: number of successful agent updates.
    :param episode_count: number of completed episodes.
    :param duration_seconds: elapsed training time in seconds.
    :param steps_per_second: overall environment throughput, or None for zero duration.
    :param updates_per_second: overall update throughput, or None for zero duration.
    :param episode_metrics: history of completed episodes.
    :param update_metrics: history of successful agent updates.
    :param partial_episode: unfinished episode metrics, or None if no steps are pending.
    """

    environment_steps: int
    update_count: int
    episode_count: int
    duration_seconds: float
    steps_per_second: float | None
    updates_per_second: float | None
    episode_metrics: list[EpisodeMetrics]
    update_metrics: list[UpdateMetrics]
    partial_episode: EpisodeMetrics | None


class Trainer:
    """Coordinate a single training run using an existing environment and agent.

    The caller owns the environment and is responsible for closing it. The
    agent and environment must already be compatible with each other.

    :param env: prepared Gymnasium environment.
    :param agent: prepared off-policy agent.
    :param config: validated training configuration.
    """

    def __init__(self, env: gym.Env, agent: OffPolicyAgent, config: TrainingConfig) -> None:
        self._env = env
        self._agent = agent
        self._config = config
        self._has_started = False

    def _initialize_environment(self) -> np.ndarray:
        """Seed the action space independently and obtain the initial observation."""
        action_seed = int(np.random.SeedSequence([self._config.seed, 1]).generate_state(1)[0])
        self._env.action_space.seed(action_seed)
        observation, _ = self._env.reset(seed = self._config.seed)
        return observation

    @staticmethod
    def _build_result(
            *,
            start_time: float,
            global_step: int,
            episode_reward: float,
            episode_length: int,
            episode_metrics: list[EpisodeMetrics],
            update_metrics: list[UpdateMetrics],
    ) -> TrainingResult:
        """Build a result from the currently recorded training state without copying histories."""
        duration = time.perf_counter() - start_time
        update_count = len(update_metrics)
        episode_count = len(episode_metrics)

        partial_episode: EpisodeMetrics | None = None
        if episode_length > 0:
            partial_episode = {
                "episode": episode_count + 1,
                "global_step": global_step,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "terminated": False,
                "truncated": False,
            }

        return {
            "environment_steps": global_step,
            "update_count": update_count,
            "episode_count": episode_count,
            "duration_seconds": duration,
            "steps_per_second": global_step / duration if duration > 0 else None,
            "updates_per_second": update_count / duration if duration > 0 else None,
            "episode_metrics": episode_metrics,
            "update_metrics": update_metrics,
            "partial_episode": partial_episode,
        }

    def train(self) -> TrainingResult:
        """Run training once within the configured environment step budget.

        :return: complete training result, including any unfinished episode.
        :raises RuntimeError: if this trainer has already been started.
        :raises TrainingInterruptedError: if training is interrupted by the user.
        :raises TrainingError: if training fails, carrying the recorded partial result.
        """
        if self._has_started:
            raise RuntimeError("Trainer can only be started once")
        self._has_started = True

        global_step = 0
        episode_reward = 0.0
        episode_length = 0
        episode_metrics: list[EpisodeMetrics] = []
        update_metrics: list[UpdateMetrics] = []
        config = self._config
        start_time = time.perf_counter()

        try:
            observation = self._initialize_environment()

            for step in range(1, config.total_steps + 1):
                if step <= config.warmup_steps:
                    action = self._env.action_space.sample()
                else:
                    action = self._agent.select_action(observation, explore = True)

                next_obs, reward, terminated, truncated, _ = self._env.step(action)
                self._agent.store_transition(
                    observation, action, reward, next_obs, terminated, truncated
                )

                # Count a transition only after the agent has stored it successfully.
                global_step = step
                episode_reward += float(reward)
                episode_length += 1
                episode_finished = terminated or truncated

                if episode_finished:
                    episode_metrics.append({
                        "episode": len(episode_metrics) + 1,
                        "global_step": global_step,
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    })
                    episode_reward = 0.0
                    episode_length = 0

                if (
                        step >= config.learning_starts
                        and (step - config.learning_starts) % config.train_frequency == 0
                ):
                    for _ in range(config.gradient_steps):
                        metrics = self._agent.update()
                        if metrics is None:
                            break
                        update_metrics.append({
                            "global_step": global_step,
                            "update": len(update_metrics) + 1,
                            "metrics": metrics,
                        })

                if episode_finished and step < config.total_steps:
                    observation, _ = self._env.reset()
                else:
                    observation = next_obs

        except KeyboardInterrupt as exc:
            result = self._build_result(
                start_time = start_time,
                global_step = global_step,
                episode_reward = episode_reward,
                episode_length = episode_length,
                episode_metrics = episode_metrics,
                update_metrics = update_metrics,
            )
            raise TrainingInterruptedError("Training interrupted by the user", result) from exc

        except Exception as exc:
            result = self._build_result(
                start_time = start_time,
                global_step = global_step,
                episode_reward = episode_reward,
                episode_length = episode_length,
                episode_metrics = episode_metrics,
                update_metrics = update_metrics,
            )
            raise TrainingError(f"Training failed: {exc}", result) from exc

        return self._build_result(
            start_time = start_time,
            global_step = global_step,
            episode_reward = episode_reward,
            episode_length = episode_length,
            episode_metrics = episode_metrics,
            update_metrics = update_metrics,
        )
