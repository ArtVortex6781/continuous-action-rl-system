"""Sequential off-policy training with complete in-memory metric histories."""

import time
from statistics import mean
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

    @staticmethod
    def _group_update_metrics(
            records: list[UpdateMetrics],
    ) -> dict[str, list[float]]:
        """Group observed metric values by key, preserving their update order."""
        values_by_key: dict[str, list[float]] = {}
        for record in records:
            for name, value in record["metrics"].items():
                values_by_key.setdefault(name, []).append(value)
        return values_by_key

    def _report_progress(
            self,
            *,
            global_step: int,
            episode_metrics: list[EpisodeMetrics],
            update_metrics: list[UpdateMetrics],
            last_report_step: int,
            last_report_episode_count: int,
            last_report_update_count: int,
            start_time: float,
            last_report_time: float,
    ) -> float:
        """Print one progress report and return its measurement timestamp.

        Only newly recorded updates enter the interval statistics. The timestamp
        is sampled before printing so the next interval includes this report's
        output cost.
        """
        now = time.perf_counter()
        elapsed = now - start_time
        progress = self._config.progress
        total_steps = self._config.total_steps
        episode_count = len(episode_metrics)
        update_count = len(update_metrics)

        lines = [
            f"Steps: {global_step} / {total_steps} ({global_step / total_steps:.2%})",
            f"Episodes: {episode_count}",
            f"Elapsed: {elapsed:.2f} s",
        ]
        if progress.show_update_count:
            lines.append(f"Updates: {update_count}")

        if episode_count > last_report_episode_count:
            last = episode_metrics[-1]
            lines.extend([
                "",
                f"Last episode: {last['episode']}",
                f"  Reward: {last['episode_reward']:.6g} | Length: {last['episode_length']}",
                f"  Terminated: {last['terminated']} | Truncated: {last['truncated']}",
            ])

        if progress.reward_window > 0 and episode_metrics:
            rewards = [
                record["episode_reward"]
                for record in episode_metrics[-progress.reward_window:]
            ]
            lines.extend([
                "",
                f"Rewards (last {len(rewards)} completed episodes):",
                f"  min={min(rewards):.6g} mean={mean(rewards):.6g} max={max(rewards):.6g}",
            ])

        if progress.show_update_metrics and update_count > last_report_update_count:
            values_by_key = self._group_update_metrics(
                update_metrics[last_report_update_count:]
            )
            if values_by_key:
                lines.extend([
                    "",
                    f"Update metrics (steps {last_report_step + 1}–{global_step}):",
                ])
                for name, values in values_by_key.items():
                    summary = (
                        f"  {name}: min={min(values):.6g} mean={mean(values):.6g} "
                        f"max={max(values):.6g} last={values[-1]:.6g}"
                    )
                    if progress.show_metric_counts:
                        summary += f" count={len(values)}"
                    lines.append(summary)

        if progress.show_throughput:
            lines.extend(["", "Throughput:"])
            for label, steps, updates, duration in (
                    ("overall", global_step, update_count, elapsed),
                    (
                            "interval",
                            global_step - last_report_step,
                            update_count - last_report_update_count,
                            now - last_report_time,
                    ),
            ):
                if duration > 0:
                    lines.append(
                        f"  {label}: {steps / duration:.2f} steps/s | "
                        f"{updates / duration:.2f} updates/s"
                    )
                else:
                    lines.append(f"  {label}: N/A steps/s | N/A updates/s")

        print("\n".join(lines), flush = True)
        return now

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
        progress = config.progress
        reporting_enabled = progress.step_interval > 0 or progress.episode_interval > 0
        last_report_step = 0
        last_report_episode_count = 0
        last_report_update_count = 0
        last_report_time = start_time

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

                if reporting_enabled:
                    episode_count = len(episode_metrics)
                    step_due = (
                            progress.step_interval > 0
                            and global_step - last_report_step >= progress.step_interval
                    )
                    episode_due = (
                            episode_finished
                            and progress.episode_interval > 0
                            and episode_count >= progress.episode_start
                            and (episode_count - progress.episode_start) % progress.episode_interval == 0
                    )
                    if step_due or episode_due or step == config.total_steps:
                        last_report_time = self._report_progress(
                            global_step = global_step,
                            episode_metrics = episode_metrics,
                            update_metrics = update_metrics,
                            last_report_step = last_report_step,
                            last_report_episode_count = last_report_episode_count,
                            last_report_update_count = last_report_update_count,
                            start_time = start_time,
                            last_report_time = last_report_time,
                        )
                        last_report_step = global_step
                        last_report_episode_count = episode_count
                        last_report_update_count = len(update_metrics)

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
