"""Minimal contract for an off-policy reinforcement learning agent."""

from typing import Protocol

import numpy as np


class OffPolicyAgent(Protocol):
    """Operations used by a trainer without exposing agent internals."""

    def select_action(
            self,
            observation: np.ndarray,
            *,
            explore: bool,
    ) -> np.ndarray:
        """Choose an action within the environment's actual action bounds.

        :param observation: current environment observation vector.
        :param explore: whether to apply algorithm-specific exploration noise.
        :return: action vector in the environment's original action range.
        """
        ...

    def store_transition(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            terminated: bool,
            truncated: bool,
    ) -> None:
        """Pass a single experience transition to the agent's storage.

        :param obs: observation before the environment step.
        :param action: action performed in the environment.
        :param reward: reward received from the environment.
        :param next_obs: observation returned by the environment step.
        :param terminated: whether the environment reached a terminal state.
        :param truncated: whether the episode was truncated.
        :return: None.
        """
        ...

    def update(self) -> dict[str, float] | None:
        """Perform an agent update if sufficient experience is available.

        :return: update metrics, or None when no update was performed.
        """
        ...
