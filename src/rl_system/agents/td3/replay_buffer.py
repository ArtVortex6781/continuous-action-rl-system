"""Preallocated NumPy replay storage for an off-policy TD3 agent."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen = True, slots = True, eq = False)
class ReplayBatch:
    """Container for a sampled minibatch of aligned transitions.

    :param obs: observation vectors of the sampled transitions.
    :param actions: action vectors of the sampled transitions.
    :param rewards: rewards as a column vector.
    :param next_obs: successor observation vectors.
    :param terminated: environment termination flags as a column vector.
    :param truncated: episode truncation flags as a column vector.
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


def _as_float32_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    """Validate a real numeric vector and prepare its independent storage copy.

    :param value: input vector, as an array or array-like sequence.
    :param dimension: required vector length.
    :param name: field name used in validation errors.
    :return: independent finite ``float32`` vector of the required shape.
    :raises TypeError: if the elements do not have a real numeric dtype.
    :raises ValueError: if the vector shape is wrong or values are nonfinite.
    """
    array = np.asarray(value)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},), got {array.shape}")
    if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must contain real numeric values, got {array.dtype}")

    with np.errstate(over = "ignore", invalid = "ignore"):
        converted = array.astype(np.float32, copy = True)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite float32 values")
    return converted


def _as_float32_reward(value: Any) -> np.float32:
    """Validate a real numeric scalar reward before storage.

    :param value: scalar reward supplied with a transition.
    :return: finite ``float32`` reward.
    :raises TypeError: if the reward has a nonnumeric or complex dtype.
    :raises ValueError: if it is not scalar or becomes nonfinite in float32.
    """
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"reward must be a scalar, got shape {array.shape}")
    if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"reward must be a real numeric scalar, got {array.dtype}")

    with np.errstate(over = "ignore", invalid = "ignore"):
        converted = array.astype(np.float32)
    if not bool(np.isfinite(converted)):
        raise ValueError("reward must be finite after conversion to float32")
    return np.float32(converted.item())


def _as_boolean_flag(value: Any, name: str) -> bool:
    """Validate an actual boolean, without accepting numeric zero or one.

    :param value: transition termination or truncation flag.
    :param name: flag name used in validation errors.
    :return: the flag as a Python boolean.
    :raises TypeError: if the flag is not ``bool`` or ``numpy.bool_``.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


class ReplayBuffer:
    """Fixed-capacity ring buffer with six aligned NumPy storage arrays.

    :param capacity: maximum number of stored transitions.
    :param obs_dim: positive length of the observation vector.
    :param act_dim: positive length of the action vector.
    :param seed: nonnegative seed reserved for the buffer's local RNG.
    :raises TypeError: if a constructor argument is not an ordinary integer.
    :raises ValueError: if capacity or dimensions are nonpositive, or seed is negative.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int) -> None:
        for name, value in (
                ("capacity", capacity),
                ("obs_dim", obs_dim),
                ("act_dim", act_dim),
                ("seed", seed),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer (not bool)")
            minimum = 0 if name == "seed" else 1
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}")

        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0

        self.obs = np.empty((capacity, obs_dim), dtype = np.float32)
        self.actions = np.empty((capacity, act_dim), dtype = np.float32)
        self.rewards = np.empty((capacity, 1), dtype = np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype = np.float32)
        self.terminated = np.empty((capacity, 1), dtype = np.bool_)
        self.truncated = np.empty((capacity, 1), dtype = np.bool_)

        self._rng = np.random.default_rng(seed)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            terminated: bool,
            truncated: bool,
    ) -> None:
        """Validate and store one complete transition at the ring write position.

        All fields are validated and converted before modifying any stored row.
        The oldest transition is overwritten once the buffer becomes full.

        :param obs: observation vector of shape ``(obs_dim,)``.
        :param action: action vector of shape ``(act_dim,)``.
        :param reward: finite real numeric scalar.
        :param next_obs: successor observation vector of shape ``(obs_dim,)``.
        :param terminated: true environment termination flag.
        :param truncated: time-limit or other episode truncation flag.
        :return: None.
        :raises TypeError: if values or flags have unsupported types.
        :raises ValueError: if shapes or numeric values are invalid.
        """
        obs_value = _as_float32_vector(obs, self.obs_dim, "obs")
        action_value = _as_float32_vector(action, self.act_dim, "action")
        reward_value = _as_float32_reward(reward)
        next_obs_value = _as_float32_vector(next_obs, self.obs_dim, "next_obs")
        terminated_value = _as_boolean_flag(terminated, "terminated")
        truncated_value = _as_boolean_flag(truncated, "truncated")

        index = self.ptr
        self.obs[index] = obs_value
        self.actions[index] = action_value
        self.rewards[index, 0] = reward_value
        self.next_obs[index] = next_obs_value
        self.terminated[index, 0] = terminated_value
        self.truncated[index, 0] = truncated_value

        self.ptr = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        """Return the number of currently stored transitions.

        :return: count between zero and the configured capacity.
        """
        return self.size

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample aligned transitions uniformly without replacement.

        Uses the buffer's own random generator and returns independent NumPy
        arrays so changes to a sampled batch cannot modify stored transitions.

        :param batch_size: positive number of distinct stored transitions to sample.
        :return: batch of six arrays indexed by the same sampled positions.
        :raises TypeError: if batch_size is not an ordinary integer.
        :raises ValueError: if batch_size is nonpositive or exceeds stored size.
        """
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer (not bool)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds stored transitions ({self.size})"
            )

        indices = self._rng.choice(self.size, size = batch_size, replace = False)
        return ReplayBatch(
            obs = self.obs[indices],
            actions = self.actions[indices],
            rewards = self.rewards[indices],
            next_obs = self.next_obs[indices],
            terminated = self.terminated[indices],
            truncated = self.truncated[indices],
        )
