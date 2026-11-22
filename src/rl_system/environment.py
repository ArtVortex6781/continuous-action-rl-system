"""Validation and description of supported Gymnasium environments."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from gymnasium import error as gym_error

from rl_system.config import EnvironmentConfig
from rl_system.exceptions import (
    EnvironmentCreationError,
    EnvironmentDependencyError,
    InvalidConfigError,
    UnsupportedConfigError,
    UnsupportedSpaceError,
)


def _copy_action_bound(value: np.ndarray, name: str, action_dim: int) -> np.ndarray:
    """Validate and copy a vector of action bounds.

    :param value: lower or upper action bounds
    :param name: bound name used in error messages
    :param action_dim: expected number of action coordinates
    :return: independent, read-only NumPy array of validated bounds
    :raises ValueError: if the bounds have an invalid shape, dtype, or value
    """
    bounds = np.array(value, copy = True)

    if bounds.ndim != 1 or bounds.shape[0] != action_dim:
        raise ValueError(f"{name} must have shape ({action_dim},), got {bounds.shape}")
    if bounds.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must have a real numeric dtype, got {bounds.dtype}")
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} must contain only finite numbers")

    bounds.setflags(write = False)
    return bounds


@dataclass(frozen = True, slots = True, eq = False)
class EnvironmentSpec:
    """Validated vector dimensions and action bounds of an environment.

    :param observation_dim: positive number of observation coordinates
    :param action_dim: positive number of action coordinates
    :param action_low: lower bound for each action coordinate
    :param action_high: upper bound for each action coordinate
    :raises ValueError: if dimensions or bounds violate the spec invariants
    """

    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def __post_init__(self) -> None:
        """Validate invariants and retain independent read-only bounds.

        :raises ValueError: if dimensions or action bounds are invalid
        """
        for name in ("observation_dim", "action_dim"):
            dimension = getattr(self, name)
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(f"{name} must be a positive integer")

        low = _copy_action_bound(self.action_low, "action_low", self.action_dim)
        high = _copy_action_bound(self.action_high, "action_high", self.action_dim)
        if not np.all(low < high):
            raise ValueError("action bounds must satisfy low < high at every coordinate")

        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)


def _box_vector_dim(space: gym.Space, name: str, *, allow_integer: bool) -> int:
    """Validate the supported Box structure and extract vector length.

    :param space: Gymnasium observation or action space
    :param name: space name used in error messages
    :param allow_integer: whether integer Box dtypes are supported
    :return: positive vector dimension
    :raises UnsupportedSpaceError: if the space shape, type, or dtype is unsupported
    """
    if not isinstance(space, gym.spaces.Box):
        raise UnsupportedSpaceError(
            f"{name} must be a Box space, got {type(space).__name__}"
        )
    if len(space.shape) != 1 or space.shape[0] <= 0:
        raise UnsupportedSpaceError(
            f"{name} must have a nonempty 1-D shape, got {space.shape}"
        )

    allowed_kinds = ("i", "u", "f") if allow_integer else ("f",)
    if space.dtype.kind not in allowed_kinds:
        expected = "real or integer" if allow_integer else "real floating-point"
        raise UnsupportedSpaceError(
            f"{name} must have a {expected} dtype, got {space.dtype}"
        )
    return space.shape[0]


def extract_environment_spec(env: gym.Env) -> EnvironmentSpec:
    """Extract validated vector dimensions and action bounds from an environment.

    This function does not reset, step, seed, modify, or close the environment.

    :param env: an already created Gymnasium environment
    :return: independent specification of its dimensions and action bounds
    :raises UnsupportedSpaceError: if observation or action spaces are unsupported
    """
    observation_dim = _box_vector_dim(
        env.observation_space, "observation_space", allow_integer = True
    )
    action_dim = _box_vector_dim(
        env.action_space, "action_space", allow_integer = False
    )

    try:
        return EnvironmentSpec(
            observation_dim = observation_dim,
            action_dim = action_dim,
            action_low = env.action_space.low,
            action_high = env.action_space.high,
        )
    except (TypeError, ValueError) as exc:
        raise UnsupportedSpaceError(f"Invalid action_space bounds: {exc}") from exc


def make_environment(
        config: EnvironmentConfig,
) -> tuple[gym.Env, EnvironmentSpec]:
    """Create a Gymnasium environment and validate its supported spaces.

    Ownership of a successfully returned environment passes to the caller.
    If space validation fails, the created environment is closed before the
    original error is re-raised.

    :param config: validated environment identifier and constructor arguments
    :return: created Gymnasium environment and its validated specification
    :raises InvalidConfigError: if the environment ID or known creation arguments are invalid
    :raises UnsupportedConfigError: if a requested environment capability is unsupported
    :raises EnvironmentDependencyError: if Gymnasium reports a missing dependency
    :raises EnvironmentCreationError: if Gymnasium reports another creation error
    :raises UnsupportedSpaceError: if the created environment has unsupported spaces
    """
    if "id" in config.kwargs:
        raise InvalidConfigError("environment.kwargs must not contain 'id'")

    try:
        env = gym.make(config.id, **config.kwargs)
    except gym_error.DependencyNotInstalled as exc:
        raise EnvironmentDependencyError(
            f"Missing dependency for environment {config.id!r}: {exc}"
        ) from exc
    except (gym_error.UnregisteredEnv, gym_error.DeprecatedEnv) as exc:
        raise InvalidConfigError(
            f"Invalid environment ID {config.id!r}: {exc}"
        ) from exc
    except (gym_error.MissingArgument, gym_error.InvalidBound,
            gym_error.InvalidProbability) as exc:
        raise InvalidConfigError(
            f"Invalid creation parameters for environment {config.id!r}: {exc}"
        ) from exc
    except gym_error.UnsupportedMode as exc:
        raise UnsupportedConfigError(
            f"Unsupported capability for environment {config.id!r}: {exc}"
        ) from exc
    except gym_error.Error as exc:
        raise EnvironmentCreationError(
            f"Failed to create environment {config.id!r}: {exc}"
        ) from exc

    try:
        spec = extract_environment_spec(env)
    except BaseException as exc:
        try:
            env.close()
        except Exception as close_exc:
            exc.add_note(f"Environment cleanup also failed: {close_exc!r}")
        raise

    return env, spec
