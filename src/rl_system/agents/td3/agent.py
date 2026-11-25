"""TD3 agent and its owned learning components."""

from typing import TYPE_CHECKING

import keras
import numpy as np
import tensorflow as tf

from rl_system.agents.base import OffPolicyAgent
from rl_system.agents.td3.config import TD3Config
from rl_system.agents.td3.networks import (
    build_actor,
    build_critic,
    create_target_network,
)
from rl_system.agents.td3.replay_buffer import ReplayBuffer
from rl_system.exceptions import (
    AgentInputError,
    AgentNumericalError,
    AgentUpdateError,
    UnsupportedSpaceError,
)

if TYPE_CHECKING:
    from rl_system.environment import EnvironmentSpec


class TD3Agent(OffPolicyAgent):
    """Off-policy TD3 agent owning its networks, optimizers and replay storage.

    :param spec: validated environment dimensions and action bounds.
    :param config: validated TD3 hyperparameters and network configurations.
    :param seed: nonnegative integer seed for independent local random streams.
    :raises AgentInputError: if seed is not a nonnegative Python integer.
    :raises UnsupportedSpaceError: if action scaling is incompatible with float32.
    """

    def __init__(
            self,
            spec: "EnvironmentSpec",
            config: TD3Config,
            *,
            seed: int,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise AgentInputError("seed must be a nonnegative integer (not bool)")

        self._spec = spec
        self._config = config
        self._initialize_action_scaling()

        self._actor = build_actor(
            spec, config.actor, output_init_limit = config.actor_output_init_limit
        )
        self._critic_1 = build_critic(spec, config.critic)
        self._critic_2 = build_critic(spec, config.critic)
        self._target_actor = create_target_network(self._actor)
        self._target_critic_1 = create_target_network(self._critic_1)
        self._target_critic_2 = create_target_network(self._critic_2)

        self._actor_optimizer = keras.optimizers.Adam(
            learning_rate = config.actor_learning_rate
        )
        self._critic_1_optimizer = keras.optimizers.Adam(
            learning_rate = config.critic_learning_rate
        )
        self._critic_2_optimizer = keras.optimizers.Adam(
            learning_rate = config.critic_learning_rate
        )
        self._actor_optimizer.build(self._actor.trainable_variables)
        self._critic_1_optimizer.build(self._critic_1.trainable_variables)
        self._critic_2_optimizer.build(self._critic_2.trainable_variables)

        buffer_seed, exploration_seed, target_noise_seed = (
            np.random.SeedSequence(seed).spawn(3)
        )
        self._replay_buffer = ReplayBuffer(
            capacity = config.replay_capacity,
            obs_dim = spec.observation_dim,
            act_dim = spec.action_dim,
            seed = int(buffer_seed.generate_state(1)[0]),
        )
        self._exploration_rng = np.random.default_rng(exploration_seed)
        self._target_noise_rng = np.random.default_rng(target_noise_seed)
        self._update_count = 0

    def _initialize_action_scaling(self) -> None:
        """Prepare affine scaling and inward-rounded float32 action bounds.

        :raises UnsupportedSpaceError: if finite, nondegenerate action ranges
            cannot be represented by the float32 scaling computation.
        """
        low = self._spec.action_low.astype(np.longdouble)
        high = self._spec.action_high.astype(np.longdouble)

        # Halve before adding/subtracting to avoid intermediate overflow.
        with np.errstate(over = "ignore", under = "ignore", invalid = "ignore"):
            low32 = low.astype(np.float32)
            high32 = high.astype(np.float32)
            midpoint = (high * 0.5 + low * 0.5).astype(np.float32)
            half_range = (high * 0.5 - low * 0.5).astype(np.float32)

        if not all(np.all(np.isfinite(array)) for array in (
                low32, high32, midpoint, half_range
        )) or np.any(half_range <= 0):
            raise UnsupportedSpaceError(
                "Action bounds and scaling must be finite with positive float32 ranges"
            )

        # Round inward so a returned float32 action cannot exceed the original bounds.
        with np.errstate(over = "ignore", under = "ignore", invalid = "ignore"):
            low32 = np.where(
                low32.astype(np.longdouble) < low,
                np.nextafter(low32, np.float32(np.inf)),
                low32,
            )
            high32 = np.where(
                high32.astype(np.longdouble) > high,
                np.nextafter(high32, np.float32(-np.inf)),
                high32,
            )
        if np.any(low32 >= high32):
            raise UnsupportedSpaceError(
                "Action bounds must contain a nonzero float32 interval"
            )

        self._action_midpoint = tf.convert_to_tensor(midpoint, dtype = tf.float32)
        self._action_half_range = tf.convert_to_tensor(half_range, dtype = tf.float32)
        scaled_low = (self._action_midpoint - self._action_half_range).numpy()
        scaled_high = (self._action_midpoint + self._action_half_range).numpy()
        if (
                not np.all(np.isfinite(scaled_low))
                or not np.all(np.isfinite(scaled_high))
                or np.any(scaled_low >= scaled_high)
                or np.any(np.clip(scaled_low, low32, high32) >= np.clip(scaled_high, low32, high32))
        ):
            raise UnsupportedSpaceError(
                "Action scaling must retain a finite nonzero range in float32"
            )

        self._action_low = low32
        self._action_high = high32

    def _scale_action(self, action: tf.Tensor) -> tf.Tensor:
        """Map normalized actions to the environment's action units.

        :param action: normalized action tensor with action_dim on its last axis.
        :return: scaled action tensor, preserving the gradient path.
        """
        return self._action_midpoint + self._action_half_range * action

    @staticmethod
    def _assert_finite(value: tf.Tensor, name: str) -> None:
        """Reject nonfinite internal computation results.

        :param value: tensor to check.
        :param name: operation or value name used in the error message.
        :raises AgentNumericalError: if the tensor contains NaN or infinity.
        """
        try:
            tf.debugging.assert_all_finite(value, f"{name} must be finite")
        except tf.errors.InvalidArgumentError as exc:
            raise AgentNumericalError(f"Nonfinite values in {name}") from exc

    def select_action(
            self, observation: np.ndarray, *, explore: bool
    ) -> np.ndarray:
        """Choose an action in the environment's actual action range.

        :param observation: current observation vector.
        :param explore: whether to apply exploration noise.
        :return: finite float32 action vector in the original action bounds.
        :raises AgentInputError: if observation or explore is invalid.
        :raises AgentUpdateError: if the Actor output has an unexpected shape.
        :raises AgentNumericalError: if action computation produces nonfinite values.
        """
        if not isinstance(explore, (bool, np.bool_)):
            raise AgentInputError("explore must be a boolean")

        try:
            obs = np.asarray(observation)
            if obs.shape != (self._spec.observation_dim,):
                raise ValueError(
                    f"observation must have shape ({self._spec.observation_dim},), "
                    f"got {obs.shape}"
                )
            if obs.dtype.kind not in ("i", "u", "f"):
                raise TypeError("observation must contain real numeric values")
            with np.errstate(over = "ignore", invalid = "ignore"):
                obs = obs.astype(np.float32)
            if not np.all(np.isfinite(obs)):
                raise ValueError("observation must contain finite float32 values")
        except (TypeError, ValueError, OverflowError) as exc:
            raise AgentInputError(f"Invalid observation: {exc}") from exc

        obs_batch = tf.convert_to_tensor(obs[None, :], dtype = tf.float32)
        normalized = self._actor(obs_batch, training = False)
        if normalized.shape != (1, self._spec.action_dim):
            raise AgentUpdateError(
                f"Actor output must have shape (1, {self._spec.action_dim}), "
                f"got {normalized.shape}"
            )
        self._assert_finite(normalized, "Actor output")

        if explore:
            with np.errstate(over = "ignore", invalid = "ignore"):
                noise = self._exploration_rng.normal(
                    loc = 0.0,
                    scale = self._config.exploration_noise_std,
                    size = (1, self._spec.action_dim),
                )
                noisy_action = normalized.numpy() + noise
            if not np.all(np.isfinite(noisy_action)):
                raise AgentNumericalError("Nonfinite values in exploration action")
            normalized = tf.convert_to_tensor(
                np.clip(noisy_action, -1.0, 1.0), dtype = tf.float32
            )
        else:
            normalized = tf.clip_by_value(normalized, -1.0, 1.0)

        scaled = self._scale_action(normalized)
        self._assert_finite(scaled, "Scaled action")
        # Only the external action is rounded/clipped to the inward float32 bounds.
        return np.clip(scaled.numpy()[0], self._action_low, self._action_high)

    def store_transition(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_obs: np.ndarray,
            terminated: bool,
            truncated: bool,
    ) -> None:
        """Store the actual transition observed in the environment.

        :param obs: observation before the environment step.
        :param action: action performed in the environment.
        :param reward: reward received from the environment.
        :param next_obs: observation returned by the environment step.
        :param terminated: whether the environment reached a terminal state.
        :param truncated: whether the episode was truncated.
        :raises AgentInputError: if the transition violates the input contract.
        """
        try:
            action_array = np.asarray(action)
        except (TypeError, ValueError) as exc:
            raise AgentInputError(f"Invalid transition action: {exc}") from exc

        # Shape, dtype and finiteness validation belong to ReplayBuffer.
        # Only compare bounds when the input supports a coordinatewise comparison.
        if (
                action_array.shape == (self._spec.action_dim,)
                and action_array.dtype.kind in ("i", "u", "f")
                and (
                np.any(action_array < self._spec.action_low)
                or np.any(action_array > self._spec.action_high)
        )
        ):
            raise AgentInputError("Transition action is outside the environment bounds")

        try:
            self._replay_buffer.add(
                obs, action_array, reward, next_obs, terminated, truncated
            )
        except (TypeError, ValueError) as exc:
            raise AgentInputError(f"Invalid transition: {exc}") from exc

    def update(self) -> dict[str, float] | None:
        """Perform a TD3 update if sufficient experience is available.

        :return: update metrics or None if the replay buffer is not ready.
        :raises NotImplementedError: until step 5/5 is implemented.
        """
        raise NotImplementedError("TD3 update is not implemented yet")
