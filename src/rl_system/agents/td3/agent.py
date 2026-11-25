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
from rl_system.exceptions import AgentInputError, UnsupportedSpaceError

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

    def select_action(
            self, observation: np.ndarray, *, explore: bool
    ) -> np.ndarray:
        """Choose an action in the environment's actual action range.

        :param observation: current observation vector.
        :param explore: whether to apply exploration noise.
        :return: action vector in the environment's actual action range.
        :raises NotImplementedError: until step 2/5 is implemented.
        """
        raise NotImplementedError("Action selection is not implemented yet")

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
        :raises NotImplementedError: until step 2/5 is implemented.
        """
        raise NotImplementedError("Transition storage is not implemented yet")

    def update(self) -> dict[str, float] | None:
        """Perform a TD3 update if sufficient experience is available.

        :return: update metrics or None if the replay buffer is not ready.
        :raises NotImplementedError: until step 5/5 is implemented.
        """
        raise NotImplementedError("TD3 update is not implemented yet")
