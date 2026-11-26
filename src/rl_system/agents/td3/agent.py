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
        """Prepare float32 affine scaling constants and action bounds.

        :raises UnsupportedSpaceError: if scaling cannot retain a finite,
            nondegenerate action range in float32.
        """
        low = self._spec.action_low
        high = self._spec.action_high
        low64 = low.astype(np.float64)
        high64 = high.astype(np.float64)
        midpoint = ((high64 + low64) / 2.0).astype(np.float32)
        half_range = ((high64 - low64) / 2.0).astype(np.float32)

        if np.any(half_range <= 0):
            raise UnsupportedSpaceError("Action scaling requires a positive float32 half range")

        self._action_midpoint = tf.convert_to_tensor(midpoint, dtype = tf.float32)
        self._action_half_range = tf.convert_to_tensor(half_range, dtype = tf.float32)
        self._action_low = tf.convert_to_tensor(low, dtype = tf.float32)
        self._action_high = tf.convert_to_tensor(high, dtype = tf.float32)

        scaled_low = self._action_midpoint - self._action_half_range
        scaled_high = self._action_midpoint + self._action_half_range
        if (
                not np.all(np.isfinite(scaled_low.numpy()))
                or not np.all(np.isfinite(scaled_high.numpy()))
        ):
            raise UnsupportedSpaceError("Action scaling endpoints must be finite in float32")

        clipped_low = tf.clip_by_value(scaled_low, self._action_low, self._action_high)
        clipped_high = tf.clip_by_value(scaled_high, self._action_low, self._action_high)
        if np.any(clipped_low.numpy() >= clipped_high.numpy()):
            raise UnsupportedSpaceError("Action scaling must retain a nonzero float32 range")

    def _scale_action(self, action: tf.Tensor) -> tf.Tensor:
        """Map normalized actions to the environment's action units with clipping.

        :param action: normalized action tensor with action_dim on its last axis.
        :return: scaled action tensor, preserving the gradient path.
        """
        return tf.clip_by_value(
            self._action_midpoint + self._action_half_range * action,
            self._action_low,
            self._action_high,
        )

    @staticmethod
    def _assert_finite(value: tf.Tensor, name: str) -> tf.Tensor:
        """Reject nonfinite internal computation results.

        :param value: tensor to check.
        :param name: operation or value name used in the error message.
        :return: checked tensor.
        :raises AgentNumericalError: if the tensor contains NaN or infinity.
        """
        try:
            return tf.debugging.assert_all_finite(value, f"{name} must be finite")
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
        if explore:
            with np.errstate(over = "ignore", invalid = "ignore"):
                noise = self._exploration_rng.normal(
                    loc = 0.0,
                    scale = self._config.exploration_noise_std,
                    size = (1, self._spec.action_dim),
                ).astype(np.float32)
            noise = tf.convert_to_tensor(noise, dtype = tf.float32)
            noisy_action = self._assert_finite(normalized + noise, "Exploration action")
            normalized = tf.clip_by_value(noisy_action, -1.0, 1.0)
        else:
            normalized = self._assert_finite(normalized, "Actor output")

        return self._scale_action(normalized).numpy()[0]

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

    def _compute_bellman_target(
            self,
            next_obs: tf.Tensor,
            rewards: tf.Tensor,
            terminated: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the shared, detached TD3 Bellman target for both Critics.

        :param next_obs: float32 next observations of shape (batch, observation_dim).
        :param rewards: float32 rewards of shape (batch, 1).
        :param terminated: float32 terminal flags (0 or 1) of shape (batch, 1).
        :return: detached float32 Bellman targets of shape (batch, 1).
        :raises AgentNumericalError: if an internal result is nonfinite.
        """
        action_shape = (next_obs.shape[0], self._spec.action_dim)
        normalized = self._target_actor(next_obs, training = False)

        with np.errstate(over = "ignore", invalid = "ignore"):
            noise = self._target_noise_rng.normal(
                loc = 0.0,
                scale = self._config.target_policy_noise_std,
                size = action_shape,
            ).astype(np.float32)
        noise = tf.convert_to_tensor(noise, dtype = tf.float32)
        noise = self._assert_finite(noise, "Target policy noise")

        noise_clip = self._config.target_noise_clip
        noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
        noisy_action = self._assert_finite(normalized + noise, "Noisy target action")
        normalized = tf.clip_by_value(noisy_action, -1.0, 1.0)
        target_action = self._scale_action(normalized)

        q1 = self._target_critic_1([next_obs, target_action], training = False)
        q2 = self._target_critic_2([next_obs, target_action], training = False)
        q1 = self._assert_finite(q1, "Target Critic 1 output")
        q2 = self._assert_finite(q2, "Target Critic 2 output")

        target = rewards + self._config.gamma * (1.0 - terminated) * tf.minimum(q1, q2)
        return tf.stop_gradient(self._assert_finite(target, "Bellman target"))

    def _compute_critic_gradients(
            self,
            critic: keras.Model,
            obs: tf.Tensor,
            actions: tf.Tensor,
            target: tf.Tensor,
            *,
            name: str,
    ) -> tuple[tf.Tensor, list[tf.Tensor | None]]:
        """Compute one Critic's MSE and gradients without applying an optimizer.

        :param critic: main Critic to differentiate.
        :param obs: batch of observations.
        :param actions: actual actions sampled from the replay buffer.
        :param target: detached Bellman targets of shape (batch, 1).
        :param name: Critic identifier for error messages.
        :return: scalar MSE and gradients in trainable-variable order.
        :raises AgentNumericalError: if the loss is nonfinite.
        """
        with tf.GradientTape() as tape:
            q = critic([obs, actions], training = True)
            loss = tf.reduce_mean(tf.square(q - target))
        self._assert_finite(loss, f"{name} loss")
        gradients = tape.gradient(loss, critic.trainable_variables)
        return loss, gradients

    def _validate_gradients(
            self,
            gradients: list[tf.Tensor | None],
            variables: list[keras.Variable],
            *,
            name: str,
    ) -> None:
        """Check gradient presence and finiteness before an optimizer update.

        :param gradients: gradients in trainable-variable order.
        :param variables: corresponding trainable variables.
        :param name: network identifier for error messages.
        :raises AgentUpdateError: if a gradient is missing.
        :raises ValueError: if gradient and variable counts do not match.
        :raises AgentNumericalError: if a gradient is nonfinite.
        """
        for gradient, variable in zip(gradients, variables, strict = True):
            label = f"{name} gradient for {variable.path}"
            if gradient is None:
                raise AgentUpdateError(f"Missing {label}")
            self._assert_finite(gradient, label)

    def _validate_optimization_state(
            self,
            model: keras.Model,
            optimizer: keras.optimizers.Optimizer,
            *,
            name: str,
    ) -> None:
        """Check model weights and floating-point optimizer state after an update.

        :param model: updated main network.
        :param optimizer: optimizer owning the corresponding update state.
        :param name: network identifier for error messages.
        :raises AgentNumericalError: if a checked variable is nonfinite.
        """
        for variable in model.weights:
            self._assert_finite(tf.convert_to_tensor(variable), f"{name} weight {variable.path}")
        for variable in optimizer.variables:
            if tf.as_dtype(variable.dtype).is_floating:
                self._assert_finite(
                    tf.convert_to_tensor(variable), f"{name} Adam state {variable.path}"
                )

    def _update_critics(
            self,
            obs: tf.Tensor,
            actions: tf.Tensor,
            target: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Update both main Critics against the same Bellman target.

        Both losses and gradient sets are checked before either Adam is applied.
        A failed post-update check is fatal; optimizer changes are not rolled back.

        :param obs: batch of observations.
        :param actions: actual actions sampled from the replay buffer.
        :param target: detached Bellman targets.
        :return: the two scalar losses used for optimization, in Critic order.
        :raises AgentUpdateError: if a gradient is missing.
        :raises AgentNumericalError: if a loss, gradient or updated state is nonfinite.
        """
        loss_1, gradients_1 = self._compute_critic_gradients(
            self._critic_1, obs, actions, target, name = "Critic 1"
        )
        loss_2, gradients_2 = self._compute_critic_gradients(
            self._critic_2, obs, actions, target, name = "Critic 2"
        )

        variables_1 = self._critic_1.trainable_variables
        variables_2 = self._critic_2.trainable_variables
        self._validate_gradients(gradients_1, variables_1, name = "Critic 1")
        self._validate_gradients(gradients_2, variables_2, name = "Critic 2")

        self._critic_1_optimizer.apply_gradients(zip(gradients_1, variables_1, strict = True))
        self._validate_optimization_state(
            self._critic_1, self._critic_1_optimizer, name = "Critic 1"
        )
        self._critic_2_optimizer.apply_gradients(zip(gradients_2, variables_2, strict = True))
        self._validate_optimization_state(
            self._critic_2, self._critic_2_optimizer, name = "Critic 2"
        )
        return loss_1, loss_2

    def _update_actor(self, obs: tf.Tensor) -> tf.Tensor:
        """Update the main Actor using the first main Critic.

        :param obs: observations from the same batch used for Critic updates.
        :return: scalar Actor loss used for optimization.
        :raises AgentUpdateError: if a gradient is missing.
        :raises AgentNumericalError: if an internal result or updated state
            is nonfinite.
        """
        with tf.GradientTape() as tape:
            normalized = self._actor(obs, training = True)
            self._assert_finite(normalized, "Actor output")
            actions = self._scale_action(normalized)
            q = self._critic_1([obs, actions], training = False)
            loss = -tf.reduce_mean(q)
        self._assert_finite(loss, "Actor loss")

        variables = self._actor.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._validate_gradients(gradients, variables, name = "Actor")
        self._actor_optimizer.apply_gradients(zip(gradients, variables, strict = True))
        self._validate_optimization_state(
            self._actor, self._actor_optimizer, name = "Actor"
        )
        return loss

    def _soft_update(
            self,
            source: keras.Model,
            target: keras.Model,
            *,
            name: str,
    ) -> None:
        """Move target weights toward their corresponding source weights.

        :param source: main network whose weights remain unchanged.
        :param target: corresponding target network to update.
        :param name: target network identifier for error messages.
        :raises AgentNumericalError: if an updated target weight is nonfinite.
        """
        tau = self._config.tau
        for source_weight, target_weight in zip(source.weights, target.weights, strict = True):
            target_weight.assign((1.0 - tau) * target_weight + tau * source_weight)

    def _update_target_networks(self) -> None:
        """Soft-update all three target networks after a successful Actor update.

        :raises AgentNumericalError: if an updated target weight is nonfinite.
        """
        self._soft_update(self._actor, self._target_actor, name = "Target Actor")
        self._soft_update(self._critic_1, self._target_critic_1, name = "Target Critic 1")
        self._soft_update(self._critic_2, self._target_critic_2, name = "Target Critic 2")

    def update(self) -> dict[str, float] | None:
        """Perform a TD3 update if sufficient experience is available.

        :return: update metrics or None if the replay buffer is not ready.
        :raises NotImplementedError: until step 4/4 is implemented.
        """
        raise NotImplementedError("TD3 update is not implemented yet")
