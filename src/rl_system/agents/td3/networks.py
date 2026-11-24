"""Keras network builders for TD3."""

import math
from typing import TYPE_CHECKING

import keras

from rl_system.agents.td3.config import MLPConfig

if TYPE_CHECKING:
    from rl_system.environment import EnvironmentSpec


def build_actor(
        spec: "EnvironmentSpec",
        config: MLPConfig,
        *,
        output_init_limit: float,
) -> keras.Model:
    """Build an Actor mapping observations to normalized continuous actions.

    Hidden Dense layers follow ``config``. The output uses ``tanh`` and a
    small uniform kernel initializer; action scaling belongs to TD3Agent.

    :param spec: validated environment dimensions.
    :param config: validated hidden-layer sizes and activations.
    :param output_init_limit: positive finite half-width of the output kernel
        initialization interval.
    :return: uncompiled Keras Actor with output shape ``(batch, action_dim)``.
    :raises ValueError: if the output initialization limit is invalid.
    """
    if (
            isinstance(output_init_limit, bool)
            or not isinstance(output_init_limit, (int, float))
            or not math.isfinite(output_init_limit)
            or output_init_limit <= 0
    ):
        raise ValueError("output_init_limit must be a positive finite number")

    observation = keras.Input(
        shape = (spec.observation_dim,), name = "observation"
    )
    hidden = observation
    for index, (units, activation) in enumerate(
            zip(config.hidden_units, config.hidden_activations, strict = True),
            start = 1,
    ):
        hidden = keras.layers.Dense(
            units, activation = activation, name = f"actor_hidden_{index}"
        )(hidden)

    action = keras.layers.Dense(
        spec.action_dim,
        activation = "tanh",
        kernel_initializer = keras.initializers.RandomUniform(
            minval = -output_init_limit, maxval = output_init_limit
        ),
        bias_initializer = "zeros",
        name = "actor_action",
    )(hidden)

    return keras.Model(inputs = observation, outputs = action, name = "actor")


def build_critic(spec: "EnvironmentSpec", config: MLPConfig) -> keras.Model:
    """Build a Critic estimating a scalar Q-value for observation-action pairs.

    Observations and actions enter through separate inputs and are concatenated
    before all configurable hidden Dense layers. Actions must use the actual
    environment scale rather than the Actor's normalized ``[-1, 1]`` scale.

    :param spec: validated environment observation and action dimensions.
    :param config: validated hidden-layer sizes and activations.
    :return: uncompiled Keras Critic with output shape ``(batch, 1)``.
    """
    observation = keras.Input(
        shape = (spec.observation_dim,), name = "observation"
    )
    action = keras.Input(shape = (spec.action_dim,), name = "action")

    hidden = keras.layers.Concatenate(name = "critic_concat")([observation, action])
    for index, (units, activation) in enumerate(
            zip(config.hidden_units, config.hidden_activations, strict = True),
            start = 1,
    ):
        hidden = keras.layers.Dense(
            units, activation = activation, name = f"critic_hidden_{index}"
        )(hidden)

    q_value = keras.layers.Dense(
        1, activation = "linear", name = "critic_q"
    )(hidden)

    return keras.Model(
        inputs = [observation, action], outputs = q_value, name = "critic"
    )


def create_target_network(model: keras.Model) -> keras.Model:
    """Create an independent target network with the main network's weights.

    Only the initial full synchronization is performed here. Subsequent
    soft updates belong to TD3Agent.

    :param model: built Keras Actor or Critic to clone.
    :return: independent Keras model with identical initial weight values.
    """
    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())
    return target_model
