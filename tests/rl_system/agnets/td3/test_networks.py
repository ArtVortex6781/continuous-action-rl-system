"""Unit tests for TD3 Actor and Critic network builders."""

import keras
import numpy as np
import pytest

from rl_system.agents.td3.config import MLPConfig
from rl_system.agents.td3.networks import build_actor, build_critic
from rl_system.environment import EnvironmentSpec


def _environment_spec(observation_dim: int, action_dim: int) -> EnvironmentSpec:
    """Create a valid spec with deliberately asymmetric action bounds."""
    action_low = np.arange(action_dim, dtype = np.float32) + 2.0
    action_high = action_low + np.arange(1, action_dim + 1, dtype = np.float32)
    return EnvironmentSpec(
        observation_dim = observation_dim,
        action_dim = action_dim,
        action_low = action_low,
        action_high = action_high,
    )


@pytest.fixture
def actor_config() -> MLPConfig:
    return MLPConfig(
        hidden_units = [32, 16], hidden_activations = ["relu", "swish"]
    )


@pytest.mark.parametrize(
    ("observation_dim", "action_dim", "batch_size"),
    [(3, 1, 1), (5, 2, 4), (1, 4, 7)],
)
def test_actor_accepts_vector_observations_and_returns_normalized_actions(
        actor_config, observation_dim, action_dim, batch_size
):
    spec = _environment_spec(observation_dim, action_dim)
    actor = build_actor(spec, actor_config, output_init_limit = 0.003)

    assert isinstance(actor, keras.Model)
    assert actor.input_shape == (None, observation_dim)
    assert actor.output_shape == (None, action_dim)

    observations = np.linspace(
        -1.0, 1.0, num = batch_size * observation_dim, dtype = np.float32
    ).reshape(batch_size, observation_dim)
    actions = keras.ops.convert_to_numpy(actor(observations, training = False))

    assert actions.shape == (batch_size, action_dim)
    assert np.all(np.isfinite(actions))
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


@pytest.mark.parametrize(
    ("units", "activations"),
    [
        ([8], ["relu"]),
        ([16, 8], ["tanh", "elu"]),
        ([32, 16, 8, 4], ["relu", "swish", "elu", "tanh"]),
    ],
)
def test_actor_creates_exactly_the_configured_hidden_dense_layers(units, activations):
    spec = _environment_spec(observation_dim = 3, action_dim = 2)
    config = MLPConfig(hidden_units = units, hidden_activations = activations)
    actor = build_actor(spec, config, output_init_limit = 0.003)

    assert len(actor.layers) == len(units) + 2  # InputLayer, hidden Dense, output Dense
    assert isinstance(actor.layers[0], keras.layers.InputLayer)
    assert all(isinstance(layer, keras.layers.Dense) for layer in actor.layers[1:])
    assert actor.inputs[0].name == "observation"

    hidden_layers = actor.layers[1:-1]
    for index, (layer, expected_units, expected_activation) in enumerate(
            zip(hidden_layers, units, activations, strict = True), start = 1
    ):
        assert layer.name == f"actor_hidden_{index}"
        assert layer.units == expected_units
        assert layer.activation is keras.activations.get(expected_activation)
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)

    output = actor.layers[-1]
    assert output.name == "actor_action"
    assert output.units == spec.action_dim
    assert output.activation is keras.activations.tanh
    assert actor.compiled is False


@pytest.mark.parametrize("limit", [0.003, 0.02])
def test_actor_output_kernel_uses_small_symmetric_uniform_initialization(
        actor_config, limit
):
    actor = build_actor(
        _environment_spec(observation_dim = 3, action_dim = 2),
        actor_config,
        output_init_limit = limit,
    )
    output = actor.get_layer("actor_action")

    assert isinstance(output.kernel_initializer, keras.initializers.RandomUniform)
    assert output.kernel_initializer.minval == pytest.approx(-limit)
    assert output.kernel_initializer.maxval == pytest.approx(limit)
    assert isinstance(output.bias_initializer, keras.initializers.Zeros)

    kernel, bias = output.get_weights()
    assert np.all(np.isfinite(kernel))
    assert np.all(kernel >= -limit)
    assert np.all(kernel <= limit)
    np.testing.assert_array_equal(bias, np.zeros_like(bias))


def test_actor_outputs_normalized_actions_without_environment_bound_scaling(actor_config):
    spec = EnvironmentSpec(
        observation_dim = 3,
        action_dim = 2,
        action_low = np.array([10.0, 20.0], dtype = np.float32),
        action_high = np.array([12.0, 25.0], dtype = np.float32),
    )
    actor = build_actor(spec, actor_config, output_init_limit = 0.003)

    # Force a known nonzero tanh output, independently of hidden-layer weights.
    output = actor.get_layer("actor_action")
    kernel, bias = output.get_weights()
    output.set_weights(
        [np.zeros_like(kernel), np.full_like(bias, np.arctanh(0.5))]
    )

    actions = keras.ops.convert_to_numpy(
        actor(np.ones((3, spec.observation_dim), dtype = np.float32), training = False)
    )
    np.testing.assert_allclose(
        actions, np.full((3, spec.action_dim), 0.5), rtol = 0, atol = 1e-6
    )
    assert np.all(actions < spec.action_low)


@pytest.mark.parametrize(
    "invalid_limit",
    [0, 0.0, -0.003, float("nan"), float("inf"), -float("inf"),
     True, False, None, "0.003"],
)
def test_actor_rejects_invalid_output_initializer_limits(actor_config, invalid_limit):
    with pytest.raises(ValueError, match = "output_init_limit"):
        build_actor(
            _environment_spec(observation_dim = 3, action_dim = 1),
            actor_config,
            output_init_limit = invalid_limit,
        )


@pytest.fixture
def critic_config() -> MLPConfig:
    return MLPConfig(
        hidden_units = [32, 16], hidden_activations = ["relu", "elu"]
    )


@pytest.mark.parametrize(
    ("observation_dim", "action_dim", "batch_size"),
    [(3, 1, 1), (5, 2, 4), (1, 4, 7)],
)
def test_critic_accepts_observations_and_real_scale_actions(
        critic_config, observation_dim, action_dim, batch_size
):
    spec = _environment_spec(observation_dim, action_dim)
    critic = build_critic(spec, critic_config)

    assert isinstance(critic, keras.Model)
    assert len(critic.inputs) == 2
    assert critic.input_shape == [(None, observation_dim), (None, action_dim)]
    assert critic.output_shape == (None, 1)
    assert [tensor.name for tensor in critic.inputs] == ["observation", "action"]

    observations = np.linspace(
        -1.0, 1.0, batch_size * observation_dim, dtype = np.float32
    ).reshape(batch_size, observation_dim)
    # Use actual environment bounds rather than Actor's normalized [-1, 1] values.
    actions = np.broadcast_to(
        (spec.action_low + spec.action_high) / 2,
        (batch_size, action_dim),
    ).copy()
    q_values = keras.ops.convert_to_numpy(
        critic([observations, actions], training = False)
    )

    assert q_values.shape == (batch_size, 1)
    assert np.all(np.isfinite(q_values))
    assert critic.compiled is False


@pytest.mark.parametrize(
    ("units", "activations"),
    [
        ([8], ["relu"]),
        ([16, 8], ["tanh", "elu"]),
        ([32, 16, 8, 4], ["relu", "swish", "elu", "tanh"]),
    ],
)
def test_critic_uses_early_concatenation_and_configured_hidden_layers(
        units, activations
):
    spec = _environment_spec(observation_dim = 3, action_dim = 2)
    config = MLPConfig(hidden_units = units, hidden_activations = activations)
    critic = build_critic(spec, config)

    assert len(critic.layers) == len(units) + 4
    assert all(isinstance(layer, keras.layers.InputLayer) for layer in critic.layers[:2])

    concatenation = critic.layers[2]
    assert isinstance(concatenation, keras.layers.Concatenate)
    assert concatenation.name == "critic_concat"
    assert len(concatenation.input) == 2
    assert concatenation.output.shape == (None, spec.observation_dim + spec.action_dim)

    hidden_layers = critic.layers[3:-1]
    for index, (layer, expected_units, expected_activation) in enumerate(
            zip(hidden_layers, units, activations, strict = True), start = 1
    ):
        assert isinstance(layer, keras.layers.Dense)
        assert layer.name == f"critic_hidden_{index}"
        assert layer.units == expected_units
        assert layer.activation is keras.activations.get(expected_activation)
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)

    output = critic.layers[-1]
    assert isinstance(output, keras.layers.Dense)
    assert output.name == "critic_q"
    assert output.units == 1
    assert output.activation is keras.activations.linear


def test_critic_does_not_normalize_actions_or_restrict_q_values():
    spec = EnvironmentSpec(
        observation_dim = 2,
        action_dim = 1,
        action_low = np.array([10.0], dtype = np.float32),
        action_high = np.array([20.0], dtype = np.float32),
    )
    config = MLPConfig(hidden_units = [1], hidden_activations = ["relu"])
    critic = build_critic(spec, config)

    # Q = -action: observation weights are zero; the action weight is one.
    hidden = critic.get_layer("critic_hidden_1")
    hidden_kernel, hidden_bias = hidden.get_weights()
    hidden_kernel.fill(0.0)
    hidden_kernel[spec.observation_dim, 0] = 1.0
    hidden.set_weights([hidden_kernel, np.zeros_like(hidden_bias)])

    output = critic.get_layer("critic_q")
    output_kernel, output_bias = output.get_weights()
    output.set_weights([
        np.full_like(output_kernel, -1.0),
        np.zeros_like(output_bias),
    ])

    observations = np.zeros((2, spec.observation_dim), dtype = np.float32)
    actions = np.array([[12.0], [15.0]], dtype = np.float32)
    q_values = keras.ops.convert_to_numpy(
        critic([observations, actions], training = False)
    )

    np.testing.assert_allclose(
        q_values, np.array([[-12.0], [-15.0]], dtype = np.float32),
        rtol = 0, atol = 1e-5,
    )


def test_separately_built_critics_have_independent_trainable_variables(critic_config):
    spec = _environment_spec(observation_dim = 3, action_dim = 2)
    critic_1 = build_critic(spec, critic_config)
    critic_2 = build_critic(spec, critic_config)

    assert critic_1 is not critic_2
    assert len(critic_1.trainable_variables) == len(critic_2.trainable_variables)
    assert critic_1.trainable_variables
    assert all(
        left is not right
        for left, right in zip(
            critic_1.trainable_variables, critic_2.trainable_variables, strict = True
        )
    )

    critic_2_weights_before = [value.copy() for value in critic_2.get_weights()]
    critic_1_weights = critic_1.get_weights()
    critic_1_weights[0] = critic_1_weights[0] + 1.0
    critic_1.set_weights(critic_1_weights)

    for before, after in zip(
            critic_2_weights_before, critic_2.get_weights(), strict = True
    ):
        np.testing.assert_array_equal(after, before)


def test_actor_and_critic_can_use_different_mlp_configs():
    spec = _environment_spec(observation_dim = 4, action_dim = 2)
    actor_config = MLPConfig(hidden_units = [8], hidden_activations = ["relu"])
    critic_config = MLPConfig(
        hidden_units = [12, 6], hidden_activations = ["swish", "elu"]
    )

    actor = build_actor(spec, actor_config, output_init_limit = 0.003)
    critic = build_critic(spec, critic_config)

    assert [layer.units for layer in actor.layers if isinstance(layer, keras.layers.Dense)] == [8, 2]
    assert [layer.units for layer in critic.layers if isinstance(layer, keras.layers.Dense)] == [12, 6, 1]
