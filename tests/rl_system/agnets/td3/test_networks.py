"""Unit tests for TD3 network builders, beginning with the Actor."""

import keras
import numpy as np
import pytest

from rl_system.agents.td3.config import MLPConfig
from rl_system.agents.td3.networks import build_actor
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
