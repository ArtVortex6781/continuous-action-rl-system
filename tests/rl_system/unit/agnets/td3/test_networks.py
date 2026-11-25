"""Unit tests for TD3 Actor, Critic and target network builders."""

import keras
import numpy as np
import pytest

from rl_system.agents.td3.config import MLPConfig
from rl_system.agents.td3.networks import build_actor, build_critic, create_target_network
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


def _network_inputs(spec, kind):
    observations = np.linspace(-0.5, 0.5, 3 * spec.observation_dim,
                               dtype = np.float32).reshape(3, spec.observation_dim)
    if kind == "actor":
        return observations
    actions = np.broadcast_to(
        (spec.action_low + spec.action_high) / 2, (3, spec.action_dim)
    ).copy()
    return [observations, actions]


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_target_network_matches_main_initially_and_has_independent_weights(kind):
    spec = _environment_spec(4, 2)
    config = MLPConfig(hidden_units = [12, 8], hidden_activations = ["tanh", "swish"])
    main = (build_actor(spec, config, output_init_limit = 0.003)
            if kind == "actor" else build_critic(spec, config))
    target = create_target_network(main)

    assert target is not main
    assert target.input_shape == main.input_shape
    assert target.output_shape == main.output_shape
    assert target.compiled is False
    assert len(target.layers) == len(main.layers)
    assert all(a is not b for a, b in zip(main.layers, target.layers, strict = True))
    assert main.weights and target.weights
    assert {id(v) for v in main.weights}.isdisjoint(id(v) for v in target.weights)
    for expected, actual in zip(main.get_weights(), target.get_weights(), strict = True):
        np.testing.assert_array_equal(actual, expected)

    inputs = _network_inputs(spec, kind)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(target(inputs, training = False)),
        keras.ops.convert_to_numpy(main(inputs, training = False)),
        rtol = 1e-6, atol = 1e-7,
    )

    target_before = target.get_weights()
    main.set_weights([w + 0.25 for w in main.get_weights()])
    for before, after in zip(target_before, target.get_weights(), strict = True):
        np.testing.assert_array_equal(after, before)

    main_before = main.get_weights()
    target.set_weights([w - 0.5 for w in target.get_weights()])
    for before, after in zip(main_before, main.get_weights(), strict = True):
        np.testing.assert_array_equal(after, before)


def _build_six_networks(spec, actor_config, critic_config):
    actor = build_actor(spec, actor_config, output_init_limit = 0.003)
    critic_1 = build_critic(spec, critic_config)
    critic_2 = build_critic(spec, critic_config)
    main = (actor, critic_1, critic_2)
    targets = tuple(create_target_network(model) for model in main)
    return main, targets


@pytest.mark.parametrize(("observation_dim", "action_dim"), [(3, 1), (8, 2)])
def test_all_six_networks_can_be_built_without_shared_variables(
        actor_config, critic_config, observation_dim, action_dim
):
    spec = _environment_spec(observation_dim, action_dim)
    main, targets = _build_six_networks(spec, actor_config, critic_config)
    models = main + targets
    assert len({id(model) for model in models}) == 6
    variables = [v for model in models for v in model.weights]
    assert len({id(v) for v in variables}) == len(variables)

    for index, (source, target) in enumerate(zip(main, targets, strict = True)):
        kind = "actor" if index == 0 else "critic"
        inputs = _network_inputs(spec, kind)
        expected_shape = (3, action_dim if index == 0 else 1)
        source_output = keras.ops.convert_to_numpy(source(inputs, training = False))
        target_output = keras.ops.convert_to_numpy(target(inputs, training = False))
        assert source_output.shape == target_output.shape == expected_shape
        assert np.all(np.isfinite(source_output))
        assert np.all(np.isfinite(target_output))
        for a, b in zip(source.get_weights(), target.get_weights(), strict = True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(target_output, source_output, rtol = 1e-6, atol = 1e-7)


@pytest.mark.parametrize("network_index", range(6), ids = [
    "actor", "critic_1", "critic_2", "target_actor", "target_critic_1", "target_critic_2"
])
def test_network_tensorflow_gradients_are_connected_and_finite(network_index):
    import tensorflow as tf

    assert keras.backend.backend() == "tensorflow"
    spec = _environment_spec(4, 2)
    config = MLPConfig(hidden_units = [8, 4], hidden_activations = ["tanh", "swish"])
    main, targets = _build_six_networks(spec, config, config)
    model = (main + targets)[network_index]
    kind = "actor" if network_index in (0, 3) else "critic"
    inputs = _network_inputs(spec, kind)
    inputs = [tf.convert_to_tensor(x) for x in inputs] if isinstance(inputs, list) else tf.convert_to_tensor(inputs)
    # A scalar reduction is sufficient to check autodiff connectivity; no TD3 loss.
    with tf.GradientTape() as tape:
        output = model(inputs, training = True)
        scalar = tf.reduce_sum(output)
    variables = model.trainable_variables
    assert variables
    gradients = tape.gradient(scalar, variables)
    assert len(gradients) == len(variables)
    for gradient, variable in zip(gradients, variables, strict = True):
        assert gradient is not None, f"Disconnected gradient: {variable.name}"
        assert gradient.shape == variable.shape
        assert bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
