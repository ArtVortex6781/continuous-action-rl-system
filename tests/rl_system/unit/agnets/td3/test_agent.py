"""Unit tests for TD3 agent initialization and interaction contracts."""

import random

import keras
import numpy as np
import pytest
import tensorflow as tf

from rl_system.agents.base import OffPolicyAgent
from rl_system.agents.td3.agent import TD3Agent
from rl_system.agents.td3.config import MLPConfig, TD3Config
from rl_system.agents.td3.replay_buffer import ReplayBuffer
from rl_system.environment import EnvironmentSpec
from rl_system.exceptions import (
    AgentInputError,
    AgentNumericalError,
    AgentSystemError,
    AgentUpdateError,
    RLSystemError,
    UnsupportedSpaceError,
)


@pytest.fixture
def config() -> TD3Config:
    return TD3Config(
        replay_capacity = 16,
        batch_size = 4,
        gamma = 0.99,
        tau = 0.005,
        actor_learning_rate = 0.001,
        critic_learning_rate = 0.002,
        policy_delay = 2,
        exploration_noise_std = 0.1,
        target_policy_noise_std = 0.2,
        target_noise_clip = 0.5,
        actor_output_init_limit = 0.003,
        actor = MLPConfig(hidden_units = [8, 8], hidden_activations = ["relu", "relu"]),
        critic = MLPConfig(hidden_units = [8, 8], hidden_activations = ["relu", "relu"]),
    )


def _environment_spec(low, high, observation_dim = 3) -> EnvironmentSpec:
    return EnvironmentSpec(
        observation_dim = observation_dim,
        action_dim = len(low),
        action_low = np.asarray(low, dtype = np.float32),
        action_high = np.asarray(high, dtype = np.float32),
    )


@pytest.fixture
def spec() -> EnvironmentSpec:
    return _environment_spec([-2.0, 1.0], [4.0, 5.0])


@pytest.fixture
def agent(spec, config) -> TD3Agent:
    keras.utils.set_random_seed(42)
    return TD3Agent(spec, config, seed = 42)


def _network_pairs(agent):
    return (
        (agent._actor, agent._target_actor),
        (agent._critic_1, agent._target_critic_1),
        (agent._critic_2, agent._target_critic_2),
    )


def _optimization_pairs(agent):
    return (
        (agent._actor, agent._actor_optimizer),
        (agent._critic_1, agent._critic_1_optimizer),
        (agent._critic_2, agent._critic_2_optimizer),
    )


def test_agent_initial_state(agent, spec, config):
    assert OffPolicyAgent in TD3Agent.__bases__
    assert agent._spec is spec
    assert agent._config is config
    assert agent._update_count == 0
    assert isinstance(agent._replay_buffer, ReplayBuffer)
    assert len(agent._replay_buffer) == 0
    assert agent._replay_buffer.capacity == config.replay_capacity
    assert agent._replay_buffer.obs_dim == spec.observation_dim
    assert agent._replay_buffer.act_dim == spec.action_dim
    assert agent._replay_buffer.ptr == 0


def test_agent_creates_six_independent_models(agent, spec):
    models = [model for pair in _network_pairs(agent) for model in pair]
    assert len({id(model) for model in models}) == 6
    variables = [variable for model in models for variable in model.weights]
    assert len({id(variable) for variable in variables}) == len(variables)
    for model in models:
        assert isinstance(model, keras.Model)
        assert not model.compiled
        assert model.weights
    for actor in (agent._actor, agent._target_actor):
        assert actor.input_shape == (None, spec.observation_dim)
        assert actor.output_shape == (None, spec.action_dim)
    for critic in models[2:]:
        assert list(critic.input_shape) == [
            (None, spec.observation_dim), (None, spec.action_dim)
        ]
        assert critic.output_shape == (None, 1)


def test_target_weights_are_exact_initial_copies(agent):
    for source, target in _network_pairs(agent):
        for source_weight, target_weight in zip(source.weights, target.weights, strict = True):
            np.testing.assert_array_equal(source_weight.numpy(), target_weight.numpy())
            assert np.all(np.isfinite(source_weight.numpy()))
        original_target = target.weights[0].numpy().copy()
        source.weights[0].assign_add(tf.ones_like(source.weights[0]))
        np.testing.assert_array_equal(target.weights[0].numpy(), original_target)


def test_agent_builds_three_independent_adam_optimizers(agent, config):
    pairs = _optimization_pairs(agent)
    assert len({id(optimizer) for _, optimizer in pairs}) == 3
    all_variables = []
    learning_rates = [config.actor_learning_rate, config.critic_learning_rate,
                      config.critic_learning_rate]
    for (model, optimizer), learning_rate in zip(pairs, learning_rates, strict = True):
        assert isinstance(optimizer, keras.optimizers.Adam)
        assert optimizer.built
        assert int(optimizer.iterations.numpy()) == 0
        assert float(optimizer.learning_rate.numpy()) == pytest.approx(learning_rate)
        # Adam has two moment slots for each trainable variable.
        assert len(optimizer.variables) == 2 + 2 * len(model.trainable_variables)
        for variable in optimizer.variables:
            assert np.all(np.isfinite(variable.numpy()))
        all_variables.extend(optimizer.variables)
    assert len({id(variable) for variable in all_variables}) == len(all_variables)
    for _, target in _network_pairs(agent):
        assert getattr(target, "optimizer", None) is None


@pytest.mark.parametrize("seed", [0, 42, 2 ** 40])
def test_agent_rng_streams_match_spawned_children(spec, config, seed):
    keras.utils.set_random_seed(42)
    agent = TD3Agent(spec, config, seed = seed)
    children = np.random.SeedSequence(seed).spawn(3)
    actual = (agent._replay_buffer._rng, agent._exploration_rng, agent._target_noise_rng)
    expected = (
        np.random.default_rng(int(children[0].generate_state(1)[0])),
        np.random.default_rng(children[1]),
        np.random.default_rng(children[2]),
    )
    assert len({id(rng) for rng in actual}) == 3
    for rng, reference in zip(actual, expected, strict = True):
        assert isinstance(rng, np.random.Generator)
        np.testing.assert_array_equal(rng.random(16), reference.random(16))


def test_rng_streams_do_not_affect_each_other(agent):
    buffer_state = agent._replay_buffer._rng.bit_generator.state
    target_state = agent._target_noise_rng.bit_generator.state
    agent._exploration_rng.normal(size = 32)
    assert agent._replay_buffer._rng.bit_generator.state == buffer_state
    assert agent._target_noise_rng.bit_generator.state == target_state
    exploration_state = agent._exploration_rng.bit_generator.state
    agent._replay_buffer._rng.random(32)
    assert agent._exploration_rng.bit_generator.state == exploration_state
    assert agent._target_noise_rng.bit_generator.state == target_state


def test_agent_does_not_reset_global_seed(spec, config, monkeypatch):
    keras.utils.set_random_seed(42)
    numpy_state = np.random.get_state()

    def unexpected_seed(*args, **kwargs):
        pytest.fail("Agent must not reset global RNG seeds")

    monkeypatch.setattr(keras.utils, "set_random_seed", unexpected_seed)
    monkeypatch.setattr(np.random, "seed", unexpected_seed)
    monkeypatch.setattr(random, "seed", unexpected_seed)
    monkeypatch.setattr(tf.random, "set_seed", unexpected_seed)
    TD3Agent(spec, config, seed = 123)
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


@pytest.mark.parametrize("seed", [-1, True, False, 1.0, "42", None, np.int64(42), np.bool_(True)])
def test_agent_rejects_invalid_seed(spec, config, seed):
    with pytest.raises(AgentInputError):
        TD3Agent(spec, config, seed = seed)


@pytest.mark.parametrize(
    ("low", "high", "midpoint", "half_range"),
    [
        ([-2.0], [2.0], [0.0], [2.0]),
        ([-2.0, 1.0], [4.0, 5.0], [1.0, 3.0], [3.0, 2.0]),
        ([-8.0, 2.0, -5.0], [-2.0, 6.0, 7.0], [-5.0, 4.0, 1.0], [3.0, 2.0, 6.0]),
    ],
)
def test_agent_prepares_float32_scaling(config, low, high, midpoint, half_range):
    keras.utils.set_random_seed(42)
    agent = TD3Agent(_environment_spec(low, high), config, seed = 42)
    for tensor, expected in ((agent._action_midpoint, midpoint),
                             (agent._action_half_range, half_range)):
        assert tf.is_tensor(tensor)
        assert tensor.dtype == tf.float32
        assert tuple(tensor.shape) == (len(low),)
        np.testing.assert_array_equal(tensor.numpy(), expected)
    np.testing.assert_array_equal(agent._action_low, low)
    np.testing.assert_array_equal(agent._action_high, high)


def test_agent_preserves_float32_bounds(config):
    keras.utils.set_random_seed(42)
    spec = _environment_spec([0.1, -0.2], [0.3, 0.2])
    agent = TD3Agent(spec, config, seed = 42)
    for actual, expected in ((agent._action_low, spec.action_low),
                             (agent._action_high, spec.action_high)):
        assert tf.is_tensor(actual)
        assert actual.dtype == tf.float32
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("low,high", [
    ([0.0], [np.nextafter(np.float32(0), np.float32(1))]),
    ([1.0], [np.nextafter(np.float32(1), np.float32(2))]),
    ([0.0, 1.0], [2.0, np.nextafter(np.float32(1), np.float32(2))]),
])
def test_agent_rejects_incompatible_float32_scaling(config, low, high):
    spec = _environment_spec(low, high)
    with pytest.raises(UnsupportedSpaceError):
        TD3Agent(spec, config, seed = 42)


def test_agent_supports_large_finite_symmetric_bounds(config):
    keras.utils.set_random_seed(42)
    limit = float(np.finfo(np.float32).max)
    agent = TD3Agent(_environment_spec([-limit], [limit]), config, seed = 42)
    np.testing.assert_array_equal(agent._action_midpoint.numpy(), [0.0])
    np.testing.assert_array_equal(agent._action_half_range.numpy(), [limit])


@pytest.mark.parametrize("error", [AgentInputError, AgentUpdateError, AgentNumericalError])
def test_agent_exception_hierarchy(error):
    assert error.__bases__ == (AgentSystemError,)
    assert AgentSystemError.__bases__ == (RLSystemError,)
    assert isinstance(error("test"), RLSystemError)
    assert not isinstance(error("test"), UnsupportedSpaceError)


def _set_actor_output(agent, normalized):
    """Set a constant normalized policy using real Actor weights."""
    weights = [np.zeros_like(weight) for weight in agent._actor.get_weights()]
    weights[-1] = np.arctanh(np.asarray(normalized, dtype = np.float32))
    agent._actor.set_weights(weights)


def _learning_state(agent):
    variables = [v for pair in _network_pairs(agent) for model in pair
                 for v in model.weights]
    variables.extend(v for _, optimizer in _optimization_pairs(agent)
                     for v in optimizer.variables)
    return [v.numpy().copy() for v in variables]


def _assert_learning_state_unchanged(agent, before):
    for actual, expected in zip(_learning_state(agent), before, strict = True):
        np.testing.assert_array_equal(actual, expected)
    assert agent._update_count == 0


@pytest.mark.parametrize("low,high,normalized,expected", [
    ([-2.0], [2.0], [0.5], [1.0]),
    ([-2.0, 1.0], [4.0, 5.0], [-0.5, 0.5], [-0.5, 4.0]),
    ([-8.0, 2.0, -5.0], [-2.0, 6.0, 7.0], [0.5, -0.5, 0.0], [-3.5, 3.0, 1.0]),
])
def test_select_action_scales_real_actor(config, low, high, normalized, expected):
    keras.utils.set_random_seed(42)
    agent = TD3Agent(_environment_spec(low, high), config, seed = 42)
    _set_actor_output(agent, normalized)
    action = agent.select_action(np.array([1, 2, 3]), explore = False)
    assert action.shape == (len(low),)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
    np.testing.assert_allclose(action, expected, atol = 1e-6, rtol = 0)
    assert np.all(action >= low) and np.all(action <= high)


def test_scale_action_preserves_gradient(agent):
    normalized = tf.Variable([[0.25, -0.5], [-0.5, 0.25]], dtype = tf.float32)
    with tf.GradientTape() as tape:
        scaled = agent._scale_action(normalized)
        loss = tf.reduce_sum(scaled)
    np.testing.assert_allclose(scaled.numpy(), [[1.75, 2.0], [-0.5, 3.5]])
    np.testing.assert_array_equal(tape.gradient(loss, normalized).numpy(), [[3., 2.], [3., 2.]])


def test_select_action_without_exploration_preserves_state(agent):
    before = _learning_state(agent)
    rngs = (agent._exploration_rng, agent._target_noise_rng, agent._replay_buffer._rng)
    states = [rng.bit_generator.state for rng in rngs]
    obs = np.array([1., 2., 3.], dtype = np.float64)
    first = agent.select_action(obs, explore = False)
    second = agent.select_action(obs, explore = np.bool_(False))
    np.testing.assert_array_equal(first, second)
    assert [rng.bit_generator.state for rng in rngs] == states
    assert len(agent._replay_buffer) == 0
    _assert_learning_state_unchanged(agent, before)


def test_select_action_exploration_matches_independent_rng(agent, config):
    _set_actor_output(agent, [0., 0.])
    reference = np.random.default_rng(np.random.SeedSequence(42).spawn(3)[1])
    target_state = agent._target_noise_rng.bit_generator.state
    buffer_state = agent._replay_buffer._rng.bit_generator.state
    before = _learning_state(agent)
    for _ in range(3):
        noise = reference.normal(0., config.exploration_noise_std, size = (2,))
        expected = np.array([1., 3.]) + np.array([3., 2.]) * np.clip(noise, -1., 1.)
        action = agent.select_action(np.zeros(3), explore = True)
        assert action.dtype == np.float32
        assert action.shape == (2,)
        np.testing.assert_allclose(action, expected, atol = 1e-6, rtol = 0)
    assert agent._exploration_rng.bit_generator.state == reference.bit_generator.state
    assert agent._target_noise_rng.bit_generator.state == target_state
    assert agent._replay_buffer._rng.bit_generator.state == buffer_state
    assert len(agent._replay_buffer) == 0
    _assert_learning_state_unchanged(agent, before)


class _FixedNoise:
    def __init__(self, values, std):
        self.values = np.asarray(values, dtype = np.float64)
        self.std = std

    def normal(self, loc, scale, size):
        assert loc == 0.0
        assert scale == self.std
        assert size == self.values.shape
        return self.values.copy()


def test_exploration_clips_normalized_action_before_scaling(agent, config, monkeypatch):
    _set_actor_output(agent, [0.5, -0.5])
    monkeypatch.setattr(agent, "_exploration_rng", _FixedNoise([[2., -2.]], config.exploration_noise_std))
    original_scale = agent._scale_action

    def checked_scale(action):
        np.testing.assert_array_equal(action.numpy(), [[1., -1.]])
        return original_scale(action)

    monkeypatch.setattr(agent, "_scale_action", checked_scale)
    np.testing.assert_array_equal(agent.select_action(np.zeros(3), explore = True), [4., 1.])


@pytest.mark.parametrize("normalized", [-1., 1.])
def test_select_action_respects_float32_bounds(config, normalized):
    spec = _environment_spec([0.100000003, -0.200000004], [0.300000010, 0.200000004])
    keras.utils.set_random_seed(42)
    agent = TD3Agent(spec, config, seed = 42)
    weights = [np.zeros_like(weight) for weight in agent._actor.get_weights()]
    weights[-1].fill(100. * normalized)
    agent._actor.set_weights(weights)
    action = agent.select_action(np.zeros(3), explore = False)
    assert action.dtype == np.float32
    assert np.all(action >= spec.action_low)
    assert np.all(action <= spec.action_high)
    expected = spec.action_low if normalized < 0 else spec.action_high
    np.testing.assert_allclose(action, expected, rtol = 1e-6, atol = 0)


@pytest.mark.parametrize("observation", [
    1., [1., 2.], [[1., 2., 3.]], [[1.], [2.], [3.]],
    [True, False, True], ["1", "2", "3"], [1j, 2j, 3j],
    [np.nan, 0., 0.], [np.inf, 0., 0.], [1e40, 0., 0.],
    np.array([1, 2, 3], dtype = object),
])
def test_select_action_rejects_invalid_observation(agent, observation):
    state = agent._exploration_rng.bit_generator.state
    with pytest.raises(AgentInputError) as error:
        agent.select_action(observation, explore = True)
    assert isinstance(error.value.__cause__, (TypeError, ValueError, OverflowError))
    assert agent._exploration_rng.bit_generator.state == state


@pytest.mark.parametrize("explore", [0, 1, None, "true", 1.0])
def test_select_action_rejects_invalid_explore(agent, explore):
    with pytest.raises(AgentInputError):
        agent.select_action(np.zeros(3), explore = explore)


def test_select_action_calls_actor_with_batched_float32_inference(agent, monkeypatch):
    def actor(observation, *, training):
        assert observation.dtype == tf.float32
        assert training is False
        np.testing.assert_array_equal(observation.numpy(), [[1., 2., 3.]])
        return tf.constant([[0., 0.]], dtype = tf.float32)

    monkeypatch.setattr(agent, "_actor", actor)
    np.testing.assert_array_equal(agent.select_action([1, 2, 3], explore = False), [1., 3.])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("explore", [False, True])
def test_select_action_rejects_nonfinite_actor_output(agent, monkeypatch, value, explore):
    monkeypatch.setattr(agent, "_actor", lambda *args, **kwargs: tf.constant([[value, 0.]], tf.float32))
    with pytest.raises(AgentNumericalError) as error:
        agent.select_action(np.zeros(3), explore = explore)
    assert isinstance(error.value.__cause__, tf.errors.InvalidArgumentError)


def test_select_action_rejects_nonfinite_exploration(agent, config, monkeypatch):
    monkeypatch.setattr(agent, "_exploration_rng", _FixedNoise([[np.inf, 0.]], config.exploration_noise_std))
    with pytest.raises(AgentNumericalError):
        agent.select_action(np.zeros(3), explore = True)


def test_scale_action_clips_to_original_bounds(agent):
    normalized = tf.constant([[-2., 2.], [2., -2.]], tf.float32)
    np.testing.assert_array_equal(agent._scale_action(normalized).numpy(),
                                  [[-2., 5.], [4., 1.]])


def test_scale_action_corrects_float32_rounding(config):
    spec = _environment_spec([0.1], [0.3])
    agent = TD3Agent(spec, config, seed = 42)
    scaled = agent._scale_action(tf.constant([[-1.], [1.]], tf.float32)).numpy()
    assert np.all(scaled >= spec.action_low)
    assert np.all(scaled <= spec.action_high)
    np.testing.assert_allclose(scaled[:, 0], [0.1, 0.3], rtol = 1e-6)


@pytest.mark.parametrize("terminated,truncated", [(False, False), (True, False), (False, True), (True, True)])
def test_store_transition_preserves_actual_experience(agent, terminated, truncated):
    obs = np.array([1., 2., 3.])
    action = np.array([-1.5, 4.5])
    next_obs = np.array([4., 5., 6.])
    before = _learning_state(agent)
    rngs = (agent._exploration_rng, agent._target_noise_rng, agent._replay_buffer._rng)
    states = [rng.bit_generator.state for rng in rngs]
    assert agent.store_transition(obs, action, -2.5, next_obs, terminated, truncated) is None
    buffer = agent._replay_buffer
    assert len(buffer) == 1
    assert buffer.ptr == 1
    obs.fill(0.)
    action.fill(0.)
    next_obs.fill(0.)
    np.testing.assert_array_equal(buffer.obs[0], [1., 2., 3.])
    np.testing.assert_array_equal(buffer.actions[0], [-1.5, 4.5])
    np.testing.assert_array_equal(buffer.next_obs[0], [4., 5., 6.])
    np.testing.assert_array_equal(buffer.rewards[0], [-2.5])
    np.testing.assert_array_equal(buffer.terminated[0], [terminated])
    np.testing.assert_array_equal(buffer.truncated[0], [truncated])
    assert [rng.bit_generator.state for rng in rngs] == states
    _assert_learning_state_unchanged(agent, before)


@pytest.mark.parametrize("action", [[-2.01, 3.], [4.01, 3.], [0., 0.99], [0., 5.01], [4.00000001, 3.]])
def test_store_transition_rejects_out_of_bounds_action(agent, action):
    with pytest.raises(AgentInputError):
        agent.store_transition(np.zeros(3), action, 1., np.ones(3), False, False)
    assert len(agent._replay_buffer) == 0
    assert agent._replay_buffer.ptr == 0


@pytest.mark.parametrize("field,value", [
    ("obs", [1., 2.]), ("obs", [np.nan, 0., 0.]),
    ("action", [[0., 3.]]), ("action", [np.nan, 3.]),
    ("action", ["0", "3"]), ("reward", np.inf), ("reward", [1.]),
    ("next_obs", [1e40, 0., 0.]), ("terminated", 1), ("truncated", None),
])
def test_store_transition_translates_buffer_validation_errors(agent, field, value):
    kwargs = dict(obs = np.zeros(3), action = [0., 3.], reward = 1., next_obs = np.ones(3),
                  terminated = False, truncated = False)
    kwargs[field] = value
    with pytest.raises(AgentInputError) as error:
        agent.store_transition(**kwargs)
    assert isinstance(error.value.__cause__, (TypeError, ValueError))
    assert len(agent._replay_buffer) == 0
    assert agent._replay_buffer.ptr == 0


def test_store_transition_does_not_mask_unexpected_buffer_error(agent, monkeypatch):
    def broken_add(*args, **kwargs):
        raise RuntimeError("unexpected buffer error")

    monkeypatch.setattr(agent._replay_buffer, "add", broken_add)
    with pytest.raises(RuntimeError, match = "unexpected buffer error"):
        agent.store_transition(np.zeros(3), [0., 3.], 1., np.ones(3), False, False)
