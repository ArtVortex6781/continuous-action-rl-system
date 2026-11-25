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
        action_low = np.asarray(low, dtype = np.float64),
        action_high = np.asarray(high, dtype = np.float64),
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


def test_agent_rounds_float32_bounds_inward(config):
    keras.utils.set_random_seed(42)
    spec = _environment_spec([0.100000003, -0.200000004], [0.300000010, 0.200000004])
    agent = TD3Agent(spec, config, seed = 42)
    assert agent._action_low.dtype == np.float32
    assert agent._action_high.dtype == np.float32
    assert np.all(agent._action_low.astype(np.float64) >= spec.action_low)
    assert np.all(agent._action_high.astype(np.float64) <= spec.action_high)
    assert np.all(agent._action_low < agent._action_high)
    # Inward bounds must not discard any other representable valid action.
    previous = np.nextafter(agent._action_low, np.float32(-np.inf))
    following = np.nextafter(agent._action_high, np.float32(np.inf))
    assert np.all(previous.astype(np.float64) < spec.action_low)
    assert np.all(following.astype(np.float64) > spec.action_high)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ([-1e40], [1.0]),
        ([-1.0], [1e40]),
        ([0.0], [1e-50]),
        ([1.0], [1.0 + 1e-10]),
        ([0.0, 1.0], [2.0, 1.0 + 1e-10]),
        ([1.0], [float(np.nextafter(np.float32(1), np.float32(2)))]),
    ],
)
def test_agent_rejects_incompatible_float32_bounds(config, low, high):
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
