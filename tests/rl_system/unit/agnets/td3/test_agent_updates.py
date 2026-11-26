"""Mathematical and component tests for TD3 learning operations."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from rl_system.agents.td3.agent import TD3Agent
from rl_system.agents.td3.config import MLPConfig, TD3Config
from rl_system.environment import EnvironmentSpec
from rl_system.exceptions import AgentNumericalError, AgentUpdateError


@pytest.fixture
def agent():
    keras.utils.set_random_seed(42)
    spec = EnvironmentSpec(3, 2, np.array([-2., 1.], np.float32),
                           np.array([4., 5.], np.float32))
    mlp = MLPConfig(hidden_units = [8, 8], hidden_activations = ["relu", "relu"])
    config = TD3Config(
        replay_capacity = 16, batch_size = 4, gamma = 0.9, tau = 0.005,
        actor_learning_rate = 0.001, critic_learning_rate = 0.002,
        policy_delay = 2, exploration_noise_std = 0.1,
        target_policy_noise_std = 0.2, target_noise_clip = 0.5,
        actor_output_init_limit = 0.003, actor = mlp, critic = mlp,
    )
    return TD3Agent(spec, config, seed = 42)


@pytest.fixture
def batch():
    return (tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 1., 2.]]),
            tf.constant([[1., 3.], [-1., 4.], [2., 2.], [4., 5.]]),
            tf.constant([[1.], [2.], [-1.], [3.]]))


class _FixedNoise:
    def __init__(self, values):
        self.values = np.array(values, dtype = np.float64)
        self.calls = 0

    def normal(self, loc, scale, size):
        assert loc == 0.
        assert scale == 0.2
        assert size == self.values.shape
        self.calls += 1
        return self.values.copy()


def _constant_output(model, value):
    weights = [np.zeros_like(w) for w in model.get_weights()]
    weights[-1].fill(value)
    model.set_weights(weights)


def _snapshot(variables):
    return [v.numpy().copy() for v in variables]


def _unchanged(variables, before):
    for variable, expected in zip(variables, before, strict = True):
        np.testing.assert_array_equal(variable.numpy(), expected)


def _main_critic_state(agent):
    return [*agent._critic_1.weights, *agent._critic_2.weights,
            *agent._critic_1_optimizer.variables, *agent._critic_2_optimizer.variables]


def test_bellman_target_smoothing_minimum_and_terminal_mask(agent, batch, monkeypatch):
    obs, _, rewards = batch
    terminated = tf.constant([[0.], [1.], [0.], [1.]])
    normalized = tf.constant([[0.8, -0.8], [0., 0.], [-0.9, 0.9], [0.2, -0.2]])
    noise = _FixedNoise([[2., -2.], [0.25, -0.25], [-2., 2.], [-0.1, 0.1]])
    monkeypatch.setattr(agent, "_target_noise_rng", noise)
    expected_actions = [[4., 1.], [1.75, 2.5], [-2., 5.], [1.3, 2.8]]

    def actor(inputs, *, training):
        assert training is False
        assert inputs is obs
        return normalized

    def critic(values):
        def forward(inputs, *, training):
            assert training is False
            assert inputs[0] is obs
            np.testing.assert_allclose(inputs[1].numpy(), expected_actions, atol = 1e-6)
            return tf.constant(values, tf.float32)

        return forward

    monkeypatch.setattr(agent, "_target_actor", actor)
    monkeypatch.setattr(agent, "_target_critic_1", critic([[5.], [8.], [-2.], [4.]]))
    monkeypatch.setattr(agent, "_target_critic_2", critic([[3.], [6.], [4.], [-3.]]))
    result = agent._compute_bellman_target(obs, rewards, terminated)
    assert result.shape == (4, 1)
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result.numpy(), [[3.7], [2.], [-2.8], [3.]], atol = 1e-6)
    assert noise.calls == 1


def test_bellman_target_bootstraps_truncated_transitions(agent, batch):
    obs, actions, _ = batch
    for index, (terminated, truncated) in enumerate([(False, False), (False, True),
                                                     (True, False), (True, True)]):
        agent.store_transition(obs[index].numpy(), actions[index].numpy(), 1.,
                               obs[index].numpy(), terminated, truncated)
    replay = agent._replay_buffer.sample(4)
    _constant_output(agent._target_critic_1, 2.)
    _constant_output(agent._target_critic_2, 3.)
    result = agent._compute_bellman_target(tf.constant(replay.next_obs),
                                           tf.constant(replay.rewards),
                                           tf.cast(replay.terminated, tf.float32))
    expected = np.where(replay.terminated, 1., 2.8)
    np.testing.assert_allclose(result.numpy(), expected, atol = 1e-6)
    assert np.any(replay.truncated & ~replay.terminated)


def test_bellman_target_uses_independent_rng_and_is_detached(agent, batch):
    obs, _, rewards = batch
    other_states = [r.bit_generator.state for r in
                    (agent._exploration_rng, agent._replay_buffer._rng)]
    reference = np.random.default_rng(np.random.SeedSequence(42).spawn(3)[2])
    reference.normal(0., 0.2, size = (4, 2))
    with tf.GradientTape() as tape:
        result = agent._compute_bellman_target(obs, rewards, tf.zeros((4, 1)))
        loss = tf.reduce_sum(result)
    variables = [v for model in (agent._target_actor, agent._target_critic_1,
                                 agent._target_critic_2) for v in model.trainable_variables]
    assert all(g is None for g in tape.gradient(loss, variables))
    assert agent._target_noise_rng.bit_generator.state == reference.bit_generator.state
    assert [r.bit_generator.state for r in
            (agent._exploration_rng, agent._replay_buffer._rng)] == other_states


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("point", ["noise", "actor", "critic_1", "critic_2"])
def test_bellman_target_rejects_nonfinite_before_masking(agent, batch, monkeypatch, point, value):
    obs, _, rewards = batch
    if point == "noise":
        monkeypatch.setattr(agent, "_target_noise_rng", _FixedNoise(np.full((4, 2), value)))
    elif point == "actor":
        monkeypatch.setattr(agent, "_target_actor", lambda *a, **k: tf.fill((4, 2), tf.constant(value, tf.float32)))
    else:
        monkeypatch.setattr(agent, "_target_" + point, lambda *a, **k: tf.fill((4, 1), tf.constant(value, tf.float32)))
    with pytest.raises(AgentNumericalError) as error:
        agent._compute_bellman_target(obs, rewards, tf.ones((4, 1)))
    assert isinstance(error.value.__cause__, tf.errors.InvalidArgumentError)


def test_bellman_target_rejects_overflow_of_finite_operands(agent, batch):
    obs, _, _ = batch
    limit = np.finfo(np.float32).max
    _constant_output(agent._target_critic_1, limit)
    _constant_output(agent._target_critic_2, limit)
    with pytest.raises(AgentNumericalError, match = "Bellman target"):
        agent._compute_bellman_target(obs, tf.fill((4, 1), tf.constant(limit)), tf.zeros((4, 1)))


@pytest.mark.parametrize("index,bias", [(1, 2.), (2, -1.)])
def test_critic_mse_and_known_gradients(agent, batch, index, bias):
    obs, actions, target = batch
    critic = getattr(agent, f"_critic_{index}")
    _constant_output(critic, bias)
    before = _snapshot(critic.weights)
    loss, gradients = agent._compute_critic_gradients(critic, obs, actions, target, name = f"Critic {index}")
    residual = bias - target.numpy()
    assert loss.shape == ()
    assert float(loss.numpy()) == pytest.approx(np.mean(residual ** 2))
    for gradient, variable in zip(gradients, critic.trainable_variables, strict = True):
        assert gradient is not None
        assert gradient.shape == variable.shape
        assert np.all(np.isfinite(gradient.numpy()))
    for gradient in gradients[:-1]:
        np.testing.assert_array_equal(gradient.numpy(), np.zeros(gradient.shape))
    np.testing.assert_allclose(gradients[-1].numpy(), [2 * np.mean(residual)])
    _unchanged(critic.weights, before)


def test_critic_receives_actual_actions_and_training_flag(agent, batch, monkeypatch):
    obs, actions, target = batch
    critic = agent._critic_1
    original_call = critic.call
    calls = []

    def call(inputs, training = None, **kwargs):
        assert training is True
        np.testing.assert_array_equal(inputs[0].numpy(), obs.numpy())
        np.testing.assert_array_equal(inputs[1].numpy(), actions.numpy())
        calls.append(True)
        return original_call(inputs, training = training, **kwargs)

    monkeypatch.setattr(critic, "call", call)
    agent._compute_critic_gradients(critic, obs, actions, target, name = "Critic 1")
    assert calls


def test_update_critics_updates_only_own_networks_and_adam(agent, batch):
    obs, actions, target = batch
    _constant_output(agent._critic_1, 2.)
    _constant_output(agent._critic_2, -1.)
    untouched = [*agent._actor.weights, *agent._actor_optimizer.variables,
                 *agent._target_actor.weights, *agent._target_critic_1.weights,
                 *agent._target_critic_2.weights]
    before = _snapshot(untouched)
    biases = [model.weights[-1].numpy().copy() for model in (agent._critic_1, agent._critic_2)]
    losses = agent._update_critics(obs, actions, target)
    np.testing.assert_allclose([float(loss.numpy()) for loss in losses],
                               [np.mean((2. - target.numpy()) ** 2),
                                np.mean((-1. - target.numpy()) ** 2)])
    for model, optimizer, bias in zip((agent._critic_1, agent._critic_2),
                                      (agent._critic_1_optimizer, agent._critic_2_optimizer), biases, strict = True):
        assert int(optimizer.iterations.numpy()) == 1
        assert not np.array_equal(model.weights[-1].numpy(), bias)
        assert all(np.all(np.isfinite(v.numpy())) for v in [*model.weights, *optimizer.variables])
    _unchanged(untouched, before)
    assert agent._update_count == 0


@pytest.mark.parametrize("index", [1, 2])
def test_nonfinite_critic_loss_prevents_both_updates(agent, batch, index):
    _constant_output(getattr(agent, f"_critic_{index}"), np.finfo(np.float32).max)
    state = _main_critic_state(agent)
    before = _snapshot(state)
    with pytest.raises(AgentNumericalError, match = f"Critic {index} loss"):
        agent._update_critics(*batch)
    _unchanged(state, before)


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_invalid_gradient_prevents_both_updates(agent, batch, monkeypatch, index, value):
    original = agent._compute_critic_gradients

    def compute(critic, *args, name):
        loss, gradients = original(critic, *args, name = name)
        if name == f"Critic {index}":
            gradients[-1] = None if value is None else tf.fill(gradients[-1].shape, tf.constant(value, tf.float32))
        return loss, gradients

    monkeypatch.setattr(agent, "_compute_critic_gradients", compute)
    state = _main_critic_state(agent)
    before = _snapshot(state)
    error = AgentUpdateError if value is None else AgentNumericalError
    with pytest.raises(error, match = f"Critic {index}"):
        agent._update_critics(*batch)
    _unchanged(state, before)


@pytest.mark.parametrize("index", [1, 2])
@pytest.mark.parametrize("component", ["weight", "adam"])
def test_nonfinite_post_update_state_is_fatal(agent, batch, monkeypatch, index, component):
    critic = getattr(agent, f"_critic_{index}")
    optimizer = getattr(agent, f"_critic_{index}_optimizer")
    original = optimizer.apply_gradients

    def apply(gradients):
        result = original(gradients)
        variable = critic.weights[-1] if component == "weight" else optimizer.variables[-1]
        variable.assign(tf.fill(variable.shape, tf.constant(np.inf, tf.float32)))
        return result

    monkeypatch.setattr(optimizer, "apply_gradients", apply)
    with pytest.raises(AgentNumericalError, match = f"Critic {index}"):
        agent._update_critics(*batch)
    assert int(agent._critic_1_optimizer.iterations.numpy()) == 1
    assert int(agent._critic_2_optimizer.iterations.numpy()) == (index == 2)
    assert agent._update_count == 0
