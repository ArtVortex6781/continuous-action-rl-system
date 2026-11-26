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


def test_actor_loss_and_gradient_through_scaling(agent, batch, monkeypatch):
    obs, _, _ = batch
    # Zero Actor parameters give u=(0, 0), hence actions=(1, 3).
    _constant_output(agent._actor, 0.)
    untouched = [*_main_critic_state(agent), *agent._target_actor.weights,
                 *agent._target_critic_1.weights, *agent._target_critic_2.weights]
    before = _snapshot(untouched)
    # Q = 2*a_1 - 3*a_2 + s_1. At u=0, da/du=(3, 2).
    # d(-mean(Q))/d(actor output bias)=(-6, 6), since tanh'(0)=1.
    calls = []

    def critic(inputs, *, training):
        assert training is False
        assert inputs[0] is obs
        np.testing.assert_allclose(inputs[1].numpy(), [[1., 3.]] * 4)
        calls.append(True)
        return 2. * inputs[1][:, :1] - 3. * inputs[1][:, 1:] + obs[:, :1]

    monkeypatch.setattr(agent, "_critic_1", critic)
    original = agent._actor_optimizer.apply_gradients

    def apply(pairs):
        pairs = list(pairs)
        for (gradient, variable), expected in zip(pairs, agent._actor.trainable_variables, strict = True):
            assert variable is expected
            assert gradient is not None
        for gradient, _ in pairs[:-1]:
            np.testing.assert_array_equal(gradient.numpy(), np.zeros(gradient.shape))
        np.testing.assert_allclose(pairs[-1][0].numpy(), [-6., 6.], atol = 1e-6)
        return original(pairs)

    monkeypatch.setattr(agent._actor_optimizer, "apply_gradients", apply)
    loss = agent._update_actor(obs)
    assert loss.shape == ()
    assert float(loss.numpy()) == pytest.approx(7. - np.mean(obs.numpy()[:, 0]))
    assert len(calls) == 1
    assert int(agent._actor_optimizer.iterations.numpy()) == 1
    assert np.all(agent._actor.weights[-1].numpy() * np.array([1., -1.]) > 0)
    _unchanged(untouched, before)
    assert agent._update_count == 0


def test_actor_uses_current_critic_after_critic_update(agent, batch):
    obs, actions, target = batch
    _constant_output(agent._actor, 0.)
    _constant_output(agent._critic_1, 2.)
    _constant_output(agent._critic_2, -1.)
    old_q = agent._critic_1([obs, agent._scale_action(agent._actor(obs))], training = False)
    agent._update_critics(obs, actions, target)
    new_q = agent._critic_1([obs, agent._scale_action(agent._actor(obs))], training = False)
    assert not np.array_equal(old_q.numpy(), new_q.numpy())
    state = _main_critic_state(agent)
    before = _snapshot(state)
    loss = agent._update_actor(obs)
    assert float(loss.numpy()) == pytest.approx(-float(tf.reduce_mean(new_q).numpy()))
    _unchanged(state, before)


def test_scaling_clips_and_preserves_interior_gradient(agent):
    normalized = tf.Variable([[-2., 2.], [0., 0.], [2., -2.]], dtype = tf.float32)
    with tf.GradientTape() as tape:
        scaled = agent._scale_action(normalized)
        loss = tf.reduce_sum(scaled)
    np.testing.assert_allclose(scaled.numpy(), [[-2., 5.], [1., 3.], [4., 1.]])
    np.testing.assert_allclose(tape.gradient(loss, normalized).numpy(),
                               [[0., 0.], [3., 2.], [0., 0.]])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("point", ["output", "loss"])
def test_actor_nonfinite_forward_prevents_update(agent, batch, monkeypatch, point, value):
    obs, _, _ = batch
    state = [*agent._actor.weights, *agent._actor_optimizer.variables]
    before = _snapshot(state)
    if point == "output":
        monkeypatch.setattr(agent, "_actor", lambda *a, **k: tf.fill((4, 2), tf.constant(value, tf.float32)))
    else:
        monkeypatch.setattr(agent, "_critic_1", lambda *a, **k: tf.fill((4, 1), tf.constant(value, tf.float32)))
    with pytest.raises(AgentNumericalError, match = f"Actor {point}"):
        agent._update_actor(obs)
    _unchanged(state, before)


@pytest.mark.parametrize("value", [None, np.nan, np.inf, -np.inf])
def test_actor_invalid_gradient_prevents_update(agent, batch, monkeypatch, value):
    obs, _, _ = batch
    state = [*agent._actor.weights, *agent._actor_optimizer.variables]
    before = _snapshot(state)
    if value is None:
        def critic(inputs, *, training):
            return tf.stop_gradient(tf.reduce_sum(inputs[1], axis = 1, keepdims = True))
    else:
        @tf.custom_gradient
        def bad_gradient(actions):
            def backward(dy):
                return tf.fill(tf.shape(actions), tf.constant(value, tf.float32))

            return tf.reduce_sum(actions, axis = 1, keepdims = True), backward

        def critic(inputs, *, training):
            return bad_gradient(inputs[1])

    monkeypatch.setattr(agent, "_critic_1", critic)
    error = AgentUpdateError if value is None else AgentNumericalError
    with pytest.raises(error, match = "Actor"):
        agent._update_actor(obs)
    _unchanged(state, before)


@pytest.mark.parametrize("component", ["weight", "adam"])
def test_actor_nonfinite_post_update_state_is_fatal(agent, batch, monkeypatch, component):
    obs, _, _ = batch
    optimizer = agent._actor_optimizer
    original = optimizer.apply_gradients
    untouched = [*_main_critic_state(agent), *agent._target_actor.weights,
                 *agent._target_critic_1.weights, *agent._target_critic_2.weights]
    before = _snapshot(untouched)

    def apply(pairs):
        result = original(pairs)
        variable = agent._actor.weights[-1] if component == "weight" else optimizer.variables[-1]
        variable.assign(tf.fill(variable.shape, tf.constant(np.inf, tf.float32)))
        return result

    monkeypatch.setattr(optimizer, "apply_gradients", apply)
    with pytest.raises(AgentNumericalError, match = "Actor"):
        agent._update_actor(obs)
    assert int(optimizer.iterations.numpy()) == 1
    _unchanged(untouched, before)


@pytest.mark.parametrize("tau", [1e-6, 0.005, 0.25, 1.])
def test_soft_updates_all_targets(agent, tau, monkeypatch):
    monkeypatch.setattr(agent, "_config", agent._config.model_copy(update = {"tau": tau}))
    pairs = [(agent._actor, agent._target_actor),
             (agent._critic_1, agent._target_critic_1),
             (agent._critic_2, agent._target_critic_2)]
    expected = []
    for index, (source, target) in enumerate(pairs, start = 1):
        for j, (s, t) in enumerate(zip(source.weights, target.weights, strict = True)):
            s.assign(tf.fill(s.shape, tf.constant(float(index + j), tf.float32)))
            t.assign(tf.fill(t.shape, tf.constant(float(-index - j), tf.float32)))
            assert s is not t
        expected.append([(1. - tau) * t + tau * s for s, t in
                         zip(_snapshot(source.weights), _snapshot(target.weights), strict = True)])
    untouched = [v for source, _ in pairs for v in source.weights]
    untouched += [*agent._actor_optimizer.variables, *agent._critic_1_optimizer.variables,
                  *agent._critic_2_optimizer.variables]
    before = _snapshot(untouched)
    agent._update_target_networks()
    for (source, target), values in zip(pairs, expected, strict = True):
        for s, t, value in zip(source.weights, target.weights, values, strict = True):
            np.testing.assert_allclose(t.numpy(), value, rtol = 1e-6, atol = 1e-7)
            if tau == 1.:
                np.testing.assert_array_equal(t.numpy(), s.numpy())
            assert t is not s
    _unchanged(untouched, before)
    assert agent._update_count == 0


# Public update coordination.


def _fill_replay(agent, count = 4):
    for i in range(count):
        obs = np.array([i + 1., 0.5, -0.25], dtype = np.float32)
        action = np.array([1., 3.], dtype = np.float32)
        agent.store_transition(obs, action, float(i + 1), obs + 0.5,
                               i % 4 in (2, 3), i % 4 in (1, 3))


def _target_state(agent):
    return [*agent._target_actor.weights, *agent._target_critic_1.weights,
            *agent._target_critic_2.weights]


def _actor_state(agent):
    return [*agent._actor.weights, *agent._actor_optimizer.variables]


def _learning_state(agent):
    return [*_main_critic_state(agent), *_actor_state(agent), *_target_state(agent)]


def _rng_states(agent):
    return [rng.bit_generator.state for rng in
            (agent._replay_buffer._rng, agent._exploration_rng, agent._target_noise_rng)]


def _replay_state(agent):
    buffer = agent._replay_buffer
    return (buffer.ptr, buffer.size, [getattr(buffer, name)[:buffer.size].copy()
                                      for name in ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated')])


def _assert_replay_unchanged(agent, before):
    after = _replay_state(agent)
    assert after[:2] == before[:2]
    for actual, expected in zip(after[2], before[2], strict = True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('count', [0, 3])
def test_update_not_ready_is_noop(agent, monkeypatch, count):
    _fill_replay(agent, count)
    state = _learning_state(agent)
    before = _snapshot(state)
    rng_before = _rng_states(agent)
    replay_before = _replay_state(agent)

    def unexpected(*args, **kwargs):
        pytest.fail('An unready update must not sample or compute')

    monkeypatch.setattr(agent._replay_buffer, 'sample', unexpected)
    monkeypatch.setattr(agent, '_compute_bellman_target', unexpected)
    for _ in range(2):
        assert agent.update() is None
    assert agent._update_count == 0
    _unchanged(state, before)
    assert _rng_states(agent) == rng_before
    _assert_replay_unchanged(agent, replay_before)


@pytest.mark.parametrize('delay', [1, 2])
def test_update_schedule_and_metrics(agent, monkeypatch, delay):
    monkeypatch.setattr(agent, '_config', agent._config.model_copy(update = {'policy_delay': delay}))
    _fill_replay(agent)  # Exactly batch_size: first update must be ready.
    # Ensure the Critics have nonzero bias gradients on the first update.
    for model in (agent._critic_1, agent._critic_2, agent._target_critic_1, agent._target_critic_2):
        _constant_output(model, 0.)
    original_sample = agent._replay_buffer.sample
    original_target = agent._compute_bellman_target
    original_critics = agent._update_critics
    original_actor = agent._update_actor
    original_targets = agent._update_target_networks
    calls = []
    captured = {}

    def sample(size):
        calls.append('sample')
        assert size == agent._config.batch_size
        captured['batch'] = original_sample(size)
        return captured['batch']

    def target(next_obs, rewards, terminated):
        calls.append('target')
        replay = captured['batch']
        for tensor, expected in ((next_obs, replay.next_obs), (rewards, replay.rewards),
                                 (terminated, replay.terminated.astype(np.float32))):
            assert tensor.dtype == tf.float32
            np.testing.assert_array_equal(tensor.numpy(), expected)
        captured['target'] = original_target(next_obs, rewards, terminated)
        return captured['target']

    def critics(obs, actions, target):
        calls.append('critics')
        assert agent._update_count == step - 1
        assert target is captured['target']
        for tensor, expected in ((obs, captured['batch'].obs), (actions, captured['batch'].actions)):
            assert tensor.dtype == tf.float32
            np.testing.assert_array_equal(tensor.numpy(), expected)
        captured['obs'] = obs
        losses = original_critics(obs, actions, target)
        captured['losses'] = {'critic_1_loss': float(losses[0].numpy()),
                              'critic_2_loss': float(losses[1].numpy())}
        return losses

    def actor(obs):
        calls.append('actor')
        assert obs is captured['obs']
        assert agent._update_count == step
        assert int(agent._critic_1_optimizer.iterations.numpy()) == step
        loss = original_actor(obs)
        captured['losses']['actor_loss'] = float(loss.numpy())
        return loss

    def targets():
        calls.append('targets')
        assert int(agent._actor_optimizer.iterations.numpy()) == step // delay
        original_targets()

    monkeypatch.setattr(agent._replay_buffer, 'sample', sample)
    monkeypatch.setattr(agent, '_compute_bellman_target', target)
    monkeypatch.setattr(agent, '_update_critics', critics)
    monkeypatch.setattr(agent, '_update_actor', actor)
    monkeypatch.setattr(agent, '_update_target_networks', targets)
    replay_before = _replay_state(agent)
    exploration_before = agent._exploration_rng.bit_generator.state
    for step in range(1, 5):
        calls.clear()
        actor_before = _snapshot(_actor_state(agent))
        targets_before = _snapshot(_target_state(agent))
        metrics = agent.update()
        due = step % delay == 0
        assert calls == ['sample', 'target', 'critics'] + (['actor', 'targets'] if due else [])
        assert metrics == captured['losses']
        assert set(metrics) == {'critic_1_loss', 'critic_2_loss'} | ({'actor_loss'} if due else set())
        assert all(type(v) is float and np.isfinite(v) for v in metrics.values())
        assert agent._update_count == step
        assert int(agent._critic_1_optimizer.iterations.numpy()) == step
        assert int(agent._critic_2_optimizer.iterations.numpy()) == step
        assert int(agent._actor_optimizer.iterations.numpy()) == step // delay
        if due:
            assert any(not np.array_equal(v.numpy(), old) for v, old in
                       zip(_target_state(agent), targets_before, strict = True))
        else:
            _unchanged(_actor_state(agent), actor_before)
            _unchanged(_target_state(agent), targets_before)
        assert agent._exploration_rng.bit_generator.state == exploration_before
        _assert_replay_unchanged(agent, replay_before)


def test_update_reproducible_with_independent_equal_seeds(agent):
    keras.utils.set_random_seed(42)
    other = TD3Agent(agent._spec, agent._config, seed = 42)
    _fill_replay(agent, 8)
    _fill_replay(other, 8)
    for _ in range(4):
        a = agent.update()
        b = other.update()
        assert a.keys() == b.keys()
        assert a == pytest.approx(b, rel = 1e-6, abs = 1e-7)
        for left, right in zip(_learning_state(agent), _learning_state(other), strict = True):
            np.testing.assert_allclose(left.numpy(), right.numpy(), rtol = 1e-6, atol = 1e-7)
        assert _rng_states(agent) == _rng_states(other)
        assert agent._update_count == other._update_count


def test_update_samples_and_draws_noise_once(agent):
    _fill_replay(agent, 8)
    reference = np.random.SeedSequence(42).spawn(3)
    buffer_rng = np.random.default_rng(int(reference[0].generate_state(1)[0]))
    noise_rng = np.random.default_rng(reference[2])
    exploration_before = agent._exploration_rng.bit_generator.state
    for _ in range(4):
        buffer_rng.choice(8, size = 4, replace = False)
        noise_rng.normal(0., 0.2, size = (4, 2))
        assert agent.update() is not None
        assert agent._replay_buffer._rng.bit_generator.state == buffer_rng.bit_generator.state
        assert agent._target_noise_rng.bit_generator.state == noise_rng.bit_generator.state
        assert agent._exploration_rng.bit_generator.state == exploration_before


@pytest.mark.parametrize('stage', ['target', 'critic_2', 'actor', 'targets'])
def test_update_failure_is_fatal_and_preserves_completed_work(agent, monkeypatch, stage):
    monkeypatch.setattr(agent, '_config', agent._config.model_copy(update = {'policy_delay': 1}))
    _fill_replay(agent)
    error = AgentNumericalError('injected numerical failure')
    targets_before = _snapshot(_target_state(agent))
    actor_before = _snapshot(_actor_state(agent))
    critics_before = _snapshot(_main_critic_state(agent))

    def fail(*args, **kwargs):
        raise error

    if stage == 'target':
        monkeypatch.setattr(agent, '_compute_bellman_target', fail)
    elif stage == 'critic_2':
        # The first Adam has already succeeded when the second raises.
        monkeypatch.setattr(agent._critic_2_optimizer, 'apply_gradients', fail)
    elif stage == 'actor':
        monkeypatch.setattr(agent, '_update_actor', fail)
    else:
        monkeypatch.setattr(agent, '_update_target_networks', fail)

    with pytest.raises(AgentNumericalError) as caught:
        agent.update()
    assert caught.value is error
    assert agent._update_count == int(stage in ('actor', 'targets'))
    assert int(agent._critic_1_optimizer.iterations.numpy()) == int(stage != 'target')
    assert int(agent._critic_2_optimizer.iterations.numpy()) == int(stage in ('actor', 'targets'))
    assert int(agent._actor_optimizer.iterations.numpy()) == int(stage == 'targets')
    _unchanged(_target_state(agent), targets_before)
    if stage != 'targets':
        _unchanged(_actor_state(agent), actor_before)
    if stage == 'target':
        _unchanged(_main_critic_state(agent), critics_before)


def test_update_propagates_unexpected_exception(agent, monkeypatch):
    _fill_replay(agent)
    error = RuntimeError('unexpected sampling failure')

    def fail(size):
        raise error

    monkeypatch.setattr(agent._replay_buffer, 'sample', fail)
    with pytest.raises(RuntimeError) as caught:
        agent.update()
    assert caught.value is error
    assert agent._update_count == 0
