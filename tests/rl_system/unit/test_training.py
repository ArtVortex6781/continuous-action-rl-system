"""Component tests for the sequential Trainer (subphase 1.6, step 2)."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import rl_system.training as training_module
from rl_system.config import ProgressConfig, TrainingConfig
from rl_system.exceptions import (
    AgentUpdateError,
    RLSystemError,
    TrainingError,
    TrainingInterruptedError,
)
from rl_system.training import Trainer


class _Trace:
    """Record calls and optionally fail on a specific invocation of an operation."""

    def __init__(self, failure = None, error = None):
        self.events = []
        self.counts = Counter()
        self.failure = failure
        self.error = error

    def record(self, operation, *args):
        self.events.append((operation, *args))
        self.counts[operation] += 1
        if (operation, self.counts[operation]) == self.failure:
            raise self.error


class _ActionSpace:
    def __init__(self, trace):
        self.trace = trace
        self.actions = []

    def seed(self, seed):
        self.trace.record("seed", seed)

    def sample(self):
        self.trace.record("sample")
        action = np.array([-0.5], dtype = np.float32)
        self.actions.append(action)
        return action


class _Environment:
    """Return a fresh observation per step, with configurable episode boundaries."""

    def __init__(self, trace, endings = None):
        self.trace = trace
        self.action_space = _ActionSpace(trace)
        self.endings = {} if endings is None else endings
        self.steps = 0
        self.initial_observations = []
        self.transitions = []
        self.actions = []
        self.closed = False

    def reset(self, *, seed = None):
        self.trace.record("reset", seed)
        obs = np.array([-len(self.initial_observations) - 1], dtype = np.float32)
        self.initial_observations.append(obs)
        return obs, {"ignored": True}

    def step(self, action):
        self.trace.record("step")
        self.steps += 1
        self.actions.append(action)
        next_obs = np.array([self.steps], dtype = np.float32)
        reward = np.float32(self.steps / 2)
        terminated, truncated = self.endings.get(self.steps, (False, False))
        transition = (next_obs, reward, terminated, truncated)
        self.transitions.append(transition)
        return (*transition, {"ignored": True})

    def close(self):
        self.trace.record("close")
        self.closed = True


class _Agent:
    """Record public Agent calls without exposing any implementation-specific state."""

    def __init__(self, trace, update_results = None):
        self.trace = trace
        self.selections = []
        self.actions = []
        self.transitions = []
        self.update_results = update_results or {}
        self.update_steps = []
        self.returned_metrics = []

    def select_action(self, observation, *, explore):
        self.trace.record("select", explore)
        self.selections.append((observation, explore))
        action = np.array([0.5], dtype = np.float32)
        self.actions.append(action)
        return action

    def store_transition(self, obs, action, reward, next_obs, terminated, truncated):
        self.trace.record("store")
        self.transitions.append((obs, action, reward, next_obs, terminated, truncated))

    def update(self):
        self.trace.record("update")
        self.update_steps.append(len(self.transitions))
        call = len(self.update_steps)
        metrics = self.update_results.get(call, {"loss": float(call)})
        self.returned_metrics.append(metrics)
        return metrics


def _config(**overrides):
    data = {
        "total_steps": 6,
        "seed": 42,
        "warmup_steps": 0,
        "learning_starts": 1,
        "train_frequency": 1,
        "gradient_steps": 1,
        "progress": ProgressConfig(
            step_interval = 0,
            episode_interval = 0,
            episode_start = 1,
            reward_window = 0,
            show_update_count = False,
            show_update_metrics = False,
            show_metric_counts = False,
            show_throughput = False,
        ),
    }
    data.update(overrides)
    return TrainingConfig(**data)


def _make_trainer(*, endings = None, update_results = None, failure = None, error = None, **config):
    trace = _Trace(failure, error)
    env = _Environment(trace, endings)
    agent = _Agent(trace, update_results)
    return Trainer(env, agent, _config(**config)), env, agent, trace


def _episode(number, step, reward, length, terminated = False, truncated = False):
    return {
        "episode": number,
        "global_step": step,
        "episode_reward": reward,
        "episode_length": length,
        "terminated": terminated,
        "truncated": truncated,
    }


@pytest.mark.parametrize("total_steps,warmup_steps", [(1, 0), (1, 1), (6, 0), (6, 3), (6, 6)])
def test_step_budget_and_warmup(total_steps, warmup_steps):
    trainer, env, agent, trace = _make_trainer(
        total_steps = total_steps, warmup_steps = warmup_steps, learning_starts = total_steps,
    )

    result = trainer.train()

    assert result["environment_steps"] == total_steps
    assert env.steps == len(agent.transitions) == total_steps
    assert trace.counts["sample"] == warmup_steps
    assert trace.counts["select"] == total_steps - warmup_steps
    assert [name for name, *_ in trace.events if name in ("sample", "select")] == (
            ["sample"] * warmup_steps + ["select"] * (total_steps - warmup_steps)
    )
    assert all(explore is True for _, explore in agent.selections)
    assert len(env.initial_observations) == 1
    assert agent.update_steps == [total_steps]  # The last step is not skipped.
    assert result["update_count"] == 1
    assert result["episode_count"] == 0
    assert result["partial_episode"] == _episode(
        1, total_steps, total_steps * (total_steps + 1) / 4, total_steps,
    )
    assert not env.closed


def test_update_schedule_is_global_and_relative_to_learning_starts():
    trainer, _, agent, _ = _make_trainer(
        total_steps = 9, warmup_steps = 6, learning_starts = 3, train_frequency = 2,
        gradient_steps = 2, endings = {4: (True, False), 7: (False, True)},
    )

    result = trainer.train()

    expected_steps = [3, 3, 5, 5, 7, 7, 9, 9]
    assert agent.update_steps == expected_steps
    assert result["update_count"] == len(expected_steps)
    assert result["update_metrics"] == [
        {"global_step": step, "update": number, "metrics": {"loss": float(number)}}
        for number, step in enumerate(expected_steps, 1)
    ]


def test_none_stops_only_the_current_update_series():
    trainer, _, agent, _ = _make_trainer(
        total_steps = 7, learning_starts = 1, train_frequency = 2, gradient_steps = 3,
        update_results = {1: None, 3: None},
    )

    result = trainer.train()

    # Step 1: None immediately; step 3: one update then None;
    # steps 5 and 7: all three requests succeed.
    assert agent.update_steps == [1, 3, 3, 5, 5, 5, 7, 7, 7]
    assert result["update_count"] == 7
    assert result["update_metrics"] == [
        {"global_step": step, "update": number, "metrics": {"loss": float(call)}}
        for number, (step, call) in enumerate(
            [(3, 2), (5, 4), (5, 5), (5, 6), (7, 7), (7, 8), (7, 9)], 1,
        )
    ]


def test_transition_data_and_call_order_across_episode_boundary():
    trainer, env, agent, trace = _make_trainer(
        total_steps = 3, warmup_steps = 1, endings = {2: (False, True)},
    )

    trainer.train()

    assert [event[0] for event in trace.events] == [
        "seed", "reset",
        "sample", "step", "store", "update",
        "select", "step", "store", "update", "reset",
        "select", "step", "store", "update",
    ]
    observations = [env.initial_observations[0], env.transitions[0][0], env.initial_observations[1]]
    actions = [env.action_space.actions[0], *agent.actions]
    for stored, obs, action, actual, sent in zip(
            agent.transitions, observations, actions, env.transitions, env.actions, strict = True,
    ):
        next_obs, reward, terminated, truncated = actual
        assert stored[0] is obs
        assert stored[1] is action is sent
        assert stored[2] is reward
        assert stored[3] is next_obs
        assert stored[4:] == (terminated, truncated)
    assert agent.selections[0][0] is observations[1]
    assert agent.selections[1][0] is observations[2]


def test_episode_history_and_partial_episode():
    trainer, env, agent, _ = _make_trainer(
        total_steps = 9, endings = {2: (True, False), 5: (False, True), 7: (True, True)},
    )

    result = trainer.train()

    assert result["episode_metrics"] == [
        _episode(1, 2, 1.5, 2, terminated = True),
        _episode(2, 5, 6.0, 3, truncated = True),
        _episode(3, 7, 6.5, 2, terminated = True, truncated = True),
    ]
    assert result["episode_count"] == 3
    assert result["partial_episode"] == _episode(4, 9, 8.5, 2)
    assert len(env.initial_observations) == 4
    assert agent.transitions[-1][4:] == (False, False)  # No artificial truncation.
    assert result["environment_steps"] == 9
    assert sum(ep["episode_length"] for ep in result["episode_metrics"]) + result["partial_episode"][
        "episode_length"] == 9


@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True)])
def test_last_step_finishes_episode_without_extra_reset(flags):
    trainer, env, agent, trace = _make_trainer(total_steps = 2, endings = {2: flags})

    result = trainer.train()

    assert result["episode_metrics"] == [_episode(1, 2, 1.5, 2, *flags)]
    assert result["episode_count"] == 1
    assert result["partial_episode"] is None
    assert len(env.initial_observations) == 1
    assert trace.counts["reset"] == 1
    assert agent.update_steps == [1, 2]


def test_sparse_algorithm_metrics_are_preserved_and_json_serializable():
    metrics = [{"custom": 1.25}, {"custom": -2.0, "optional": 3.5}, {"other": 0.0}, {}]
    trainer, _, _, _ = _make_trainer(
        total_steps = 4, endings = {2: (True, False)},
        update_results = dict(enumerate(metrics, 1)),
    )

    result = trainer.train()

    assert result["update_metrics"] == [
        {"global_step": number, "update": number, "metrics": value}
        for number, value in enumerate(metrics, 1)
    ]
    assert result["update_count"] == 4  # An empty dict is not None.
    assert type(result["episode_metrics"][0]["episode_reward"]) is float
    assert type(result["partial_episode"]["episode_reward"]) is float
    assert json.loads(json.dumps(result, allow_nan = False)) == result


def test_no_successful_updates():
    trainer, _, agent, _ = _make_trainer(
        total_steps = 3, gradient_steps = 4, update_results = {1: None, 2: None, 3: None},
    )

    result = trainer.train()

    assert agent.update_steps == [1, 2, 3]
    assert result["update_count"] == 0
    assert result["update_metrics"] == []
    assert result["environment_steps"] == 3


@pytest.mark.parametrize("seed", [0, 42, 2 ** 32 - 1])
def test_seeding_once_and_resets_without_reseeding(seed):
    histories = []
    for _ in range(2):
        trainer, _, _, trace = _make_trainer(
            seed = seed, total_steps = 5, endings = {2: (True, False), 4: (False, True)},
        )
        trainer.train()
        calls = [event for event in trace.events if event[0] in ("seed", "reset")]
        assert calls[1:] == [("reset", seed), ("reset", None), ("reset", None)]
        assert calls[0][0] == "seed"
        action_seed = calls[0][1]
        assert type(action_seed) is int
        assert 0 <= action_seed <= 2 ** 32 - 1
        assert action_seed != seed
        histories.append(calls)
    assert histories[0] == histories[1]


@pytest.mark.parametrize("duration,expected_rates", [(4.0, (1.0, 0.5)), (0.0, (None, None))])
def test_duration_and_overall_throughput(monkeypatch, duration, expected_rates):
    times = iter([100.0, 100.0 + duration])
    monkeypatch.setattr(training_module, "time", SimpleNamespace(perf_counter = lambda: next(times)))
    trainer, _, _, _ = _make_trainer(total_steps = 4, learning_starts = 2, train_frequency = 2)

    result = trainer.train()

    assert result["duration_seconds"] == duration
    assert result["environment_steps"] == 4
    assert result["update_count"] == 2
    assert (result["steps_per_second"], result["updates_per_second"]) == expected_rates


def test_trainer_is_single_use_after_success():
    trainer, env, _, trace = _make_trainer(total_steps = 1)
    trainer.train()
    events = list(trace.events)

    with pytest.raises(RuntimeError):
        trainer.train()

    assert trace.events == events
    assert not env.closed


@pytest.mark.parametrize(
    "failure,confirmed_steps,successful_updates,completed,partial",
    [
        (("seed", 1), 0, 0, [], None),
        (("reset", 1), 0, 0, [], None),
        (("sample", 1), 0, 0, [], None),
        (("select", 1), 1, 1, [], _episode(1, 1, 0.5, 1)),
        (("step", 2), 1, 1, [], _episode(1, 1, 0.5, 1)),
        (("store", 2), 1, 1, [], _episode(1, 1, 0.5, 1)),
        (("update", 2), 2, 1, [], _episode(1, 2, 1.5, 2)),
        (("update", 3), 3, 2, [_episode(1, 3, 3.0, 3, truncated = True)], None),
        (("reset", 2), 3, 3, [_episode(1, 3, 3.0, 3, truncated = True)], None),
        (("store", 4), 3, 3, [_episode(1, 3, 3.0, 3, truncated = True)], None),
    ],
)
def test_failure_preserves_only_confirmed_state(
        monkeypatch, failure, confirmed_steps, successful_updates, completed, partial,
):
    times = iter([10.0, 12.0])
    monkeypatch.setattr(training_module, "time", SimpleNamespace(perf_counter = lambda: next(times)))
    cause = ValueError("injected unexpected failure")
    trainer, env, agent, trace = _make_trainer(
        total_steps = 5, warmup_steps = 1, endings = {3: (False, True)}, failure = failure, error = cause,
    )

    with pytest.raises(TrainingError) as caught:
        trainer.train()

    error = caught.value
    assert type(error) is TrainingError
    assert isinstance(error, RLSystemError)
    assert error.__cause__ is cause
    result = error.result
    assert result["environment_steps"] == confirmed_steps == len(agent.transitions)
    assert result["update_count"] == successful_updates
    assert result["update_metrics"] == [
        {"global_step": i, "update": i, "metrics": {"loss": float(i)}}
        for i in range(1, successful_updates + 1)
    ]
    assert result["episode_count"] == len(completed)
    assert result["episode_metrics"] == completed
    assert result["partial_episode"] == partial
    assert result["duration_seconds"] == 2.0
    assert result["steps_per_second"] == confirmed_steps / 2
    assert result["updates_per_second"] == successful_updates / 2
    assert json.loads(json.dumps(result, allow_nan = False)) == result
    assert not env.closed
    assert trace.events[-1][0] == failure[0]
    if failure[0] == "store":
        assert env.steps == confirmed_steps + 1
    events = list(trace.events)
    with pytest.raises(RuntimeError):
        trainer.train()
    assert trace.events == events


@pytest.mark.parametrize("error_type,wrapper", [
    (AgentUpdateError, TrainingError), (KeyboardInterrupt, TrainingInterruptedError),
])
def test_failure_during_update_series_preserves_previous_successes(error_type, wrapper):
    cause = error_type("injected failure")
    trainer, env, agent, trace = _make_trainer(
        total_steps = 5, learning_starts = 2, gradient_steps = 3,
        failure = ("update", 2), error = cause,
    )

    with pytest.raises(wrapper) as caught:
        trainer.train()

    assert type(caught.value) is wrapper
    assert caught.value.__cause__ is cause
    assert isinstance(caught.value, TrainingError)
    result = caught.value.result
    assert result["environment_steps"] == 2
    assert result["update_count"] == 1
    assert result["update_metrics"] == [{"global_step": 2, "update": 1, "metrics": {"loss": 1.0}}]
    assert result["episode_metrics"] == []
    assert result["partial_episode"] == _episode(1, 2, 1.5, 2)
    assert agent.update_steps == [2]
    assert trace.counts["update"] == 2
    assert not env.closed
    with pytest.raises(RuntimeError):
        trainer.train()


def test_system_exit_is_not_wrapped():
    cause = SystemExit(1)
    trainer, env, _, _ = _make_trainer(failure = ("reset", 1), error = cause)

    with pytest.raises(SystemExit) as caught:
        trainer.train()

    assert caught.value is cause
    assert not env.closed
