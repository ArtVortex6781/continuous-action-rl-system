"""Unit tests for the off-policy contract and Replay Buffer write path."""

from dataclasses import FrozenInstanceError, fields, is_dataclass
from inspect import Parameter, signature

import numpy as np
import pytest

from rl_system.agents.base import OffPolicyAgent
from rl_system.agents.td3.replay_buffer import ReplayBatch, ReplayBuffer

STORAGE_FIELDS = (
    "obs",
    "actions",
    "rewards",
    "next_obs",
    "terminated",
    "truncated",
)


def _transition(step: int) -> dict:
    """Build one distinguishable transition with two observation/action features."""
    return {
        "obs": np.array([step + 0.25, step + 0.5], dtype = np.float64),
        "action": np.array([step + 1.0, -step - 1.5], dtype = np.float64),
        "reward": step + 0.75,
        "next_obs": np.array([step + 1.25, step + 1.5], dtype = np.float64),
        "terminated": step % 2 == 0,
        "truncated": step % 3 == 0,
    }


def _assert_row_matches(buffer: ReplayBuffer, index: int, step: int) -> None:
    """Compare the six aligned arrays at one physical storage index."""
    expected = _transition(step)
    np.testing.assert_array_equal(buffer.obs[index], expected["obs"].astype(np.float32))
    np.testing.assert_array_equal(buffer.actions[index], expected["action"].astype(np.float32))
    np.testing.assert_array_equal(
        buffer.next_obs[index], expected["next_obs"].astype(np.float32)
    )
    assert buffer.rewards[index, 0] == np.float32(expected["reward"])
    assert buffer.terminated[index, 0] == expected["terminated"]
    assert buffer.truncated[index, 0] == expected["truncated"]


@pytest.fixture
def buffer() -> ReplayBuffer:
    return ReplayBuffer(capacity = 3, obs_dim = 2, act_dim = 2, seed = 42)


def test_off_policy_agent_declares_only_three_required_operations():
    assert getattr(OffPolicyAgent, "_is_protocol", False)
    operations = {
        name
        for name, value in vars(OffPolicyAgent).items()
        if callable(value) and not name.startswith("_")
    }
    assert operations == {"select_action", "store_transition", "update"}

    action_signature = signature(OffPolicyAgent.select_action)
    assert list(action_signature.parameters) == ["self", "observation", "explore"]
    assert action_signature.parameters["explore"].kind is Parameter.KEYWORD_ONLY

    transition_signature = signature(OffPolicyAgent.store_transition)
    assert list(transition_signature.parameters) == [
        "self", "obs", "action", "reward", "next_obs", "terminated", "truncated"
    ]
    assert list(signature(OffPolicyAgent.update).parameters) == ["self"]


def test_replay_batch_is_a_frozen_slotted_container():
    assert is_dataclass(ReplayBatch)
    assert tuple(field.name for field in fields(ReplayBatch)) == STORAGE_FIELDS
    assert ReplayBatch.__dataclass_params__.eq is False
    assert set(ReplayBatch.__slots__) == set(STORAGE_FIELDS)

    arrays = [np.empty((2, 1)) for _ in STORAGE_FIELDS]
    batch = ReplayBatch(*arrays)
    assert all(getattr(batch, name) is array for name, array in zip(STORAGE_FIELDS, arrays))
    with pytest.raises(FrozenInstanceError):
        batch.obs = np.empty((2, 1))


def test_initialization_allocates_six_arrays_with_approved_shapes_and_dtypes(buffer):
    expected = {
        "obs": ((3, 2), np.dtype("float32")),
        "actions": ((3, 2), np.dtype("float32")),
        "rewards": ((3, 1), np.dtype("float32")),
        "next_obs": ((3, 2), np.dtype("float32")),
        "terminated": ((3, 1), np.dtype("bool")),
        "truncated": ((3, 1), np.dtype("bool")),
    }
    for name, (shape, dtype) in expected.items():
        array = getattr(buffer, name)
        assert isinstance(array, np.ndarray)
        assert array.shape == shape
        assert array.dtype == dtype

    assert buffer.capacity == 3
    assert buffer.obs_dim == 2
    assert buffer.act_dim == 2
    assert buffer.ptr == 0
    assert buffer.size == 0
    assert len(buffer) == 0
    assert isinstance(buffer._rng, np.random.Generator)


def test_add_one_transition_preserves_all_six_fields(buffer):
    assert buffer.add(**_transition(0)) is None
    assert len(buffer) == 1
    assert buffer.size == 1
    assert buffer.ptr == 1
    _assert_row_matches(buffer, index = 0, step = 0)


def test_add_multiple_transitions_advances_pointer_and_keeps_rows_aligned(buffer):
    buffer.add(**_transition(0))
    buffer.add(**_transition(1))

    assert len(buffer) == 2
    assert buffer.size == 2
    assert buffer.ptr == 2
    _assert_row_matches(buffer, index = 0, step = 0)
    _assert_row_matches(buffer, index = 1, step = 1)


def test_ring_buffer_overwrites_oldest_transition_without_mixing_fields(buffer):
    for step in range(3):
        buffer.add(**_transition(step))

    assert len(buffer) == 3
    assert buffer.ptr == 0

    buffer.add(**_transition(3))
    assert len(buffer) == buffer.capacity
    assert buffer.size == buffer.capacity
    assert buffer.ptr == 1
    _assert_row_matches(buffer, index = 0, step = 3)
    _assert_row_matches(buffer, index = 1, step = 1)
    _assert_row_matches(buffer, index = 2, step = 2)

    buffer.add(**_transition(4))
    assert buffer.ptr == 2
    _assert_row_matches(buffer, index = 0, step = 3)
    _assert_row_matches(buffer, index = 1, step = 4)
    _assert_row_matches(buffer, index = 2, step = 2)


def test_length_never_exceeds_capacity():
    buffer = ReplayBuffer(capacity = 2, obs_dim = 2, act_dim = 2, seed = 0)
    for step in range(10):
        buffer.add(**_transition(step))
        assert len(buffer) == min(step + 1, buffer.capacity)
        assert buffer.size == len(buffer)
        assert buffer.ptr == (step + 1) % buffer.capacity

    _assert_row_matches(buffer, index = 0, step = 8)
    _assert_row_matches(buffer, index = 1, step = 9)


@pytest.mark.parametrize("name", ["capacity", "obs_dim", "act_dim"])
@pytest.mark.parametrize("value", [0, -1, -10])
def test_constructor_rejects_nonpositive_capacity_or_dimensions(name, value):
    kwargs = dict(capacity = 3, obs_dim = 2, act_dim = 2, seed = 42)
    kwargs[name] = value
    with pytest.raises(ValueError):
        ReplayBuffer(**kwargs)


@pytest.mark.parametrize("name", ["capacity", "obs_dim", "act_dim", "seed"])
@pytest.mark.parametrize("value", [True, False, 2.0, "2", None, np.int64(2)])
def test_constructor_requires_plain_integers_not_bool(name, value):
    kwargs = dict(capacity = 3, obs_dim = 2, act_dim = 2, seed = 42)
    kwargs[name] = value
    with pytest.raises(TypeError):
        ReplayBuffer(**kwargs)


def test_constructor_rejects_negative_seed():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity = 3, obs_dim = 2, act_dim = 2, seed = -1)


@pytest.mark.parametrize("field", ["obs", "next_obs", "action"])
@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0]]),
        np.array([[1.0], [2.0]]),
        np.array(1.0),
    ],
)
def test_add_rejects_incorrect_vector_shapes(buffer, field, value):
    transition = _transition(0)
    transition[field] = value
    with pytest.raises(ValueError, match = "shape"):
        buffer.add(**transition)
    assert len(buffer) == 0
    assert buffer.ptr == 0


@pytest.mark.parametrize("field", ["obs", "next_obs", "action"])
@pytest.mark.parametrize(
    "value",
    [
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([-np.inf, 1.0]),
        np.array([1e40, 1.0]),  # finite float64, overflows on float32 conversion
    ],
)
def test_add_rejects_nonfinite_vectors_in_storage_dtype(buffer, field, value):
    transition = _transition(1)
    transition[field] = value
    with pytest.raises(ValueError, match = "finite"):
        buffer.add(**transition)


@pytest.mark.parametrize("field", ["obs", "next_obs", "action"])
@pytest.mark.parametrize(
    "value",
    [
        np.array([True, False]),
        np.array([1 + 2j, 3 + 4j]),
        np.array(["1", "2"]),
        np.array([object(), object()], dtype = object),
    ],
)
def test_add_rejects_unsupported_vector_dtypes(buffer, field, value):
    transition = _transition(1)
    transition[field] = value
    with pytest.raises(TypeError, match = "numeric"):
        buffer.add(**transition)


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf, 1e40])
def test_add_rejects_nonfinite_or_float32_overflowing_reward(buffer, reward):
    transition = _transition(0)
    transition["reward"] = reward
    with pytest.raises(ValueError, match = "finite"):
        buffer.add(**transition)


@pytest.mark.parametrize("reward", [[1.0], np.array([1.0]), np.array([[1.0]])])
def test_add_rejects_nonscalar_reward(buffer, reward):
    transition = _transition(0)
    transition["reward"] = reward
    with pytest.raises(ValueError, match = "scalar"):
        buffer.add(**transition)


@pytest.mark.parametrize("reward", [True, False, "1", 1 + 2j, None])
def test_add_rejects_reward_with_unsupported_type(buffer, reward):
    transition = _transition(0)
    transition["reward"] = reward
    with pytest.raises(TypeError):
        buffer.add(**transition)


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
@pytest.mark.parametrize("value", [0, 1, 0.0, "false", None, np.int64(1), np.array(True)])
def test_add_rejects_nonboolean_flags(buffer, flag, value):
    transition = _transition(0)
    transition[flag] = value
    with pytest.raises(TypeError, match = "boolean"):
        buffer.add(**transition)


def test_add_accepts_numpy_boolean_flags(buffer):
    transition = _transition(0)
    transition["terminated"] = np.bool_(True)
    transition["truncated"] = np.bool_(False)
    buffer.add(**transition)
    assert bool(buffer.terminated[0, 0]) is True
    assert bool(buffer.truncated[0, 0]) is False


def test_add_stores_copies_instead_of_references(buffer):
    transition = _transition(1)
    obs = transition["obs"]
    action = transition["action"]
    next_obs = transition["next_obs"]
    buffer.add(**transition)

    obs[:] = -100
    action[:] = -100
    next_obs[:] = -100

    _assert_row_matches(buffer, index = 0, step = 1)
    assert not np.shares_memory(buffer.obs, obs)
    assert not np.shares_memory(buffer.actions, action)
    assert not np.shares_memory(buffer.next_obs, next_obs)


@pytest.mark.parametrize(
    "invalid_field,invalid_value,expected_exception",
    [
        ("obs", np.array([1.0]), ValueError),
        ("action", np.array([np.inf, 0.0]), ValueError),
        ("reward", np.nan, ValueError),
        ("next_obs", np.array(["bad", "data"]), TypeError),
        ("terminated", 1, TypeError),
        ("truncated", 0, TypeError),
    ],
)
@pytest.mark.parametrize("filled", [False, True])
def test_failed_add_does_not_modify_existing_transitions_or_indices(
        buffer, invalid_field, invalid_value, expected_exception, filled
):
    count = buffer.capacity if filled else 2
    for step in range(count):
        buffer.add(**_transition(step))

    existing = {name: getattr(buffer, name)[:count].copy() for name in STORAGE_FIELDS}
    initial_state = (buffer.ptr, buffer.size, len(buffer))

    transition = _transition(99)
    transition[invalid_field] = invalid_value
    with pytest.raises(expected_exception):
        buffer.add(**transition)

    assert (buffer.ptr, buffer.size, len(buffer)) == initial_state
    for name, expected in existing.items():
        np.testing.assert_array_equal(getattr(buffer, name)[:count], expected)
