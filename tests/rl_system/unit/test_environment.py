"""Tests for EnvironmentSpec, Factory, and Gymnasium integration (subphase 1.2)."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
from gymnasium import error as gym_error
import numpy as np
import pytest

from rl_system.config import EnvironmentConfig
from rl_system.environment import (
    EnvironmentSpec,
    extract_environment_spec,
    make_environment,
)
from rl_system.exceptions import (
    ConfigurationError,
    EnvironmentCreationError,
    EnvironmentDependencyError,
    EnvironmentSystemError,
    InvalidConfigError,
    RLSystemError,
    UnsupportedConfigError,
    UnsupportedSpaceError,
)


def _box(shape = (3,), *, low = -1.0, high = 1.0, dtype = np.float32):
    return gym.spaces.Box(low = low, high = high, shape = shape, dtype = dtype)


def _env(*, observation_space = None, action_space = None):
    return SimpleNamespace(
        observation_space = (
            _box(shape = (3,)) if observation_space is None else observation_space
        ),
        action_space = _box(shape = (1,)) if action_space is None else action_space,
        reset = Mock(),
        step = Mock(),
        close = Mock(),
    )


def _spec(**overrides):
    values = {
        "observation_dim": 3,
        "action_dim": 2,
        "action_low": np.array([-2.0, 0.5], dtype = np.float64),
        "action_high": np.array([1.0, 4.5], dtype = np.float64),
    }
    values.update(overrides)
    return EnvironmentSpec(**values)


def test_environment_exception_hierarchy():
    assert issubclass(UnsupportedSpaceError, UnsupportedConfigError)
    assert issubclass(UnsupportedSpaceError, ConfigurationError)
    assert issubclass(EnvironmentSystemError, RLSystemError)
    assert issubclass(EnvironmentDependencyError, EnvironmentSystemError)
    assert issubclass(EnvironmentCreationError, EnvironmentSystemError)
    assert not issubclass(UnsupportedSpaceError, EnvironmentSystemError)


def test_direct_spec_preserves_valid_dimensions_and_bounds():
    spec = _spec()
    assert spec.observation_dim == 3
    assert spec.action_dim == 2
    assert spec.action_low.dtype == np.dtype("float64")
    np.testing.assert_array_equal(spec.action_low, [-2.0, 0.5])
    np.testing.assert_array_equal(spec.action_high, [1.0, 4.5])


def test_direct_spec_accepts_integer_bounds_without_dtype_coercion():
    spec = _spec(
        action_low = np.array([-2, 0], dtype = np.int64),
        action_high = np.array([1, 5], dtype = np.int64),
    )
    assert spec.action_low.dtype == np.dtype("int64")
    assert spec.action_high.dtype == np.dtype("int64")


def test_direct_spec_copies_bounds_and_blocks_ordinary_mutation():
    source_low = np.array([-2.0, 0.5])
    source_high = np.array([1.0, 4.5])
    spec = _spec(action_low = source_low, action_high = source_high)

    assert not np.shares_memory(source_low, spec.action_low)
    assert not np.shares_memory(source_high, spec.action_high)
    assert not spec.action_low.flags.writeable
    assert not spec.action_high.flags.writeable

    source_low[0] = -99.0
    source_high[0] = 99.0
    assert spec.action_low[0] == -2.0
    assert spec.action_high[0] == 1.0

    with pytest.raises(ValueError):
        spec.action_low[0] = -10.0
    with pytest.raises(ValueError):
        spec.action_high[0] = 10.0
    with pytest.raises(FrozenInstanceError):
        spec.action_dim = 3


def test_spec_equality_is_not_numpy_elementwise_comparison():
    first = _spec()
    second = _spec()
    assert first is first
    assert first is not second
    assert first != second


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("observation_dim", 0),
        ("observation_dim", -1),
        ("observation_dim", True),
        ("observation_dim", 2.0),
        ("action_dim", 0),
        ("action_dim", -1),
        ("action_dim", False),
        ("action_dim", 2.0),
    ],
)
def test_direct_spec_rejects_invalid_dimensions(field, invalid):
    with pytest.raises(ValueError, match = field):
        _spec(**{field: invalid})


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("action_low", np.array([-2.0])),
        ("action_high", np.array([[1.0, 4.5]])),
        ("action_low", np.array([-np.inf, 0.5])),
        ("action_high", np.array([1.0, np.nan])),
        ("action_low", np.array([False, True])),
        ("action_high", np.array([1 + 0j, 4 + 0j])),
        ("action_low", np.array(["-2", "0.5"])),
    ],
)
def test_direct_spec_rejects_invalid_bound_arrays(field, invalid):
    with pytest.raises(ValueError, match = field):
        _spec(**{field: invalid})


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ([-2.0, 4.5], [1.0, 4.5]),  # equal at one coordinate
        ([2.0, 0.5], [1.0, 4.5]),  # reversed at one coordinate
    ],
)
def test_direct_spec_rejects_unordered_bounds(low, high):
    with pytest.raises(ValueError, match = "low < high"):
        _spec(action_low = np.array(low), action_high = np.array(high))


@pytest.mark.parametrize("observation_dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("action_dtype", [np.float32, np.float64])
def test_extract_supports_numeric_vector_boxes(observation_dtype, action_dtype):
    env = _env(
        observation_space = _box(shape = (4,), dtype = observation_dtype),
        action_space = _box(shape = (2,), low = -2.0, high = 3.0, dtype = action_dtype),
    )
    spec = extract_environment_spec(env)
    assert (spec.observation_dim, spec.action_dim) == (4, 2)
    np.testing.assert_array_equal(spec.action_low, [-2.0, -2.0])
    np.testing.assert_array_equal(spec.action_high, [3.0, 3.0])
    assert spec.action_low.dtype == np.dtype(action_dtype)


def test_extract_allows_infinite_observation_bounds_and_asymmetric_actions():
    env = _env(
        observation_space = _box(shape = (3,), low = -np.inf, high = np.inf),
        action_space = _box(
            shape = (2,),
            low = np.array([-3.0, 0.5]),
            high = np.array([1.0, 8.0]),
            dtype = np.float64,
        ),
    )
    spec = extract_environment_spec(env)
    np.testing.assert_array_equal(spec.action_low, [-3.0, 0.5])
    np.testing.assert_array_equal(spec.action_high, [1.0, 8.0])


def test_extract_does_not_modify_spaces_or_call_environment_methods():
    env = _env()
    obs_low_before = env.observation_space.low.copy()
    obs_high_before = env.observation_space.high.copy()
    action_low_before = env.action_space.low.copy()
    action_high_before = env.action_space.high.copy()
    low_writeable_before = env.action_space.low.flags.writeable
    high_writeable_before = env.action_space.high.flags.writeable

    spec = extract_environment_spec(env)

    np.testing.assert_array_equal(env.observation_space.low, obs_low_before)
    np.testing.assert_array_equal(env.observation_space.high, obs_high_before)
    np.testing.assert_array_equal(env.action_space.low, action_low_before)
    np.testing.assert_array_equal(env.action_space.high, action_high_before)
    assert not np.shares_memory(spec.action_low, env.action_space.low)
    assert not np.shares_memory(spec.action_high, env.action_space.high)
    assert env.action_space.low.flags.writeable == low_writeable_before
    assert env.action_space.high.flags.writeable == high_writeable_before
    env.reset.assert_not_called()
    env.step.assert_not_called()
    env.close.assert_not_called()

    env.action_space.low = np.array([-99.0], dtype = np.float32)
    env.action_space.high = np.array([99.0], dtype = np.float32)
    assert spec.action_low[0] == -1.0
    assert spec.action_high[0] == 1.0


@pytest.mark.parametrize("space_name", ["observation_space", "action_space"])
def test_extract_rejects_discrete_spaces(space_name):
    env = _env()
    setattr(env, space_name, gym.spaces.Discrete(5))
    with pytest.raises(UnsupportedSpaceError, match = f"{space_name}.*Box"):
        extract_environment_spec(env)


@pytest.mark.parametrize("space_name", ["observation_space", "action_space"])
def test_extract_rejects_multidimensional_boxes(space_name):
    env = _env()
    setattr(env, space_name, _box(shape = (2, 2)))
    with pytest.raises(UnsupportedSpaceError, match = f"{space_name}.*1-D"):
        extract_environment_spec(env)


@pytest.mark.parametrize("space_name", ["observation_space", "action_space"])
def test_extract_rejects_empty_box_vector(space_name):
    env = _env()
    # Gymnasium rejects zero-length Box dimensions during normal construction.
    # Model a malformed third-party space to test the extractor's own guard.
    getattr(env, space_name)._shape = (0,)
    with pytest.raises(UnsupportedSpaceError, match = f"{space_name}.*nonempty"):
        extract_environment_spec(env)


@pytest.mark.parametrize(
    ("space_name", "dtype"),
    [
        ("observation_space", np.bool_),
        ("observation_space", np.complex64),
        ("observation_space", object),
        ("action_space", np.int32),
        ("action_space", np.bool_),
        ("action_space", np.complex64),
    ],
)
def test_extract_rejects_unsupported_box_dtypes(space_name, dtype):
    env = _env()
    box = getattr(env, space_name)
    # Set the reported dtype directly: Gymnasium itself may reject some of
    # these types before our extractor gets a chance to validate them.
    box.dtype = np.dtype(dtype)
    with pytest.raises(UnsupportedSpaceError, match = f"{space_name}.*dtype"):
        extract_environment_spec(env)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ([-np.inf, -1.0], [2.0, 1.0]),
        ([-1.0, -1.0], [np.inf, 1.0]),
        ([np.nan, -1.0], [2.0, 1.0]),
        ([-1.0, -1.0], [2.0, np.nan]),
        ([2.0, -1.0], [2.0, 1.0]),
        ([3.0, -1.0], [2.0, 1.0]),
    ],
)
def test_extract_rejects_invalid_action_bound_values(low, high):
    env = _env(action_space = _box(shape = (2,)))
    # Use an otherwise valid Box and inject invalid bounds. Gymnasium may
    # disallow constructing an inconsistent Box directly.
    env.action_space.low = np.asarray(low, dtype = np.float32)
    env.action_space.high = np.asarray(high, dtype = np.float32)
    with pytest.raises(UnsupportedSpaceError, match = "action_space bounds") as error:
        extract_environment_spec(env)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("bound_name", ["low", "high"])
def test_extract_rejects_inconsistent_action_bound_shapes(bound_name):
    env = _env(action_space = _box(shape = (2,)))
    setattr(env.action_space, bound_name, np.array([0.0], dtype = np.float32))
    with pytest.raises(UnsupportedSpaceError, match = "action_space bounds"):
        extract_environment_spec(env)


# Factory tests (subphase 1.2, step 2).


def test_factory_forwards_id_and_kwargs_and_transfers_ownership(monkeypatch):
    env = _env(
        observation_space = _box(shape = (4,)),
        action_space = _box(shape = (2,), low = -3.0, high = 1.5),
    )
    make_mock = Mock(return_value = env)
    monkeypatch.setattr(gym, "make", make_mock)
    config = EnvironmentConfig(
        id = "Example-v0", kwargs = {"render_mode": "rgb_array", "option": 7}
    )

    created, spec = make_environment(config)

    make_mock.assert_called_once_with(
        "Example-v0", render_mode = "rgb_array", option = 7
    )
    assert created is env
    assert (spec.observation_dim, spec.action_dim) == (4, 2)
    np.testing.assert_array_equal(spec.action_low, [-3.0, -3.0])
    np.testing.assert_array_equal(spec.action_high, [1.5, 1.5])
    assert not np.shares_memory(spec.action_low, env.action_space.low)
    assert not np.shares_memory(spec.action_high, env.action_space.high)
    env.reset.assert_not_called()
    env.step.assert_not_called()
    env.close.assert_not_called()


def test_factory_uses_empty_kwargs_by_default(monkeypatch):
    env = _env()
    make_mock = Mock(return_value = env)
    monkeypatch.setattr(gym, "make", make_mock)

    created, spec = make_environment(EnvironmentConfig(id = "Example-v0"))

    make_mock.assert_called_once_with("Example-v0")
    assert created is env
    assert isinstance(spec, EnvironmentSpec)
    env.close.assert_not_called()


def test_factory_rejects_duplicate_id_in_kwargs_before_creation(monkeypatch):
    make_mock = Mock()
    monkeypatch.setattr(gym, "make", make_mock)
    config = EnvironmentConfig(id = "Example-v0", kwargs = {"id": "Other-v0"})

    with pytest.raises(InvalidConfigError, match = "kwargs.*id"):
        make_environment(config)

    make_mock.assert_not_called()


@pytest.mark.parametrize("bad_space", ["observation_space", "action_space"])
def test_factory_closes_environment_when_spaces_are_unsupported(monkeypatch, bad_space):
    env = _env()
    setattr(env, bad_space, gym.spaces.Discrete(4))
    monkeypatch.setattr(gym, "make", Mock(return_value = env))

    with pytest.raises(UnsupportedSpaceError, match = bad_space):
        make_environment(EnvironmentConfig(id = "Example-v0"))

    env.close.assert_called_once_with()
    env.reset.assert_not_called()
    env.step.assert_not_called()


def test_factory_closes_on_unexpected_spec_error_without_masking(monkeypatch):
    env = _env()
    failure = RuntimeError("unexpected spec failure")
    monkeypatch.setattr(gym, "make", Mock(return_value = env))
    monkeypatch.setattr(
        "rl_system.environment.extract_environment_spec", Mock(side_effect = failure)
    )

    with pytest.raises(RuntimeError, match = "unexpected spec failure") as error:
        make_environment(EnvironmentConfig(id = "Example-v0"))

    assert error.value is failure
    env.close.assert_called_once_with()


def test_factory_preserves_validation_error_if_close_also_fails(monkeypatch):
    env = _env(action_space = gym.spaces.Discrete(3))
    env.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(gym, "make", Mock(return_value = env))

    with pytest.raises(UnsupportedSpaceError, match = "action_space") as error:
        make_environment(EnvironmentConfig(id = "Example-v0"))

    env.close.assert_called_once_with()
    assert any("cleanup failed" in note for note in error.value.__notes__)


@pytest.mark.parametrize(
    ("gym_exception", "expected_exception"),
    [
        (gym_error.UnregisteredEnv, InvalidConfigError),
        (gym_error.DeprecatedEnv, InvalidConfigError),
        (gym_error.MissingArgument, InvalidConfigError),
        (gym_error.InvalidBound, InvalidConfigError),
        (gym_error.InvalidProbability, InvalidConfigError),
        (gym_error.UnsupportedMode, UnsupportedConfigError),
        (gym_error.DependencyNotInstalled, EnvironmentDependencyError),
        (gym_error.Error, EnvironmentCreationError),
    ],
)
def test_factory_classifies_known_gymnasium_errors(
        monkeypatch, gym_exception, expected_exception
):
    original = gym_exception("simulated Gymnasium failure")
    make_mock = Mock(side_effect = original)
    monkeypatch.setattr(gym, "make", make_mock)

    with pytest.raises(expected_exception) as error:
        make_environment(EnvironmentConfig(id = "Example-v0"))

    assert type(error.value) is expected_exception
    assert error.value.__cause__ is original
    make_mock.assert_called_once_with("Example-v0")


@pytest.mark.parametrize("exception_type", [TypeError, ValueError, RuntimeError])
def test_factory_does_not_mask_unexpected_creation_errors(
        monkeypatch, exception_type
):
    original = exception_type("unexpected constructor failure")
    make_mock = Mock(side_effect = original)
    monkeypatch.setattr(gym, "make", make_mock)

    with pytest.raises(exception_type) as error:
        make_environment(EnvironmentConfig(id = "Example-v0"))

    assert error.value is original
    make_mock.assert_called_once_with("Example-v0")


# Real Gymnasium integration tests (subphase 1.2, step 3).


def test_factory_pendulum_integration_reset_step_close():
    env, spec = make_environment(EnvironmentConfig(id = "Pendulum-v1"))
    try:
        assert isinstance(env, gym.Env)
        assert (spec.observation_dim, spec.action_dim) == (3, 1)
        np.testing.assert_array_equal(spec.action_low, [-2.0])
        np.testing.assert_array_equal(spec.action_high, [2.0])
        assert not spec.action_low.flags.writeable
        assert not spec.action_high.flags.writeable
        assert not np.shares_memory(spec.action_low, env.action_space.low)
        assert not np.shares_memory(spec.action_high, env.action_space.high)

        observation, info = env.reset(seed = 42)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

        action = np.zeros((spec.action_dim,), dtype = env.action_space.dtype)
        assert env.action_space.contains(action)
        next_observation, reward, terminated, truncated, step_info = env.step(action)

        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()


def test_factory_pendulum_accepts_constructor_kwargs():
    env, spec = make_environment(
        EnvironmentConfig(id = "Pendulum-v1", kwargs = {"g": 9.81})
    )
    try:
        assert (spec.observation_dim, spec.action_dim) == (3, 1)
        assert env.unwrapped.g == pytest.approx(9.81)
    finally:
        env.close()


def test_factory_rejects_unknown_real_environment_id():
    with pytest.raises(InvalidConfigError) as error:
        make_environment(EnvironmentConfig(id = "NoSuchRLEnvironment-987654-v0"))

    assert isinstance(error.value.__cause__, gym_error.UnregisteredEnv)


def test_factory_rejects_real_discrete_action_space():
    # CartPole-v1 is available without optional Box2D dependencies.
    with pytest.raises(UnsupportedSpaceError, match = "action_space.*Box"):
        make_environment(EnvironmentConfig(id = "CartPole-v1"))
