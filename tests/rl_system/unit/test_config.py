"""Unit tests for Configuration models and YAML loading (subphases 1.1 and 1.6)."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rl_system.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    ProgressConfig,
    RunConfig,
    TrainingConfig,
    load_config,
)
from rl_system.exceptions import InvalidConfigError


def _progress_data():
    return {
        "step_interval": 1000,
        "episode_interval": 2,
        "episode_start": 30,
        "reward_window": 10,
        "show_update_count": True,
        "show_update_metrics": True,
        "show_metric_counts": True,
        "show_throughput": True,
    }


def _training_data(**overrides):
    data = {
        "total_steps": 35_000,
        "seed": 42,
        "warmup_steps": 1000,
        "learning_starts": 1000,
        "train_frequency": 1,
        "gradient_steps": 1,
        "progress": _progress_data(),
    }
    data.update(overrides)
    return data


@pytest.fixture
def valid_data():
    return {
        "environment": {"id": "Pendulum-v1", "kwargs": {}},
        "algorithm": {"name": "td3", "params": {}},
        "training": _training_data(),
    }


def test_run_config_valid(valid_data):
    config = RunConfig.model_validate(valid_data)

    assert isinstance(config.environment, EnvironmentConfig)
    assert isinstance(config.algorithm, AlgorithmConfig)
    assert isinstance(config.training, TrainingConfig)
    assert config.environment.id == "Pendulum-v1"
    assert config.algorithm.params == {}
    assert config.training.total_steps == 35_000
    assert isinstance(config.training.progress, ProgressConfig)
    assert config.training.seed == 42
    assert config.model_dump(mode = "json") == valid_data


def test_environment_kwargs_default_is_independent():
    first = EnvironmentConfig(id = "Pendulum-v1")
    second = EnvironmentConfig(id = "Pendulum-v1")

    assert first.kwargs == second.kwargs == {}
    assert first.kwargs is not second.kwargs
    first.kwargs["example"] = 1  # frozen=True does not freeze nested dicts.
    assert second.kwargs == {}


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("environment", "id"),
        ("algorithm", "name"),
        ("algorithm", "params"),
        ("training", "total_steps"),
        ("training", "seed"),
        ("training", "warmup_steps"),
        ("training", "learning_starts"),
        ("training", "train_frequency"),
        ("training", "gradient_steps"),
        ("training", "progress"),
    ],
)
def test_required_fields(valid_data, section, field):
    del valid_data[section][field]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(valid_data)


@pytest.mark.parametrize("section", ["environment", "algorithm", "training"])
def test_required_sections(valid_data, section):
    del valid_data[section]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(valid_data)


@pytest.mark.parametrize("invalid", [0, -1, 3.5, True, "100", None])
def test_total_steps_must_be_positive_strict_int(invalid):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_training_data(total_steps = invalid, warmup_steps = 0, learning_starts = 1))


@pytest.mark.parametrize("invalid", [-1, 2 ** 32, 1.5, False, "42", None])
def test_seed_must_be_bounded_strict_int(invalid):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_training_data(seed = invalid))


@pytest.mark.parametrize("seed", [0, 2 ** 32 - 1])
def test_seed_boundaries_are_valid(seed):
    assert TrainingConfig.model_validate(_training_data(seed = seed)).seed == seed


@pytest.mark.parametrize("invalid", ["", "   ", 123, None])
def test_environment_id_must_be_nonblank_string(invalid):
    with pytest.raises(ValidationError):
        EnvironmentConfig(id = invalid)


@pytest.mark.parametrize("invalid", ["", "\t  ", 123, None])
def test_algorithm_name_must_be_nonblank_string(invalid):
    with pytest.raises(ValidationError):
        AlgorithmConfig(name = invalid, params = {})


@pytest.mark.parametrize("section", [None, "environment", "algorithm", "training"])
def test_extra_fields_are_forbidden(valid_data, section):
    target = valid_data if section is None else valid_data[section]
    target["unexpected"] = "value"
    with pytest.raises(ValidationError) as exc_info:
        RunConfig.model_validate(valid_data)
    assert any(error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.parametrize(
    ("model", "field", "replacement"),
    [
        (EnvironmentConfig(id = "Pendulum-v1"), "id", "Other-v1"),
        (AlgorithmConfig(name = "td3", params = {}), "name", "sac"),
        (TrainingConfig.model_validate(_training_data()), "seed", 4),
        (ProgressConfig.model_validate(_progress_data()), "step_interval", 1),
        (
                RunConfig(
                    environment = EnvironmentConfig(id = "Pendulum-v1"),
                    algorithm = AlgorithmConfig(name = "td3", params = {}),
                    training = TrainingConfig.model_validate(_training_data()),
                ),
                "training",
                TrainingConfig.model_validate(_training_data(total_steps = 40_000)),
        ),
    ],
)
def test_models_are_frozen(model, field, replacement):
    with pytest.raises(ValidationError) as exc_info:
        setattr(model, field, replacement)
    assert exc_info.value.errors()[0]["type"] == "frozen_instance"


def test_nested_json_values_are_accepted():
    payload = {"items": [None, True, False, 0, 1.5, "text", {"nested": [1, 2]}]}

    environment = EnvironmentConfig(id = "Pendulum-v1", kwargs = deepcopy(payload))
    algorithm = AlgorithmConfig(name = "td3", params = deepcopy(payload))

    assert environment.kwargs == payload
    assert algorithm.params == payload


@pytest.mark.parametrize(
    "invalid",
    [
        {"value": float("nan")},
        {"value": [float("inf")]},
        {"value": {"nested": -float("inf")}},
        {"value": (1, 2)},
        {"value": {1, 2}},
        {"value": object()},
        {1: "non-string-key"},
        {"value": [{False: "non-string-key"}]},
    ],
)
@pytest.mark.parametrize("model_class, field", [(EnvironmentConfig, "kwargs"), (AlgorithmConfig, "params")])
def test_nested_non_json_values_rejected(invalid, model_class, field):
    data = {"id": "Pendulum-v1"} if model_class is EnvironmentConfig else {"name": "td3"}
    data[field] = invalid
    with pytest.raises(ValidationError):
        model_class.model_validate(data)


@pytest.mark.parametrize("invalid", [None, [], "string", 5])
@pytest.mark.parametrize("model_class, field", [(EnvironmentConfig, "kwargs"), (AlgorithmConfig, "params")])
def test_json_fields_must_be_dictionaries(invalid, model_class, field):
    data = {"id": "Pendulum-v1"} if model_class is EnvironmentConfig else {"name": "td3"}
    data[field] = invalid
    with pytest.raises(ValidationError):
        model_class.model_validate(data)


def test_cyclic_json_structure_is_rejected():
    payload = {}
    payload["self"] = payload
    with pytest.raises(ValidationError):
        AlgorithmConfig(name = "td3", params = payload)


def test_run_config_rejects_wrong_nested_types(valid_data):
    valid_data["environment"] = "Pendulum-v1"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(valid_data)


# Step 3/3: YAML loader and public load_config() contract.

VALID_YAML = """\
environment:
  id: Pendulum-v1
  kwargs: {}

algorithm:
  name: td3
  params: {}

training:
  total_steps: 35000
  seed: 42
  warmup_steps: 1000
  learning_starts: 1000
  train_frequency: 1
  gradient_steps: 1
  progress:
    step_interval: 1000
    episode_interval: 2
    episode_start: 30
    reward_window: 10
    show_update_count: true
    show_update_metrics: true
    show_metric_counts: true
    show_throughput: true
"""


def test_load_project_example_yaml(valid_data):
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "configs" / "td3_pendulum.yaml"
    config = load_config(config_path)
    valid_data["algorithm"]["params"] = yaml.safe_load(
        config_path.read_text(encoding = "utf-8")
    )["algorithm"]["params"]

    assert isinstance(config, RunConfig)
    assert config.model_dump(mode = "json") == valid_data


@pytest.mark.parametrize("use_str_path", [False, True])
def test_load_config_accepts_path_and_string(tmp_path, valid_data, use_str_path):
    config_file = tmp_path / "run.yaml"
    config_file.write_text(VALID_YAML, encoding = "utf-8")

    path = str(config_file) if use_str_path else config_file
    config = load_config(path)

    assert isinstance(config, RunConfig)
    assert config.model_dump(mode = "json") == valid_data


@pytest.mark.parametrize(
    "content",
    [
        "",  # Empty YAML file.
        "# only a comment\n",  # YAML parsing yields None.
        "[]\n",  # Invalid root type.
        VALID_YAML.replace("  seed: 42", "  seed: true"),
        VALID_YAML + "unexpected: 1\n",  # Unknown top-level field.
    ],
)
def test_load_config_rejects_invalid_schema(tmp_path, content):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(content, encoding = "utf-8")

    with pytest.raises(InvalidConfigError) as exc_info:
        load_config(config_file)

    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_load_config_rejects_malformed_yaml(tmp_path):
    config_file = tmp_path / "malformed.yaml"
    config_file.write_text("environment: [\n", encoding = "utf-8")

    with pytest.raises(InvalidConfigError) as exc_info:
        load_config(config_file)

    assert isinstance(exc_info.value.__cause__, yaml.YAMLError)


@pytest.mark.parametrize(
    "content",
    [
        VALID_YAML + "training:\n  total_steps: 1\n  seed: 1\n",
        VALID_YAML.replace("  seed: 42", "  seed: 42\n  seed: 17"),
        VALID_YAML.replace(
            "  params: {}",
            "  params:\n    network:\n      width: 64\n      width: 128",
        ),
        VALID_YAML.replace(
            "  params: {}",
            "  params:\n    layers:\n      - width: 64\n        width: 128",
        ),
    ],
)
def test_load_config_rejects_duplicate_yaml_keys(tmp_path, content):
    config_file = tmp_path / "duplicate.yaml"
    config_file.write_text(content, encoding = "utf-8")

    with pytest.raises(InvalidConfigError) as exc_info:
        load_config(config_file)

    assert isinstance(exc_info.value.__cause__, yaml.constructor.ConstructorError)
    assert "duplicate key" in str(exc_info.value.__cause__)


def test_load_config_rejects_invalid_utf8(tmp_path):
    config_file = tmp_path / "bad_encoding.yaml"
    config_file.write_bytes(b"environment: \xff\n")

    with pytest.raises(InvalidConfigError) as exc_info:
        load_config(config_file)

    assert isinstance(exc_info.value.__cause__, UnicodeError)


def test_load_config_preserves_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_preserves_other_os_errors(tmp_path):
    # Opening a directory as a file fails as OSError on Windows and POSIX.
    with pytest.raises(OSError):
        load_config(tmp_path)


def test_load_config_defers_algorithm_support_check(tmp_path):
    config_file = tmp_path / "unknown_algorithm.yaml"
    config_file.write_text(
        VALID_YAML.replace("name: td3", "name: not_implemented_yet"),
        encoding = "utf-8",
    )

    config = load_config(config_file)

    assert config.algorithm.name == "not_implemented_yet"


# Subphase 1.6: training schedules and progress configuration.


@pytest.mark.parametrize("field", ["warmup_steps", "learning_starts", "train_frequency", "gradient_steps"])
@pytest.mark.parametrize("invalid", [True, False, 1.5, "1", None])
def test_training_schedule_fields_require_strict_integers(field, invalid):
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig.model_validate(_training_data(**{field: invalid}))
    assert any(error["loc"] == (field,) for error in exc_info.value.errors())


@pytest.mark.parametrize(
    "field, invalid",
    [("warmup_steps", -1), ("warmup_steps", 35_001),
     ("learning_starts", 0), ("learning_starts", 35_001),
     ("train_frequency", 0), ("train_frequency", -1),
     ("gradient_steps", 0), ("gradient_steps", -1)],
)
def test_training_schedule_bounds(field, invalid):
    with pytest.raises(ValidationError, match = field):
        TrainingConfig.model_validate(_training_data(**{field: invalid}))


@pytest.mark.parametrize("warmup, starts", [(0, 1), (1, 1), (10, 1), (0, 10), (10, 10)])
def test_training_schedule_boundaries_and_independent_thresholds(warmup, starts):
    config = TrainingConfig.model_validate(_training_data(
        total_steps = 10, warmup_steps = warmup, learning_starts = starts,
        train_frequency = 11, gradient_steps = 12,
    ))
    assert (config.warmup_steps, config.learning_starts) == (warmup, starts)
    assert (config.train_frequency, config.gradient_steps) == (11, 12)


def test_single_step_budget_is_valid():
    config = TrainingConfig.model_validate(_training_data(
        total_steps = 1, warmup_steps = 0, learning_starts = 1,
    ))
    assert config.total_steps == 1


@pytest.mark.parametrize("field", list(_progress_data()))
def test_progress_fields_are_required(valid_data, field):
    del valid_data["training"]["progress"][field]
    with pytest.raises(ValidationError) as exc_info:
        RunConfig.model_validate(valid_data)
    assert any(error["loc"] == ("training", "progress", field)
               and error["type"] == "missing" for error in exc_info.value.errors())


@pytest.mark.parametrize("field", ["step_interval", "episode_interval", "episode_start", "reward_window"])
@pytest.mark.parametrize("invalid", [True, False, 1.5, "1", None, -1])
def test_progress_integer_fields_reject_invalid_values(field, invalid):
    data = _progress_data()
    data[field] = invalid
    with pytest.raises(ValidationError, match = field):
        ProgressConfig.model_validate(data)


def test_progress_episode_start_must_be_positive():
    with pytest.raises(ValidationError, match = "episode_start"):
        ProgressConfig.model_validate({**_progress_data(), "episode_start": 0})


@pytest.mark.parametrize("field", ["show_update_count", "show_update_metrics", "show_metric_counts", "show_throughput"])
@pytest.mark.parametrize("invalid", [0, 1, 1.0, "true", None])
def test_progress_flags_require_strict_booleans(field, invalid):
    with pytest.raises(ValidationError, match = field):
        ProgressConfig.model_validate({**_progress_data(), field: invalid})


def test_progress_can_disable_both_schedules_and_reward_statistics():
    data = {**_progress_data(), "step_interval": 0, "episode_interval": 0,
            "episode_start": 1, "reward_window": 0, "show_update_count": False,
            "show_update_metrics": False, "show_metric_counts": True,
            "show_throughput": False}
    config = TrainingConfig.model_validate(_training_data(progress = data))
    assert config.progress.model_dump(mode = "json") == data


def test_progress_rejects_extra_fields(valid_data):
    valid_data["training"]["progress"]["unexpected"] = True
    with pytest.raises(ValidationError) as exc_info:
        RunConfig.model_validate(valid_data)
    assert any(error["loc"] == ("training", "progress", "unexpected")
               and error["type"] == "extra_forbidden" for error in exc_info.value.errors())


@pytest.mark.parametrize("invalid", [None, [], "disabled", True])
def test_training_requires_progress_mapping(invalid):
    with pytest.raises(ValidationError, match = "progress"):
        TrainingConfig.model_validate(_training_data(progress = invalid))
