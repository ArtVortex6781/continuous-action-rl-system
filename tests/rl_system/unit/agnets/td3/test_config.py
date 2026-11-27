"""Unit and YAML-integration tests for TD3-specific Pydantic configuration."""

import math
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from rl_system.agents.td3.config import MLPConfig, TD3Config
from rl_system.config import RunConfig, load_config

SUPPORTED_ACTIVATIONS = ("relu", "tanh", "elu", "swish")
INTEGER_FIELDS = ("replay_capacity", "batch_size", "policy_delay")
FLOAT_FIELDS = (
    "gamma",
    "tau",
    "actor_learning_rate",
    "critic_learning_rate",
    "exploration_noise_std",
    "target_policy_noise_std",
    "target_noise_clip",
    "actor_output_init_limit",
)
REQUIRED_TD3_FIELDS = (*INTEGER_FIELDS, *FLOAT_FIELDS, "actor", "critic")


def _valid_td3_params() -> dict:
    """Return independent, complete TD3 parameters matching the example YAML."""
    return {
        "replay_capacity": 100_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "actor_learning_rate": 0.001,
        "critic_learning_rate": 0.001,
        "policy_delay": 2,
        "exploration_noise_std": 0.1,
        "target_policy_noise_std": 0.2,
        "target_noise_clip": 0.5,
        "actor_output_init_limit": 0.003,
        "actor": {
            "hidden_units": [512, 256],
            "hidden_activations": ["relu", "relu"],
        },
        "critic": {
            "hidden_units": [512, 256],
            "hidden_activations": ["relu", "relu"],
        },
    }


@pytest.mark.parametrize(
    ("units", "activations"),
    [
        ([32], ["relu"]),
        ([64, 32], ["tanh", "elu"]),
        ((128, 64, 32, 16), ("relu", "swish", "tanh", "elu")),
    ],
)
def test_mlp_accepts_one_or_multiple_layers_and_converts_lists_to_tuples(
        units, activations
):
    config = MLPConfig.model_validate(
        {"hidden_units": units, "hidden_activations": activations}
    )

    assert config.hidden_units == tuple(units)
    assert config.hidden_activations == tuple(activations)
    assert type(config.hidden_units) is tuple
    assert type(config.hidden_activations) is tuple


@pytest.mark.parametrize("activation", SUPPORTED_ACTIVATIONS)
def test_mlp_accepts_each_supported_activation(activation):
    config = MLPConfig.model_validate(
        {"hidden_units": [16], "hidden_activations": [activation]}
    )
    assert config.hidden_activations == (activation,)


@pytest.mark.parametrize("field", ["hidden_units", "hidden_activations"])
def test_mlp_rejects_empty_sequences(field):
    params = {"hidden_units": [64], "hidden_activations": ["relu"]}
    params[field] = []
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(params)


@pytest.mark.parametrize(
    ("units", "activations"),
    [([64], ["relu", "tanh"]), ([64, 32], ["relu"])],
)
def test_mlp_rejects_mismatched_layer_and_activation_counts(units, activations):
    with pytest.raises(ValidationError, match = "equal lengths"):
        MLPConfig.model_validate(
            {"hidden_units": units, "hidden_activations": activations}
        )


@pytest.mark.parametrize("bad_unit", [0, -1, -128, True, False, 3.5, "64", None])
def test_mlp_rejects_invalid_hidden_unit_values(bad_unit):
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(
            {"hidden_units": [bad_unit], "hidden_activations": ["relu"]}
        )


@pytest.mark.parametrize("bad_activation", ["sigmoid", "ReLU", "linear", "", 1, True, None])
def test_mlp_rejects_unsupported_activations_and_types(bad_activation):
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(
            {"hidden_units": [32], "hidden_activations": [bad_activation]}
        )


@pytest.mark.parametrize("field", ["hidden_units", "hidden_activations"])
@pytest.mark.parametrize("bad_container", [None, 32, "relu", {"key": "value"}])
def test_mlp_rejects_invalid_sequence_container_types(field, bad_container):
    params = {"hidden_units": [32], "hidden_activations": ["relu"]}
    params[field] = bad_container
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(params)


@pytest.mark.parametrize("missing", ["hidden_units", "hidden_activations"])
def test_mlp_requires_both_fields(missing):
    params = {"hidden_units": [32], "hidden_activations": ["relu"]}
    del params[missing]
    with pytest.raises(ValidationError):
        MLPConfig.model_validate(params)


def test_mlp_rejects_unknown_fields():
    with pytest.raises(ValidationError, match = "extra_forbidden"):
        MLPConfig.model_validate(
            {"hidden_units": [32], "hidden_activations": ["relu"], "dropout": 0.1}
        )


def test_mlp_is_frozen_and_sequence_fields_are_tuples():
    config = MLPConfig(hidden_units = [32], hidden_activations = ["relu"])
    with pytest.raises(ValidationError, match = "frozen_instance"):
        config.hidden_units = (64,)
    with pytest.raises(ValidationError, match = "frozen_instance"):
        config.hidden_activations = ("tanh",)


def test_mlp_has_no_field_defaults():
    assert all(field.is_required() for field in MLPConfig.model_fields.values())


def test_td3_accepts_all_required_fields_and_nested_mlp_configs():
    config = TD3Config.model_validate(_valid_td3_params())

    assert config.replay_capacity == 100_000
    assert config.batch_size == 256
    assert config.gamma == 0.99
    assert config.tau == 0.005
    assert config.actor_learning_rate == 0.001
    assert config.critic_learning_rate == 0.001
    assert config.policy_delay == 2
    assert config.exploration_noise_std == 0.1
    assert config.target_policy_noise_std == 0.2
    assert config.target_noise_clip == 0.5
    assert config.actor_output_init_limit == 0.003
    assert isinstance(config.actor, MLPConfig)
    assert isinstance(config.critic, MLPConfig)
    assert config.actor.hidden_units == (512, 256)
    assert config.critic.hidden_activations == ("relu", "relu")


@pytest.mark.parametrize("missing", REQUIRED_TD3_FIELDS)
def test_td3_requires_every_specialized_field(missing):
    params = _valid_td3_params()
    del params[missing]
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


def test_td3_has_no_field_defaults():
    assert set(TD3Config.model_fields) == set(REQUIRED_TD3_FIELDS)
    assert all(field.is_required() for field in TD3Config.model_fields.values())


@pytest.mark.parametrize("field", INTEGER_FIELDS)
@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_td3_rejects_nonpositive_integer_fields(field, bad_value):
    params = _valid_td3_params()
    params[field] = bad_value
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


@pytest.mark.parametrize("field", INTEGER_FIELDS)
@pytest.mark.parametrize("bad_value", [True, False, 2.5, "2", None])
def test_td3_rejects_noninteger_fields_and_bool(field, bad_value):
    params = _valid_td3_params()
    params[field] = bad_value
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


@pytest.mark.parametrize("field", FLOAT_FIELDS)
@pytest.mark.parametrize("bad_value", [True, False, "0.5", None, [0.5]])
def test_td3_rejects_invalid_numeric_types(field, bad_value):
    params = _valid_td3_params()
    params[field] = bad_value
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


@pytest.mark.parametrize("field", FLOAT_FIELDS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_td3_rejects_nonfinite_numbers(field, bad_value):
    params = _valid_td3_params()
    params[field] = bad_value
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


NUMERIC_RANGES = (
    ("gamma", 0.0, 0.999999),
    ("tau", 1e-6, 1.0),
    ("actor_learning_rate", 1e-8, 1.0),
    ("critic_learning_rate", 1e-8, 1.0),
    ("exploration_noise_std", 0.0, 1.0),
    ("target_policy_noise_std", 0.0, 1.0),
    ("target_noise_clip", 0.0, 1.0),
    ("actor_output_init_limit", 1e-8, 0.1),
)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [(field, math.nextafter(low, -math.inf)) for field, low, high in NUMERIC_RANGES]
    + [(field, math.nextafter(high, math.inf)) for field, low, high in NUMERIC_RANGES],
)
def test_td3_rejects_numeric_values_outside_allowed_ranges(field, bad_value):
    params = _valid_td3_params()
    params[field] = bad_value
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


@pytest.mark.parametrize(
    ("field", "value"),
    [(field, low) for field, low, high in NUMERIC_RANGES]
    + [(field, high) for field, low, high in NUMERIC_RANGES],
)
def test_td3_accepts_approved_numeric_boundaries(field, value):
    params = _valid_td3_params()
    params[field] = value
    config = TD3Config.model_validate(params)
    assert getattr(config, field) == value


def test_td3_accepts_minimum_integer_fields():
    params = _valid_td3_params()
    params.update(replay_capacity = 1, batch_size = 1, policy_delay = 1)
    config = TD3Config.model_validate(params)
    assert config.replay_capacity == config.batch_size == config.policy_delay == 1


@pytest.mark.parametrize("capacity,batch", [(1, 2), (255, 256), (2, 3)])
def test_td3_rejects_batch_larger_than_replay_capacity(capacity, batch):
    params = _valid_td3_params()
    params["replay_capacity"] = capacity
    params["batch_size"] = batch
    with pytest.raises(ValidationError, match = "replay_capacity"):
        TD3Config.model_validate(params)


def test_td3_accepts_batch_equal_to_replay_capacity():
    params = _valid_td3_params()
    params["replay_capacity"] = 256
    config = TD3Config.model_validate(params)
    assert config.replay_capacity == config.batch_size == 256


@pytest.mark.parametrize("field", ["actor", "critic"])
def test_td3_accepts_different_actor_and_critic_architectures(field):
    params = _valid_td3_params()
    params[field] = {
        "hidden_units": [128, 64, 16],
        "hidden_activations": ["swish", "tanh", "elu"],
    }
    config = TD3Config.model_validate(params)
    nested = getattr(config, field)
    assert nested.hidden_units == (128, 64, 16)
    assert nested.hidden_activations == ("swish", "tanh", "elu")


@pytest.mark.parametrize("field", ["actor", "critic"])
def test_td3_rejects_invalid_nested_mlp_configuration(field):
    params = _valid_td3_params()
    params[field]["hidden_units"] = [0]
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


@pytest.mark.parametrize("field", ["actor", "critic"])
def test_td3_rejects_extra_nested_mlp_fields(field):
    params = _valid_td3_params()
    params[field]["batch_norm"] = True
    with pytest.raises(ValidationError, match = "extra_forbidden"):
        TD3Config.model_validate(params)


def test_td3_rejects_extra_top_level_fields():
    params = _valid_td3_params()
    params["seed"] = 42
    with pytest.raises(ValidationError, match = "extra_forbidden"):
        TD3Config.model_validate(params)


def test_td3_is_frozen():
    config = TD3Config.model_validate(_valid_td3_params())
    with pytest.raises(ValidationError, match = "frozen_instance"):
        config.batch_size = 512
    with pytest.raises(ValidationError, match = "frozen_instance"):
        config.actor = MLPConfig(hidden_units = (64,), hidden_activations = ("relu",))


def test_td3_model_validate_reports_pydantic_error_without_project_wrapper():
    params = _valid_td3_params()
    params["gamma"] = 1.0
    with pytest.raises(ValidationError):
        TD3Config.model_validate(params)


def test_example_yaml_loads_through_general_and_td3_specific_configuration():
    project_root = Path(__file__).resolve().parents[5]
    run_config = load_config(project_root / "configs" / "td3_pendulum.yaml")

    assert isinstance(run_config, RunConfig)
    assert run_config.environment.id == "Pendulum-v1"
    assert run_config.environment.kwargs == {}
    assert run_config.training.total_steps == 35_000
    assert run_config.training.seed == 42
    assert run_config.algorithm.name == "td3"
    assert run_config.algorithm.params == _valid_td3_params()

    td3_config = TD3Config.model_validate(run_config.algorithm.params)
    assert isinstance(td3_config, TD3Config)
    assert td3_config.actor.hidden_units == (512, 256)
    assert td3_config.critic.hidden_units == (512, 256)
    assert td3_config.actor.hidden_activations == ("relu", "relu")
    assert td3_config.critic.hidden_activations == ("relu", "relu")


def test_td3_validation_does_not_modify_general_config_params():
    project_root = Path(__file__).resolve().parents[5]
    run_config = load_config(project_root / "configs" / "td3_pendulum.yaml")
    original_params = deepcopy(run_config.algorithm.params)

    TD3Config.model_validate(run_config.algorithm.params)

    assert run_config.algorithm.params == original_params
    assert isinstance(run_config.algorithm.params["actor"]["hidden_units"], list)
    assert isinstance(run_config.algorithm.params["critic"]["hidden_units"], list)
