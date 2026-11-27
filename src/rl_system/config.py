import math
from pathlib import Path
from typing import Any, Self

import yaml

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rl_system.exceptions import InvalidConfigError


def _check_json_value(value: Any, path: str, active: set[int]) -> None:
    """Reject non-JSON Python values, nonfinite numbers, and reference cycles.

    :param value: value or nested container to check.
    :param path: field path used to identify invalid nested values in errors.
    :param active: IDs of containers currently being traversed, for cycle detection.
    :return: None when the value and all nested values are JSON-compatible.
    :raises ValueError: if a value, dictionary key, number, or container cycle is invalid.
    """
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: number must be finite")
        return
    if type(value) not in (dict, list):
        raise ValueError(f"{path}: value is not JSON-compatible")

    identity = id(value)
    if identity in active:
        raise ValueError(f"{path}: cyclic container is not JSON-compatible")

    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{path}: dictionary keys must be strings")
                _check_json_value(item, f"{path}.{key}", active)
        else:
            for index, item in enumerate(value):
                _check_json_value(item, f"{path}[{index}]", active)
    finally:
        active.remove(identity)


def _validate_json_mapping(value: Any) -> dict[str, Any]:
    """Validate a mapping and all its nested values for JSON compatibility.

    :param value: mapping to validate without modifying it.
    :return: the original dictionary after successful validation.
    :raises ValueError: if the value is not a dict, contains invalid JSON values,
        or has nesting deep enough to trigger a RecursionError.
    """
    if type(value) is not dict:
        raise ValueError("expected a dictionary with JSON-compatible values")
    try:
        _check_json_value(value, "value", set())
    except RecursionError as exc:
        raise ValueError("JSON structure nesting is too deep") from exc
    return value


class EnvironmentConfig(BaseModel):
    """Gymnasium environment identifier and constructor keyword arguments.

    :param id: nonempty environment identifier (e.g. ``Pendulum-v1``).
    :param kwargs: optional JSON-compatible keyword arguments for environment creation.
    :raises pydantic.ValidationError: if a field or nested value is invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    id: str = Field(min_length = 1, pattern = r"\S")
    kwargs: dict[str, Any] = Field(default_factory = dict)

    @field_validator("kwargs", mode = "before")
    @classmethod
    def validate_kwargs(cls, value: Any) -> dict[str, Any]:
        """Validate environment keyword arguments before field parsing.

        :param value: raw ``kwargs`` field value.
        :return: the original validated dictionary.
        :raises ValueError: if the mapping contains non-JSON values or cycles.
        """
        return _validate_json_mapping(value)


class AlgorithmConfig(BaseModel):
    """Algorithm identifier and generic, as-yet-unchecked parameters.

    :param name: nonempty algorithm identifier; support is checked later.
    :param params: required JSON-compatible mapping of algorithm parameters.
    :raises pydantic.ValidationError: if a field or nested value is invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    name: str = Field(min_length = 1, pattern = r"\S")
    params: dict[str, Any]

    @field_validator("params", mode = "before")
    @classmethod
    def validate_params(cls, value: Any) -> dict[str, Any]:
        """Validate algorithm parameters before field parsing.

        :param value: raw ``params`` field value.
        :return: the original validated dictionary.
        :raises ValueError: if the mapping contains non-JSON values or cycles.
        """
        return _validate_json_mapping(value)


class ProgressConfig(BaseModel):
    """Console progress reporting schedule and diagnostic options.

    :param step_interval: maximum steps between reports; zero disables step scheduling.
    :param episode_interval: episode reporting period; zero disables episode scheduling.
    :param episode_start: first episode eligible for episode-based reporting.
    :param reward_window: number of completed episodes in reward statistics; zero disables them.
    :param show_update_count: include the total number of successful updates.
    :param show_update_metrics: include statistics of new agent metrics.
    :param show_metric_counts: include counts in the update metrics block when enabled.
    :param show_throughput: include overall and interval throughput.
    :raises pydantic.ValidationError: if a field is missing or invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    step_interval: int = Field(ge = 0)
    episode_interval: int = Field(ge = 0)
    episode_start: int = Field(ge = 1)
    reward_window: int = Field(ge = 0)
    show_update_count: bool
    show_update_metrics: bool
    show_metric_counts: bool
    show_throughput: bool


class TrainingConfig(BaseModel):
    """Common training budget, reproducibility seed, schedules and diagnostics.

    :param total_steps: positive number of environment steps.
    :param seed: run seed in the inclusive range [0, 2**32 - 1].
    :param warmup_steps: initial random-action steps, from zero to total_steps.
    :param learning_starts: first step eligible for updates, from one to total_steps.
    :param train_frequency: positive interval between update series in environment steps.
    :param gradient_steps: maximum number of agent update calls per series.
    :param progress: console progress reporting settings.
    :raises pydantic.ValidationError: if a field or cross-field constraint is invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    total_steps: int = Field(gt = 0)
    seed: int = Field(ge = 0, le = 2 ** 32 - 1)
    warmup_steps: int = Field(ge = 0)
    learning_starts: int = Field(ge = 1)
    train_frequency: int = Field(ge = 1)
    gradient_steps: int = Field(ge = 1)
    progress: ProgressConfig

    @model_validator(mode = "after")
    def validate_step_limits(self) -> Self:
        """Ensure warm-up and learning start fit within the training budget.

        :return: the validated training configuration.
        :raises ValueError: if either step limit exceeds total_steps.
        """
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must not exceed total_steps")
        if self.learning_starts > self.total_steps:
            raise ValueError("learning_starts must not exceed total_steps")
        return self


class RunConfig(BaseModel):
    """Validated user specification, not the effective runtime configuration.

    :param environment: validated environment settings.
    :param algorithm: validated algorithm settings.
    :param training: validated training settings.
    :raises pydantic.ValidationError: if a section is missing or invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    environment: EnvironmentConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Load YAML safely while rejecting duplicate mapping keys at any level."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Construct a mapping with unique keys.

        :param node: YAML node representing a mapping.
        :param deep: whether to construct nested keys and values deeply.
        :return: mapping with validated, unique YAML keys.
        :raises yaml.constructor.ConstructorError: if a key is duplicated or unhashable.
        """
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, "expected a mapping node", node.start_mark
            )

        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep = deep)
            try:
                if key in mapping:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"duplicate key {key!r}",
                        key_node.start_mark,
                    )
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                ) from exc
            mapping[key] = self.construct_object(value_node, deep = deep)
        return mapping


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML run specification and validate its general schema.

    :param path: path to a YAML configuration file.
    :return: validated user RunConfig, without runtime or TD3-specific checks.
    :raises InvalidConfigError: if YAML is malformed, contains duplicate keys,
        has invalid encoding, or fails RunConfig validation.
    :raises OSError: if the configuration file cannot be opened or read.
    """
    try:
        with Path(path).open("r", encoding = "utf-8") as stream:
            data = yaml.load(stream, Loader = _UniqueKeySafeLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise InvalidConfigError(f"Invalid YAML configuration: {exc}") from exc

    try:
        return RunConfig.model_validate(data)
    except ValidationError as exc:
        raise InvalidConfigError(f"Invalid run configuration: {exc}") from exc
