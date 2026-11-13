import math
from pathlib import Path
from typing import Any

import yaml

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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


class TrainingConfig(BaseModel):
    """Common training limits and required reproducibility seed.

    :param total_steps: strictly positive integer number of training steps.
    :param seed: required nonnegative integer random seed.
    :raises pydantic.ValidationError: if a field is missing or invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    total_steps: int = Field(gt = 0)
    seed: int = Field(ge = 0)


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
