"""Validated TD3 hyperparameters and configurable hidden MLP layers."""

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PositiveInt = Annotated[int, Field(gt = 0)]
_HiddenActivation = Literal["relu", "tanh", "elu", "swish"]


class MLPConfig(BaseModel):
    """Hidden Dense-layer configuration shared by Actor and Critic builders.

    :param hidden_units: positive neuron counts for each hidden layer.
    :param hidden_activations: activation name for each corresponding hidden layer;
        one of ``relu``, ``tanh``, ``elu`` or ``swish``.
    :raises pydantic.ValidationError: if either sequence is empty, their lengths
        differ, or any entry is invalid.
    """

    model_config = ConfigDict(extra = "forbid", frozen = True, strict = True)

    hidden_units: tuple[_PositiveInt, ...] = Field(min_length = 1)
    hidden_activations: tuple[_HiddenActivation, ...] = Field(min_length = 1)

    @field_validator("hidden_units", "hidden_activations", mode = "before")
    @classmethod
    def convert_yaml_lists(cls, value: Any) -> Any:
        """Accept YAML lists while keeping tuple fields strictly typed.

        :param value: original field value, normally a list from YAML or a tuple.
        :return: a tuple for list inputs; all other inputs unchanged.
        """
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode = "after")
    def validate_layer_counts(self) -> Self:
        """Require one activation per hidden layer.

        :return: the validated MLP configuration.
        :raises ValueError: if units and activations have different lengths.
        """
        if len(self.hidden_units) != len(self.hidden_activations):
            raise ValueError(
                "hidden_units and hidden_activations must have equal lengths"
            )
        return self


class TD3Config(BaseModel):
    """Required TD3 parameters validated independently of the general RunConfig.

    :param replay_capacity: maximum number of transitions in the Replay Buffer.
    :param batch_size: number of unique transitions per sampled minibatch.
    :param gamma: reward discount factor, in ``[0, 1)``.
    :param tau: target-network soft-update coefficient, in ``(0, 1]``.
    :param actor_learning_rate: positive Adam learning rate for the Actor.
    :param critic_learning_rate: positive Adam learning rate for both Critics.
    :param policy_delay: number of Critic gradient updates per Actor update.
    :param exploration_noise_std: nonnegative standard deviation of exploration noise.
    :param target_policy_noise_std: nonnegative standard deviation of target noise.
    :param target_noise_clip: nonnegative magnitude limit for target noise.
    :param actor_output_init_limit: positive half-width of the Actor output kernel
        initialization range.
    :param actor: hidden-layer configuration for the Actor.
    :param critic: hidden-layer configuration used by both independent Critics.
    :raises pydantic.ValidationError: if a required field or cross-field constraint
        is invalid.
    """

    model_config = ConfigDict(
        extra = "forbid", frozen = True, strict = True, allow_inf_nan = False
    )

    replay_capacity: _PositiveInt
    batch_size: _PositiveInt

    gamma: float = Field(ge = 0.0, lt = 1.0)
    tau: float = Field(gt = 0.0, le = 1.0)

    actor_learning_rate: float = Field(gt = 0.0)
    critic_learning_rate: float = Field(gt = 0.0)

    policy_delay: _PositiveInt

    exploration_noise_std: float = Field(ge = 0.0)
    target_policy_noise_std: float = Field(ge = 0.0)
    target_noise_clip: float = Field(ge = 0.0)

    actor_output_init_limit: float = Field(gt = 0.0)

    actor: MLPConfig
    critic: MLPConfig

    @model_validator(mode = "after")
    def validate_replay_capacity(self) -> Self:
        """Ensure the Replay Buffer can sample a full batch without replacement.

        :return: the validated TD3 configuration.
        :raises ValueError: if the batch exceeds replay capacity.
        """
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        return self
