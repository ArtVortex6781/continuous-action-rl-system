from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_system.training import TrainingResult


class RLSystemError(Exception):
    """Базовое исключение RL-системы."""


class ConfigurationError(RLSystemError):
    """Базовая ошибка конфигурации запуска."""


class InvalidConfigError(ConfigurationError):
    """Некорректная структура, тип или значение конфигурации."""


class UnsupportedConfigError(ConfigurationError):
    """Запрошенная конфигурация не поддерживается системой."""


class UnsupportedSpaceError(UnsupportedConfigError):
    """Неподдерживаемое пространство наблюдений или действий."""


class EnvironmentSystemError(RLSystemError):
    """Базовая ошибка подсистемы Environment."""


class EnvironmentDependencyError(EnvironmentSystemError):
    """Отсутствует зависимость, необходимая для среды."""


class EnvironmentCreationError(EnvironmentSystemError):
    """Ошибка создания Gymnasium-среды."""


class AgentSystemError(RLSystemError):
    """Базовая ошибка подсистемы Agent."""


class AgentInputError(AgentSystemError):
    """Нарушение входного контракта агента."""


class AgentUpdateError(AgentSystemError):
    """Нарушение структурного или вычислительного контракта обновления."""


class AgentNumericalError(AgentSystemError):
    """Неконечное значение внутренних вычислений или состояния агента."""


class TrainingError(RLSystemError):
    """Ошибка обучения с доступным частичным результатом.

    :param message: описание ошибки обучения.
    :param result: частичный результат без копирования истории метрик.
    """

    def __init__(self, message: str, result: "TrainingResult") -> None:
        super().__init__(message)
        self.result = result


class TrainingInterruptedError(TrainingError):
    """Обучение прервано пользователем; доступен частичный результат."""
