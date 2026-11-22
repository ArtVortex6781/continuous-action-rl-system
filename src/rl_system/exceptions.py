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
