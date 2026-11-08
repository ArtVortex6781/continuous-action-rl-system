class RLSystemError(Exception):
    """Базовое исключение RL-системы."""


class ConfigurationError(RLSystemError):
    """Базовая ошибка конфигурации запуска."""


class InvalidConfigError(ConfigurationError):
    """Некорректная структура, тип или значение конфигурации."""


class UnsupportedConfigError(ConfigurationError):
    """Запрошенная конфигурация не поддерживается системой."""
