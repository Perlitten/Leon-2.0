"""
Модуль с базовыми исключениями для проекта Leon Trading Bot.
Определяет иерархию исключений для различных компонентов системы.
"""

from core.constants import ERROR_CODES


class LeonError(Exception):
    """Базовое исключение для всех ошибок Leon Trading Bot."""
    
    def __init__(self, message: str, code: int = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            code: Код ошибки
        """
        self.message = message
        self.code = code
        super().__init__(self.message)


# Ошибки конфигурации
class ConfigError(LeonError):
    """Ошибка в конфигурации."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Ошибка в конфигурации"
        super().__init__(self.message, **kwargs)


class ConfigValidationError(ConfigError):
    """Ошибка валидации конфигурации."""
    
    def __init__(self, message: str = None, invalid_fields=None, **kwargs):
        self.invalid_fields = invalid_fields or []
        self.message = message or "Ошибка валидации конфигурации"
        super().__init__(self.message, invalid_fields=self.invalid_fields, **kwargs)


class ConfigLoadError(ConfigError):
    """Ошибка загрузки конфигурации."""
    
    def __init__(self, message: str = None, file_path=None, **kwargs):
        self.file_path = file_path
        self.message = message or "Ошибка загрузки конфигурации"
        super().__init__(self.message, file_path=self.file_path, **kwargs)


# Ошибки API
class APIError(LeonError):
    """Ошибка при взаимодействии с внешним API."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            api_name: Имя API, вызвавшего ошибку
            status_code: HTTP-код состояния
        """
        self.api_name = api_name
        self.status_code = status_code
        code = ERROR_CODES["API_ERROR"]
        super().__init__(message, code)


class APIConnectionError(APIError):
    """Ошибка соединения с API."""
    
    def __init__(self, message: str = None, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.message = message or "Ошибка соединения с API"
        super().__init__(self.message, endpoint=self.endpoint, **kwargs)


class APIResponseError(APIError):
    """Ошибка в ответе API."""
    
    def __init__(self, message: str = None, response=None, **kwargs):
        self.response = response
        self.message = message or "Ошибка в ответе API"
        super().__init__(self.message, response=self.response, **kwargs)


class APIAuthenticationError(APIError):
    """Ошибка аутентификации при взаимодействии с API."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Ошибка аутентификации при взаимодействии с API"
        super().__init__(self.message, **kwargs)


# Ошибки данных
class DataError(LeonError):
    """Ошибка при работе с данными."""
    
    def __init__(self, message: str, data_source: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            data_source: Источник данных, вызвавший ошибку
        """
        self.data_source = data_source
        code = ERROR_CODES["DATA_ERROR"]
        super().__init__(message, code)


class DataValidationError(DataError):
    """Ошибка валидации данных."""
    
    def __init__(self, message: str = None, invalid_data=None, **kwargs):
        self.invalid_data = invalid_data
        self.message = message or "Ошибка валидации данных"
        super().__init__(self.message, invalid_data=self.invalid_data, **kwargs)


class DataStorageError(DataError):
    """Ошибка при сохранении или загрузке данных."""
    
    def __init__(self, message: str = None, storage_path=None, **kwargs):
        self.storage_path = storage_path
        self.message = message or "Ошибка при сохранении или загрузке данных"
        super().__init__(self.message, storage_path=self.storage_path, **kwargs)


class DataNotFoundError(DataError):
    """Ошибка при попытке доступа к несуществующим данным."""
    
    def __init__(self, message: str = None, data_id=None, **kwargs):
        self.data_id = data_id
        self.message = message or "Данные не найдены"
        super().__init__(self.message, data_id=self.data_id, **kwargs)


# Ошибки торговли
class TradingError(LeonError):
    """Ошибка в торговой логике."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Ошибка в торговой логике"
        super().__init__(self.message, **kwargs)


class OrderError(TradingError):
    """Ошибка при создании или управлении ордером."""
    
    def __init__(self, message: str = None, order_id=None, **kwargs):
        self.order_id = order_id
        self.message = message or "Ошибка при создании или управлении ордером"
        super().__init__(self.message, order_id=self.order_id, **kwargs)


class PositionError(TradingError):
    """Ошибка при управлении позицией."""
    
    def __init__(self, message: str = None, position_id=None, **kwargs):
        self.position_id = position_id
        self.message = message or "Ошибка позиции"
        super().__init__(self.message, position_id=self.position_id, **kwargs)


class InsufficientFundsError(TradingError):
    """Ошибка недостаточного баланса для выполнения операции."""
    
    def __init__(self, message: str = None, required=None, available=None, **kwargs):
        self.required = required
        self.available = available
        self.message = message or "Недостаточно средств для выполнения операции"
        super().__init__(self.message, required=self.required, available=self.available, **kwargs)


class PositionLimitError(TradingError):
    """Ошибка превышения лимита позиций."""
    
    def __init__(self, message: str = None, limit: int = None, **kwargs):
        self.limit = limit
        self.message = message or "Превышен лимит позиций"
        super().__init__(self.message, limit=self.limit, **kwargs)


# Ошибки ML
class MLError(LeonError):
    """Ошибка в ML-компоненте."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Ошибка в ML-компоненте"
        super().__init__(self.message, **kwargs)


class ModelLoadError(MLError):
    """Ошибка загрузки ML-модели."""
    
    def __init__(self, message: str, model_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            model_name: Имя модели, вызвавшей ошибку
        """
        self.model_name = model_name
        code = ERROR_CODES["MODEL_LOAD_ERROR"]
        super().__init__(message, code)


class PredictionError(MLError):
    """Ошибка при выполнении предсказания."""
    
    def __init__(self, message: str, model_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            model_name: Имя модели, вызвавшей ошибку
        """
        self.model_name = model_name
        code = ERROR_CODES["PREDICTION_ERROR"]
        super().__init__(message, code)


class EvaluationError(MLError):
    """Ошибка оценки модели."""
    
    def __init__(self, message: str, model_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            model_name: Имя модели, вызвавшей ошибку
        """
        self.model_name = model_name
        code = ERROR_CODES["EVALUATION_ERROR"]
        super().__init__(message, code)


# Ошибки уведомлений
class NotificationError(LeonError):
    """Ошибка при отправке уведомления."""
    
    def __init__(self, message: str, notification_type: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            notification_type: Тип уведомления, вызвавшего ошибку
        """
        self.notification_type = notification_type
        code = ERROR_CODES["NOTIFICATION_ERROR"]
        super().__init__(message, code)


class TelegramError(NotificationError):
    """Ошибка при отправке уведомления в Telegram."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Ошибка при отправке уведомления в Telegram"
        super().__init__(self.message, **kwargs)


# Ошибки системы
class SystemError(LeonError):
    """Системная ошибка."""
    
    def __init__(self, message: str = None, **kwargs):
        self.message = message or "Системная ошибка"
        super().__init__(self.message, **kwargs)


class ResourceExhaustedError(SystemError):
    """Ошибка исчерпания ресурсов."""
    
    def __init__(self, message: str = None, resource_type: str = None, **kwargs):
        self.resource_type = resource_type
        self.message = message or "Исчерпаны ресурсы"
        super().__init__(self.message, resource_type=self.resource_type, **kwargs)


class TimeoutError(SystemError):
    """Ошибка превышения времени ожидания."""
    
    def __init__(self, message: str = None, operation: str = None, timeout: float = None, **kwargs):
        self.operation = operation
        self.timeout = timeout
        self.message = message or "Превышено время ожидания"
        super().__init__(self.message, operation=self.operation, timeout=self.timeout, **kwargs)


# Ошибки оркестратора
class OrchestratorError(LeonError):
    """Базовое исключение для ошибок оркестратора."""
    
    def __init__(self, message: str, code: int = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            code: Код ошибки
        """
        super().__init__(message, code)


class InitializationError(OrchestratorError):
    """Ошибка инициализации компонентов."""
    
    def __init__(self, message: str, component: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            component: Компонент, вызвавший ошибку
        """
        self.component = component
        code = ERROR_CODES["INITIALIZATION_ERROR"]
        super().__init__(message, code)


class OperationError(OrchestratorError):
    """Ошибка выполнения операции."""
    
    def __init__(self, message: str, operation: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            operation: Операция, вызвавшая ошибку
        """
        self.operation = operation
        code = ERROR_CODES["OPERATION_ERROR"]
        super().__init__(message, code)


class InvalidModeError(OrchestratorError):
    """Недопустимый режим работы."""
    
    def __init__(self, message: str, mode: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            mode: Недопустимый режим работы
        """
        self.mode = mode
        code = ERROR_CODES["INVALID_MODE_ERROR"]
        super().__init__(message, code)


class CommandError(OrchestratorError):
    """Ошибка выполнения команды."""
    
    def __init__(self, message: str, command: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            command: Команда, вызвавшая ошибку
        """
        self.command = command
        code = ERROR_CODES["COMMAND_ERROR"]
        super().__init__(message, code)


class NetworkError(LeonError):
    """Исключение для ошибок сети."""
    
    def __init__(self, message: str, url: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            url: URL, вызвавший ошибку
        """
        self.url = url
        code = ERROR_CODES["NETWORK_ERROR"]
        super().__init__(message, code)


class StrategyError(LeonError):
    """Исключение для ошибок стратегий."""
    
    def __init__(self, message: str, strategy_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            strategy_name: Имя стратегии, вызвавшей ошибку
        """
        self.strategy_name = strategy_name
        code = ERROR_CODES["STRATEGY_ERROR"]
        super().__init__(message, code)


class TraderError(LeonError):
    """Исключение для ошибок трейдеров."""
    
    def __init__(self, message: str, trader_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            trader_name: Имя трейдера, вызвавшего ошибку
        """
        self.trader_name = trader_name
        code = ERROR_CODES["TRADER_ERROR"]
        super().__init__(message, code)


class VisualizationError(LeonError):
    """Исключение для ошибок визуализации."""
    
    def __init__(self, message: str, visualizer_name: str = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке
            visualizer_name: Имя визуализатора, вызвавшего ошибку
        """
        self.visualizer_name = visualizer_name
        code = ERROR_CODES["VISUALIZATION_ERROR"]
        super().__init__(message, code) 