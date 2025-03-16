# Модуль обработки ошибок (core.exceptions)

## Обзор

Модуль `core.exceptions` содержит иерархию исключений, используемых в проекте Leon Trading Bot. Эти исключения обеспечивают структурированный подход к обработке ошибок и позволяют точно определить причину сбоя.

## Классы и функции

### Базовые исключения

```python
class LeonError(Exception):
    def __init__(self, message="Произошла ошибка в системе Leon Trading Bot", **kwargs)
```

Базовое исключение для всех ошибок в системе Leon Trading Bot.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `**kwargs`: Дополнительные атрибуты, которые будут добавлены к исключению

### Исключения конфигурации

```python
class ConfigError(LeonError):
    def __init__(self, message="Ошибка конфигурации", **kwargs)
```

Базовое исключение для ошибок, связанных с конфигурацией.

```python
class ConfigValidationError(ConfigError):
    def __init__(self, message="Ошибка валидации конфигурации", invalid_fields=None, **kwargs)
```

Исключение, возникающее при ошибке валидации конфигурации.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `invalid_fields` (list): Список полей, не прошедших валидацию
- `**kwargs`: Дополнительные атрибуты

```python
class ConfigLoadError(ConfigError):
    def __init__(self, message="Ошибка загрузки конфигурации", file_path=None, **kwargs)
```

Исключение, возникающее при ошибке загрузки конфигурации.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `file_path` (str): Путь к файлу конфигурации
- `**kwargs`: Дополнительные атрибуты

### Исключения API

```python
class APIError(LeonError):
    def __init__(self, message="Ошибка API", status_code=None, **kwargs)
```

Базовое исключение для ошибок, связанных с API.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `status_code` (int): Код состояния HTTP
- `**kwargs`: Дополнительные атрибуты

```python
class BinanceAPIError(APIError):
    def __init__(self, message="Ошибка Binance API", error_code=None, **kwargs)
```

Исключение, возникающее при ошибке Binance API.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `error_code` (int): Код ошибки Binance API
- `**kwargs`: Дополнительные атрибуты

```python
class APIRateLimitError(APIError):
    def __init__(self, message="Превышен лимит запросов API", retry_after=None, **kwargs)
```

Исключение, возникающее при превышении лимита запросов API.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `retry_after` (int): Время в секундах до следующей попытки
- `**kwargs`: Дополнительные атрибуты

```python
class APIAuthenticationError(APIError):
    def __init__(self, message="Ошибка аутентификации API", **kwargs)
```

Исключение, возникающее при ошибке аутентификации API.

### Исключения данных

```python
class DataError(LeonError):
    def __init__(self, message="Ошибка данных", **kwargs)
```

Базовое исключение для ошибок, связанных с данными.

```python
class DataValidationError(DataError):
    def __init__(self, message="Ошибка валидации данных", **kwargs)
```

Исключение, возникающее при ошибке валидации данных.

```python
class DataStorageError(DataError):
    def __init__(self, message="Ошибка хранения данных", storage_type=None, **kwargs)
```

Исключение, возникающее при ошибке хранения данных.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `storage_type` (str): Тип хранилища (CSV, SQLite, JSON)
- `**kwargs`: Дополнительные атрибуты

```python
class DataNotFoundError(DataError):
    def __init__(self, message="Данные не найдены", **kwargs)
```

Исключение, возникающее, когда данные не найдены.

### Исключения торговли

```python
class TradingError(LeonError):
    def __init__(self, message="Ошибка торговли", **kwargs)
```

Базовое исключение для ошибок, связанных с торговлей.

```python
class OrderError(TradingError):
    def __init__(self, message="Ошибка ордера", order_id=None, **kwargs)
```

Исключение, возникающее при ошибке ордера.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `order_id` (str): Идентификатор ордера
- `**kwargs`: Дополнительные атрибуты

```python
class InsufficientFundsError(TradingError):
    def __init__(self, message="Недостаточно средств", required=None, available=None, **kwargs)
```

Исключение, возникающее при недостаточном количестве средств.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `required` (float): Требуемое количество средств
- `available` (float): Доступное количество средств
- `**kwargs`: Дополнительные атрибуты

```python
class PositionLimitError(TradingError):
    def __init__(self, message="Превышен лимит позиций", **kwargs)
```

Исключение, возникающее при превышении лимита позиций.

### Исключения машинного обучения

```python
class MLError(LeonError):
    def __init__(self, message="Ошибка машинного обучения", **kwargs)
```

Базовое исключение для ошибок, связанных с машинным обучением.

```python
class ModelLoadError(MLError):
    def __init__(self, message="Ошибка загрузки модели", model_path=None, **kwargs)
```

Исключение, возникающее при ошибке загрузки модели.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `model_path` (str): Путь к файлу модели
- `**kwargs`: Дополнительные атрибуты

```python
class PredictionError(MLError):
    def __init__(self, message="Ошибка предсказания", **kwargs)
```

Исключение, возникающее при ошибке предсказания.

### Исключения уведомлений

```python
class NotificationError(LeonError):
    def __init__(self, message="Ошибка уведомления", **kwargs)
```

Базовое исключение для ошибок, связанных с уведомлениями.

```python
class TelegramError(NotificationError):
    def __init__(self, message="Ошибка Telegram", **kwargs)
```

Исключение, возникающее при ошибке Telegram.

### Системные исключения

```python
class SystemError(LeonError):
    def __init__(self, message="Системная ошибка", **kwargs)
```

Базовое исключение для системных ошибок.

```python
class ResourceExhaustedError(SystemError):
    def __init__(self, message="Ресурсы исчерпаны", resource_type=None, **kwargs)
```

Исключение, возникающее при исчерпании ресурсов.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `resource_type` (str): Тип ресурса (память, диск, CPU)
- `**kwargs`: Дополнительные атрибуты

```python
class TimeoutError(SystemError):
    def __init__(self, message="Превышено время ожидания", operation=None, timeout=None, **kwargs)
```

Исключение, возникающее при превышении времени ожидания.

#### Параметры:

- `message` (str): Сообщение об ошибке
- `operation` (str): Операция, которая вызвала таймаут
- `timeout` (float): Время ожидания в секундах
- `**kwargs`: Дополнительные атрибуты

## Примеры использования

### Базовое использование

```python
from core.exceptions import LeonError, APIError, DataNotFoundError

# Создание и вызов исключения
try:
    raise APIError("Не удалось подключиться к API", status_code=500)
except APIError as e:
    print(f"Произошла ошибка API: {e}, код состояния: {e.status_code}")
except LeonError as e:
    print(f"Произошла общая ошибка: {e}")
```

### Использование с дополнительными атрибутами

```python
from core.exceptions import OrderError

# Создание исключения с дополнительными атрибутами
try:
    raise OrderError(
        message="Не удалось создать ордер",
        order_id="12345",
        symbol="BTCUSDT",
        side="BUY",
        price=50000
    )
except OrderError as e:
    print(f"Ошибка ордера: {e}")
    print(f"ID ордера: {e.order_id}")
    print(f"Символ: {e.symbol}")
    print(f"Сторона: {e.side}")
    print(f"Цена: {e.price}")
```

### Использование в обработчике исключений

```python
from core.exceptions import APIRateLimitError, BinanceAPIError, APIError

async def fetch_data_safely():
    try:
        return await api_client.get_ticker(symbol="BTCUSDT")
    except APIRateLimitError as e:
        logger.warning(f"Превышен лимит запросов: {e}, повтор через {e.retry_after} секунд")
        await asyncio.sleep(e.retry_after)
        return await fetch_data_safely()  # Рекурсивный вызов после ожидания
    except BinanceAPIError as e:
        logger.error(f"Ошибка Binance API: {e}, код ошибки: {e.error_code}")
        return None
    except APIError as e:
        logger.error(f"Общая ошибка API: {e}")
        return None
```

## Исключения

Модуль сам по себе не генерирует исключений, а предоставляет классы исключений для использования в других модулях.

## Зависимости

Модуль не имеет внешних зависимостей, кроме стандартной библиотеки Python. 