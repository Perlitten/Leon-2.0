# Модуль отказоустойчивости (core.resilience)

## Обзор

Модуль `core.resilience` предоставляет механизмы для обеспечения отказоустойчивости приложения, включая шаблон Circuit Breaker (Предохранитель) и механизм повторных попыток (Retry). Эти механизмы помогают системе справляться с временными сбоями и предотвращать каскадные отказы.

## Классы и функции

### CircuitBreaker

```python
class CircuitBreaker:
    def __init__(self, name, failure_threshold=5, recovery_timeout=60, 
                 half_open_max_calls=3, exception_types=None, 
                 on_open=None, on_close=None, on_half_open=None)
```

Реализует шаблон Circuit Breaker для защиты системы от каскадных отказов.

#### Параметры:

- `name` (str): Уникальное имя для идентификации предохранителя
- `failure_threshold` (int): Количество последовательных сбоев, после которых предохранитель переходит в состояние OPEN
- `recovery_timeout` (int): Время в секундах, после которого предохранитель переходит из состояния OPEN в HALF-OPEN
- `half_open_max_calls` (int): Максимальное количество вызовов в состоянии HALF-OPEN
- `exception_types` (list): Список типов исключений, которые считаются сбоями
- `on_open` (callable): Функция обратного вызова при переходе в состояние OPEN
- `on_close` (callable): Функция обратного вызова при переходе в состояние CLOSED
- `on_half_open` (callable): Функция обратного вызова при переходе в состояние HALF-OPEN

#### Методы:

```python
async def execute(self, func, *args, fallback=None, **kwargs)
```

Выполняет функцию с учетом текущего состояния предохранителя.

- `func` (callable): Функция для выполнения
- `*args`: Позиционные аргументы для функции
- `fallback` (callable, optional): Функция, которая будет вызвана, если предохранитель открыт
- `**kwargs`: Именованные аргументы для функции
- Возвращает: Результат выполнения функции или fallback

```python
def get_state(self)
```

Возвращает текущее состояние предохранителя.

- Возвращает: Одно из значений `CircuitBreaker.CLOSED`, `CircuitBreaker.OPEN`, `CircuitBreaker.HALF_OPEN`

```python
def reset(self)
```

Сбрасывает предохранитель в закрытое состояние.

### RetryConfig

```python
class RetryConfig:
    def __init__(self, max_retries=3, delay=1, max_delay=60, 
                 backoff_factor=2, jitter=0.1, retry_on=None)
```

Конфигурация для механизма повторных попыток.

#### Параметры:

- `max_retries` (int): Максимальное количество повторных попыток
- `delay` (float): Начальная задержка между попытками в секундах
- `max_delay` (float): Максимальная задержка между попытками в секундах
- `backoff_factor` (float): Множитель для экспоненциального увеличения задержки
- `jitter` (float): Коэффициент случайного отклонения для задержки (0-1)
- `retry_on` (list): Список исключений, при которых следует повторить попытку

### retry

```python
async def retry(func, *args, config=None, logger=None, **kwargs)
```

Декоратор для выполнения функции с повторными попытками при сбоях.

#### Параметры:

- `func` (callable): Функция для выполнения
- `*args`: Позиционные аргументы для функции
- `config` (RetryConfig, optional): Конфигурация повторных попыток
- `logger` (Logger, optional): Логгер для записи информации о повторных попытках
- `**kwargs`: Именованные аргументы для функции
- Возвращает: Результат выполнения функции
- Вызывает: Исключение последней неудачной попытки, если все попытки не удались

## Примеры использования

### Использование Circuit Breaker

```python
from core.resilience import CircuitBreaker
from core.exceptions import APIError

# Создание предохранителя
breaker = CircuitBreaker(
    name="binance_api",
    failure_threshold=3,
    recovery_timeout=30,
    exception_types=[APIError]
)

# Использование предохранителя
async def get_market_data():
    async def fallback():
        return {"status": "error", "message": "Service unavailable"}
    
    result = await breaker.execute(
        api_client.get_ticker,
        symbol="BTCUSDT",
        fallback=fallback
    )
    return result
```

### Использование механизма повторных попыток

```python
from core.resilience import retry, RetryConfig
from core.exceptions import APIRateLimitError, APIError

# Создание конфигурации повторных попыток
retry_config = RetryConfig(
    max_retries=5,
    delay=1,
    backoff_factor=2,
    retry_on=[APIRateLimitError, ConnectionError]
)

# Использование механизма повторных попыток
@retry(config=retry_config)
async def fetch_data(symbol):
    return await api_client.get_klines(symbol=symbol, interval="1h")

# Или без декоратора
async def fetch_historical_data(symbol, interval):
    return await retry(
        api_client.get_klines,
        symbol=symbol,
        interval=interval,
        config=retry_config
    )
```

### Комбинирование Circuit Breaker и Retry

```python
from core.resilience import CircuitBreaker, retry, RetryConfig
from core.exceptions import APIError

# Создание предохранителя и конфигурации повторных попыток
breaker = CircuitBreaker(name="binance_api", failure_threshold=3)
retry_config = RetryConfig(max_retries=3, delay=1)

# Использование обоих механизмов
async def get_exchange_info():
    async def fetch_with_retry():
        return await retry(
            api_client.get_exchange_info,
            config=retry_config
        )
    
    return await breaker.execute(fetch_with_retry)
```

## Исключения

Модуль не генерирует собственных исключений, но перехватывает исключения, указанные в параметрах `exception_types` для `CircuitBreaker` и `retry_on` для `RetryConfig`.

## Зависимости

- `logging`: Для логирования состояний и событий
- `time`: Для работы с временными интервалами
- `random`: Для добавления случайного отклонения (jitter) в задержки
- `asyncio`: Для асинхронного выполнения операций
- `core.exceptions`: Для типов исключений, используемых в примерах 