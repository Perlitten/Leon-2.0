# Руководство по интеграции с модулем отказоустойчивости

## Обзор

Модуль `core.resilience` предоставляет набор инструментов для обеспечения надежности и отказоустойчивости при взаимодействии с внешними сервисами и API. Он включает в себя следующие компоненты:

1. **Circuit Breaker** - паттерн для предотвращения каскадных отказов
2. **Retry** - механизм повторных попыток с экспоненциальной задержкой
3. **Bulkhead** - паттерн для изоляции отказов и ограничения одновременных операций
4. **Rate Limiter** - ограничитель скорости запросов

## Требования для интеграции

Для интеграции с модулем отказоустойчивости необходимо:

1. Использовать асинхронное программирование (async/await)
2. Настроить логирование для отслеживания работы компонентов
3. Импортировать необходимые классы и функции из модуля `core.resilience`
4. Правильно обрабатывать исключения, определенные в `core.exceptions`

## Примеры интеграции

### 1. Интеграция с Circuit Breaker

Circuit Breaker предотвращает повторные вызовы операций, которые с высокой вероятностью завершатся неудачей.

#### Прямое использование

```python
from core.resilience import CircuitBreaker

# Создание экземпляра Circuit Breaker
circuit_breaker = CircuitBreaker(
    name="binance_api",
    failure_threshold=5,  # Количество ошибок до перехода в состояние OPEN
    recovery_timeout=30,  # Время в секундах до перехода из OPEN в HALF-OPEN
    reset_timeout=60,     # Время в секундах до сброса счетчика ошибок
    success_threshold=2   # Количество успешных вызовов для перехода в CLOSED
)

# Использование Circuit Breaker
async def get_market_data():
    async def operation():
        # Основная операция
        return await api_client.get_ticker("BTCUSDT")
    
    async def fallback():
        # Резервная операция при отказе
        return {"symbol": "BTCUSDT", "price": "0.0", "is_fallback": True}
    
    return await circuit_breaker.execute(operation, fallback)
```

#### Использование декоратора

```python
from core.resilience import CircuitBreaker, with_circuit_breaker

# Создание экземпляра Circuit Breaker
circuit_breaker = CircuitBreaker(name="binance_api")

# Резервная функция
async def ticker_fallback(symbol):
    return {"symbol": symbol, "price": "0.0", "is_fallback": True}

# Применение декоратора к функции
@with_circuit_breaker(circuit_breaker, fallback=ticker_fallback)
async def get_ticker(symbol):
    return await api_client.get_ticker(symbol)
```

### 2. Интеграция с механизмом повторных попыток

Механизм повторных попыток автоматически повторяет операцию при возникновении определенных исключений.

#### Прямое использование

```python
from core.resilience import retry, RetryConfig
from core.exceptions import APIError

# Настройка конфигурации повторных попыток
retry_config = RetryConfig(
    max_retries=3,           # Максимальное количество повторных попыток
    base_delay=1.0,          # Базовая задержка в секундах
    max_delay=10.0,          # Максимальная задержка в секундах
    jitter_factor=0.5,       # Коэффициент случайного отклонения
    retry_on=[APIError],     # Список исключений для повторных попыток
    timeout=30.0             # Общий таймаут для всех попыток
)

# Использование механизма повторных попыток
async def get_account_info():
    return await retry(api_client.get_account_info, retry_config)
```

#### Использование декоратора

```python
from core.resilience import RetryConfig, with_retry
from core.exceptions import APIError

# Настройка конфигурации повторных попыток
retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    retry_on=[APIError]
)

# Применение декоратора к функции
@with_retry(retry_config)
async def get_account_info():
    return await api_client.get_account_info()
```

### 3. Интеграция с Bulkhead

Bulkhead ограничивает количество одновременных вызовов операций, что позволяет изолировать отказы.

#### Прямое использование

```python
from core.resilience import Bulkhead, BulkheadConfig

# Настройка конфигурации Bulkhead
bulkhead_config = BulkheadConfig(
    max_concurrent=10,    # Максимальное количество одновременных выполнений
    max_queue_size=100,   # Максимальный размер очереди ожидания
    queue_timeout=30.0    # Таймаут ожидания в очереди в секундах
)

# Создание экземпляра Bulkhead
bulkhead = Bulkhead(name="api_requests", config=bulkhead_config)

# Использование Bulkhead
async def get_market_data(symbol):
    return await bulkhead.execute(api_client.get_ticker, symbol)
```

#### Использование декоратора

```python
from core.resilience import Bulkhead, BulkheadConfig, with_bulkhead

# Настройка и создание Bulkhead
bulkhead_config = BulkheadConfig(max_concurrent=5)
bulkhead = Bulkhead(name="api_requests", config=bulkhead_config)

# Применение декоратора к функции
@with_bulkhead(bulkhead)
async def get_ticker(symbol):
    return await api_client.get_ticker(symbol)
```

### 4. Интеграция с Rate Limiter

Rate Limiter ограничивает скорость запросов к внешним сервисам.

#### Прямое использование

```python
from core.resilience import RateLimiter

# Создание экземпляра Rate Limiter
rate_limiter = RateLimiter(
    name="binance_api",
    rate=10,     # Максимальное количество запросов в единицу времени
    period=1.0   # Период времени в секундах
)

# Использование Rate Limiter
async def get_ticker(symbol):
    if await rate_limiter.acquire():
        return await api_client.get_ticker(symbol)
    else:
        # Обработка превышения лимита
        raise APIRateLimitError("Превышен лимит запросов")
```

#### Использование декоратора

```python
from core.resilience import RateLimiter, with_rate_limit

# Создание экземпляра Rate Limiter
rate_limiter = RateLimiter(name="binance_api", rate=10, period=1.0)

# Применение декоратора к функции
@with_rate_limit(rate_limiter)
async def get_ticker(symbol):
    return await api_client.get_ticker(symbol)
```

## Комбинирование компонентов

Компоненты отказоустойчивости можно комбинировать для создания надежных операций:

```python
from core.resilience import (
    CircuitBreaker, RetryConfig, Bulkhead, RateLimiter,
    with_circuit_breaker, with_retry, with_bulkhead, with_rate_limit
)

# Создание экземпляров компонентов
circuit_breaker = CircuitBreaker(name="binance_api")
retry_config = RetryConfig(max_retries=3)
bulkhead = Bulkhead(name="api_requests")
rate_limiter = RateLimiter(name="binance_api", rate=10, period=1.0)

# Комбинирование декораторов
@with_circuit_breaker(circuit_breaker)
@with_retry(retry_config)
@with_bulkhead(bulkhead)
@with_rate_limit(rate_limiter)
async def get_ticker(symbol):
    return await api_client.get_ticker(symbol)
```

## Рекомендации по использованию

1. **Circuit Breaker** - используйте для защиты от каскадных отказов при взаимодействии с внешними сервисами.
2. **Retry** - применяйте для временных ошибок, которые могут быть устранены при повторной попытке.
3. **Bulkhead** - используйте для ограничения количества одновременных запросов и изоляции отказов.
4. **Rate Limiter** - применяйте для соблюдения ограничений API и предотвращения блокировки.

## Обработка ошибок

При использовании модуля отказоустойчивости могут возникать следующие исключения:

- `LeonError` - базовое исключение для всех ошибок
- `APIError` - ошибка при взаимодействии с внешним API
- `APIRateLimitError` - ошибка превышения лимита запросов к API
- `TimeoutError` - ошибка превышения времени ожидания
- `ResourceExhaustedError` - ошибка исчерпания ресурсов

Пример обработки ошибок:

```python
from core.exceptions import LeonError, APIError, TimeoutError, ResourceExhaustedError

try:
    result = await get_ticker("BTCUSDT")
except APIError as e:
    logger.error(f"Ошибка API: {e}")
    # Обработка ошибки API
except TimeoutError as e:
    logger.error(f"Превышено время ожидания: {e}")
    # Обработка таймаута
except ResourceExhaustedError as e:
    logger.error(f"Исчерпаны ресурсы: {e}")
    # Обработка исчерпания ресурсов
except LeonError as e:
    logger.error(f"Общая ошибка: {e}")
    # Обработка общей ошибки
```

## Настройка логирования

Для эффективного мониторинга работы компонентов отказоустойчивости рекомендуется настроить логирование:

```python
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/resilience.log"),
        logging.StreamHandler()
    ]
)

# Настройка уровня логирования для отдельных компонентов
logging.getLogger("CircuitBreaker").setLevel(logging.DEBUG)
logging.getLogger("Retry").setLevel(logging.INFO)
logging.getLogger("Bulkhead").setLevel(logging.INFO)
logging.getLogger("RateLimiter").setLevel(logging.INFO)
```

## Тестирование

При тестировании компонентов отказоустойчивости рекомендуется:

1. Создавать моки для внешних сервисов
2. Тестировать поведение при различных сценариях отказов
3. Проверять корректность работы механизмов восстановления
4. Тестировать комбинации компонентов

Пример тестирования Circuit Breaker:

```python
import unittest
from unittest.mock import AsyncMock, patch
from core.resilience import CircuitBreaker
from core.exceptions import APIError

class TestCircuitBreaker(unittest.TestCase):
    
    async def test_circuit_breaker_open_after_failures(self):
        # Создание Circuit Breaker с низким порогом ошибок
        cb = CircuitBreaker(name="test", failure_threshold=2)
        
        # Создание мока операции, которая всегда завершается с ошибкой
        operation = AsyncMock(side_effect=APIError("Test error"))
        fallback = AsyncMock(return_value="Fallback result")
        
        # Первый вызов - ошибка, но Circuit Breaker остается закрытым
        with self.assertRaises(APIError):
            await cb.execute(operation)
        
        # Второй вызов - ошибка, Circuit Breaker должен открыться
        with self.assertRaises(APIError):
            await cb.execute(operation)
        
        # Третий вызов - должен использоваться fallback
        result = await cb.execute(operation, fallback)
        self.assertEqual(result, "Fallback result")
        
        # Проверка, что операция была вызвана только 2 раза
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(fallback.call_count, 1)
``` 