"""
Модуль отказоустойчивости для Leon Trading Bot.

Предоставляет инструменты для обеспечения надежности и отказоустойчивости
при взаимодействии с внешними сервисами и API. Включает реализацию паттерна
Circuit Breaker и механизмы повторных попыток с экспоненциальной задержкой.
"""

import logging
import time
import random
import asyncio
from typing import Callable, Any, Optional, TypeVar, Dict, List, Union
from enum import Enum
from functools import wraps

from core.exceptions import APIError, TimeoutError, LeonError

# Типовые переменные для аннотаций
T = TypeVar('T')
R = TypeVar('R')

# Константы
DEFAULT_RETRY_COUNT = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_JITTER_FACTOR = 0.5
DEFAULT_TIMEOUT = 30.0


class CircuitBreakerState(Enum):
    """Состояния для паттерна Circuit Breaker."""
    CLOSED = "CLOSED"  # Нормальное состояние, запросы проходят
    OPEN = "OPEN"      # Состояние отказа, запросы блокируются
    HALF_OPEN = "HALF-OPEN"  # Пробное состояние, пропускается ограниченное число запросов


class CircuitBreaker:
    """
    Реализация паттерна Circuit Breaker для защиты от каскадных отказов.
    
    Паттерн Circuit Breaker предотвращает повторные вызовы операций, которые
    с высокой вероятностью завершатся неудачей, что позволяет избежать
    перегрузки системы и ускорить восстановление после сбоев.
    
    Attributes:
        name: Имя Circuit Breaker для идентификации в логах
        failure_threshold: Количество ошибок до перехода в состояние OPEN
        recovery_timeout: Время в секундах до перехода из OPEN в HALF-OPEN
        reset_timeout: Время в секундах до сброса счетчика ошибок
        state: Текущее состояние Circuit Breaker
        failure_count: Текущее количество последовательных ошибок
        last_failure_time: Время последней ошибки
        success_threshold: Количество успешных вызовов в состоянии HALF-OPEN для перехода в CLOSED
        success_count: Текущее количество последовательных успешных вызовов
        logger: Логгер для записи событий
    """
    
    def __init__(self, 
                 name: str = "default",
                 failure_threshold: int = 5,
                 recovery_timeout: int = 30,
                 reset_timeout: int = 60,
                 success_threshold: int = 2):
        """
        Инициализация Circuit Breaker.
        
        Args:
            name: Имя Circuit Breaker для идентификации в логах
            failure_threshold: Количество ошибок до перехода в состояние OPEN
            recovery_timeout: Время в секундах до перехода из OPEN в HALF-OPEN
            reset_timeout: Время в секундах до сброса счетчика ошибок
            success_threshold: Количество успешных вызовов в состоянии HALF-OPEN для перехода в CLOSED
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        self.logger.info(f"Circuit Breaker '{name}' инициализирован")
    
    async def execute(self, operation: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """
        Выполнение операции с защитой Circuit Breaker.
        
        Args:
            operation: Асинхронная функция для выполнения
            fallback: Функция для выполнения при открытом Circuit Breaker
            
        Returns:
            Результат операции или fallback
            
        Raises:
            Exception: Если операция завершилась с ошибкой и fallback не предоставлен
        """
        # Проверка состояния Circuit Breaker
        if self.state == CircuitBreakerState.OPEN:
            # Проверка времени восстановления
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.logger.info(f"Circuit Breaker '{self.name}' переходит в состояние HALF-OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
            else:
                self.logger.warning(f"Circuit Breaker '{self.name}' в состоянии OPEN, операция не выполняется")
                if fallback:
                    return await fallback()
                raise LeonError(f"Circuit Breaker '{self.name}' открыт, операция не выполняется")
        
        # Сброс счетчика ошибок, если прошло достаточно времени с последней ошибки
        if (self.state == CircuitBreakerState.CLOSED and 
            self.last_failure_time and 
            (time.time() - self.last_failure_time) > self.reset_timeout):
            self.failure_count = 0
        
        try:
            # Выполнение операции
            result = await operation()
            
            # Обработка успешного выполнения
            self._handle_success()
            return result
            
        except Exception as e:
            # Обработка ошибки
            self._handle_failure(e)
            
            # Выполнение fallback или повторное возбуждение исключения
            if fallback:
                return await fallback()
            raise
    
    def _handle_success(self):
        """Обработка успешного выполнения операции."""
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                self.logger.info(f"Circuit Breaker '{self.name}' восстановлен, переход в CLOSED")
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
    
    def _handle_failure(self, exception: Exception):
        """
        Обработка ошибки при выполнении операции.
        
        Args:
            exception: Возникшее исключение
        """
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.logger.warning(f"Circuit Breaker '{self.name}' переходит в состояние OPEN после {self.failure_count} ошибок")
            self.state = CircuitBreakerState.OPEN
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.logger.warning(f"Circuit Breaker '{self.name}' возвращается в состояние OPEN после ошибки в HALF-OPEN")
            self.state = CircuitBreakerState.OPEN
            
        self.logger.error(f"Ошибка в Circuit Breaker '{self.name}': {exception}")
    
    def reset(self):
        """Сброс Circuit Breaker в исходное состояние."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.logger.info(f"Circuit Breaker '{self.name}' сброшен в исходное состояние")


class RetryConfig:
    """
    Конфигурация для механизма повторных попыток.
    
    Attributes:
        max_retries: Максимальное количество повторных попыток
        base_delay: Базовая задержка в секундах
        max_delay: Максимальная задержка в секундах
        jitter_factor: Коэффициент случайного отклонения (0.0 - 1.0)
        retry_on: Список исключений, при которых выполняется повторная попытка
        timeout: Общий таймаут для всех попыток в секундах
    """
    
    def __init__(self,
                 max_retries: int = DEFAULT_RETRY_COUNT,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 max_delay: float = DEFAULT_MAX_DELAY,
                 jitter_factor: float = DEFAULT_JITTER_FACTOR,
                 retry_on: Optional[List[type]] = None,
                 timeout: Optional[float] = DEFAULT_TIMEOUT):
        """
        Инициализация конфигурации повторных попыток.
        
        Args:
            max_retries: Максимальное количество повторных попыток
            base_delay: Базовая задержка в секундах
            max_delay: Максимальная задержка в секундах
            jitter_factor: Коэффициент случайного отклонения (0.0 - 1.0)
            retry_on: Список исключений, при которых выполняется повторная попытка
            timeout: Общий таймаут для всех попыток в секундах
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = min(max(jitter_factor, 0.0), 1.0)  # Ограничение в диапазоне [0.0, 1.0]
        self.retry_on = retry_on or [Exception]
        self.timeout = timeout


async def retry(operation: Callable[..., T], 
                config: Optional[RetryConfig] = None, 
                **kwargs) -> T:
    """
    Выполнение операции с повторными попытками при неудаче.
    
    Args:
        operation: Асинхронная функция для выполнения
        config: Конфигурация повторных попыток
        **kwargs: Аргументы для передачи в operation
        
    Returns:
        Результат операции
        
    Raises:
        Exception: Если все попытки завершились неудачей
        TimeoutError: Если превышен общий таймаут
    """
    config = config or RetryConfig()
    logger = logging.getLogger("Retry")
    
    retries = 0
    last_exception = None
    start_time = time.time()
    
    while retries <= config.max_retries:
        # Проверка общего таймаута
        if config.timeout and (time.time() - start_time) > config.timeout:
            logger.error(f"Превышен общий таймаут {config.timeout} сек.")
            raise TimeoutError(
                message=f"Превышен общий таймаут {config.timeout} сек.",
                operation=operation.__name__,
                timeout=config.timeout
            )
        
        try:
            return await operation(**kwargs)
        except tuple(config.retry_on) as e:
            last_exception = e
            retries += 1
            
            if retries > config.max_retries:
                logger.error(f"Все {config.max_retries} попытки завершились неудачей")
                break
                
            # Экспоненциальная задержка с случайным отклонением
            delay = min(
                config.base_delay * (2 ** (retries - 1)),
                config.max_delay
            )
            
            # Добавление случайного отклонения (jitter)
            if config.jitter_factor > 0:
                jitter = config.jitter_factor * delay * (random.random() * 2 - 1)
                delay = max(0.1, delay + jitter)
            
            logger.warning(f"Попытка {retries}/{config.max_retries} не удалась. "
                          f"Повтор через {delay:.2f} сек. Ошибка: {e}")
            
            await asyncio.sleep(delay)
        except Exception as e:
            # Для исключений, не входящих в retry_on, сразу прерываем
            logger.error(f"Неожиданная ошибка, не входящая в список retry_on: {e}")
            raise
    
    logger.error(f"Все {config.max_retries} попытки завершились неудачей")
    raise last_exception


def with_retry(config: Optional[RetryConfig] = None):
    """
    Декоратор для выполнения функции с повторными попытками.
    
    Args:
        config: Конфигурация повторных попыток
        
    Returns:
        Декорированная функция
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry(func, config, *args, **kwargs)
        return wrapper
    return decorator


def with_circuit_breaker(circuit_breaker: CircuitBreaker, fallback: Optional[Callable[..., T]] = None):
    """
    Декоратор для выполнения функции с защитой Circuit Breaker.
    
    Args:
        circuit_breaker: Экземпляр CircuitBreaker
        fallback: Функция для выполнения при открытом Circuit Breaker
        
    Returns:
        Декорированная функция
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def operation():
                return await func(*args, **kwargs)
            
            async def fallback_wrapper():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise LeonError(f"Circuit Breaker '{circuit_breaker.name}' открыт, операция не выполняется")
            
            return await circuit_breaker.execute(operation, fallback_wrapper)
        return wrapper
    return decorator


class BulkheadConfig:
    """
    Конфигурация для паттерна Bulkhead (изоляция отказов).
    
    Attributes:
        max_concurrent: Максимальное количество одновременных выполнений
        max_queue_size: Максимальный размер очереди ожидания
        queue_timeout: Таймаут ожидания в очереди в секундах
    """
    
    def __init__(self,
                 max_concurrent: int = 10,
                 max_queue_size: int = 100,
                 queue_timeout: float = 30.0):
        """
        Инициализация конфигурации Bulkhead.
        
        Args:
            max_concurrent: Максимальное количество одновременных выполнений
            max_queue_size: Максимальный размер очереди ожидания
            queue_timeout: Таймаут ожидания в очереди в секундах
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue_timeout = queue_timeout


class Bulkhead:
    """
    Реализация паттерна Bulkhead для изоляции отказов.
    
    Паттерн Bulkhead ограничивает количество одновременных вызовов операций,
    что позволяет изолировать отказы и предотвратить исчерпание ресурсов.
    
    Attributes:
        name: Имя Bulkhead для идентификации в логах
        semaphore: Семафор для ограничения одновременных выполнений
        queue: Очередь ожидания
        config: Конфигурация Bulkhead
        logger: Логгер для записи событий
    """
    
    def __init__(self, name: str = "default", config: Optional[BulkheadConfig] = None):
        """
        Инициализация Bulkhead.
        
        Args:
            name: Имя Bulkhead для идентификации в логах
            config: Конфигурация Bulkhead
        """
        self.name = name
        self.config = config or BulkheadConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.logger = logging.getLogger(f"Bulkhead.{name}")
        self.logger.info(f"Bulkhead '{name}' инициализирован с max_concurrent={self.config.max_concurrent}, "
                        f"max_queue_size={self.config.max_queue_size}")
    
    async def execute(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """
        Выполнение операции с ограничением Bulkhead.
        
        Args:
            operation: Асинхронная функция для выполнения
            *args: Позиционные аргументы для operation
            **kwargs: Именованные аргументы для operation
            
        Returns:
            Результат операции
            
        Raises:
            asyncio.QueueFull: Если очередь ожидания заполнена
            asyncio.TimeoutError: Если превышен таймаут ожидания в очереди
            Exception: Если операция завершилась с ошибкой
        """
        # Добавление в очередь ожидания
        try:
            # Используем put_nowait для немедленной проверки заполненности очереди
            self.queue.put_nowait(None)
        except asyncio.QueueFull:
            self.logger.error(f"Очередь Bulkhead '{self.name}' заполнена")
            raise ResourceExhaustedError(
                message=f"Очередь Bulkhead '{self.name}' заполнена",
                resource_type="queue"
            )
        
        try:
            # Ожидание доступного слота с таймаутом
            try:
                # Ожидание освобождения семафора с таймаутом
                acquired = False
                async with asyncio.timeout(self.config.queue_timeout):
                    acquired = await self.semaphore.acquire()
                    
                if not acquired:
                    self.logger.error(f"Не удалось получить семафор Bulkhead '{self.name}'")
                    raise ResourceExhaustedError(
                        message=f"Не удалось получить семафор Bulkhead '{self.name}'",
                        resource_type="semaphore"
                    )
                
                # Выполнение операции
                self.logger.debug(f"Выполнение операции в Bulkhead '{self.name}'")
                return await operation(*args, **kwargs)
                
            except asyncio.TimeoutError:
                self.logger.error(f"Превышен таймаут ожидания в очереди Bulkhead '{self.name}'")
                raise TimeoutError(
                    message=f"Превышен таймаут ожидания в очереди Bulkhead '{self.name}'",
                    operation=operation.__name__,
                    timeout=self.config.queue_timeout
                )
        finally:
            # Освобождение ресурсов
            if acquired:
                self.semaphore.release()
            await self.queue.get()
            self.queue.task_done()


def with_bulkhead(bulkhead: Bulkhead):
    """
    Декоратор для выполнения функции с ограничением Bulkhead.
    
    Args:
        bulkhead: Экземпляр Bulkhead
        
    Returns:
        Декорированная функция
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await bulkhead.execute(func, *args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """
    Реализация ограничителя скорости запросов.
    
    Attributes:
        name: Имя ограничителя для идентификации в логах
        rate: Максимальное количество запросов в единицу времени
        period: Период времени в секундах
        tokens: Текущее количество доступных токенов
        last_refill: Время последнего пополнения токенов
        logger: Логгер для записи событий
    """
    
    def __init__(self, name: str = "default", rate: int = 10, period: float = 1.0):
        """
        Инициализация ограничителя скорости.
        
        Args:
            name: Имя ограничителя для идентификации в логах
            rate: Максимальное количество запросов в единицу времени
            period: Период времени в секундах
        """
        self.name = name
        self.rate = rate
        self.period = period
        self.tokens = rate
        self.last_refill = time.time()
        self.logger = logging.getLogger(f"RateLimiter.{name}")
        self.logger.info(f"RateLimiter '{name}' инициализирован с rate={rate}, period={period}")
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Получение токенов для выполнения запроса.
        
        Args:
            tokens: Количество требуемых токенов
            
        Returns:
            True, если токены получены, иначе False
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        # Расчет времени ожидания до следующего пополнения
        wait_time = self.period - (time.time() - self.last_refill)
        if wait_time > 0:
            self.logger.debug(f"Ожидание {wait_time:.2f} сек. до следующего пополнения токенов")
            await asyncio.sleep(wait_time)
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
        
        self.logger.warning(f"Превышен лимит запросов для '{self.name}'")
        return False
    
    def _refill(self):
        """Пополнение токенов на основе прошедшего времени."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed >= self.period:
            # Расчет количества периодов, прошедших с последнего пополнения
            periods = elapsed / self.period
            self.tokens = min(self.rate, self.tokens + int(periods * self.rate))
            self.last_refill = now


def with_rate_limit(rate_limiter: RateLimiter, tokens: int = 1):
    """
    Декоратор для выполнения функции с ограничением скорости.
    
    Args:
        rate_limiter: Экземпляр RateLimiter
        tokens: Количество требуемых токенов
        
    Returns:
        Декорированная функция
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if await rate_limiter.acquire(tokens):
                return await func(*args, **kwargs)
            raise APIRateLimitError(
                message=f"Превышен лимит запросов для '{rate_limiter.name}'",
                retry_after=rate_limiter.period
            )
        return wrapper
    return decorator


# Пример использования
if __name__ == "__main__":
    import sys
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    
    async def example_operation(success: bool = True):
        """Пример операции для тестирования."""
        if not success:
            raise APIError("Тестовая ошибка")
        return "Успешный результат"
    
    async def fallback_operation():
        """Пример fallback операции."""
        return "Результат fallback"
    
    async def main():
        logger = logging.getLogger("Main")
        
        # Пример использования Circuit Breaker
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=5)
        
        # Успешное выполнение
        result = await cb.execute(lambda: example_operation(True))
        logger.info(f"Результат: {result}")
        
        # Неудачное выполнение с fallback
        result = await cb.execute(
            lambda: example_operation(False),
            lambda: fallback_operation()
        )
        logger.info(f"Результат с fallback: {result}")
        
        # Пример использования retry
        try:
            config = RetryConfig(max_retries=2, base_delay=0.5)
            result = await retry(example_operation, config, success=True)
            logger.info(f"Результат retry: {result}")
        except Exception as e:
            logger.error(f"Ошибка retry: {e}")
        
        # Пример использования Bulkhead
        bulkhead = Bulkhead(name="test", config=BulkheadConfig(max_concurrent=2))
        
        # Запуск нескольких операций через Bulkhead
        tasks = [bulkhead.execute(example_operation, True) for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Результаты Bulkhead: {results}")
        
        # Пример использования RateLimiter
        rate_limiter = RateLimiter(name="test", rate=2, period=1.0)
        
        # Запуск нескольких операций через RateLimiter
        for i in range(3):
            if await rate_limiter.acquire():
                logger.info(f"Операция {i+1} выполнена")
            else:
                logger.warning(f"Операция {i+1} отклонена из-за превышения лимита")
    
    # Запуск примера
    asyncio.run(main())
