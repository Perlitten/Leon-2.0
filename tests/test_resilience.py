"""
Тесты для модуля отказоустойчивости core.resilience.
"""

import unittest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

from core.resilience import (
    CircuitBreaker, CircuitBreakerState, RetryConfig, Bulkhead, 
    BulkheadConfig, RateLimiter, retry, with_retry, with_circuit_breaker,
    with_bulkhead, with_rate_limit
)
from core.exceptions import (
    LeonError, APIError, APIRateLimitError, TimeoutError, ResourceExhaustedError
)


class TestCircuitBreaker(unittest.TestCase):
    """Тесты для класса CircuitBreaker."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создание Circuit Breaker с низким порогом ошибок для быстрого тестирования
        self.circuit_breaker = CircuitBreaker(
            name="test",
            failure_threshold=2,
            recovery_timeout=0.1,  # Короткий таймаут для быстрого тестирования
            reset_timeout=0.2,
            success_threshold=1
        )
        
        # Создание моков для операций
        self.success_operation = AsyncMock(return_value="Success")
        self.failure_operation = AsyncMock(side_effect=APIError("Test error"))
        self.fallback_operation = AsyncMock(return_value="Fallback")
    
    async def test_circuit_breaker_initial_state(self):
        """Тест начального состояния Circuit Breaker."""
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.success_count, 0)
        self.assertIsNone(self.circuit_breaker.last_failure_time)
        self.assertIsNone(self.circuit_breaker.last_success_time)
    
    async def test_circuit_breaker_success(self):
        """Тест успешного выполнения операции."""
        result = await self.circuit_breaker.execute(self.success_operation)
        
        self.assertEqual(result, "Success")
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertIsNotNone(self.circuit_breaker.last_success_time)
        self.success_operation.assert_called_once()
    
    async def test_circuit_breaker_failure(self):
        """Тест неудачного выполнения операции."""
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 1)
        self.assertIsNotNone(self.circuit_breaker.last_failure_time)
        self.failure_operation.assert_called_once()
    
    async def test_circuit_breaker_open_after_failures(self):
        """Тест перехода в состояние OPEN после нескольких ошибок."""
        # Первый вызов - ошибка, но Circuit Breaker остается закрытым
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 1)
        
        # Второй вызов - ошибка, Circuit Breaker должен открыться
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        self.assertEqual(self.circuit_breaker.failure_count, 2)
        self.assertEqual(self.failure_operation.call_count, 2)
    
    async def test_circuit_breaker_fallback(self):
        """Тест использования fallback при открытом Circuit Breaker."""
        # Открываем Circuit Breaker
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        # Проверяем, что Circuit Breaker открыт
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        
        # Вызов с fallback
        result = await self.circuit_breaker.execute(
            self.failure_operation, 
            self.fallback_operation
        )
        
        self.assertEqual(result, "Fallback")
        self.assertEqual(self.failure_operation.call_count, 2)  # Не должен вызываться снова
        self.fallback_operation.assert_called_once()
    
    async def test_circuit_breaker_half_open_after_timeout(self):
        """Тест перехода в состояние HALF-OPEN после таймаута."""
        # Открываем Circuit Breaker
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        # Проверяем, что Circuit Breaker открыт
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        
        # Ждем, пока истечет recovery_timeout
        await asyncio.sleep(0.2)
        
        # Вызов после таймаута должен перевести в HALF-OPEN
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        self.assertEqual(self.failure_operation.call_count, 3)
    
    async def test_circuit_breaker_closed_after_success_in_half_open(self):
        """Тест перехода в состояние CLOSED после успеха в HALF-OPEN."""
        # Открываем Circuit Breaker
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        # Ждем, пока истечет recovery_timeout
        await asyncio.sleep(0.2)
        
        # Устанавливаем состояние HALF-OPEN вручную
        self.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        
        # Успешный вызов в HALF-OPEN должен перевести в CLOSED
        result = await self.circuit_breaker.execute(self.success_operation)
        
        self.assertEqual(result, "Success")
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.success_operation.assert_called_once()
    
    async def test_circuit_breaker_reset(self):
        """Тест сброса Circuit Breaker."""
        # Открываем Circuit Breaker
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        with self.assertRaises(APIError):
            await self.circuit_breaker.execute(self.failure_operation)
        
        # Проверяем, что Circuit Breaker открыт
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.OPEN)
        
        # Сбрасываем Circuit Breaker
        self.circuit_breaker.reset()
        
        # Проверяем, что Circuit Breaker сброшен
        self.assertEqual(self.circuit_breaker.state, CircuitBreakerState.CLOSED)
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.success_count, 0)
        self.assertIsNone(self.circuit_breaker.last_failure_time)
        self.assertIsNone(self.circuit_breaker.last_success_time)


class TestRetry(unittest.TestCase):
    """Тесты для функции retry и декоратора with_retry."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создание конфигурации retry с низкими значениями для быстрого тестирования
        self.retry_config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            max_delay=0.1,
            jitter_factor=0.1,
            retry_on=[APIError],
            timeout=0.5
        )
        
        # Создание моков для операций
        self.success_operation = AsyncMock(return_value="Success")
        self.failure_operation = AsyncMock(side_effect=APIError("Test error"))
        self.eventual_success_operation = AsyncMock(side_effect=[
            APIError("Test error"),
            APIError("Test error"),
            "Success"
        ])
    
    async def test_retry_success_first_attempt(self):
        """Тест успешного выполнения с первой попытки."""
        result = await retry(self.success_operation, self.retry_config)
        
        self.assertEqual(result, "Success")
        self.success_operation.assert_called_once()
    
    async def test_retry_eventual_success(self):
        """Тест успешного выполнения после нескольких попыток."""
        result = await retry(self.eventual_success_operation, self.retry_config)
        
        self.assertEqual(result, "Success")
        self.assertEqual(self.eventual_success_operation.call_count, 3)
    
    async def test_retry_max_retries_exceeded(self):
        """Тест превышения максимального количества попыток."""
        with self.assertRaises(APIError):
            await retry(self.failure_operation, self.retry_config)
        
        self.assertEqual(self.failure_operation.call_count, 3)  # 1 основная + 2 повторные
    
    async def test_retry_timeout(self):
        """Тест превышения общего таймаута."""
        # Создаем операцию, которая выполняется долго
        slow_operation = AsyncMock(side_effect=lambda: asyncio.sleep(0.2))
        
        # Создаем конфигурацию с коротким таймаутом
        config = RetryConfig(
            max_retries=5,
            base_delay=0.1,
            timeout=0.1
        )
        
        with self.assertRaises(TimeoutError):
            await retry(slow_operation, config)
    
    async def test_retry_non_retryable_exception(self):
        """Тест исключения, не входящего в список retry_on."""
        # Создаем операцию, которая вызывает исключение, не входящее в retry_on
        operation = AsyncMock(side_effect=ValueError("Non-retryable error"))
        
        with self.assertRaises(ValueError):
            await retry(operation, self.retry_config)
        
        operation.assert_called_once()
    
    async def test_with_retry_decorator(self):
        """Тест декоратора with_retry."""
        # Создаем функцию с декоратором
        @with_retry(self.retry_config)
        async def test_function():
            return await self.eventual_success_operation()
        
        # Вызываем функцию
        result = await test_function()
        
        self.assertEqual(result, "Success")
        self.assertEqual(self.eventual_success_operation.call_count, 3)


class TestBulkhead(unittest.TestCase):
    """Тесты для класса Bulkhead."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создание конфигурации Bulkhead с низкими значениями для быстрого тестирования
        self.bulkhead_config = BulkheadConfig(
            max_concurrent=2,
            max_queue_size=1,
            queue_timeout=0.1
        )
        
        # Создание Bulkhead
        self.bulkhead = Bulkhead(name="test", config=self.bulkhead_config)
        
        # Создание моков для операций
        self.fast_operation = AsyncMock(return_value="Fast")
        self.slow_operation = AsyncMock(side_effect=lambda: asyncio.sleep(0.2) or "Slow")
    
    async def test_bulkhead_execute_success(self):
        """Тест успешного выполнения операции через Bulkhead."""
        result = await self.bulkhead.execute(self.fast_operation)
        
        self.assertEqual(result, "Fast")
        self.fast_operation.assert_called_once()
    
    async def test_bulkhead_max_concurrent(self):
        """Тест ограничения количества одновременных операций."""
        # Запускаем 3 медленные операции (max_concurrent=2, max_queue_size=1)
        tasks = [
            asyncio.create_task(self.bulkhead.execute(self.slow_operation))
            for _ in range(3)
        ]
        
        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Проверяем, что 2 операции выполнились успешно, а 1 вызвала исключение
        success_count = sum(1 for r in results if r == "Slow")
        error_count = sum(1 for r in results if isinstance(r, Exception))
        
        self.assertEqual(success_count, 2)
        self.assertEqual(error_count, 1)
        self.assertEqual(self.slow_operation.call_count, 2)
    
    async def test_bulkhead_queue_timeout(self):
        """Тест таймаута ожидания в очереди."""
        # Запускаем 2 медленные операции, которые займут все слоты
        task1 = asyncio.create_task(self.bulkhead.execute(self.slow_operation))
        task2 = asyncio.create_task(self.bulkhead.execute(self.slow_operation))
        
        # Даем время на запуск операций
        await asyncio.sleep(0.05)
        
        # Запускаем еще одну операцию, которая должна ждать в очереди и вызвать таймаут
        with self.assertRaises(TimeoutError):
            await self.bulkhead.execute(self.fast_operation)
        
        # Ждем завершения запущенных задач
        await task1
        await task2
    
    async def test_with_bulkhead_decorator(self):
        """Тест декоратора with_bulkhead."""
        # Создаем функцию с декоратором
        @with_bulkhead(self.bulkhead)
        async def test_function():
            return await self.fast_operation()
        
        # Вызываем функцию
        result = await test_function()
        
        self.assertEqual(result, "Fast")
        self.fast_operation.assert_called_once()


class TestRateLimiter(unittest.TestCase):
    """Тесты для класса RateLimiter."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создание RateLimiter с низкими значениями для быстрого тестирования
        self.rate_limiter = RateLimiter(name="test", rate=2, period=0.1)
    
    async def test_rate_limiter_acquire(self):
        """Тест получения токенов."""
        # Первые два запроса должны пройти
        self.assertTrue(await self.rate_limiter.acquire())
        self.assertTrue(await self.rate_limiter.acquire())
        
        # Третий запрос должен быть отклонен
        self.assertFalse(await self.rate_limiter.acquire())
    
    async def test_rate_limiter_refill(self):
        """Тест пополнения токенов."""
        # Используем все токены
        self.assertTrue(await self.rate_limiter.acquire())
        self.assertTrue(await self.rate_limiter.acquire())
        self.assertFalse(await self.rate_limiter.acquire())
        
        # Ждем пополнения токенов
        await asyncio.sleep(0.2)
        
        # Теперь должны быть доступны новые токены
        self.assertTrue(await self.rate_limiter.acquire())
    
    async def test_with_rate_limit_decorator(self):
        """Тест декоратора with_rate_limit."""
        # Создаем функцию с декоратором
        @with_rate_limit(self.rate_limiter)
        async def test_function():
            return "Success"
        
        # Первые два вызова должны пройти
        self.assertEqual(await test_function(), "Success")
        self.assertEqual(await test_function(), "Success")
        
        # Третий вызов должен вызвать исключение
        with self.assertRaises(APIRateLimitError):
            await test_function()


class TestCombinedPatterns(unittest.TestCase):
    """Тесты для комбинирования паттернов отказоустойчивости."""
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        # Создание компонентов
        self.circuit_breaker = CircuitBreaker(
            name="test_cb",
            failure_threshold=2,
            recovery_timeout=0.1
        )
        
        self.retry_config = RetryConfig(
            max_retries=1,
            base_delay=0.01,
            retry_on=[APIError]
        )
        
        self.bulkhead = Bulkhead(
            name="test_bh",
            config=BulkheadConfig(max_concurrent=1)
        )
        
        self.rate_limiter = RateLimiter(
            name="test_rl",
            rate=1,
            period=0.1
        )
        
        # Создание моков для операций
        self.success_operation = AsyncMock(return_value="Success")
        self.failure_operation = AsyncMock(side_effect=APIError("Test error"))
    
    async def test_combined_decorators(self):
        """Тест комбинирования декораторов."""
        # Создаем функцию с комбинированными декораторами
        @with_circuit_breaker(self.circuit_breaker)
        @with_retry(self.retry_config)
        @with_bulkhead(self.bulkhead)
        @with_rate_limit(self.rate_limiter)
        async def test_function():
            return await self.success_operation()
        
        # Вызываем функцию
        result = await test_function()
        
        self.assertEqual(result, "Success")
        self.success_operation.assert_called_once()
    
    async def test_combined_decorators_failure(self):
        """Тест комбинирования декораторов при ошибке."""
        # Создаем функцию с комбинированными декораторами
        @with_circuit_breaker(self.circuit_breaker)
        @with_retry(self.retry_config)
        @with_bulkhead(self.bulkhead)
        @with_rate_limit(self.rate_limiter)
        async def test_function():
            return await self.failure_operation()
        
        # Первый вызов должен вызвать исключение после retry
        with self.assertRaises(APIError):
            await test_function()
        
        # Проверяем, что retry был выполнен
        self.assertEqual(self.failure_operation.call_count, 2)  # 1 основная + 1 повторная
        
        # Второй вызов должен вызвать исключение после retry
        with self.assertRaises(APIError):
            await test_function()
        
        # Проверяем, что retry был выполнен
        self.assertEqual(self.failure_operation.call_count, 4)  # 2 основные + 2 повторные
        
        # Третий вызов должен использовать Circuit Breaker и не вызывать операцию
        with self.assertRaises(LeonError):
            await test_function()
        
        # Проверяем, что операция не была вызвана снова
        self.assertEqual(self.failure_operation.call_count, 4)


if __name__ == "__main__":
    unittest.main() 