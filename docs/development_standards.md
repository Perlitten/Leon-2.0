# Стандарты разработки Leon Trading Bot

## 1. Структура кода

### 1.1 Модульность
- Каждый модуль должен иметь единую ответственность (принцип единственной ответственности)
- Модули должны быть слабо связаны между собой
- Интерфейсы модулей должны быть четко определены и документированы
- Избегайте циклических зависимостей между модулями

### 1.2 Именование
- Используйте осмысленные имена для классов, методов и переменных
- Классы: `CamelCase` (например, `DataStorage`, `BinanceClient`)
- Методы и функции: `snake_case` (например, `get_klines`, `save_to_json`)
- Переменные: `snake_case` (например, `api_key`, `trading_mode`)
- Константы: `UPPER_SNAKE_CASE` (например, `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`)

### 1.3 Документация кода
- Каждый модуль, класс и метод должен иметь docstring в формате Google Style
- Документируйте параметры, возвращаемые значения и исключения
- Добавляйте примеры использования для сложных функций
- Комментируйте сложные алгоритмы и неочевидные решения

## 2. Логирование

### 2.1 Уровни логирования
- `DEBUG`: Детальная информация для отладки
- `INFO`: Подтверждение нормальной работы
- `WARNING`: Индикация потенциальных проблем
- `ERROR`: Ошибки, которые не прерывают работу программы
- `CRITICAL`: Критические ошибки, приводящие к остановке программы

### 2.2 Стандарты логирования
- Каждый модуль должен использовать свой логгер с именем модуля
- Логи должны быть информативными и содержать контекст
- Избегайте логирования чувствительной информации (API ключи, пароли)
- Используйте структурированное логирование для сложных объектов
- Логируйте начало и завершение важных операций

### 2.3 Пример настройки логирования
```python
import logging

# Настройка логгера для модуля
logger = logging.getLogger("ModuleName")

# Пример использования
logger.debug("Детальная информация для отладки")
logger.info("Операция успешно выполнена")
logger.warning("Предупреждение: возможны проблемы")
logger.error("Произошла ошибка: %s", error_message)
logger.critical("Критическая ошибка, требуется вмешательство")
```

## 3. Обработка ошибок

### 3.1 Принципы обработки ошибок
- Используйте исключения для обработки ошибок, а не коды возврата
- Создавайте собственные классы исключений для специфичных ошибок
- Обрабатывайте исключения на соответствующем уровне абстракции
- Логируйте исключения с полным стеком вызовов
- Не подавляйте исключения без явной необходимости

### 3.2 Иерархия исключений
```python
# Базовое исключение для всех ошибок в проекте
class LeonError(Exception):
    """Базовое исключение для всех ошибок в Leon Trading Bot."""
    pass

# Специфичные исключения
class APIError(LeonError):
    """Ошибка при взаимодействии с внешним API."""
    pass

class ConfigError(LeonError):
    """Ошибка в конфигурации."""
    pass

class DataError(LeonError):
    """Ошибка при работе с данными."""
    pass

class TradingError(LeonError):
    """Ошибка в торговой логике."""
    pass
```

### 3.3 Пример обработки ошибок
```python
try:
    # Код, который может вызвать исключение
    result = api_client.get_data()
except APIError as e:
    # Обработка специфичной ошибки
    logger.error("Ошибка API: %s", e)
    # Повторная попытка или альтернативное действие
except Exception as e:
    # Обработка непредвиденных ошибок
    logger.exception("Непредвиденная ошибка: %s", e)
    # Корректное завершение или уведомление
finally:
    # Код, который выполняется всегда
    cleanup_resources()
```

## 4. Отказоустойчивость

### 4.1 Стратегии повторных попыток
- Используйте экспоненциальную задержку между повторными попытками
- Ограничивайте количество повторных попыток
- Добавляйте случайное отклонение (jitter) для предотвращения синхронизированных повторных запросов

```python
async def retry_operation(operation, max_retries=3, base_delay=1.0):
    """
    Выполнение операции с повторными попытками при неудаче.
    
    Args:
        operation: Асинхронная функция для выполнения
        max_retries: Максимальное количество повторных попыток
        base_delay: Базовая задержка в секундах
        
    Returns:
        Результат операции
        
    Raises:
        Exception: Если все попытки завершились неудачей
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            retries += 1
            
            if retries > max_retries:
                break
                
            # Экспоненциальная задержка с случайным отклонением
            delay = base_delay * (2 ** (retries - 1)) * (0.5 + random.random())
            logger.warning(f"Попытка {retries} не удалась. Повтор через {delay:.2f} сек. Ошибка: {e}")
            await asyncio.sleep(delay)
    
    logger.error(f"Все {max_retries} попытки завершились неудачей")
    raise last_exception
```

### 4.2 Мониторинг состояния
- Реализуйте проверки работоспособности (health checks)
- Отслеживайте использование ресурсов (память, CPU, диск)
- Мониторьте время отклика внешних сервисов
- Реализуйте механизм самовосстановления при обнаружении проблем

### 4.3 Обработка сетевых ошибок
- Устанавливайте таймауты для всех сетевых операций
- Обрабатывайте временные сетевые ошибки с повторными попытками
- Реализуйте механизм переключения на резервные эндпоинты
- Используйте паттерн Circuit Breaker для предотвращения каскадных отказов

```python
class CircuitBreaker:
    """
    Реализация паттерна Circuit Breaker для защиты от каскадных отказов.
    """
    
    def __init__(self, failure_threshold=5, recovery_timeout=30, name="default"):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
    
    async def execute(self, operation, fallback=None):
        """
        Выполнение операции с защитой Circuit Breaker.
        
        Args:
            operation: Асинхронная функция для выполнения
            fallback: Функция для выполнения при открытом Circuit Breaker
            
        Returns:
            Результат операции или fallback
        """
        if self.state == "OPEN":
            # Проверка времени восстановления
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.logger.info(f"Circuit Breaker {self.name} переходит в состояние HALF-OPEN")
                self.state = "HALF-OPEN"
            else:
                self.logger.warning(f"Circuit Breaker {self.name} OPEN, операция не выполняется")
                return await fallback() if fallback else None
        
        try:
            result = await operation()
            
            # Успешное выполнение в состоянии HALF-OPEN сбрасывает счетчик
            if self.state == "HALF-OPEN":
                self.logger.info(f"Circuit Breaker {self.name} восстановлен, переход в CLOSED")
                self.failure_count = 0
                self.state = "CLOSED"
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                self.logger.warning(f"Circuit Breaker {self.name} переходит в состояние OPEN")
                self.state = "OPEN"
            
            if self.state == "HALF-OPEN":
                self.logger.warning(f"Circuit Breaker {self.name} возвращается в состояние OPEN")
                self.state = "OPEN"
                
            self.logger.error(f"Ошибка в Circuit Breaker {self.name}: {e}")
            
            if fallback:
                return await fallback()
            raise
```

## 5. Тестирование

### 5.1 Типы тестов
- **Модульные тесты**: Тестирование отдельных компонентов в изоляции
- **Интеграционные тесты**: Тестирование взаимодействия между компонентами
- **Функциональные тесты**: Тестирование функциональности системы в целом
- **Нагрузочные тесты**: Тестирование производительности под нагрузкой

### 5.2 Стандарты тестирования
- Каждый модуль должен иметь соответствующие тесты
- Целевое покрытие кода тестами: не менее 80%
- Тесты должны быть независимыми друг от друга
- Используйте моки и стабы для изоляции тестируемого кода
- Тесты должны быть быстрыми и детерминированными

### 5.3 Структура тестов
```python
import unittest
from unittest.mock import patch, MagicMock

class TestBinanceClient(unittest.TestCase):
    
    def setUp(self):
        """Подготовка перед каждым тестом."""
        self.api_key = "test_key"
        self.api_secret = "test_secret"
        self.client = BinanceClient(self.api_key, self.api_secret, testnet=True)
    
    def tearDown(self):
        """Очистка после каждого теста."""
        pass
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_ticker(self, mock_get):
        """Тест получения тикера."""
        # Настройка мока
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json.return_value = {"symbol": "BTCUSDT", "price": "50000.00"}
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Вызов тестируемого метода
        result = await self.client.get_ticker("BTCUSDT")
        
        # Проверка результата
        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["price"], "50000.00")
        
        # Проверка вызова API
        mock_get.assert_called_once()
    
    @patch('aiohttp.ClientSession.get')
    async def test_get_ticker_error(self, mock_get):
        """Тест обработки ошибки при получении тикера."""
        # Настройка мока для имитации ошибки
        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text.return_value = "Invalid symbol"
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Проверка, что метод вызывает исключение
        with self.assertRaises(Exception):
            await self.client.get_ticker("INVALID")
```

### 5.4 Автоматизация тестирования
- Настройте CI/CD для автоматического запуска тестов при коммите
- Используйте инструменты для измерения покрытия кода тестами
- Интегрируйте статический анализ кода в процесс тестирования
- Автоматизируйте тестирование производительности

## 6. Безопасность

### 6.1 Защита чувствительных данных
- Никогда не храните API ключи и секреты в коде
- Используйте переменные окружения или защищенные хранилища секретов
- Шифруйте чувствительные данные при хранении
- Не логируйте чувствительную информацию

### 6.2 Валидация входных данных
- Всегда проверяйте входные данные перед использованием
- Используйте типизацию и проверку типов
- Применяйте принцип наименьших привилегий
- Защищайтесь от инъекций и других атак

### 6.3 Безопасность API
- Используйте HTTPS для всех внешних API
- Реализуйте ограничение скорости запросов
- Проверяйте подлинность запросов
- Минимизируйте поверхность атаки

## 7. Производительность

### 7.1 Оптимизация кода
- Профилируйте код для выявления узких мест
- Оптимизируйте алгоритмы и структуры данных
- Используйте кэширование для часто запрашиваемых данных
- Минимизируйте блокирующие операции

### 7.2 Асинхронное программирование
- Используйте асинхронное программирование для I/O-bound операций
- Избегайте блокировки event loop
- Группируйте асинхронные операции для параллельного выполнения
- Контролируйте количество одновременных задач

### 7.3 Управление ресурсами
- Освобождайте ресурсы после использования
- Используйте пулы соединений для баз данных и HTTP-клиентов
- Мониторьте использование памяти и утечки
- Реализуйте механизмы ограничения использования ресурсов

## 8. Документация

### 8.1 Документация кода
- Используйте docstrings для всех модулей, классов и методов
- Следуйте единому стилю документации (Google Style)
- Обновляйте документацию при изменении кода
- Генерируйте API документацию автоматически

### 8.2 Проектная документация
- Поддерживайте актуальную архитектурную документацию
- Документируйте принятые решения и их обоснование
- Создавайте руководства по установке и настройке
- Включайте примеры использования и сценарии

### 8.3 Документация для пользователей
- Создавайте понятные руководства пользователя
- Документируйте все функции и параметры
- Включайте примеры и сценарии использования
- Поддерживайте FAQ и раздел устранения неполадок

## 9. Версионирование и релизы

### 9.1 Семантическое версионирование
- Используйте формат MAJOR.MINOR.PATCH
- MAJOR: несовместимые изменения API
- MINOR: новая функциональность с обратной совместимостью
- PATCH: исправления ошибок с обратной совместимостью

### 9.2 Управление релизами
- Создавайте теги для каждого релиза
- Поддерживайте журнал изменений (CHANGELOG)
- Тестируйте релизы перед публикацией
- Автоматизируйте процесс сборки и публикации

### 9.3 Обратная совместимость
- Стремитесь к сохранению обратной совместимости
- Документируйте устаревшие функции и планы их удаления
- Предоставляйте миграционные пути для пользователей
- Тестируйте обратную совместимость

## 10. Непрерывная интеграция и доставка (CI/CD)

### 10.1 Непрерывная интеграция
- Автоматически запускайте тесты при каждом коммите
- Проверяйте стиль кода и статический анализ
- Измеряйте покрытие кода тестами
- Уведомляйте команду о проблемах

### 10.2 Непрерывная доставка
- Автоматизируйте сборку и упаковку
- Реализуйте автоматическое развертывание в тестовую среду
- Проводите автоматизированное тестирование в тестовой среде
- Упростите процесс развертывания в производственную среду

### 10.3 Мониторинг и обратная связь
- Мониторьте производительность и ошибки в производственной среде
- Собирайте метрики использования и обратную связь
- Анализируйте данные для улучшения продукта
- Быстро реагируйте на критические проблемы 