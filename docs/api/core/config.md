# Модуль конфигурации (core.config_manager)

## Обзор

Модуль `core.config_manager` предоставляет функциональность для управления конфигурацией приложения Leon Trading Bot. Он обеспечивает загрузку настроек из YAML-файлов и переменных окружения, их валидацию и предоставление доступа к конфигурационным параметрам.

## Классы и функции

### ConfigManager

```python
class ConfigManager:
    def __init__(self, config_path="config/config.yaml", env_file=".env")
```

Класс для управления конфигурацией приложения.

#### Параметры:

- `config_path` (str): Путь к основному файлу конфигурации в формате YAML
- `env_file` (str): Путь к файлу с переменными окружения

#### Методы:

```python
def load_config(self)
```

Загружает конфигурацию из YAML-файла и переменных окружения.

- Возвращает: Словарь с конфигурацией
- Вызывает: `ConfigLoadError`, если не удалось загрузить конфигурацию

```python
def load_env_variables(self)
```

Загружает переменные окружения из файла .env.

- Возвращает: Словарь с переменными окружения
- Вызывает: `ConfigLoadError`, если не удалось загрузить переменные окружения

```python
def validate_config(self, config)
```

Проверяет конфигурацию на наличие всех необходимых параметров.

- Параметры:
  - `config` (dict): Словарь с конфигурацией
- Возвращает: `True`, если конфигурация валидна
- Вызывает: `ConfigValidationError`, если конфигурация не валидна

```python
def get_config(self)
```

Возвращает полную конфигурацию.

- Возвращает: Словарь с конфигурацией

```python
def get_value(self, key, default=None)
```

Получает значение конфигурации по ключу.

- Параметры:
  - `key` (str): Ключ для получения значения (поддерживает вложенные ключи через точку)
  - `default`: Значение по умолчанию, если ключ не найден
- Возвращает: Значение конфигурации или значение по умолчанию

```python
def update_config(self, key, value)
```

Обновляет значение конфигурации по ключу.

- Параметры:
  - `key` (str): Ключ для обновления (поддерживает вложенные ключи через точку)
  - `value`: Новое значение
- Возвращает: `True`, если обновление успешно
- Вызывает: `ConfigError`, если не удалось обновить конфигурацию

```python
def save_config(self)
```

Сохраняет текущую конфигурацию в файл.

- Возвращает: `True`, если сохранение успешно
- Вызывает: `ConfigError`, если не удалось сохранить конфигурацию

## Примеры использования

### Базовое использование

```python
from core.config_manager import ConfigManager

# Создание экземпляра ConfigManager
config_manager = ConfigManager()

# Загрузка конфигурации
config = config_manager.load_config()

# Получение полной конфигурации
full_config = config_manager.get_config()
print(f"Полная конфигурация: {full_config}")

# Получение значения по ключу
api_key = config_manager.get_value("binance.api_key")
print(f"API ключ Binance: {api_key}")

# Получение значения с значением по умолчанию
debug_mode = config_manager.get_value("app.debug", False)
print(f"Режим отладки: {debug_mode}")

# Обновление значения
config_manager.update_config("trading.max_open_positions", 5)
```

### Использование с обработкой исключений

```python
from core.config_manager import ConfigManager
from core.exceptions import ConfigLoadError, ConfigValidationError, ConfigError

try:
    # Создание экземпляра ConfigManager с нестандартными путями
    config_manager = ConfigManager(
        config_path="custom_config.yaml",
        env_file="custom.env"
    )
    
    # Загрузка конфигурации
    config = config_manager.load_config()
    
    # Получение значений
    trading_mode = config_manager.get_value("trading.mode")
    symbols = config_manager.get_value("trading.symbols")
    
    print(f"Режим торговли: {trading_mode}")
    print(f"Торговые пары: {symbols}")
    
    # Обновление и сохранение конфигурации
    config_manager.update_config("trading.risk_level", "medium")
    config_manager.save_config()
    
except ConfigLoadError as e:
    print(f"Ошибка загрузки конфигурации: {e}")
    if hasattr(e, 'file_path'):
        print(f"Проблемный файл: {e.file_path}")
except ConfigValidationError as e:
    print(f"Ошибка валидации конфигурации: {e}")
    if hasattr(e, 'invalid_fields'):
        print(f"Недопустимые поля: {e.invalid_fields}")
except ConfigError as e:
    print(f"Общая ошибка конфигурации: {e}")
```

### Использование в асинхронном контексте

```python
import asyncio
from core.config_manager import ConfigManager

async def initialize_app():
    # Создание экземпляра ConfigManager
    config_manager = ConfigManager()
    
    # Загрузка конфигурации
    await asyncio.to_thread(config_manager.load_config)
    
    # Получение настроек для инициализации компонентов
    db_settings = config_manager.get_value("database")
    api_settings = config_manager.get_value("binance")
    
    # Инициализация компонентов с настройками
    db = await initialize_database(db_settings)
    api_client = await initialize_api_client(api_settings)
    
    return db, api_client

async def main():
    db, api_client = await initialize_app()
    # Дальнейшая логика приложения...

if __name__ == "__main__":
    asyncio.run(main())
```

## Исключения

Модуль может вызывать следующие исключения:

- `ConfigLoadError`: Если не удалось загрузить конфигурацию из файла или переменных окружения
- `ConfigValidationError`: Если конфигурация не прошла валидацию
- `ConfigError`: Общее исключение для ошибок, связанных с конфигурацией

## Зависимости

- `os`: Для работы с переменными окружения и файловой системой
- `yaml`: Для чтения и записи YAML-файлов
- `dotenv`: Для загрузки переменных окружения из файла .env
- `logging`: Для логирования операций
- `core.exceptions`: Для использования специфичных исключений 