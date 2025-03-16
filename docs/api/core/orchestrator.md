# Оркестратор (Orchestrator)

Оркестратор является центральным компонентом Leon Trading Bot, который управляет жизненным циклом системы, координирует работу всех компонентов и обеспечивает переключение между различными режимами работы.

## Основные классы

### LeonOrchestrator

Главный класс оркестратора, который управляет всеми подсистемами.

```python
from core.orchestrator import LeonOrchestrator

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator(config_path="config/config.yaml", dry_mode=False)

# Запуск оркестратора
orchestrator.start()

# Получение статуса оркестратора
status = orchestrator.get_status()
print(f"Статус оркестратора: {status}")

# Обработка команды
result = orchestrator.process_command("set_mode", ["dry"])
print(f"Результат выполнения команды: {result}")

# Остановка оркестратора
orchestrator.stop()
```

#### Параметры конструктора

- `config_path` (str, опционально): Путь к файлу конфигурации. По умолчанию: "config/config.yaml".
- `dry_mode` (bool, опционально): Режим тестирования без сохранения изменений. По умолчанию: False.

#### Методы

- `start()`: Запуск оркестратора.
- `stop()`: Остановка оркестратора.
- `get_status()`: Получение текущего статуса оркестратора.
- `process_command(command: str, args: List[str] = None)`: Обработка команды.

### EventBus

Шина событий для обмена сообщениями между компонентами системы. Реализует паттерн "Наблюдатель" для асинхронной обработки событий.

```python
from core.orchestrator import EventBus

# Создание экземпляра шины событий
event_bus = EventBus()

# Подписка на событие
def handle_event(data):
    print(f"Получено событие: {data}")

event_bus.subscribe("SYSTEM_STARTED", handle_event)

# Публикация события
event_bus.publish("SYSTEM_STARTED", {"timestamp": 1625097600})

# Отписка от события
event_bus.unsubscribe("SYSTEM_STARTED", handle_event)
```

#### Методы

- `subscribe(event_type: str, callback: Callable)`: Подписка на событие.
- `unsubscribe(event_type: str, callback: Callable)`: Отписка от события.
- `publish(event_type: str, data: Any = None)`: Публикация события.

### TradingModeManager

Менеджер режимов торговли. Управляет переключением между различными режимами работы системы.

```python
from core.orchestrator import TradingModeManager
from core.config_manager import ConfigManager
from core.orchestrator import EventBus

# Создание экземпляра менеджера режимов торговли
config_manager = ConfigManager("config/config.yaml")
event_bus = EventBus()
trading_mode_manager = TradingModeManager(config_manager, event_bus)

# Инициализация менеджера режимов торговли
trading_mode_manager.initialize()

# Установка режима работы
trading_mode_manager.set_mode("dry")

# Получение текущего режима работы
current_mode = trading_mode_manager.get_current_mode()
print(f"Текущий режим работы: {current_mode}")
```

#### Параметры конструктора

- `config_manager` (ConfigManager): Менеджер конфигурации.
- `event_bus` (EventBus): Шина событий.

#### Методы

- `initialize()`: Инициализация менеджера режимов торговли.
- `set_mode(mode: str)`: Установка режима работы.
- `get_current_mode()`: Получение текущего режима работы.

### MLIntegrationManager

Менеджер интеграции с ML-моделями. Управляет загрузкой, оценкой и использованием ML-моделей для прогнозирования.

```python
from core.orchestrator import MLIntegrationManager
from core.config_manager import ConfigManager
from core.orchestrator import EventBus

# Создание экземпляра менеджера интеграции с ML-моделями
config_manager = ConfigManager("config/config.yaml")
event_bus = EventBus()
ml_integration_manager = MLIntegrationManager(config_manager, event_bus)

# Инициализация менеджера интеграции с ML-моделями
ml_integration_manager.initialize()

# Установка активной модели
ml_integration_manager.set_active_model("lstm")

# Получение имени активной модели
active_model = ml_integration_manager.get_active_model()
print(f"Активная модель: {active_model}")

# Получение предсказания от активной модели
data = {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
prediction = ml_integration_manager.get_prediction(data)
print(f"Предсказание: {prediction}")
```

#### Параметры конструктора

- `config_manager` (ConfigManager): Менеджер конфигурации.
- `event_bus` (EventBus): Шина событий.

#### Методы

- `initialize()`: Инициализация менеджера интеграции с ML-моделями.
- `set_active_model(model_name: str)`: Установка активной модели.
- `get_active_model()`: Получение имени активной модели.
- `get_prediction(data: Any)`: Получение предсказания от активной модели.

### CommandProcessor

Процессор команд. Обрабатывает команды пользователя и маршрутизирует их к соответствующим обработчикам.

```python
from core.orchestrator import CommandProcessor
from core.orchestrator import EventBus

# Создание экземпляра процессора команд
event_bus = EventBus()
command_processor = CommandProcessor(event_bus)

# Регистрация обработчика команды
def handle_start_command(args):
    print(f"Выполнение команды 'start' с аргументами: {args}")
    return {"success": True, "message": "Система запущена"}

command_processor.register_handler("start", handle_start_command)

# Обработка команды
result = command_processor.process_command("start", ["--verbose"])
print(f"Результат выполнения команды: {result}")
```

#### Параметры конструктора

- `event_bus` (EventBus): Шина событий.

#### Методы

- `register_handler(command: str, handler: Callable)`: Регистрация обработчика команды.
- `process_command(command: str, args: List[str] = None)`: Обработка команды.

## Режимы работы

Оркестратор поддерживает следующие режимы работы:

- **Dry Mode (тестирование без реальных сделок)**: Режим для тестирования стратегий без реальных сделок. Все операции выполняются на виртуальном балансе.
- **Backtesting (тестирование на исторических данных)**: Режим для тестирования стратегий на исторических данных. Позволяет оценить эффективность стратегии на прошлых данных.
- **Real Trading (реальная торговля)**: Режим для реальной торговли на бирже. Все операции выполняются с реальными средствами.

## События

Оркестратор использует систему событий для обмена сообщениями между компонентами. Основные типы событий:

- **Системные события**: События, связанные с жизненным циклом системы (запуск, остановка, изменение режима работы и т.д.).
- **Торговые события**: События, связанные с торговыми операциями (открытие позиции, закрытие позиции, изменение цены и т.д.).
- **ML-события**: События, связанные с ML-моделями (загрузка модели, получение предсказания и т.д.).
- **События визуализации**: События, связанные с визуализацией данных (обновление графика, изменение масштаба и т.д.).

## Команды

Оркестратор поддерживает следующие команды:

- **start**: Запуск системы.
- **stop**: Остановка системы.
- **status**: Получение текущего статуса системы.
- **set_mode**: Установка режима работы.
- **set_model**: Установка активной ML-модели.

## Примеры использования

### Запуск системы в режиме тестирования

```python
from core.orchestrator import LeonOrchestrator

# Создание экземпляра оркестратора в режиме тестирования
orchestrator = LeonOrchestrator(dry_mode=True)

# Запуск оркестратора
orchestrator.start()

# Установка режима работы "dry"
orchestrator.process_command("set_mode", ["dry"])

# Установка активной модели
orchestrator.process_command("set_model", ["lstm"])

# Получение статуса системы
status = orchestrator.process_command("status")
print(f"Статус системы: {status}")

# Остановка оркестратора
orchestrator.stop()
```

### Обработка событий

```python
from core.orchestrator import LeonOrchestrator, EventBus
from core.constants import EVENT_TYPES

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator()

# Получение шины событий
event_bus = orchestrator._event_bus

# Подписка на событие открытия позиции
def handle_position_opened(data):
    position = data.get("position")
    print(f"Открыта позиция: {position}")

event_bus.subscribe(EVENT_TYPES["TRADE"]["POSITION_OPENED"], handle_position_opened)

# Запуск оркестратора
orchestrator.start()

# Остановка оркестратора
orchestrator.stop()
```

### Создание собственного обработчика команд

```python
from core.orchestrator import LeonOrchestrator

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator()

# Регистрация обработчика команды
def handle_custom_command(args):
    print(f"Выполнение пользовательской команды с аргументами: {args}")
    return {"success": True, "message": "Пользовательская команда выполнена"}

orchestrator._command_processor.register_handler("custom", handle_custom_command)

# Запуск оркестратора
orchestrator.start()

# Обработка пользовательской команды
result = orchestrator.process_command("custom", ["arg1", "arg2"])
print(f"Результат выполнения команды: {result}")

# Остановка оркестратора
orchestrator.stop()
``` 