# API оркестратора и управления состоянием

## Обзор

API оркестратора предоставляет интерфейс для управления всеми подсистемами бота Leon, включая функции управления состоянием, которые позволяют приостанавливать и возобновлять работу бота без полной остановки.

## Классы

### LeonOrchestrator

Центральный класс для управления всеми подсистемами бота.

```python
class LeonOrchestrator:
    def __init__(self, config_manager: ConfigManager, localization: LocalizationManager):
        """
        Инициализирует оркестратор.
        
        Args:
            config_manager: Менеджер конфигурации
            localization: Менеджер локализации
        """
        pass
        
    async def start(self, mode: str = "dry") -> bool:
        """
        Запускает бота в указанном режиме.
        
        Args:
            mode: Режим работы бота ("dry", "backtest" или "live")
            
        Returns:
            bool: Успешность запуска
        """
        pass
        
    async def stop(self) -> bool:
        """
        Останавливает бота.
        
        Returns:
            bool: Успешность остановки
        """
        pass
        
    async def pause(self) -> bool:
        """
        Приостанавливает работу бота без полной остановки.
        
        Returns:
            bool: Успешность приостановки
        """
        pass
        
    async def resume(self) -> bool:
        """
        Возобновляет работу бота после паузы.
        
        Returns:
            bool: Успешность возобновления
        """
        pass
        
    def register_event_handler(self, event_name: str, handler: Callable) -> None:
        """
        Регистрирует обработчик события.
        
        Args:
            event_name: Имя события
            handler: Функция-обработчик события
        """
        pass
        
    def unregister_event_handler(self, event_name: str, handler: Callable) -> bool:
        """
        Отменяет регистрацию обработчика события.
        
        Args:
            event_name: Имя события
            handler: Функция-обработчик события
            
        Returns:
            bool: Успешность отмены регистрации
        """
        pass
```

### EventBus

Система событий для коммуникации между компонентами.

```python
class EventBus:
    def __init__(self):
        """
        Инициализирует шину событий.
        """
        pass
        
    async def emit(self, event_name: str, data: Dict[str, Any] = None) -> None:
        """
        Генерирует событие.
        
        Args:
            event_name: Имя события
            data: Данные события
        """
        pass
        
    def subscribe(self, event_name: str, handler: Callable) -> None:
        """
        Подписывается на событие.
        
        Args:
            event_name: Имя события
            handler: Функция-обработчик события
        """
        pass
        
    def unsubscribe(self, event_name: str, handler: Callable) -> bool:
        """
        Отписывается от события.
        
        Args:
            event_name: Имя события
            handler: Функция-обработчик события
            
        Returns:
            bool: Успешность отписки
        """
        pass
```

### CommandProcessor

Обработчик команд.

```python
class CommandProcessor:
    def __init__(self, orchestrator: LeonOrchestrator):
        """
        Инициализирует обработчик команд.
        
        Args:
            orchestrator: Оркестратор
        """
        pass
        
    async def process_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Обрабатывает команду.
        
        Args:
            command: Команда
            params: Параметры команды
            
        Returns:
            Dict[str, Any]: Результат выполнения команды
        """
        pass
        
    def register_command_handler(self, command: str, handler: Callable) -> None:
        """
        Регистрирует обработчик команды.
        
        Args:
            command: Команда
            handler: Функция-обработчик команды
        """
        pass
        
    def unregister_command_handler(self, command: str) -> bool:
        """
        Отменяет регистрацию обработчика команды.
        
        Args:
            command: Команда
            
        Returns:
            bool: Успешность отмены регистрации
        """
        pass
```

### TradingModeManager

Менеджер режимов торговли.

```python
class TradingModeManager:
    def __init__(self, orchestrator: LeonOrchestrator):
        """
        Инициализирует менеджер режимов торговли.
        
        Args:
            orchestrator: Оркестратор
        """
        pass
        
    async def set_mode(self, mode: str) -> bool:
        """
        Устанавливает режим торговли.
        
        Args:
            mode: Режим торговли ("dry", "backtest" или "live")
            
        Returns:
            bool: Успешность установки режима
        """
        pass
        
    def get_mode(self) -> str:
        """
        Возвращает текущий режим торговли.
        
        Returns:
            str: Текущий режим торговли
        """
        pass
```

## Функции управления состоянием

### pause()

Приостанавливает работу бота без полной остановки.

```python
async def pause(self) -> bool:
    """
    Приостанавливает работу бота без полной остановки.
    
    Returns:
        bool: Успешность приостановки
    """
    if not self.running:
        self.logger.warning("Оркестратор не запущен")
        return False
        
    if self.paused:
        self.logger.warning("Оркестратор уже приостановлен")
        return True
        
    try:
        self.logger.info("Приостановка работы оркестратора")
        
        # Приостанавливаем торговлю
        if hasattr(self, 'trader') and self.trader:
            await self.trader.pause()
            
        # Устанавливаем флаг паузы
        self.paused = True
        
        # Генерируем событие о приостановке
        await self.event_bus.emit("orchestrator_paused", {
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        self.logger.error(f"Ошибка при приостановке оркестратора: {str(e)}")
        self.logger.debug(traceback.format_exc())
        return False
```

### resume()

Возобновляет работу бота после паузы.

```python
async def resume(self) -> bool:
    """
    Возобновляет работу бота после паузы.
    
    Returns:
        bool: Успешность возобновления
    """
    if not self.running:
        self.logger.warning("Оркестратор не запущен")
        return False
        
    if not self.paused:
        self.logger.warning("Оркестратор не приостановлен")
        return True
        
    try:
        self.logger.info("Возобновление работы оркестратора")
        
        # Возобновляем торговлю
        if hasattr(self, 'trader') and self.trader:
            await self.trader.resume()
            
        # Сбрасываем флаг паузы
        self.paused = False
        
        # Генерируем событие о возобновлении
        await self.event_bus.emit("orchestrator_resumed", {
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        self.logger.error(f"Ошибка при возобновлении работы оркестратора: {str(e)}")
        self.logger.debug(traceback.format_exc())
        return False
```

## События

При приостановке и возобновлении работы бота генерируются следующие события:

- `orchestrator_paused` - при успешной приостановке работы бота
- `orchestrator_resumed` - при успешном возобновлении работы бота

### Формат данных события orchestrator_paused

```json
{
    "timestamp": "2023-03-16T15:30:45.123456"
}
```

### Формат данных события orchestrator_resumed

```json
{
    "timestamp": "2023-03-16T15:35:12.654321"
}
```

## Примеры использования

### Приостановка работы бота

```python
from core.orchestrator import LeonOrchestrator
from core.config_manager import ConfigManager
from core.localization import LocalizationManager

# Создание экземпляра оркестратора
config_manager = ConfigManager("config.yaml")
localization = LocalizationManager("locales")
orchestrator = LeonOrchestrator(config_manager, localization)

# Запуск бота
await orchestrator.start(mode="dry")

# Приостановка работы бота
success = await orchestrator.pause()
if success:
    print("Бот успешно приостановлен")
else:
    print("Не удалось приостановить бота")
```

### Возобновление работы бота

```python
# Возобновление работы бота
success = await orchestrator.resume()
if success:
    print("Работа бота успешно возобновлена")
else:
    print("Не удалось возобновить работу бота")
```

### Обработка событий

```python
# Регистрация обработчика события приостановки
async def on_pause(data):
    print(f"Бот приостановлен в {data['timestamp']}")
    # Дополнительные действия при приостановке

orchestrator.register_event_handler("orchestrator_paused", on_pause)

# Регистрация обработчика события возобновления
async def on_resume(data):
    print(f"Работа бота возобновлена в {data['timestamp']}")
    # Дополнительные действия при возобновлении

orchestrator.register_event_handler("orchestrator_resumed", on_resume)
```

## Ограничения

- Функции управления состоянием доступны только когда бот запущен
- При приостановке бота некоторые компоненты могут продолжать работу (например, мониторинг рынка)
- Возобновление работы бота возможно только если он был приостановлен 