# DryTrader

Модуль `dry_trader.py` предоставляет реализацию трейдера для симуляции торговли без реальных сделок. Этот трейдер позволяет тестировать стратегии и алгоритмы без риска потери реальных средств.

## Классы

### DryModeData

Встроенный менеджер для виртуального баланса и истории сделок.

#### Методы

- `__init__(symbol: str, storage_file: Optional[str] = None)` - Инициализация менеджера виртуального баланса.
- `load_data() -> bool` - Загрузка данных из файла.
- `initialize(initial_balance: float) -> None` - Инициализация данных с заданным начальным балансом.
- `save_data() -> bool` - Сохранение данных в файл.
- `get_balance() -> float` - Получение текущего баланса.
- `update_balance(amount: float) -> float` - Обновление баланса.
- `add_trade(trade_data: Dict[str, Any]) -> None` - Добавление сделки в историю.
- `add_position(position_data: Dict[str, Any]) -> None` - Добавление открытой позиции.
- `remove_position(position_id: str) -> Optional[Dict[str, Any]]` - Удаление позиции из списка открытых.
- `get_open_positions() -> List[Dict[str, Any]]` - Получение списка открытых позиций.
- `get_trades() -> List[Dict[str, Any]]` - Получение истории сделок.
- `get_performance_stats() -> Dict[str, Any]` - Получение статистики производительности.

### DryTrader

Трейдер для симуляции торговли без реальных сделок. Наследуется от `TraderBase`.

#### Методы

- `__init__(symbol: str, exchange_client, strategy, notification_service=None, risk_controller=None, initial_balance: float = 1000.0, leverage: int = 1, storage_file: Optional[str] = None, price_update_interval: float = 5.0, visualizer=None)` - Инициализация трейдера для симуляции.
- `initialize() -> bool` - Инициализация трейдера.
- `_update_price_loop()` - Фоновая задача для периодического обновления цены.
- `_check_positions()` - Проверка условий для открытых позиций (стоп-лосс, тейк-профит).
- `start() -> bool` - Запуск трейдера.
- `stop() -> bool` - Остановка трейдера.
- `enter_position(direction: str, size: float, price: Optional[float] = None, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]` - Вход в позицию.
- `exit_position(position_id: str, price: Optional[float] = None) -> Dict[str, Any]` - Выход из позиции.
- `update_position(position_id: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool` - Обновление параметров позиции.
- `get_open_positions() -> List[Dict[str, Any]]` - Получение списка открытых позиций.
- `get_balance() -> float` - Получение текущего баланса.
- `get_current_price() -> float` - Получение текущей цены торговой пары.
- `get_trade_history() -> List[Dict[str, Any]]` - Получение истории сделок.
- `get_performance_stats() -> Dict[str, Any]` - Получение статистики производительности.

## Пример использования

```python
# Создание экземпляра DryTrader
dry_trader = DryTrader(
    symbol="BTCUSDT",
    exchange_client=exchange_client,
    strategy=strategy,
    notification_service=notification_service,
    initial_balance=1000.0,
    leverage=1,
    price_update_interval=5.0,
    visualizer=visualizer
)

# Инициализация и запуск трейдера
await dry_trader.initialize()
await dry_trader.start()

# Открытие позиции
position = await dry_trader.enter_position(
    direction="LONG",
    size=0.1,
    stop_loss=19000.0,
    take_profit=21000.0
)

# Получение текущего баланса
balance = await dry_trader.get_balance()

# Получение открытых позиций
positions = await dry_trader.get_open_positions()

# Закрытие позиции
trade = await dry_trader.exit_position(position["id"])

# Остановка трейдера
await dry_trader.stop()

# Получение статистики производительности
stats = await dry_trader.get_performance_stats()
```

## Формат данных

### Позиция

```json
{
    "id": "BTCUSDT-LONG-1647345678-1234",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "size": 0.1,
    "entry_price": 20000.0,
    "entry_time": "2023-03-15T12:34:56.789Z",
    "stop_loss": 19000.0,
    "take_profit": 21000.0,
    "status": "OPEN"
}
```

### Сделка

```json
{
    "position_id": "BTCUSDT-LONG-1647345678-1234",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "size": 0.1,
    "entry_price": 20000.0,
    "entry_time": "2023-03-15T12:34:56.789Z",
    "exit_price": 21000.0,
    "exit_time": "2023-03-16T10:11:12.345Z",
    "profit": 100.0,
    "profit_percent": 5.0
}
```

### Статистика производительности

```json
{
    "initial_balance": 1000.0,
    "current_balance": 1100.0,
    "profit_loss": 100.0,
    "profit_loss_percent": 10.0,
    "total_trades": 5,
    "open_positions": 1,
    "profitable_trades": 3,
    "losing_trades": 2,
    "win_rate": 60.0,
    "avg_profit": 50.0,
    "avg_loss": -25.0,
    "max_drawdown_percent": 5.0
}
``` 