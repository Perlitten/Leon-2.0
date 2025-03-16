# Торговая панель (Trading Dashboard)

Модуль `visualization.trading_dashboard` предоставляет интерактивную консольную панель для отображения торговой информации с использованием библиотеки rich.

## Классы

### TradingDashboard

Интерактивная консольная панель для отображения торговой информации. Использует библиотеку rich для создания красивого и информативного интерфейса в консоли.

```python
from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES

# Создание экземпляра торговой панели
dashboard = TradingDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1050.0,
    "refresh_rate": 1.0,
    "trading_params": {
        "leverage": 20,
        "risk_per_trade": 1.0,
        "stop_loss": 0.5,
        "take_profit": 1.0
    }
})

# Запуск панели
dashboard.start()

# Обновление данных
dashboard.update({
    "current_balance": 1100.0,
    "total_trades": 5,
    "winning_trades": 3,
    "losing_trades": 2,
    "positions": [
        {
            "id": 1,
            "symbol": "BTCUSDT",
            "type": "LONG",
            "size": 0.01,
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "pnl": 10.0,
            "pnl_percent": 2.0
        }
    ],
    "signals": [
        {"indicator": "RSI", "value": "44.03", "signal": "NEUTRAL"},
        {"indicator": "MACD", "value": "-168.6", "signal": "SELL"},
        {"indicator": "Bollinger", "value": "87488.5", "signal": "BUY"},
        {"indicator": "MA Cross", "value": "2432.1", "signal": "NEUTRAL"}
    ],
    "recommendation": "Рекомендуется открыть LONG позицию по BTCUSDT",
    "recommendation_color": "green"
})

# Запуск в интерактивном режиме
dashboard.run()

# Остановка панели
dashboard.stop()
```

#### Параметры конструктора

- `name` (str, опционально): Имя визуализатора (по умолчанию: "trading_dashboard")
- `config` (Dict[str, Any], опционально): Конфигурация визуализатора
  - `symbol` (str): Символ торговой пары (по умолчанию: "BTCUSDT")
  - `mode` (str): Режим работы (по умолчанию: TRADING_MODES["DRY"])
  - `initial_balance` (float): Начальный баланс (по умолчанию: 1000.0)
  - `current_balance` (float): Текущий баланс (по умолчанию: 1000.0)
  - `refresh_rate` (float): Частота обновления в секундах (по умолчанию: 1.0)
  - `trading_params` (Dict[str, Any]): Параметры торговли
    - `leverage` (int): Плечо (по умолчанию: 1)
    - `risk_per_trade` (float): Риск на сделку в процентах (по умолчанию: 1.0)
    - `stop_loss` (float): Стоп-лосс в процентах (по умолчанию: 0.5)
    - `take_profit` (float): Тейк-профит в процентах (по умолчанию: 1.0)

#### Методы

- `start() -> bool`: Запуск торговой панели
- `stop() -> bool`: Остановка торговой панели
- `update(data: Dict[str, Any]) -> bool`: Обновление данных торговой панели
- `render() -> Layout`: Отрисовка торговой панели
- `run() -> None`: Запуск торговой панели в интерактивном режиме

#### Структура данных для обновления

Метод `update()` принимает словарь с данными для обновления. Поддерживаются следующие ключи:

- `symbol` (str): Символ торговой пары
- `mode` (str): Режим работы
- `initial_balance` (float): Начальный баланс
- `current_balance` (float): Текущий баланс
- `total_trades` (int): Общее количество сделок
- `winning_trades` (int): Количество выигрышных сделок
- `losing_trades` (int): Количество убыточных сделок
- `positions` (List[Dict[str, Any]]): Список активных позиций
  - `id` (int): Идентификатор позиции
  - `symbol` (str): Символ торговой пары
  - `type` (str): Тип позиции ("LONG" или "SHORT")
  - `size` (float): Размер позиции
  - `entry_price` (float): Цена входа
  - `current_price` (float): Текущая цена
  - `pnl` (float): Прибыль/убыток в абсолютном выражении
  - `pnl_percent` (float): Прибыль/убыток в процентах
- `signals` (List[Dict[str, Any]]): Список сигналов
  - `indicator` (str): Название индикатора
  - `value` (str): Значение индикатора
  - `signal` (str): Сигнал ("BUY", "SELL" или "NEUTRAL")
- `recommendation` (str): Рекомендация
- `recommendation_color` (str): Цвет рекомендации ("green", "red" или "yellow")
- `trading_params` (Dict[str, Any]): Параметры торговли
  - `leverage` (int): Плечо
  - `risk_per_trade` (float): Риск на сделку в процентах
  - `stop_loss` (float): Стоп-лосс в процентах
  - `take_profit` (float): Тейк-профит в процентах

## Пример использования

```python
import logging
import time
import random
from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создание торговой панели
dashboard = TradingDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1000.0,
    "refresh_rate": 1.0
})

# Запуск панели
dashboard.start()

# Симуляция обновления данных
try:
    while True:
        # Обновление данных
        dashboard.update({
            "current_balance": 1000.0 + random.uniform(-100, 200),
            "total_trades": random.randint(0, 20),
            "winning_trades": random.randint(0, 10),
            "losing_trades": random.randint(0, 10),
            "signals": [
                {"indicator": "RSI", "value": f"{random.uniform(30, 70):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
                {"indicator": "MACD", "value": f"{random.uniform(-200, 200):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
                {"indicator": "Bollinger", "value": f"{random.uniform(50000, 100000):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])},
                {"indicator": "MA Cross", "value": f"{random.uniform(2000, 3000):.2f}", "signal": random.choice(["BUY", "SELL", "NEUTRAL"])}
            ]
        })
        
        # Пауза перед следующим обновлением
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка панели при нажатии Ctrl+C
    dashboard.stop()
```

## Интеграция с оркестратором

```python
from core.orchestrator import LeonOrchestrator
from visualization.trading_dashboard import TradingDashboard

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator()

# Создание торговой панели
dashboard = TradingDashboard()

# Регистрация обработчиков событий
orchestrator._event_bus.subscribe("PRICE_UPDATED", lambda data: dashboard.update({"current_price": data["price"]}))
orchestrator._event_bus.subscribe("BALANCE_UPDATED", lambda data: dashboard.update({"current_balance": data["balance"]}))
orchestrator._event_bus.subscribe("POSITION_OPENED", lambda data: dashboard.update({"positions": orchestrator.get_positions()}))
orchestrator._event_bus.subscribe("POSITION_CLOSED", lambda data: dashboard.update({"positions": orchestrator.get_positions()}))
orchestrator._event_bus.subscribe("PREDICTION_RECEIVED", lambda data: dashboard.update({"signals": data["signals"]}))

# Запуск оркестратора
orchestrator.start()

# Запуск торговой панели
dashboard.start()

# Основной цикл
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка при нажатии Ctrl+C
    dashboard.stop()
    orchestrator.stop() 