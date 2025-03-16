# Веб-панель (Web Dashboard)

Модуль `visualization.web_dashboard` предоставляет веб-интерфейс для отображения торговой информации с использованием Flask и Plotly.

## Классы

### WebDashboard

Веб-панель для отображения торговой информации. Использует Flask для создания веб-сервера и Plotly для визуализации графиков и диаграмм.

```python
from visualization.web_dashboard import WebDashboard
from core.constants import TRADING_MODES

# Создание экземпляра веб-панели
dashboard = WebDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1050.0,
    "host": "127.0.0.1",
    "port": 8080,
    "debug": True,
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
    "recommendation_color": "success"
})

# Остановка панели
dashboard.stop()
```

#### Параметры конструктора

- `name` (str, опционально): Имя визуализатора (по умолчанию: "web_dashboard")
- `config` (Dict[str, Any], опционально): Конфигурация визуализатора
  - `symbol` (str): Символ торговой пары (по умолчанию: "BTCUSDT")
  - `mode` (str): Режим работы (по умолчанию: TRADING_MODES["DRY"])
  - `initial_balance` (float): Начальный баланс (по умолчанию: 1000.0)
  - `current_balance` (float): Текущий баланс (по умолчанию: 1000.0)
  - `host` (str): Хост для веб-сервера (по умолчанию: "127.0.0.1")
  - `port` (int): Порт для веб-сервера (по умолчанию: 8080)
  - `debug` (bool): Режим отладки (по умолчанию: False)
  - `max_price_history` (int): Максимальное количество записей в истории цен (по умолчанию: 1000)
  - `max_balance_history` (int): Максимальное количество записей в истории баланса (по умолчанию: 1000)
  - `max_trade_history` (int): Максимальное количество записей в истории сделок (по умолчанию: 100)
  - `trading_params` (Dict[str, Any]): Параметры торговли
    - `leverage` (int): Плечо (по умолчанию: 1)
    - `risk_per_trade` (float): Риск на сделку в процентах (по умолчанию: 1.0)
    - `stop_loss` (float): Стоп-лосс в процентах (по умолчанию: 0.5)
    - `take_profit` (float): Тейк-профит в процентах (по умолчанию: 1.0)

#### Методы

- `start() -> bool`: Запуск веб-панели
- `stop() -> bool`: Остановка веб-панели
- `update(data: Dict[str, Any]) -> bool`: Обновление данных веб-панели
- `render() -> Dict[str, Any]`: Отрисовка веб-панели

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
- `recommendation_color` (str): Цвет рекомендации ("success", "danger" или "warning")
- `current_price` (float): Текущая цена
- `new_trade` (Dict[str, Any]): Новая сделка для добавления в историю
  - `id` (int): Идентификатор сделки
  - `symbol` (str): Символ торговой пары
  - `type` (str): Тип сделки ("LONG" или "SHORT")
  - `entry_time` (str): Время входа в формате ISO
  - `entry_price` (float): Цена входа
  - `exit_time` (str, опционально): Время выхода в формате ISO
  - `exit_price` (float, опционально): Цена выхода
  - `pnl` (float, опционально): Прибыль/убыток в абсолютном выражении
  - `pnl_percent` (float, опционально): Прибыль/убыток в процентах

#### API эндпоинты

Веб-панель предоставляет следующие API эндпоинты:

- `/`: Главная страница
- `/api/data`: Получение всех данных
- `/api/chart/price`: Получение графика цены
- `/api/chart/balance`: Получение графика баланса
- `/api/positions`: Получение активных позиций
- `/api/signals`: Получение сигналов
- `/api/stats`: Получение статистики
- `/api/recommendation`: Получение рекомендации
- `/api/trading_params`: Получение параметров торговли

## Пример использования

```python
import logging
import time
from visualization.web_dashboard import WebDashboard
from core.constants import TRADING_MODES

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Создание веб-панели
dashboard = WebDashboard(config={
    "symbol": "BTCUSDT",
    "mode": TRADING_MODES["DRY"],
    "initial_balance": 1000.0,
    "current_balance": 1000.0,
    "host": "127.0.0.1",
    "port": 8080
})

# Запуск панели
dashboard.start()

# Симуляция обновления данных
try:
    for i in range(100):
        # Обновление данных
        dashboard.update({
            "current_balance": 1000.0 + i,
            "symbol": "BTCUSDT",
            "current_price": 50000.0 + i * 10
        })
        
        # Пауза перед следующим обновлением
        time.sleep(0.1)
    
    # Ожидание завершения
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка панели при нажатии Ctrl+C
    dashboard.stop()
```

## Интеграция с оркестратором

```python
from core.orchestrator import LeonOrchestrator
from visualization.web_dashboard import WebDashboard

# Создание экземпляра оркестратора
orchestrator = LeonOrchestrator()

# Создание веб-панели
dashboard = WebDashboard()

# Регистрация обработчиков событий
orchestrator._event_bus.subscribe("PRICE_UPDATED", lambda data: dashboard.update({
    "symbol": data["symbol"],
    "current_price": data["price"]
}))
orchestrator._event_bus.subscribe("BALANCE_UPDATED", lambda data: dashboard.update({
    "current_balance": data["balance"]
}))
orchestrator._event_bus.subscribe("POSITION_OPENED", lambda data: dashboard.update({
    "positions": orchestrator.get_positions(),
    "new_trade": {
        "id": data["position_id"],
        "symbol": data["symbol"],
        "type": data["type"],
        "entry_time": data["time"],
        "entry_price": data["price"]
    }
}))
orchestrator._event_bus.subscribe("POSITION_CLOSED", lambda data: dashboard.update({
    "positions": orchestrator.get_positions(),
    "new_trade": {
        "id": data["position_id"],
        "symbol": data["symbol"],
        "type": data["type"],
        "entry_time": data["entry_time"],
        "entry_price": data["entry_price"],
        "exit_time": data["time"],
        "exit_price": data["price"],
        "pnl": data["pnl"],
        "pnl_percent": data["pnl_percent"]
    }
}))
orchestrator._event_bus.subscribe("PREDICTION_RECEIVED", lambda data: dashboard.update({
    "signals": data["signals"],
    "recommendation": data["recommendation"],
    "recommendation_color": data["recommendation_color"]
}))

# Запуск оркестратора
orchestrator.start()

# Запуск веб-панели
dashboard.start()

# Основной цикл
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Остановка при нажатии Ctrl+C
    dashboard.stop()
    orchestrator.stop() 