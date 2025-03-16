# Модуль визуализации

Модуль визуализации предоставляет инструменты для отображения торговой информации в различных форматах, включая консольный интерфейс и веб-интерфейс.

## Содержание

- [Базовые классы визуализации](base.md)
- [Торговая панель](trading_dashboard.md)
- [Примеры использования](examples.md)

## Обзор

Модуль визуализации предназначен для отображения торговой информации в режиме реального времени. Он предоставляет следующие возможности:

- Отображение текущего баланса и прибыли/убытка
- Отображение открытых позиций и их статуса
- Отображение сигналов от индикаторов и рекомендаций
- Отображение исторических данных и графиков
- Интеграция с оркестратором для автоматического обновления данных

## Архитектура

Модуль визуализации построен на основе абстрактного класса `BaseVisualizer`, который определяет общий интерфейс для всех визуализаторов. Конкретные реализации, такие как `ConsoleVisualizer` и `WebVisualizer`, наследуются от этого класса и предоставляют специфичные для своего типа методы отображения.

```
BaseVisualizer
├── ConsoleVisualizer
│   └── TradingDashboard
└── WebVisualizer
```

## Использование

Для использования модуля визуализации необходимо создать экземпляр нужного визуализатора, настроить его параметры и запустить. После этого можно обновлять данные визуализатора с помощью метода `update()`.

```python
from visualization.trading_dashboard import TradingDashboard

# Создание торговой панели
dashboard = TradingDashboard(config={
    "symbol": "BTCUSDT",
    "mode": "DRY",
    "initial_balance": 1000.0,
    "current_balance": 1000.0,
    "refresh_rate": 1.0
})

# Запуск панели
dashboard.start()

# Обновление данных
dashboard.update({
    "current_balance": 1100.0,
    "total_trades": 5,
    "winning_trades": 3,
    "losing_trades": 2
})

# Остановка панели
dashboard.stop()
```

## Интеграция с оркестратором

Модуль визуализации может быть интегрирован с оркестратором для автоматического обновления данных. Для этого необходимо зарегистрировать обработчики событий оркестратора, которые будут вызывать метод `update()` визуализатора при получении новых данных.

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

# Запуск оркестратора и торговой панели
orchestrator.start()
dashboard.start()
```

## Расширение

Модуль визуализации может быть расширен путем создания новых классов, наследующихся от `BaseVisualizer`. Для этого необходимо реализовать методы `start()`, `stop()`, `update()` и `render()`.

```python
from visualization.base import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    def __init__(self, name="custom_visualizer", config=None):
        super().__init__(name, config or {})
        self.data = {}
        
    def start(self):
        # Инициализация визуализатора
        return True
        
    def stop(self):
        # Остановка визуализатора
        return True
        
    def update(self, data):
        # Обновление данных
        self.data.update(data)
        return True
        
    def render(self):
        # Отрисовка визуализации
        return str(self.data)
``` 