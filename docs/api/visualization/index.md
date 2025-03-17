# Модуль визуализации

## Обзор

Модуль визуализации предоставляет компоненты для отображения данных и результатов торговли в различных форматах.

## Структура модуля

```
visualization/
│
├── base.py - Базовый класс визуализатора
├── candle_visualizer.py - Визуализатор свечей
├── console_ui.py - Консольный интерфейс
├── manager.py - Менеджер визуализаторов
├── trading_dashboard.py - Торговая панель
└── web_dashboard.py - Веб-панель
```

## Основные компоненты

- `BaseVisualizer` - Базовый класс для всех визуализаторов
- `ConsoleVisualizer` - Визуализатор для консоли
- `TradingDashboard` - Панель для отображения торговых данных
- `CandleVisualizer` - Визуализатор свечей
- `WebDashboard` - Веб-интерфейс для отображения данных
- `VisualizationManager` - Менеджер визуализаторов

## Использование

Для использования визуализаторов рекомендуется использовать `VisualizationManager`, который предоставляет единый интерфейс для управления всеми визуализаторами.

## Архитектура

Модуль визуализации построен на основе абстрактного класса `BaseVisualizer`, который определяет общий интерфейс для всех визуализаторов. Конкретные реализации, такие как `ConsoleVisualizer` и `WebVisualizer`, наследуются от этого класса и предоставляют специфичные для своего типа методы отображения.

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