# Базовый модуль визуализации

Модуль `visualization.base` предоставляет базовые классы и интерфейсы для визуализации данных в Leon Trading Bot.

## Классы

### BaseVisualizer

Базовый абстрактный класс для всех визуализаторов. Определяет общий интерфейс для всех визуализаторов в системе.

```python
from visualization.base import BaseVisualizer

class MyVisualizer(BaseVisualizer):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
    def start(self) -> bool:
        # Реализация метода start
        return super().start()
    
    def stop(self) -> bool:
        # Реализация метода stop
        return super().stop()
    
    def update(self, data: Dict[str, Any]) -> bool:
        # Реализация метода update
        return super().update(data)
    
    def render(self) -> Any:
        # Реализация метода render
        return None
```

#### Параметры конструктора

- `name` (str): Имя визуализатора
- `config` (Dict[str, Any], опционально): Конфигурация визуализатора

#### Методы

- `start() -> bool`: Запуск визуализатора
- `stop() -> bool`: Остановка визуализатора
- `update(data: Dict[str, Any]) -> bool`: Обновление данных визуализатора
- `render() -> Any`: Отрисовка визуализации

### ConsoleVisualizer

Базовый класс для визуализаторов, работающих в консоли. Предоставляет общую функциональность для консольных визуализаторов.

```python
from visualization.base import ConsoleVisualizer

# Создание экземпляра консольного визуализатора
visualizer = ConsoleVisualizer("my_visualizer", {
    "refresh_rate": 1.0,
    "width": 80,
    "height": 24,
    "clear_screen": True
})

# Запуск визуализатора
visualizer.start()

# Обновление данных
visualizer.update({"key": "value"})

# Очистка экрана
visualizer.clear()

# Отрисовка
result = visualizer.render()

# Остановка визуализатора
visualizer.stop()
```

#### Параметры конструктора

- `name` (str): Имя визуализатора
- `config` (Dict[str, Any], опционально): Конфигурация визуализатора
  - `refresh_rate` (float): Частота обновления в секундах (по умолчанию: 1.0)
  - `width` (int): Ширина консоли (по умолчанию: 80)
  - `height` (int): Высота консоли (по умолчанию: 24)
  - `clear_screen` (bool): Очищать экран перед отрисовкой (по умолчанию: True)

#### Методы

- `clear() -> None`: Очистка экрана
- `start() -> bool`: Запуск визуализатора
- `stop() -> bool`: Остановка визуализатора
- `update(data: Dict[str, Any]) -> bool`: Обновление данных визуализатора
- `render() -> str`: Отрисовка визуализации

### WebVisualizer

Базовый класс для визуализаторов, работающих через веб-интерфейс. Предоставляет общую функциональность для веб-визуализаторов.

```python
from visualization.base import WebVisualizer

# Создание экземпляра веб-визуализатора
visualizer = WebVisualizer("my_visualizer", {
    "host": "127.0.0.1",
    "port": 8080,
    "debug": False
})

# Запуск визуализатора
visualizer.start()

# Обновление данных
visualizer.update({"key": "value"})

# Отрисовка
result = visualizer.render()

# Остановка визуализатора
visualizer.stop()
```

#### Параметры конструктора

- `name` (str): Имя визуализатора
- `config` (Dict[str, Any], опционально): Конфигурация визуализатора
  - `host` (str): Хост для веб-сервера (по умолчанию: "127.0.0.1")
  - `port` (int): Порт для веб-сервера (по умолчанию: 8080)
  - `debug` (bool): Режим отладки (по умолчанию: False)

#### Методы

- `start() -> bool`: Запуск визуализатора
- `stop() -> bool`: Остановка визуализатора
- `update(data: Dict[str, Any]) -> bool`: Обновление данных визуализатора
- `render() -> Dict[str, Any]`: Отрисовка визуализации 