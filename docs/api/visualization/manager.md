# Менеджер визуализаторов

## Обзор

Модуль `visualization.manager` предоставляет класс `VisualizationManager`, который отвечает за централизованное управление всеми визуализаторами в системе. Менеджер позволяет создавать, настраивать, запускать и останавливать различные визуализаторы, а также обновлять их данные.

## Классы

### VisualizationManager

```python
class VisualizationManager:
    def __init__(self, config: Dict[str, Any] = None):
        ...
```

Менеджер визуализаторов.

#### Параметры конструктора

- `config` (Dict[str, Any], optional): Конфигурация визуализаторов

#### Методы

##### create_visualizer

```python
def create_visualizer(self, visualizer_type: str, name: Optional[str] = None, 
                     config: Optional[Dict[str, Any]] = None) -> Optional[BaseVisualizer]:
    ...
```

Создание визуализатора.

**Параметры:**
- `visualizer_type` (str): Тип визуализатора
- `name` (Optional[str], optional): Имя визуализатора (если None, будет использовано имя по умолчанию)
- `config` (Optional[Dict[str, Any]], optional): Конфигурация визуализатора

**Возвращает:**
- `Optional[BaseVisualizer]`: Созданный визуализатор или None, если тип не найден

##### add_visualizer

```python
def add_visualizer(self, visualizer: BaseVisualizer) -> bool:
    ...
```

Добавление визуализатора в список активных.

**Параметры:**
- `visualizer` (BaseVisualizer): Визуализатор

**Возвращает:**
- `bool`: True, если визуализатор успешно добавлен, иначе False

##### remove_visualizer

```python
def remove_visualizer(self, name: str) -> bool:
    ...
```

Удаление визуализатора из списка активных.

**Параметры:**
- `name` (str): Имя визуализатора

**Возвращает:**
- `bool`: True, если визуализатор успешно удален, иначе False

##### get_visualizer

```python
def get_visualizer(self, name: str) -> Optional[BaseVisualizer]:
    ...
```

Получение визуализатора по имени.

**Параметры:**
- `name` (str): Имя визуализатора

**Возвращает:**
- `Optional[BaseVisualizer]`: Визуализатор или None, если не найден

##### start_visualizer

```python
def start_visualizer(self, name: str, background: bool = False) -> bool:
    ...
```

Запуск визуализатора.

**Параметры:**
- `name` (str): Имя визуализатора
- `background` (bool, optional): Запустить в фоновом режиме. По умолчанию False.

**Возвращает:**
- `bool`: True, если визуализатор успешно запущен, иначе False

##### stop_visualizer

```python
def stop_visualizer(self, name: str) -> bool:
    ...
```

Остановка визуализатора.

**Параметры:**
- `name` (str): Имя визуализатора

**Возвращает:**
- `bool`: True, если визуализатор успешно остановлен, иначе False

##### update_visualizer

```python
def update_visualizer(self, name: str, data: Dict[str, Any]) -> bool:
    ...
```

Обновление данных визуализатора.

**Параметры:**
- `name` (str): Имя визуализатора
- `data` (Dict[str, Any]): Данные для обновления

**Возвращает:**
- `bool`: True, если данные успешно обновлены, иначе False

##### update_all_visualizers

```python
def update_all_visualizers(self, data: Dict[str, Any]) -> Dict[str, bool]:
    ...
```

Обновление данных всех активных визуализаторов.

**Параметры:**
- `data` (Dict[str, Any]): Данные для обновления

**Возвращает:**
- `Dict[str, bool]`: Словарь с результатами обновления для каждого визуализатора

##### start_all_visualizers

```python
def start_all_visualizers(self, background: bool = True) -> Dict[str, bool]:
    ...
```

Запуск всех активных визуализаторов.

**Параметры:**
- `background` (bool, optional): Запустить в фоновом режиме. По умолчанию True.

**Возвращает:**
- `Dict[str, bool]`: Словарь с результатами запуска для каждого визуализатора

##### stop_all_visualizers

```python
def stop_all_visualizers(self) -> Dict[str, bool]:
    ...
```

Остановка всех активных визуализаторов.

**Возвращает:**
- `Dict[str, bool]`: Словарь с результатами остановки для каждого визуализатора

##### get_active_visualizers

```python
def get_active_visualizers(self) -> Dict[str, BaseVisualizer]:
    ...
```

Получение списка активных визуализаторов.

**Возвращает:**
- `Dict[str, BaseVisualizer]`: Словарь активных визуализаторов

##### get_available_visualizer_types

```python
def get_available_visualizer_types(self) -> List[str]:
    ...
```

Получение списка доступных типов визуализаторов.

**Возвращает:**
- `List[str]`: Список доступных типов визуализаторов

##### register_visualizer_type

```python
def register_visualizer_type(self, name: str, visualizer_class: Type[BaseVisualizer]) -> bool:
    ...
```

Регистрация нового типа визуализатора.

**Параметры:**
- `name` (str): Имя типа визуализатора
- `visualizer_class` (Type[BaseVisualizer]): Класс визуализатора

**Возвращает:**
- `bool`: True, если тип успешно зарегистрирован, иначе False

##### create_and_start_visualizer

```python
def create_and_start_visualizer(self, visualizer_type: str, name: Optional[str] = None,
                              config: Optional[Dict[str, Any]] = None,
                              background: bool = True) -> Optional[BaseVisualizer]:
    ...
```

Создание и запуск визуализатора.

**Параметры:**
- `visualizer_type` (str): Тип визуализатора
- `name` (Optional[str], optional): Имя визуализатора
- `config` (Optional[Dict[str, Any]], optional): Конфигурация визуализатора
- `background` (bool, optional): Запустить в фоновом режиме. По умолчанию True.

**Возвращает:**
- `Optional[BaseVisualizer]`: Созданный визуализатор или None, если произошла ошибка

## Пример использования

```python
import logging
import asyncio
import time
from visualization.manager import VisualizationManager
from exchange.binance.client import BinanceClient
from core.config_manager import ConfigManager

# Настройка логирования
logging.basicConfig(level=logging.INFO)

async def main():
    # Загрузка конфигурации
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Создание клиента Binance
    client = BinanceClient(
        api_key=config["binance"]["api_key"],
        api_secret=config["binance"]["api_secret"],
        testnet=config["binance"]["testnet"]
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Получение исторических данных
        symbol = config["trading"]["symbol"]
        timeframe = config["trading"]["timeframe"]
        klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=50)
        
        # Создание менеджера визуализаторов
        visualization_manager = VisualizationManager()
        
        # Создание и запуск визуализатора свечей
        candle_visualizer = visualization_manager.create_and_start_visualizer(
            visualizer_type="candle_visualizer",
            name="candles",
            config={
                "symbol": symbol,
                "timeframe": timeframe,
                "refresh_rate": 1.0,
                "max_candles": 20,
                "price_precision": 2
            },
            background=True
        )
        
        # Создание и запуск торговой панели
        trading_dashboard = visualization_manager.create_and_start_visualizer(
            visualizer_type="trading_dashboard",
            name="trading",
            config={
                "symbol": symbol,
                "mode": config["trading"]["mode"],
                "refresh_rate": 1.0
            },
            background=True
        )
        
        # Обновление данных визуализаторов
        visualization_manager.update_visualizer("candles", {
            "candles": klines,
            "indicators": {
                "MA(9)": 50000.0,
                "MA(21)": 49500.0,
                "RSI(14)": 55.0
            }
        })
        
        visualization_manager.update_visualizer("trading", {
            "balance": 10000.0,
            "positions": [],
            "orders": [],
            "signals": []
        })
        
        # Ожидание 30 секунд для просмотра визуализаций
        for i in range(30):
            # Обновление данных каждую секунду
            if i % 5 == 0:
                # Получение новых данных
                new_klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=50)
                
                # Обновление визуализаторов
                visualization_manager.update_visualizer("candles", {
                    "candles": new_klines
                })
            
            time.sleep(1)
        
        # Остановка всех визуализаторов
        visualization_manager.stop_all_visualizers()
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 