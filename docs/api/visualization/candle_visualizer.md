# Визуализатор свечей

## Обзор

Модуль `visualization.candle_visualizer` предоставляет класс `CandleVisualizer`, который отображает свечи, объемы и индикаторы в консоли с использованием библиотеки rich. Визуализатор создает интерактивный интерфейс для анализа ценовых данных и технических индикаторов.

## Классы

### CandleVisualizer

```python
class CandleVisualizer(ConsoleVisualizer):
    def __init__(self, name: str = "candle_visualizer", config: Dict[str, Any] = None):
        ...
```

Консольный визуализатор свечей и индикаторов.

#### Параметры конструктора

- `name` (str, optional): Имя визуализатора. По умолчанию "candle_visualizer".
- `config` (Dict[str, Any], optional): Конфигурация визуализатора, которая может включать:
  - `symbol` (str): Символ торговой пары (по умолчанию "BTCUSDT")
  - `timeframe` (str): Таймфрейм (по умолчанию "1h")
  - `price_precision` (int): Точность отображения цены (по умолчанию 2)
  - `volume_precision` (int): Точность отображения объема (по умолчанию 2)
  - `max_candles` (int): Максимальное количество отображаемых свечей (по умолчанию 20)
  - `show_volumes` (bool): Отображать ли объемы (по умолчанию True)
  - `show_indicators` (bool): Отображать ли индикаторы (по умолчанию True)
  - `refresh_rate` (float): Частота обновления в секундах (по умолчанию 1.0)

#### Методы

##### start

```python
def start(self) -> bool:
    ...
```

Запуск визуализатора свечей.

**Возвращает:**
- `bool`: True, если визуализатор успешно запущен, иначе False

##### stop

```python
def stop(self) -> bool:
    ...
```

Остановка визуализатора свечей.

**Возвращает:**
- `bool`: True, если визуализатор успешно остановлен, иначе False

##### update

```python
def update(self, data: Dict[str, Any]) -> bool:
    ...
```

Обновление данных визуализатора свечей.

**Параметры:**
- `data` (Dict[str, Any]): Данные для обновления, которые могут включать:
  - `symbol` (str): Символ торговой пары
  - `timeframe` (str): Таймфрейм
  - `candles` (List): Список свечей в формате [timestamp, open, high, low, close, volume]
  - `volumes` (List): Список объемов
  - `indicators` (Dict): Словарь с индикаторами

**Возвращает:**
- `bool`: True, если данные успешно обновлены, иначе False

##### render

```python
def render(self) -> Layout:
    ...
```

Отрисовка визуализатора свечей.

**Возвращает:**
- `Layout`: Layout объект с визуализацией

##### run

```python
def run(self) -> None:
    ...
```

Запуск визуализатора в интерактивном режиме.

## Пример использования

```python
import logging
import asyncio
from datetime import datetime
from visualization.candle_visualizer import CandleVisualizer
from exchange.binance.client import BinanceClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)

async def main():
    # Создание клиента Binance
    client = BinanceClient(api_key="your_api_key", api_secret="your_api_secret", testnet=True)
    await client.initialize()
    
    try:
        # Получение исторических данных
        symbol = "BTCUSDT"
        timeframe = "1h"
        klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=50)
        
        # Создание визуализатора
        visualizer = CandleVisualizer(config={
            "symbol": symbol,
            "timeframe": timeframe,
            "refresh_rate": 1.0,
            "max_candles": 20,
            "price_precision": 2
        })
        
        # Обновление данных визуализатора
        visualizer.update({
            "candles": klines,
            "indicators": {
                "MA(9)": 50000.0,
                "MA(21)": 49500.0,
                "RSI(14)": 55.0,
                "Bollinger": {
                    "Upper": 51000.0,
                    "Middle": 50000.0,
                    "Lower": 49000.0
                }
            }
        })
        
        # Запуск визуализатора
        visualizer.run()
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Формат данных

### Свечи

Свечи должны быть представлены в формате списка списков, где каждая свеча содержит:
```
[timestamp, open, high, low, close, volume]
```

- `timestamp`: Временная метка в миллисекундах
- `open`: Цена открытия
- `high`: Максимальная цена
- `low`: Минимальная цена
- `close`: Цена закрытия
- `volume`: Объем

### Индикаторы

Индикаторы должны быть представлены в формате словаря, где ключи - названия индикаторов, а значения - числа, списки или словари:

```python
{
    "MA(9)": 50000.0,
    "MA(21)": 49500.0,
    "RSI(14)": 55.0,
    "Bollinger": {
        "Upper": 51000.0,
        "Middle": 50000.0,
        "Lower": 49000.0
    }
}
``` 