# Модуль стратегий

## Обзор

Модуль `trading.strategies` предоставляет базовый класс и реализации различных торговых стратегий для использования в Leon Trading Bot. Стратегии отвечают за анализ рыночных данных и генерацию торговых сигналов.

## Классы

### StrategyBase

```python
class StrategyBase(ABC):
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any] = None):
        ...
```

Базовый абстрактный класс для торговых стратегий. Определяет общий интерфейс для всех стратегий в системе.

#### Параметры конструктора

- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `timeframe` (str): Таймфрейм для анализа (например, "1m", "5m", "1h", "1d")
- `params` (Dict[str, Any], optional): Дополнительные параметры стратегии

#### Методы

##### initialize

```python
@abstractmethod
async def initialize(self, historical_data: List[List[Union[int, str, float]]]) -> bool:
    ...
```

Инициализация стратегии с историческими данными.

**Параметры:**
- `historical_data` (List[List[Union[int, str, float]]]): Исторические данные для инициализации стратегии

**Возвращает:**
- `bool`: Успешность инициализации

##### update

```python
@abstractmethod
async def update(self, candle: List[Union[int, str, float]]) -> None:
    ...
```

Обновление стратегии новыми данными.

**Параметры:**
- `candle` (List[Union[int, str, float]]): Новая свеча для обновления стратегии

##### generate_signal

```python
@abstractmethod
async def generate_signal(self) -> Dict[str, Any]:
    ...
```

Генерация торгового сигнала на основе текущего состояния стратегии.

**Возвращает:**
- `Dict[str, Any]`: Торговый сигнал в виде словаря с ключами:
  - `"action"`: Действие ("BUY", "SELL", "HOLD")
  - `"price"`: Цена входа (опционально)
  - `"stop_loss"`: Уровень стоп-лосса (опционально)
  - `"take_profit"`: Уровень тейк-профита (опционально)
  - `"reason"`: Причина сигнала (опционально)
  - `"confidence"`: Уверенность в сигнале от 0 до 1 (опционально)

##### calculate_position_size

```python
@abstractmethod
async def calculate_position_size(self, balance: float, risk_per_trade: float) -> float:
    ...
```

Расчет размера позиции на основе баланса и риска.

**Параметры:**
- `balance` (float): Доступный баланс
- `risk_per_trade` (float): Процент риска на сделку (от 0 до 100)

**Возвращает:**
- `float`: Размер позиции

##### calculate_stop_loss

```python
@abstractmethod
async def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
    ...
```

Расчет уровня стоп-лосса.

**Параметры:**
- `entry_price` (float): Цена входа
- `direction` (str): Направление сделки ("BUY" или "SELL")

**Возвращает:**
- `float`: Уровень стоп-лосса

##### calculate_take_profit

```python
@abstractmethod
async def calculate_take_profit(self, entry_price: float, direction: str) -> float:
    ...
```

Расчет уровня тейк-профита.

**Параметры:**
- `entry_price` (float): Цена входа
- `direction` (str): Направление сделки ("BUY" или "SELL")

**Возвращает:**
- `float`: Уровень тейк-профита

##### should_exit

```python
@abstractmethod
async def should_exit(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
    ...
```

Проверка необходимости выхода из позиции.

**Параметры:**
- `position` (Dict[str, Any]): Информация о текущей позиции
- `current_price` (float): Текущая цена

**Возвращает:**
- `Tuple[bool, str]`: Кортеж (нужно ли выходить, причина выхода)

##### get_indicators

```python
@abstractmethod
async def get_indicators(self) -> Dict[str, Any]:
    ...
```

Получение текущих значений индикаторов стратегии.

**Возвращает:**
- `Dict[str, Any]`: Словарь с текущими значениями индикаторов

##### get_state

```python
@abstractmethod
async def get_state(self) -> Dict[str, Any]:
    ...
```

Получение текущего состояния стратегии.

**Возвращает:**
- `Dict[str, Any]`: Словарь с текущим состоянием стратегии

##### set_state

```python
@abstractmethod
async def set_state(self, state: Dict[str, Any]) -> None:
    ...
```

Установка состояния стратегии.

**Параметры:**
- `state` (Dict[str, Any]): Состояние стратегии

### SimpleMAStrategy

```python
class SimpleMAStrategy(StrategyBase):
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any] = None):
        ...
```

Простая стратегия на основе скользящих средних. Стратегия использует пересечение быстрой и медленной скользящих средних для генерации торговых сигналов.

#### Параметры конструктора

- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `timeframe` (str): Таймфрейм для анализа (например, "1m", "5m", "1h", "1d")
- `params` (Dict[str, Any], optional): Дополнительные параметры стратегии:
  - `fast_ma_period`: Период быстрой скользящей средней (по умолчанию 9)
  - `slow_ma_period`: Период медленной скользящей средней (по умолчанию 21)
  - `stop_loss_percent`: Процент стоп-лосса (по умолчанию 2.0)
  - `take_profit_percent`: Процент тейк-профита (по умолчанию 3.0)

#### Методы

Реализует все абстрактные методы базового класса `StrategyBase`.

## Пример использования

```python
import asyncio
from exchange.binance.client import BinanceClient
from trading.strategies.simple_ma import SimpleMAStrategy

async def main():
    # Создание клиента Binance
    client = BinanceClient(api_key="your_api_key", api_secret="your_api_secret", testnet=True)
    await client.initialize()
    
    try:
        # Получение исторических данных
        symbol = "BTCUSDT"
        timeframe = "1h"
        klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=100)
        
        # Создание и инициализация стратегии
        strategy = SimpleMAStrategy(
            symbol=symbol,
            timeframe=timeframe,
            params={
                "fast_ma_period": 9,
                "slow_ma_period": 21,
                "stop_loss_percent": 2.0,
                "take_profit_percent": 3.0
            }
        )
        
        # Инициализация стратегии историческими данными
        await strategy.initialize(klines)
        
        # Получение текущего сигнала
        signal = await strategy.generate_signal()
        print(f"Сигнал: {signal}")
        
        # Расчет размера позиции
        balance = 1000.0  # USDT
        risk_per_trade = 2.0  # 2% риска на сделку
        position_size = await strategy.calculate_position_size(balance, risk_per_trade)
        print(f"Размер позиции: {position_size} BTC")
        
        # Получение текущих индикаторов
        indicators = await strategy.get_indicators()
        print(f"Индикаторы: {indicators}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 