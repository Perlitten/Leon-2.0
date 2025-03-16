# Базовый класс для работы с биржами

## Обзор

Модуль `exchange.base` предоставляет базовый абстрактный класс `ExchangeBase`, который определяет общий интерфейс для всех классов, работающих с биржами. Конкретные реализации должны наследоваться от этого класса и реализовывать все абстрактные методы.

## Классы

### ExchangeBase

```python
class ExchangeBase(ABC):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        ...
```

Базовый абстрактный класс для работы с биржами.

#### Параметры конструктора

- `api_key` (str): API ключ биржи
- `api_secret` (str): API секрет биржи
- `testnet` (bool, optional): Использовать тестовую сеть биржи. По умолчанию `False`.

#### Методы

##### initialize

```python
@abstractmethod
async def initialize(self):
    ...
```

Инициализация соединения с биржей. Должна быть вызвана перед использованием других методов.

##### close

```python
@abstractmethod
async def close(self):
    ...
```

Закрытие соединения с биржей. Должна быть вызвана при завершении работы с биржей.

##### ping

```python
@abstractmethod
async def ping(self) -> Dict[str, Any]:
    ...
```

Проверка соединения с биржей.

**Возвращает:**
- `Dict[str, Any]`: Результат проверки соединения

##### get_ticker

```python
@abstractmethod
async def get_ticker(self, symbol: str) -> Dict[str, Any]:
    ...
```

Получение текущей цены для указанного символа.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")

**Возвращает:**
- `Dict[str, Any]`: Информация о текущей цене

##### get_klines

```python
@abstractmethod
async def get_klines(self, symbol: str, interval: str, 
                    start_time: Optional[int] = None, 
                    end_time: Optional[int] = None,
                    limit: int = 500) -> List[List[Union[int, str, float]]]:
    ...
```

Получение исторических данных (свечей) для указанного символа.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `interval` (str): Интервал свечей (например, "1m", "5m", "1h", "1d")
- `start_time` (Optional[int], optional): Начальное время в миллисекундах
- `end_time` (Optional[int], optional): Конечное время в миллисекундах
- `limit` (int, optional): Максимальное количество свечей. По умолчанию 500.

**Возвращает:**
- `List[List[Union[int, str, float]]]`: Список свечей

##### get_account_info

```python
@abstractmethod
async def get_account_info(self) -> Dict[str, Any]:
    ...
```

Получение информации об аккаунте.

**Возвращает:**
- `Dict[str, Any]`: Информация об аккаунте

##### place_order

```python
@abstractmethod
async def place_order(self, symbol: str, side: str, order_type: str, 
                     quantity: float, price: Optional[float] = None,
                     time_in_force: str = "GTC", **kwargs) -> Dict[str, Any]:
    ...
```

Размещение ордера.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `side` (str): Сторона ордера ("BUY" или "SELL")
- `order_type` (str): Тип ордера ("LIMIT", "MARKET", и т.д.)
- `quantity` (float): Количество
- `price` (Optional[float], optional): Цена (для LIMIT ордеров)
- `time_in_force` (str, optional): Время действия ордера (для LIMIT ордеров). По умолчанию "GTC".
- `**kwargs`: Дополнительные параметры

**Возвращает:**
- `Dict[str, Any]`: Информация о размещенном ордере

##### cancel_order

```python
@abstractmethod
async def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                      orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
    ...
```

Отмена ордера.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `order_id` (Optional[int], optional): ID ордера
- `orig_client_order_id` (Optional[str], optional): ID клиентского ордера

**Возвращает:**
- `Dict[str, Any]`: Информация об отмененном ордере

##### get_open_orders

```python
@abstractmethod
async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    ...
```

Получение открытых ордеров.

**Параметры:**
- `symbol` (Optional[str], optional): Торговая пара (например, "BTCUSDT")

**Возвращает:**
- `List[Dict[str, Any]]`: Список открытых ордеров

##### subscribe_to_klines

```python
@abstractmethod
async def subscribe_to_klines(self, symbol: str, interval: str, callback):
    ...
```

Подписка на обновления свечей.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `interval` (str): Интервал свечей (например, "1m", "5m", "1h", "1d")
- `callback`: Функция обратного вызова для обработки обновлений

##### subscribe_to_ticker

```python
@abstractmethod
async def subscribe_to_ticker(self, symbol: str, callback):
    ...
```

Подписка на обновления тикера.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `callback`: Функция обратного вызова для обработки обновлений

## Пример использования

```python
import asyncio
from exchange.binance.client import BinanceClient

async def main():
    # Создание клиента Binance
    client = BinanceClient(api_key="your_api_key", api_secret="your_api_secret", testnet=True)
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Получение текущей цены BTC/USDT
        ticker = await client.get_ticker("BTCUSDT")
        print(f"Текущая цена BTC/USDT: {ticker['price']}")
        
        # Получение информации об аккаунте
        account_info = await client.get_account_info()
        print(f"Баланс USDT: {next((asset['free'] for asset in account_info['balances'] if asset['asset'] == 'USDT'), 0)}")
        
        # Размещение ордера
        order = await client.place_order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.001
        )
        print(f"Ордер размещен: {order}")
        
    finally:
        # Закрытие соединения
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 