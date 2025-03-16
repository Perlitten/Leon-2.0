# Клиент для работы с Binance API

## Обзор

Модуль `exchange.binance.client` предоставляет класс `BinanceClient`, который реализует интерфейс `ExchangeBase` для работы с биржей Binance. Клиент поддерживает как REST API, так и WebSocket соединения.

## Классы

### BinanceClient

```python
class BinanceClient(ExchangeBase):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        ...
```

Клиент для работы с Binance API.

#### Параметры конструктора

- `api_key` (str): API ключ Binance
- `api_secret` (str): API секрет Binance
- `testnet` (bool, optional): Использовать тестовую сеть Binance. По умолчанию `False`.

#### Методы

##### initialize

```python
async def initialize(self):
    ...
```

Инициализация HTTP сессии для работы с API.

##### close

```python
async def close(self):
    ...
```

Закрытие HTTP сессии и WebSocket соединений.

##### ping

```python
async def ping(self) -> Dict[str, Any]:
    ...
```

Проверка соединения с Binance API.

**Возвращает:**
- `Dict[str, Any]`: Результат проверки соединения

##### get_server_time

```python
async def get_server_time(self) -> Dict[str, Any]:
    ...
```

Получение текущего времени сервера Binance.

**Возвращает:**
- `Dict[str, Any]`: Информация о времени сервера

##### get_exchange_info

```python
async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
    ...
```

Получение информации о бирже или конкретной торговой паре.

**Параметры:**
- `symbol` (Optional[str], optional): Торговая пара (например, "BTCUSDT")

**Возвращает:**
- `Dict[str, Any]`: Информация о бирже или торговой паре

##### get_ticker

```python
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
async def get_account_info(self) -> Dict[str, Any]:
    ...
```

Получение информации об аккаунте.

**Возвращает:**
- `Dict[str, Any]`: Информация об аккаунте

##### place_order

```python
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
async def subscribe_to_klines(self, symbol: str, interval: str, callback):
    ...
```

Подписка на обновления свечей через WebSocket.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `interval` (str): Интервал свечей (например, "1m", "5m", "1h", "1d")
- `callback`: Функция обратного вызова для обработки обновлений

##### subscribe_to_ticker

```python
async def subscribe_to_ticker(self, symbol: str, callback):
    ...
```

Подписка на обновления тикера через WebSocket.

**Параметры:**
- `symbol` (str): Торговая пара (например, "BTCUSDT")
- `callback`: Функция обратного вызова для обработки обновлений

## Пример использования

```python
import asyncio
import logging
from exchange.binance.client import BinanceClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)

async def ticker_callback(data):
    print(f"Получено обновление тикера: {data['s']} - {data['c']}")

async def main():
    # Создание клиента Binance
    client = BinanceClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Получение текущей цены BTC/USDT
        ticker = await client.get_ticker("BTCUSDT")
        print(f"Текущая цена BTC/USDT: {ticker['price']}")
        
        # Получение исторических данных
        klines = await client.get_klines(
            symbol="BTCUSDT",
            interval="1h",
            limit=10
        )
        print(f"Получено {len(klines)} свечей")
        
        # Подписка на обновления тикера
        await client.subscribe_to_ticker("BTCUSDT", ticker_callback)
        
        # Ожидание обновлений
        await asyncio.sleep(60)
        
    finally:
        # Закрытие соединения
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 