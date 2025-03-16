# Менеджер рыночных данных

## Обзор

Модуль `data.market_data` предоставляет класс `MarketDataManager`, который отвечает за получение, обработку и распространение рыночных данных в реальном времени. Менеджер позволяет подписываться на обновления данных по различным символам, получать данные через WebSocket, а также анализировать рыночные данные.

## Классы

### MarketDataManager

```python
class MarketDataManager:
    def __init__(self, storage: Optional[DataStorage] = None):
        ...
```

Менеджер рыночных данных в реальном времени.

#### Параметры конструктора

- `storage` (Optional[DataStorage]): Хранилище данных. По умолчанию создается новый экземпляр `DataStorage`.

#### Методы

##### start

```python
async def start(self) -> bool:
    ...
```

Запуск менеджера рыночных данных.

**Возвращает:**
- `bool`: True, если менеджер успешно запущен, иначе False

##### stop

```python
async def stop(self) -> bool:
    ...
```

Остановка менеджера рыночных данных.

**Возвращает:**
- `bool`: True, если менеджер успешно остановлен, иначе False

##### subscribe

```python
def subscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
    ...
```

Подписка на обновления данных по символу.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `callback` (Callable[[Dict[str, Any]], None]): Функция обратного вызова

**Возвращает:**
- `bool`: True, если подписка успешно оформлена, иначе False

##### unsubscribe

```python
def unsubscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> bool:
    ...
```

Отписка от обновлений данных по символу.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `callback` (Callable[[Dict[str, Any]], None]): Функция обратного вызова

**Возвращает:**
- `bool`: True, если отписка успешно оформлена, иначе False

##### get_latest_data

```python
def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
    ...
```

Получение последних данных по символу.

**Параметры:**
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `Optional[Dict[str, Any]]`: Словарь с последними данными или None, если данные не найдены

##### update_data

```python
async def update_data(self, symbol: str, data: Dict[str, Any]) -> bool:
    ...
```

Обновление данных по символу.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `data` (Dict[str, Any]): Новые данные

**Возвращает:**
- `bool`: True, если данные успешно обновлены, иначе False

##### start_websocket

```python
async def start_websocket(self, client, symbols: List[str], interval: str = "1m") -> bool:
    ...
```

Запуск WebSocket для получения данных в реальном времени.

**Параметры:**
- `client`: Клиент биржи
- `symbols` (List[str]): Список символов торговых пар
- `interval` (str): Интервал свечей. По умолчанию "1m".

**Возвращает:**
- `bool`: True, если WebSocket успешно запущен, иначе False

##### _process_kline_message

```python
async def _process_kline_message(self, message: Dict[str, Any]) -> None:
    ...
```

Обработка сообщения от WebSocket.

**Параметры:**
- `message` (Dict[str, Any]): Сообщение от WebSocket

##### _save_closed_kline

```python
async def _save_closed_kline(self, symbol: str, data: Dict[str, Any]) -> None:
    ...
```

Сохранение закрытой свечи.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `data` (Dict[str, Any]): Данные свечи, включая open_time, open, high, low, close, volume, close_time и interval

##### get_ticker

```python
async def get_ticker(self, client, symbol: str) -> Optional[Dict[str, Any]]:
    ...
```

Получение текущего тикера.

**Параметры:**
- `client`: Клиент биржи
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `Optional[Dict[str, Any]]`: Словарь с данными тикера или None, если данные не найдены

##### get_order_book

```python
async def get_order_book(self, client, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
    ...
```

Получение книги ордеров.

**Параметры:**
- `client`: Клиент биржи
- `symbol` (str): Символ торговой пары
- `limit` (int): Количество ордеров. По умолчанию 20.

**Возвращает:**
- `Optional[Dict[str, Any]]`: Словарь с данными книги ордеров или None, если данные не найдены

##### analyze_market_sentiment

```python
async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
    ...
```

Анализ настроения рынка.

**Параметры:**
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `Dict[str, Any]`: Словарь с результатами анализа

##### detect_market_anomalies

```python
async def detect_market_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
    ...
```

Обнаружение аномалий на рынке.

**Параметры:**
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `List[Dict[str, Any]]`: Список обнаруженных аномалий

##### calculate_market_metrics

```python
async def calculate_market_metrics(self, symbol: str) -> Dict[str, Any]:
    ...
```

Расчет рыночных метрик.

**Параметры:**
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `Dict[str, Any]`: Словарь с рыночными метриками

##### get_historical_data

```python
async def get_historical_data(self, client, symbol: str, interval: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             limit: int = 1000,
                             add_indicators: bool = True) -> pd.DataFrame:
    ...
```

Получение исторических данных.

**Параметры:**
- `client`: Клиент биржи
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `start_date` (Optional[datetime]): Начальная дата
- `end_date` (Optional[datetime]): Конечная дата
- `limit` (int): Максимальное количество свечей. По умолчанию 1000.
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию True.

**Возвращает:**
- `pd.DataFrame`: DataFrame с историческими данными

##### combine_data

```python
async def combine_data(self, client, symbol: str, interval: str, 
                      days: int = 7, add_indicators: bool = True) -> pd.DataFrame:
    ...
```

Объединение исторических и текущих рыночных данных.

**Параметры:**
- `client`: Клиент биржи
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `days` (int): Количество дней для исторических данных. По умолчанию 7.
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию True.

**Возвращает:**
- `pd.DataFrame`: DataFrame с объединенными данными

## Пример использования

```python
import asyncio
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data.storage import DataStorage
from data.market_data import MarketDataManager
from exchange.binance.client import BinanceClient
from core.config_manager import ConfigManager

async def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    client = BinanceClient(
        api_key=config["binance"]["api_key"],
        api_secret=config["binance"]["api_secret"],
        testnet=config["binance"]["testnet"]
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Создание менеджера рыночных данных
        market_data_manager = MarketDataManager(storage)
        
        # Запуск менеджера
        await market_data_manager.start()
        
        # Символ для отслеживания
        symbol = "BTCUSDT"
        
        # Получение тикера
        ticker = await market_data_manager.get_ticker(client, symbol)
        print(f"Тикер {symbol}:")
        print(json.dumps(ticker, indent=2, ensure_ascii=False))
        
        # Получение книги ордеров
        order_book = await market_data_manager.get_order_book(client, symbol)
        print(f"\nКнига ордеров {symbol}:")
        print(f"Лучшая цена покупки: {order_book['bids'][0][0]}")
        print(f"Лучшая цена продажи: {order_book['asks'][0][0]}")
        
        # Анализ настроения рынка
        sentiment = await market_data_manager.analyze_market_sentiment(symbol)
        print(f"\nНастроение рынка {symbol}:")
        print(json.dumps(sentiment, indent=2, ensure_ascii=False))
        
        # Получение исторических данных
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        historical_data = await market_data_manager.get_historical_data(
            client, symbol, "1h", start_date, end_date, limit=168, add_indicators=True
        )
        
        print(f"\nПолучено {len(historical_data)} исторических свечей")
        print(historical_data.tail(5)[['open', 'high', 'low', 'close', 'volume']])
        
        # Визуализация исторических данных
        plt.figure(figsize=(12, 6))
        plt.plot(historical_data.index, historical_data['close'], label='Цена закрытия')
        
        if 'sma_25' in historical_data.columns:
            plt.plot(historical_data.index, historical_data['sma_25'], label='SMA 25')
        
        if 'ema_7' in historical_data.columns:
            plt.plot(historical_data.index, historical_data['ema_7'], label='EMA 7')
        
        plt.title(f'{symbol} - Исторические данные')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{symbol}_historical_data.png')
        print(f"\nГрафик сохранен в {symbol}_historical_data.png")
        
        # Объединение исторических и текущих данных
        combined_data = await market_data_manager.combine_data(
            client, symbol, "1h", days=3, add_indicators=True
        )
        
        print(f"\nОбъединенные данные (последние 5 строк):")
        print(combined_data.tail(5)[['open', 'high', 'low', 'close', 'volume']])
        
        # Анализ объединенных данных
        if 'rsi_14' in combined_data.columns:
            current_rsi = combined_data['rsi_14'].iloc[-1]
            print(f"\nТекущий RSI: {current_rsi:.2f}")
            if current_rsi > 70:
                print("Рынок перекуплен (RSI > 70)")
            elif current_rsi < 30:
                print("Рынок перепродан (RSI < 30)")
            else:
                print("Нейтральная зона RSI")
        
        # Подписка на обновления
        def on_data_update(data):
            print(f"Получено обновление для {data['symbol']}: {data['close']}")
        
        market_data_manager.subscribe(symbol, on_data_update)
        
        # Запуск WebSocket
        await market_data_manager.start_websocket(client, [symbol])
        
        # Ожидание 30 секунд для получения обновлений
        print("\nОжидание обновлений...")
        await asyncio.sleep(30)
        
        # Остановка менеджера
        await market_data_manager.stop()
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 