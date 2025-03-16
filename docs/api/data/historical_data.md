# Менеджер исторических данных

## Обзор

Модуль `data.historical_data` предоставляет класс `HistoricalDataManager`, который отвечает за загрузку, обработку и анализ исторических данных для торговли. Менеджер позволяет загружать исторические данные с биржи, сохранять их в локальное хранилище, добавлять технические индикаторы и обнаруживать паттерны.

## Классы

### HistoricalDataManager

```python
class HistoricalDataManager:
    def __init__(self, storage: Optional[DataStorage] = None, client: Optional[BinanceClient] = None):
        ...
```

Менеджер исторических данных.

#### Параметры конструктора

- `storage` (Optional[DataStorage]): Хранилище данных. По умолчанию создается новый экземпляр `DataStorage`.
- `client` (Optional[BinanceClient]): Клиент Binance. По умолчанию `None`.

#### Методы

##### set_client

```python
async def set_client(self, client: BinanceClient) -> None:
    ...
```

Установка клиента Binance.

**Параметры:**
- `client` (BinanceClient): Клиент Binance

##### load_historical_data

```python
async def load_historical_data(self, symbol: str, interval: str, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             use_cache: bool = True,
                             add_indicators: bool = True,
                             detect_patterns: bool = False) -> pd.DataFrame:
    ...
```

Загрузка исторических данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `start_date` (Optional[datetime]): Начальная дата
- `end_date` (Optional[datetime]): Конечная дата
- `use_cache` (bool): Использовать кэш. По умолчанию `True`.
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию `True`.
- `detect_patterns` (bool): Обнаружить паттерны. По умолчанию `False`.

**Возвращает:**
- `pd.DataFrame`: DataFrame с историческими данными

##### update_historical_data

```python
async def update_historical_data(self, symbol: str, interval: str,
                               days_to_update: int = 1,
                               add_indicators: bool = True,
                               detect_patterns: bool = False) -> pd.DataFrame:
    ...
```

Обновление исторических данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `days_to_update` (int): Количество дней для обновления. По умолчанию `1`.
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию `True`.
- `detect_patterns` (bool): Обнаружить паттерны. По умолчанию `False`.

**Возвращает:**
- `pd.DataFrame`: DataFrame с обновленными историческими данными

##### get_latest_data

```python
async def get_latest_data(self, symbol: str, interval: str, limit: int = 100,
                        add_indicators: bool = True,
                        detect_patterns: bool = False) -> pd.DataFrame:
    ...
```

Получение последних данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `limit` (int): Количество свечей. По умолчанию `100`.
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию `True`.
- `detect_patterns` (bool): Обнаружить паттерны. По умолчанию `False`.

**Возвращает:**
- `pd.DataFrame`: DataFrame с последними данными

##### get_data_for_backtesting

```python
async def get_data_for_backtesting(self, symbol: str, interval: str,
                                 start_date: datetime, end_date: datetime,
                                 add_indicators: bool = True,
                                 detect_patterns: bool = False) -> pd.DataFrame:
    ...
```

Получение данных для бэктестинга.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `start_date` (datetime): Начальная дата
- `end_date` (datetime): Конечная дата
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию `True`.
- `detect_patterns` (bool): Обнаружить паттерны. По умолчанию `False`.

**Возвращает:**
- `pd.DataFrame`: DataFrame с данными для бэктестинга

##### get_data_for_training

```python
async def get_data_for_training(self, symbol: str, interval: str,
                              start_date: datetime, end_date: datetime,
                              add_indicators: bool = True,
                              detect_patterns: bool = False,
                              train_test_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Получение данных для обучения ML моделей.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `start_date` (datetime): Начальная дата
- `end_date` (datetime): Конечная дата
- `add_indicators` (bool): Добавить технические индикаторы. По умолчанию `True`.
- `detect_patterns` (bool): Обнаружить паттерны. По умолчанию `False`.
- `train_test_split` (float): Соотношение обучающей и тестовой выборок. По умолчанию `0.8`.

**Возвращает:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Кортеж (обучающая выборка, тестовая выборка)

##### export_data_to_csv

```python
async def export_data_to_csv(self, symbol: str, interval: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> str:
    ...
```

Экспорт данных в CSV.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `start_date` (Optional[datetime]): Начальная дата
- `end_date` (Optional[datetime]): Конечная дата

**Возвращает:**
- `str`: Путь к CSV файлу

##### import_data_from_csv

```python
async def import_data_from_csv(self, csv_path: str, symbol: str, interval: str) -> pd.DataFrame:
    ...
```

Импорт данных из CSV.

**Параметры:**
- `csv_path` (str): Путь к CSV файлу
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей

**Возвращает:**
- `pd.DataFrame`: DataFrame с импортированными данными

##### analyze_market_data

```python
async def analyze_market_data(self, symbol: str, interval: str, days: int = 30) -> Dict[str, Any]:
    ...
```

Анализ рыночных данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `days` (int): Количество дней для анализа. По умолчанию `30`.

**Возвращает:**
- `Dict[str, Any]`: Словарь с результатами анализа

## Пример использования

```python
import asyncio
import logging
from datetime import datetime, timedelta

from data.storage import DataStorage
from data.historical_data import HistoricalDataManager
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
        # Создание менеджера исторических данных
        data_manager = HistoricalDataManager(storage, client)
        
        # Загрузка исторических данных
        symbol = "BTCUSDT"
        interval = "1h"
        days = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await data_manager.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=True, detect_patterns=True
        )
        
        print(f"Загружено {len(df)} свечей для {symbol} {interval}")
        print(df.head())
        
        # Анализ рыночных данных
        analysis = await data_manager.analyze_market_data(symbol, interval, days)
        print("\nАнализ рыночных данных:")
        print(analysis)
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 