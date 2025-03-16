# Хранилище данных

## Обзор

Модуль `data.storage` предоставляет класс `DataStorage`, который отвечает за хранение и управление данными в системе Leon Trading Bot. Хранилище обеспечивает сохранение и загрузку исторических данных, кэширование, а также управление метаданными.

## Классы

### DataStorage

```python
class DataStorage:
    def __init__(self, base_path: str = None):
        ...
```

Хранилище данных для торгового бота.

#### Параметры конструктора

- `base_path` (str, optional): Базовый путь для хранения данных. По умолчанию используется директория `data` в корне проекта.

#### Методы

##### save_historical_data

```python
def save_historical_data(self, symbol: str, interval: str, data: pd.DataFrame) -> bool:
    ...
```

Сохранение исторических данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `data` (pd.DataFrame): Данные для сохранения

**Возвращает:**
- `bool`: True, если данные успешно сохранены, иначе False

##### load_historical_data

```python
def load_historical_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    ...
```

Загрузка исторических данных.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей

**Возвращает:**
- `Optional[pd.DataFrame]`: DataFrame с историческими данными или None, если данные не найдены

##### save_kline_data

```python
def save_kline_data(self, symbol: str, interval: str, kline: Dict[str, Any]) -> bool:
    ...
```

Сохранение данных свечи.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `kline` (Dict[str, Any]): Данные свечи

**Возвращает:**
- `bool`: True, если данные успешно сохранены, иначе False

##### save_kline

```python
async def save_kline(self, symbol: str, interval: str, kline: List[Any]) -> bool:
    ...
```

Сохранение одной свечи в хранилище.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечи
- `kline` (List[Any]): Данные свечи в формате Binance API

**Возвращает:**
- `bool`: True, если свеча успешно сохранена, иначе False

##### get_latest_kline

```python
def get_latest_kline(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    ...
```

Получение последней свечи.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей

**Возвращает:**
- `Optional[Dict[str, Any]]`: Данные последней свечи или None, если данные не найдены

##### save_metadata

```python
def save_metadata(self, key: str, data: Dict[str, Any]) -> bool:
    ...
```

Сохранение метаданных.

**Параметры:**
- `key` (str): Ключ метаданных
- `data` (Dict[str, Any]): Данные для сохранения

**Возвращает:**
- `bool`: True, если данные успешно сохранены, иначе False

##### load_metadata

```python
def load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
    ...
```

Загрузка метаданных.

**Параметры:**
- `key` (str): Ключ метаданных

**Возвращает:**
- `Optional[Dict[str, Any]]`: Метаданные или None, если данные не найдены

##### save_to_cache

```python
def save_to_cache(self, key: str, data: Any, ttl: int = 3600) -> bool:
    ...
```

Сохранение данных в кэш.

**Параметры:**
- `key` (str): Ключ кэша
- `data` (Any): Данные для сохранения
- `ttl` (int): Время жизни кэша в секундах. По умолчанию 3600 (1 час).

**Возвращает:**
- `bool`: True, если данные успешно сохранены, иначе False

##### load_from_cache

```python
def load_from_cache(self, key: str) -> Optional[Any]:
    ...
```

Загрузка данных из кэша.

**Параметры:**
- `key` (str): Ключ кэша

**Возвращает:**
- `Optional[Any]`: Данные из кэша или None, если данные не найдены или истек срок действия

##### clear_cache

```python
def clear_cache(self, key: str = None) -> bool:
    ...
```

Очистка кэша.

**Параметры:**
- `key` (str, optional): Ключ кэша для очистки. Если None, очищается весь кэш.

**Возвращает:**
- `bool`: True, если кэш успешно очищен, иначе False

##### export_to_csv

```python
def export_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
    ...
```

Экспорт данных в CSV-файл.

**Параметры:**
- `data` (pd.DataFrame): Данные для экспорта
- `filename` (str): Имя файла

**Возвращает:**
- `bool`: True, если данные успешно экспортированы, иначе False

##### import_from_csv

```python
def import_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
    ...
```

Импорт данных из CSV-файла.

**Параметры:**
- `filename` (str): Имя файла

**Возвращает:**
- `Optional[pd.DataFrame]`: DataFrame с данными или None, если файл не найден

##### get_available_symbols

```python
def get_available_symbols(self) -> List[str]:
    ...
```

Получение списка доступных символов.

**Возвращает:**
- `List[str]`: Список доступных символов

##### get_available_intervals

```python
def get_available_intervals(self, symbol: str) -> List[str]:
    ...
```

Получение списка доступных интервалов для символа.

**Параметры:**
- `symbol` (str): Символ торговой пары

**Возвращает:**
- `List[str]`: Список доступных интервалов

##### get_data_info

```python
def get_data_info(self) -> Dict[str, Any]:
    ...
```

Получение информации о данных.

**Возвращает:**
- `Dict[str, Any]`: Информация о данных

##### get_average_volume

```python
async def get_average_volume(self, symbol: str, interval: str, periods: int = 24) -> float:
    ...
```

Получение среднего объема за указанное количество периодов.

**Параметры:**
- `symbol` (str): Символ торговой пары
- `interval` (str): Интервал свечей
- `periods` (int): Количество периодов для расчета среднего. По умолчанию 24.

**Возвращает:**
- `float`: Средний объем

## Пример использования

```python
import pandas as pd
from data.storage import DataStorage

# Создание хранилища данных
storage = DataStorage()

# Создание тестовых данных
data = pd.DataFrame({
    'open': [10000, 10100, 10200],
    'high': [10050, 10150, 10250],
    'low': [9950, 10050, 10150],
    'close': [10100, 10200, 10300],
    'volume': [100, 150, 200],
    'timestamp': pd.date_range(start='2023-01-01', periods=3)
})

# Сохранение исторических данных
storage.save_historical_data('BTCUSDT', '1h', data)

# Загрузка исторических данных
loaded_data = storage.load_historical_data('BTCUSDT', '1h')
print(loaded_data)

# Сохранение метаданных
metadata = {
    'last_update': '2023-01-01 12:00:00',
    'source': 'binance',
    'status': 'complete'
}
storage.save_metadata('BTCUSDT_1h_info', metadata)

# Загрузка метаданных
loaded_metadata = storage.load_metadata('BTCUSDT_1h_info')
print(loaded_metadata)

# Использование кэша
storage.save_to_cache('recent_price', 10300, ttl=60)  # Кэш на 1 минуту
cached_price = storage.load_from_cache('recent_price')
print(f"Кэшированная цена: {cached_price}")

# Экспорт в CSV
storage.export_to_csv(data, 'btc_data.csv')

# Импорт из CSV
imported_data = storage.import_from_csv('btc_data.csv')
print(imported_data)

# Получение информации о данных
available_symbols = storage.get_available_symbols()
print(f"Доступные символы: {available_symbols}")

available_intervals = storage.get_available_intervals('BTCUSDT')
print(f"Доступные интервалы для BTCUSDT: {available_intervals}")

data_info = storage.get_data_info()
print("Информация о данных:")
print(data_info)

# Получение среднего объема
async def get_avg_volume():
    avg_volume = await storage.get_average_volume('BTCUSDT', '1h', 24)
    print(f"Средний объем за 24 часа: {avg_volume}")

import asyncio
asyncio.run(get_avg_volume()) 