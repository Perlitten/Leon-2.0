# Процессор данных

## Обзор

Модуль `data.data_processor` предоставляет класс `DataProcessor`, который отвечает за предобработку, трансформацию и подготовку данных для анализа и машинного обучения. Процессор данных позволяет нормализовать и стандартизировать данные, создавать признаки и целевые переменные, обрабатывать пропущенные значения и выбросы, а также подготавливать данные для машинного обучения.

## Классы

### DataProcessor

```python
class DataProcessor:
    def __init__(self, storage: Optional[DataStorage] = None):
        ...
```

Процессор данных для предобработки и трансформации.

#### Параметры конструктора

- `storage` (Optional[DataStorage]): Хранилище данных. По умолчанию создается новый экземпляр `DataStorage`.

#### Методы

##### normalize_data

```python
def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    ...
```

Нормализация данных (масштабирование в диапазон [0, 1]).

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `columns` (List[str], optional): Список столбцов для нормализации. Если None, нормализуются все числовые столбцы.

**Возвращает:**
- `pd.DataFrame`: DataFrame с нормализованными данными

##### standardize_data

```python
def standardize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    ...
```

Стандартизация данных (z-score нормализация).

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `columns` (List[str], optional): Список столбцов для стандартизации. Если None, стандартизуются все числовые столбцы.

**Возвращает:**
- `pd.DataFrame`: DataFrame со стандартизованными данными

##### create_features

```python
def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Создание признаков для машинного обучения.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными

**Возвращает:**
- `pd.DataFrame`: DataFrame с добавленными признаками

##### create_target

```python
def create_target(self, df: pd.DataFrame, periods: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    ...
```

Создание целевой переменной для машинного обучения.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `periods` (int): Количество периодов для прогнозирования. По умолчанию 1.
- `threshold` (float): Порог изменения цены для классификации. По умолчанию 0.0.

**Возвращает:**
- `pd.DataFrame`: DataFrame с добавленной целевой переменной

##### handle_missing_values

```python
def handle_missing_values(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    ...
```

Обработка пропущенных значений.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `method` (str): Метод обработки пропущенных значений ('ffill', 'bfill', 'interpolate', 'drop'). По умолчанию 'ffill'.

**Возвращает:**
- `pd.DataFrame`: DataFrame с обработанными пропущенными значениями

##### remove_outliers

```python
def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    ...
```

Удаление выбросов.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `columns` (List[str], optional): Список столбцов для обработки. Если None, обрабатываются все числовые столбцы.
- `method` (str): Метод обнаружения выбросов ('iqr', 'zscore'). По умолчанию 'iqr'.
- `threshold` (float): Порог для обнаружения выбросов. По умолчанию 1.5.

**Возвращает:**
- `pd.DataFrame`: DataFrame с удаленными выбросами

##### split_data

```python
def split_data(self, df: pd.DataFrame, train_size: float = 0.8, shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Разделение данных на обучающую и тестовую выборки.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `train_size` (float): Доля обучающей выборки. По умолчанию 0.8.
- `shuffle` (bool): Перемешивать ли данные перед разделением. По умолчанию False.

**Возвращает:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Кортеж (обучающая выборка, тестовая выборка)

##### prepare_data_for_ml

```python
def prepare_data_for_ml(self, df: pd.DataFrame, target_column: str, feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    ...
```

Подготовка данных для машинного обучения.

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `target_column` (str): Имя столбца с целевой переменной
- `feature_columns` (List[str], optional): Список столбцов с признаками. Если None, используются все числовые столбцы кроме целевой.

**Возвращает:**
- `Tuple[pd.DataFrame, pd.Series]`: Кортеж (признаки, целевая переменная)

## Пример использования

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.storage import DataStorage
from data.data_processor import DataProcessor

# Создание хранилища данных
storage = DataStorage()

# Создание процессора данных
processor = DataProcessor(storage)

# Создание тестовых данных
dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
data = {
    'open': np.random.normal(100, 5, len(dates)),
    'high': np.random.normal(105, 5, len(dates)),
    'low': np.random.normal(95, 5, len(dates)),
    'close': np.random.normal(102, 5, len(dates)),
    'volume': np.random.normal(1000, 200, len(dates))
}

df = pd.DataFrame(data, index=dates)

# Обработка пропущенных значений
df_clean = processor.handle_missing_values(df, method='interpolate')

# Создание признаков
df_features = processor.create_features(df_clean)

# Создание целевой переменной
df_target = processor.create_target(df_features, periods=5, threshold=0.01)

# Нормализация данных
df_norm = processor.normalize_data(df_target, columns=['close', 'volume'])

# Разделение данных
train_df, test_df = processor.split_data(df_target, train_size=0.8, shuffle=False)

# Подготовка данных для машинного обучения
X, y = processor.prepare_data_for_ml(df_target, target_column='target_cls_5')

print(f"Подготовлено {X.shape[0]} строк данных с {X.shape[1]} признаками") 