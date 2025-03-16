# Модели машинного обучения

В этом разделе описаны модели машинного обучения, доступные в Leon Trading Bot.

## BaseModel

`BaseModel` - это абстрактный базовый класс для всех моделей машинного обучения в системе.
Он определяет общий интерфейс, который должны реализовать все модели.

```python
from ml.models import BaseModel

# Создание экземпляра не допускается, так как это абстрактный класс
# model = BaseModel(name="my_model")  # Вызовет ошибку
```

### Методы

- `__init__(name: str, version: str = "1.0.0", **kwargs)` - Инициализация модели с именем и версией
- `train(X, y, **kwargs)` - Обучение модели на данных
- `predict(X, **kwargs)` - Получение предсказаний модели
- `evaluate(X, y, **kwargs)` - Оценка производительности модели
- `get_feature_importance()` - Получение важности признаков
- `save(path: str)` - Сохранение модели в файл
- `load(path: str)` - Загрузка модели из файла

## RegressionModel

`RegressionModel` - модель для задач регрессии, наследуется от `BaseModel`.

```python
from ml.models import RegressionModel
import numpy as np

# Создание модели
model = RegressionModel(name="price_predictor", algorithm="random_forest")

# Обучение модели
X_train = np.random.rand(100, 5)  # 100 примеров, 5 признаков
y_train = np.random.rand(100)     # 100 целевых значений
metrics = model.train(X_train, y_train)

# Предсказание
X_test = np.random.rand(10, 5)
predictions = model.predict(X_test)

# Оценка модели
X_val = np.random.rand(20, 5)
y_val = np.random.rand(20)
eval_metrics = model.evaluate(X_val, y_val)
```

### Поддерживаемые алгоритмы

- `linear` - Линейная регрессия
- `ridge` - Гребневая регрессия
- `lasso` - LASSO регрессия
- `elastic_net` - ElasticNet регрессия
- `random_forest` - Случайный лес
- `gradient_boosting` - Градиентный бустинг
- `svr` - Метод опорных векторов для регрессии
- `xgboost` - XGBoost (если установлен)
- `lightgbm` - LightGBM (если установлен)

### Методы

- `__init__(name: str, algorithm: str = "linear", version: str = "1.0.0", **kwargs)` - Инициализация модели регрессии
- `train(X, y, **kwargs)` - Обучение модели на данных
- `predict(X, **kwargs)` - Получение предсказаний модели
- `predict_proba(X, **kwargs)` - Не поддерживается для регрессии
- `evaluate(X, y, **kwargs)` - Оценка производительности модели
- `get_feature_importance()` - Получение важности признаков
- `save(path: str)` - Сохранение модели в файл
- `load(path: str)` - Загрузка модели из файла

## ClassificationModel

`ClassificationModel` - модель для задач классификации, наследуется от `BaseModel`.

```python
from ml.models import ClassificationModel
import numpy as np

# Создание модели
model = ClassificationModel(name="trade_signal_classifier", algorithm="random_forest")

# Обучение модели
X_train = np.random.rand(100, 5)  # 100 примеров, 5 признаков
y_train = np.random.randint(0, 2, 100)  # Бинарные метки классов
metrics = model.train(X_train, y_train)

# Предсказание классов
X_test = np.random.rand(10, 5)
predictions = model.predict(X_test)

# Предсказание вероятностей классов
probabilities = model.predict_proba(X_test)

# Оценка модели
X_val = np.random.rand(20, 5)
y_val = np.random.randint(0, 2, 20)
eval_metrics = model.evaluate(X_val, y_val)
```

### Поддерживаемые алгоритмы

- `logistic` - Логистическая регрессия
- `random_forest` - Случайный лес
- `gradient_boosting` - Градиентный бустинг
- `svc` - Метод опорных векторов для классификации
- `naive_bayes` - Наивный Байес
- `knn` - k-ближайших соседей
- `xgboost` - XGBoost (если установлен)
- `lightgbm` - LightGBM (если установлен)

### Методы

- `__init__(name: str, algorithm: str = "logistic", version: str = "1.0.0", **kwargs)` - Инициализация модели классификации
- `train(X, y, **kwargs)` - Обучение модели на данных
- `predict(X, **kwargs)` - Получение предсказаний классов
- `predict_proba(X, **kwargs)` - Получение вероятностей классов
- `evaluate(X, y, **kwargs)` - Оценка производительности модели
- `get_feature_importance()` - Получение важности признаков
- `save(path: str)` - Сохранение модели в файл
- `load(path: str)` - Загрузка модели из файла

## EnsembleModel

`EnsembleModel` - модель для ансамблевого обучения, объединяющая несколько моделей для улучшения качества предсказаний.

```python
from ml.models import RegressionModel, EnsembleModel
import numpy as np

# Создание базовых моделей
model1 = RegressionModel(name="model1", algorithm="random_forest")
model2 = RegressionModel(name="model2", algorithm="gradient_boosting")
model3 = RegressionModel(name="model3", algorithm="linear")

# Создание ансамбля
ensemble = EnsembleModel(
    name="price_ensemble",
    models=[model1, model2, model3],
    aggregation_method="mean",
    weights=[0.5, 0.3, 0.2]  # Веса для каждой модели
)

# Обучение ансамбля
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
metrics = ensemble.train(X_train, y_train)

# Предсказание
X_test = np.random.rand(10, 5)
predictions = ensemble.predict(X_test)

# Оценка ансамбля
X_val = np.random.rand(20, 5)
y_val = np.random.rand(20)
eval_metrics = ensemble.evaluate(X_val, y_val)
```

### Поддерживаемые методы агрегации

- `voting` - Голосование (для классификации)
- `mean` - Среднее значение предсказаний
- `median` - Медиана предсказаний
- `max` - Максимальное значение предсказаний
- `min` - Минимальное значение предсказаний
- Пользовательская функция агрегации

### Методы

- `__init__(name: str, models: list, aggregation_method: str = "mean", weights: list = None, custom_aggregation_func = None, version: str = "1.0.0", **kwargs)` - Инициализация ансамблевой модели
- `train(X, y, **kwargs)` - Обучение всех моделей в ансамбле
- `predict(X, **kwargs)` - Получение агрегированных предсказаний
- `predict_proba(X, **kwargs)` - Получение агрегированных вероятностей классов (для классификации)
- `evaluate(X, y, **kwargs)` - Оценка производительности ансамбля
- `get_feature_importance()` - Получение усредненной важности признаков
- `save(path: str)` - Сохранение ансамбля и всех моделей
- `load(path: str)` - Загрузка ансамбля и всех моделей

### Особенности реализации

`EnsembleModel` предоставляет следующие возможности:

1. **Гибкая агрегация результатов**:
   - Для классификации: голосование, среднее вероятностей
   - Для регрессии: среднее, медиана, минимум, максимум
   - Поддержка пользовательских функций агрегации

2. **Взвешенная агрегация**:
   - Возможность задать веса для каждой модели
   - Автоматическая нормализация весов

3. **Обучение и оценка**:
   - Параллельное обучение всех моделей
   - Сбор метрик от каждой модели
   - Оценка производительности ансамбля в целом

4. **Сохранение и загрузка**:
   - Сохранение всех моделей ансамбля
   - Сохранение метаданных о структуре ансамбля
   - Восстановление ансамбля из сохраненных моделей

### Пример использования с пользовательской функцией агрегации

```python
from ml.models import ClassificationModel, EnsembleModel
import numpy as np

# Создание базовых моделей
model1 = ClassificationModel(name="model1", algorithm="random_forest")
model2 = ClassificationModel(name="model2", algorithm="logistic")

# Пользовательская функция агрегации - взвешенное голосование с учетом уверенности
def custom_aggregation(predictions, weights=None):
    # predictions имеет форму [n_models, n_samples, n_classes] для вероятностей
    # Выбираем класс с максимальной взвешенной вероятностью
    if weights is None:
        weights = np.ones(predictions.shape[0])
    
    # Применяем веса к каждой модели
    weighted_preds = np.array([w * p for w, p in zip(weights, predictions)])
    
    # Суммируем по моделям и находим класс с максимальной вероятностью
    summed = np.sum(weighted_preds, axis=0)
    return np.argmax(summed, axis=1)

# Создание ансамбля с пользовательской функцией
ensemble = EnsembleModel(
    name="custom_ensemble",
    models=[model1, model2],
    custom_aggregation_func=custom_aggregation,
    weights=[0.7, 0.3]
)

# Использование ансамбля
X = np.random.rand(10, 5)
ensemble.train(X, np.random.randint(0, 2, 10))
predictions = ensemble.predict(X)
```

## TimeSeriesModel

`TimeSeriesModel` - модель для прогнозирования временных рядов, наследуется от `BaseModel`.

```python
from ml.models import TimeSeriesModel
import pandas as pd
import numpy as np

# Создание модели ARIMA
arima_model = TimeSeriesModel(
    name="btc_price_predictor",
    algorithm="arima",
    p=2,  # порядок авторегрессии
    d=1,  # порядок интегрирования
    q=2   # порядок скользящего среднего
)

# Создание тестовых данных временного ряда
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
time_series = pd.Series(values, index=dates)

# Обучение модели
metrics = arima_model.train(time_series)
print(f"Метрики обучения: {metrics}")

# Прогнозирование на 10 шагов вперед
forecast = arima_model.predict(time_series, steps=10)
print(f"Прогноз: {forecast}")

# Создание модели SARIMA с экзогенными переменными
sarima_model = TimeSeriesModel(
    name="seasonal_price_predictor",
    algorithm="sarima",
    p=1, d=1, q=1,  # несезонные параметры
    P=1, D=1, Q=1,  # сезонные параметры
    s=7             # период сезонности (недельный)
)

# Создание DataFrame с экзогенными переменными
exog_data = pd.DataFrame({
    'target': values,
    'volume': np.random.normal(1000, 100, 100),
    'sentiment': np.random.normal(0, 1, 100)
}, index=dates)

# Обучение модели SARIMA
sarima_metrics = sarima_model.train(exog_data)

# Создание данных для прогноза с экзогенными переменными
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=5, freq='D')
future_exog = pd.DataFrame({
    'target': [0] * 5,  # Заполнитель, не используется для прогноза
    'volume': np.random.normal(1000, 100, 5),
    'sentiment': np.random.normal(0, 1, 5)
}, index=future_dates)

# Прогнозирование с экзогенными переменными
sarima_forecast = sarima_model.predict(future_exog, steps=5)
```

### Поддерживаемые алгоритмы

- `arima` - Авторегрессионная интегрированная модель скользящего среднего
- `sarima` - Сезонная ARIMA
- `exp_smoothing` - Экспоненциальное сглаживание

### Методы

- `__init__(name: str, algorithm: str = "arima", version: str = "1.0.0", **kwargs)` - Инициализация модели временных рядов
- `train(X, y=None, **kwargs)` - Обучение модели на данных временного ряда
- `predict(X, steps=1, **kwargs)` - Получение прогноза на указанное количество шагов
- `evaluate(X, y=None, steps=1)` - Оценка производительности модели
- `get_feature_importance()` - Не имеет смысла для большинства моделей временных рядов
- `get_model_summary()` - Получение сводки о модели
- `get_residuals()` - Получение остатков модели
- `get_aic()` - Получение информационного критерия Акаике
- `get_bic()` - Получение байесовского информационного критерия

### Параметры алгоритмов

#### ARIMA
- `p` - Порядок авторегрессии
- `d` - Порядок интегрирования
- `q` - Порядок скользящего среднего

#### SARIMA
- `p`, `d`, `q` - Несезонные параметры
- `P`, `D`, `Q` - Сезонные параметры
- `s` - Период сезонности

#### Экспоненциальное сглаживание
- `trend` - Тип тренда (`None`, `'add'`, `'mul'`)
- `seasonal` - Тип сезонности (`None`, `'add'`, `'mul'`)
- `seasonal_periods` - Период сезонности

### Особенности реализации

`TimeSeriesModel` предоставляет следующие возможности:

1. **Поддержка различных алгоритмов прогнозирования**:
   - ARIMA для несезонных временных рядов
   - SARIMA для сезонных временных рядов
   - Экспоненциальное сглаживание для рядов с трендом и сезонностью

2. **Работа с экзогенными переменными**:
   - Поддержка внешних регрессоров для SARIMA
   - Автоматическое извлечение целевой переменной и экзогенных переменных из DataFrame

3. **Расширенные метрики оценки**:
   - MSE (среднеквадратичная ошибка)
   - RMSE (корень из среднеквадратичной ошибки)
   - MAE (средняя абсолютная ошибка)
   - MAPE (средняя абсолютная процентная ошибка)
   - R² (коэффициент детерминации)

4. **Дополнительные методы анализа**:
   - Получение сводки о модели
   - Анализ остатков
   - Информационные критерии (AIC, BIC)

### Пример визуализации прогноза

```python
from ml.models import TimeSeriesModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Создание данных
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
time_series = pd.Series(values, index=dates)

# Разделение на обучающую и тестовую выборки
train_size = 80
train_data = time_series[:train_size]
test_data = time_series[train_size:]

# Создание и обучение модели
model = TimeSeriesModel(
    name="arima_example",
    algorithm="arima",
    p=2, d=1, q=2
)
model.train(train_data)

# Прогнозирование
forecast = model.predict(train_data, steps=len(test_data))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values, label='Обучающие данные')
plt.plot(test_data.index, test_data.values, label='Тестовые данные')
plt.plot(test_data.index, forecast, label='Прогноз', color='red')
plt.title('Прогноз временного ряда с использованием ARIMA')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()

# Оценка модели
metrics = model.evaluate(test_data, test_data)
print(f"Метрики оценки: {metrics}")
``` 