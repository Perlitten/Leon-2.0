# ModelFactory

`ModelFactory` - класс для создания моделей машинного обучения различных типов.

## Назначение

`ModelFactory` предоставляет унифицированный интерфейс для создания моделей разных типов с различными параметрами. Фабрика упрощает процесс создания моделей и обеспечивает единообразный подход к их инициализации.

## API

### Инициализация

```python
from ml.model_factory import ModelFactory

# Инициализация фабрики моделей
factory = ModelFactory()
```

### Основные методы

#### create_model(model_type, name, **kwargs)

Создает модель указанного типа с заданными параметрами.

```python
# Создание модели регрессии
model = factory.create_model(
    model_type="regression",
    name="price_predictor",
    algorithm="random_forest",
    max_depth=5
)
```

#### create_regression_model(name, algorithm="linear", **kwargs)

Создает модель регрессии.

```python
# Создание модели регрессии
regression_model = factory.create_regression_model(
    name="price_predictor",
    algorithm="random_forest",
    max_depth=5,
    n_estimators=100
)
```

#### create_classification_model(name, algorithm="logistic", **kwargs)

Создает модель классификации.

```python
# Создание модели классификации
classification_model = factory.create_classification_model(
    name="signal_classifier",
    algorithm="gradient_boosting",
    learning_rate=0.1,
    n_estimators=200
)
```

#### create_time_series_model(name, algorithm="arima", **kwargs)

Создает модель временных рядов.

```python
# Создание модели ARIMA
arima_model = factory.create_time_series_model(
    name="btc_price_predictor",
    algorithm="arima",
    p=2, d=1, q=2
)

# Создание модели SARIMA
sarima_model = factory.create_time_series_model(
    name="seasonal_predictor",
    algorithm="sarima",
    p=1, d=1, q=1,
    P=1, D=1, Q=1, s=7
)
```

#### create_ensemble_model(name, models, aggregation_method="mean", weights=None, **kwargs)

Создает ансамблевую модель.

```python
from ml.models import RegressionModel

# Создание базовых моделей
model1 = RegressionModel(name="model1", algorithm="random_forest")
model2 = RegressionModel(name="model2", algorithm="gradient_boosting")
model3 = RegressionModel(name="model3", algorithm="linear")

# Создание ансамбля
ensemble = factory.create_ensemble_model(
    name="price_ensemble",
    models=[model1, model2, model3],
    aggregation_method="mean",
    weights=[0.5, 0.3, 0.2]
)
```

#### create_model_from_config(config)

Создает модель на основе конфигурации.

```python
# Конфигурация модели регрессии
regression_config = {
    "type": "regression",
    "name": "price_predictor",
    "algorithm": "random_forest",
    "max_depth": 5,
    "n_estimators": 100
}

# Создание модели из конфигурации
regression_model = factory.create_model_from_config(regression_config)

# Конфигурация ансамблевой модели
ensemble_config = {
    "type": "ensemble",
    "name": "price_ensemble",
    "aggregation_method": "mean",
    "weights": [0.5, 0.3, 0.2],
    "models_config": [
        {
            "type": "regression",
            "name": "model1",
            "algorithm": "random_forest"
        },
        {
            "type": "regression",
            "name": "model2",
            "algorithm": "gradient_boosting"
        },
        {
            "type": "regression",
            "name": "model3",
            "algorithm": "linear"
        }
    ]
}

# Создание ансамблевой модели из конфигурации
ensemble_model = factory.create_model_from_config(ensemble_config)
```

## Поддерживаемые типы моделей

- `regression` - Модели регрессии
- `classification` - Модели классификации
- `ensemble` - Ансамблевые модели
- `time_series` - Модели временных рядов

## Пример использования

### Создание и обучение различных типов моделей

```python
from ml.model_factory import ModelFactory
import numpy as np
import pandas as pd

# Инициализация фабрики моделей
factory = ModelFactory()

# Создание тестовых данных
X = np.random.rand(100, 5)
y_reg = np.random.rand(100)
y_cls = np.random.randint(0, 2, 100)

# Создание и обучение модели регрессии
regression_model = factory.create_regression_model(
    name="price_predictor",
    algorithm="random_forest"
)
regression_model.train(X, y_reg)
reg_predictions = regression_model.predict(X[:10])

# Создание и обучение модели классификации
classification_model = factory.create_classification_model(
    name="signal_classifier",
    algorithm="gradient_boosting"
)
classification_model.train(X, y_cls)
cls_predictions = classification_model.predict(X[:10])

# Создание и обучение модели временных рядов
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
time_series = pd.Series(values, index=dates)

time_series_model = factory.create_time_series_model(
    name="btc_price_predictor",
    algorithm="arima",
    p=2, d=1, q=2
)
time_series_model.train(time_series)
ts_forecast = time_series_model.predict(time_series, steps=10)

# Создание ансамблевой модели
ensemble = factory.create_ensemble_model(
    name="regression_ensemble",
    models=[
        factory.create_regression_model("model1", algorithm="random_forest"),
        factory.create_regression_model("model2", algorithm="gradient_boosting"),
        factory.create_regression_model("model3", algorithm="linear")
    ],
    aggregation_method="mean"
)
ensemble.train(X, y_reg)
ensemble_predictions = ensemble.predict(X[:10])
```

### Создание моделей из конфигурации

```python
from ml.model_factory import ModelFactory

# Инициализация фабрики моделей
factory = ModelFactory()

# Конфигурация модели
config = {
    "type": "time_series",
    "name": "btc_price_predictor",
    "algorithm": "sarima",
    "p": 1, "d": 1, "q": 1,
    "P": 1, "D": 1, "Q": 1, "s": 7
}

# Создание модели из конфигурации
model = factory.create_model_from_config(config)

# Использование модели
# ...
```

## Интеграция с другими компонентами

`ModelFactory` интегрируется с другими компонентами системы Leon:

- `ModelManager` использует `ModelFactory` для создания моделей при загрузке из конфигурации.
- `ModelTrainer` использует `ModelFactory` для создания моделей перед обучением.
- `MLIntegrationManager` использует `ModelFactory` для создания моделей при интеграции с торговыми стратегиями. 