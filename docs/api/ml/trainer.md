# Модуль обучения моделей

Модуль `ml.training.trainer` предоставляет инструменты для обучения моделей машинного обучения на исторических данных.

## Класс ModelTrainer

Класс `ModelTrainer` отвечает за подготовку данных, обучение моделей, их валидацию и сохранение.

### Инициализация

```python
trainer = ModelTrainer(config=None)
```

**Параметры:**

- `config` (dict, optional): Словарь с конфигурационными параметрами. По умолчанию `None`.
  - `models_dir` (str): Директория для сохранения моделей. По умолчанию `"ml/models"`.
  - `task_type` (str): Тип задачи ("classification" или "regression"). По умолчанию `"classification"`.
  - `test_size` (float): Доля данных для валидационной выборки. По умолчанию `0.2`.
  - `random_state` (int): Начальное значение для генератора случайных чисел. По умолчанию `42`.
  - `batch_size` (int): Размер батча для обучения. По умолчанию `32`.
  - `epochs` (int): Количество эпох обучения. По умолчанию `50`.
  - `early_stopping` (bool): Использовать ли раннюю остановку. По умолчанию `True`.
  - `patience` (int): Количество эпох без улучшения для ранней остановки. По умолчанию `10`.

### Методы

#### train

```python
results = trainer.train(model_type, X, y, model_params=None, metadata=None)
```

Обучает модель на данных.

**Параметры:**

- `model_type` (str): Тип модели ("lstm", "cnn", "mlp", "xgboost", "random_forest").
- `X` (numpy.ndarray): Признаки для обучения.
- `y` (numpy.ndarray): Целевые значения.
- `model_params` (dict, optional): Параметры модели. По умолчанию `None`.
- `metadata` (dict, optional): Метаданные модели. По умолчанию `None`.

**Возвращает:**

- `results` (dict): Словарь с результатами обучения, включающий:
  - `model_id` (str): Идентификатор модели.
  - `model_path` (str): Путь к сохраненной модели.
  - `metrics` (dict): Метрики модели.
  - `metadata` (dict): Метаданные модели.

#### update_config

```python
trainer.update_config(config)
```

Обновляет конфигурацию тренера.

**Параметры:**

- `config` (dict): Словарь с новыми конфигурационными параметрами.

### Поддерживаемые типы моделей

#### LSTM (Long Short-Term Memory)

Рекуррентная нейронная сеть с долгой краткосрочной памятью, подходит для временных рядов и последовательностей.

**Параметры:**

- `units` (list): Список с количеством нейронов в каждом слое LSTM. По умолчанию `[64, 32]`.
- `dropout` (float): Коэффициент отсева. По умолчанию `0.2`.
- `recurrent_dropout` (float): Коэффициент отсева для рекуррентных связей. По умолчанию `0.2`.
- `activation` (str): Функция активации. По умолчанию `"tanh"`.
- `recurrent_activation` (str): Функция активации для рекуррентных связей. По умолчанию `"sigmoid"`.
- `learning_rate` (float): Скорость обучения. По умолчанию `0.001`.

#### CNN (Convolutional Neural Network)

Сверточная нейронная сеть, подходит для изображений и пространственных данных.

**Параметры:**

- `filters` (list): Список с количеством фильтров в каждом сверточном слое. По умолчанию `[32, 64, 128]`.
- `kernel_size` (int или tuple): Размер ядра свертки. По умолчанию `3`.
- `pool_size` (int или tuple): Размер ядра пулинга. По умолчанию `2`.
- `dense_units` (list): Список с количеством нейронов в каждом полносвязном слое. По умолчанию `[128, 64]`.
- `dropout` (float): Коэффициент отсева. По умолчанию `0.2`.
- `activation` (str): Функция активации. По умолчанию `"relu"`.
- `learning_rate` (float): Скорость обучения. По умолчанию `0.001`.

#### MLP (Multi-Layer Perceptron)

Многослойный перцептрон, подходит для табличных данных.

**Параметры:**

- `units` (list): Список с количеством нейронов в каждом слое. По умолчанию `[128, 64, 32]`.
- `dropout` (float): Коэффициент отсева. По умолчанию `0.2`.
- `activation` (str): Функция активации. По умолчанию `"relu"`.
- `learning_rate` (float): Скорость обучения. По умолчанию `0.001`.

#### XGBoost

Градиентный бустинг на деревьях решений, подходит для табличных данных.

**Параметры:**

- `n_estimators` (int): Количество деревьев. По умолчанию `100`.
- `max_depth` (int): Максимальная глубина деревьев. По умолчанию `6`.
- `learning_rate` (float): Скорость обучения. По умолчанию `0.1`.
- `subsample` (float): Доля выборки для обучения каждого дерева. По умолчанию `0.8`.
- `colsample_bytree` (float): Доля признаков для обучения каждого дерева. По умолчанию `0.8`.

#### Random Forest

Случайный лес, подходит для табличных данных.

**Параметры:**

- `n_estimators` (int): Количество деревьев. По умолчанию `100`.
- `max_depth` (int): Максимальная глубина деревьев. По умолчанию `None`.
- `min_samples_split` (int): Минимальное количество образцов для разделения узла. По умолчанию `2`.
- `min_samples_leaf` (int): Минимальное количество образцов в листовом узле. По умолчанию `1`.
- `max_features` (str или int): Количество признаков для поиска наилучшего разделения. По умолчанию `"auto"`.

## Примеры использования

### Обучение модели LSTM

```python
import numpy as np
from ml.training.trainer import ModelTrainer

# Создаем тестовые данные для временного ряда
X = np.random.rand(100, 10, 5)  # 100 образцов, 10 временных шагов, 5 признаков
y = np.random.randint(0, 2, 100)  # Бинарная классификация

# Создаем тренер
trainer = ModelTrainer()

# Параметры модели
model_params = {
    "units": [128, 64],
    "dropout": 0.3,
    "learning_rate": 0.0005
}

# Обучаем модель
results = trainer.train("lstm", X, y, model_params=model_params)

# Выводим результаты
print(f"Модель сохранена: {results['model_path']}")
print(f"Метрики: {results['metrics']}")
```

### Обучение модели MLP для регрессии

```python
import numpy as np
from ml.training.trainer import ModelTrainer

# Создаем тестовые данные
X = np.random.rand(500, 20)  # 500 образцов, 20 признаков
y = np.random.rand(500)  # Непрерывные целевые значения

# Создаем тренер с конфигурацией для регрессии
config = {
    "task_type": "regression",
    "test_size": 0.3,
    "epochs": 100,
    "early_stopping": True,
    "patience": 15
}
trainer = ModelTrainer(config)

# Параметры модели
model_params = {
    "units": [256, 128, 64],
    "dropout": 0.3,
    "activation": "relu",
    "learning_rate": 0.001
}

# Обучаем модель
results = trainer.train("mlp", X, y, model_params=model_params)

# Выводим результаты
print(f"Модель сохранена: {results['model_path']}")
print(f"Метрики: {results['metrics']}")
```

### Обучение модели XGBoost

```python
import numpy as np
from ml.training.trainer import ModelTrainer

# Создаем тестовые данные
X = np.random.rand(1000, 50)  # 1000 образцов, 50 признаков
y = np.random.randint(0, 3, 1000)  # Многоклассовая классификация

# Создаем тренер
trainer = ModelTrainer()

# Параметры модели
model_params = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9
}

# Обучаем модель
results = trainer.train("xgboost", X, y, model_params=model_params)

# Выводим результаты
print(f"Модель сохранена: {results['model_path']}")
print(f"Метрики: {results['metrics']}")
```

### Настройка конфигурации

```python
from ml.training.trainer import ModelTrainer

# Создаем тренер с базовой конфигурацией
trainer = ModelTrainer()

# Обновляем конфигурацию
new_config = {
    "models_dir": "custom_models",
    "batch_size": 64,
    "epochs": 200,
    "early_stopping": True,
    "patience": 20
}
trainer.update_config(new_config)
``` 