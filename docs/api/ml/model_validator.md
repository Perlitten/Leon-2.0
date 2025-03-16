# Модуль валидации моделей

Модуль `ml.training.model_validator` предоставляет инструменты для валидации и сравнения моделей машинного обучения.

## Класс ModelValidator

Класс `ModelValidator` предназначен для оценки производительности моделей машинного обучения, расчета метрик, визуализации результатов и сравнения нескольких моделей.

### Инициализация

```python
validator = ModelValidator(config=None)
```

**Параметры:**

- `config` (dict, optional): Словарь с конфигурационными параметрами. По умолчанию `None`.
  - `reports_dir` (str): Директория для сохранения отчетов. По умолчанию `"reports/model_validation"`.
  - `save_reports` (bool): Флаг, указывающий, нужно ли сохранять отчеты. По умолчанию `True`.
  - `visualization` (bool): Флаг, указывающий, нужно ли создавать визуализации. По умолчанию `True`.
  - `visualizer_config` (dict): Конфигурация для визуализатора. По умолчанию `{}`.

### Методы

#### validate

```python
report = validator.validate(model, X, y, task_type=None, dataset_name="dataset")
```

Валидирует модель на заданных данных и возвращает отчет с метриками.

**Параметры:**

- `model` (BaseModel): Модель для валидации.
- `X` (numpy.ndarray или pandas.DataFrame): Признаки для валидации.
- `y` (numpy.ndarray или pandas.Series): Целевые значения для валидации.
- `task_type` (str, optional): Тип задачи ("classification" или "regression"). Если не указан, определяется автоматически.
- `dataset_name` (str, optional): Название набора данных. По умолчанию "dataset".

**Возвращает:**

- `report` (dict): Словарь с результатами валидации, включающий:
  - `model_name` (str): Название модели.
  - `dataset_name` (str): Название набора данных.
  - `task_type` (str): Тип задачи.
  - `metrics` (dict): Словарь с метриками.
  - `timestamp` (str): Временная метка.
  - `data_shape` (tuple): Форма входных данных.
  - `plots` (dict, optional): Словарь с путями к графикам (если включена визуализация).

#### compare_models

```python
comparison_report = validator.compare_models(models, X, y, task_type=None, dataset_name="comparison")
```

Сравнивает несколько моделей на одном наборе данных и возвращает отчет с сравнительными метриками.

**Параметры:**

- `models` (list): Список моделей для сравнения.
- `X` (numpy.ndarray или pandas.DataFrame): Признаки для валидации.
- `y` (numpy.ndarray или pandas.Series): Целевые значения для валидации.
- `task_type` (str, optional): Тип задачи ("classification" или "regression"). Если не указан, определяется автоматически.
- `dataset_name` (str, optional): Название набора данных. По умолчанию "comparison".

**Возвращает:**

- `comparison_report` (dict): Словарь с результатами сравнения, включающий:
  - `dataset_name` (str): Название набора данных.
  - `task_type` (str): Тип задачи.
  - `models` (list): Список названий моделей.
  - `metrics_comparison` (dict): Словарь с метриками для каждой модели.
  - `timestamp` (str): Временная метка.
  - `plots` (dict, optional): Словарь с путями к сравнительным графикам (если включена визуализация).

#### update_config

```python
validator.update_config(new_config)
```

Обновляет конфигурацию валидатора.

**Параметры:**

- `new_config` (dict): Словарь с новыми конфигурационными параметрами.

### Метрики

#### Для задач классификации:

- `accuracy`: Точность (доля правильных предсказаний).
- `precision`: Точность (для бинарной классификации).
- `recall`: Полнота (для бинарной классификации).
- `f1`: F1-мера (для бинарной классификации).
- `roc_auc`: Площадь под ROC-кривой (для бинарной классификации).
- `precision_macro`, `precision_weighted`: Макро- и взвешенная точность (для многоклассовой классификации).
- `recall_macro`, `recall_weighted`: Макро- и взвешенная полнота (для многоклассовой классификации).
- `f1_macro`, `f1_weighted`: Макро- и взвешенная F1-мера (для многоклассовой классификации).

#### Для задач регрессии:

- `mse`: Среднеквадратичная ошибка.
- `rmse`: Корень из среднеквадратичной ошибки.
- `mae`: Средняя абсолютная ошибка.
- `r2`: Коэффициент детерминации.
- `mape`: Средняя абсолютная процентная ошибка.

### Визуализации

Для визуализации результатов `ModelValidator` использует класс `ModelVisualizer` из модуля `ml.training.visualization`.

#### Для задач классификации:

- Матрица ошибок (confusion matrix).
- ROC-кривая (для бинарной классификации).
- Кривая точность-полнота (PR-кривая, для бинарной классификации).

#### Для задач регрессии:

- График сравнения предсказанных и фактических значений.
- График остатков.
- Гистограмма остатков.

#### Для сравнения моделей:

- Графики сравнения отдельных метрик.
- График сравнения нескольких метрик.

## Примеры использования

### Валидация одной модели

```python
import numpy as np
from ml.models.ensemble_model import EnsembleModel
from ml.training.model_validator import ModelValidator

# Создаем тестовые данные
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Создаем модель
model = EnsembleModel(name="ensemble_classifier", task_type="classification")
model.train(X, y)

# Создаем валидатор
validator = ModelValidator()

# Валидируем модель
report = validator.validate(model, X, y, dataset_name="test_dataset")

# Выводим метрики
print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
print(f"F1 Score: {report['metrics']['f1']:.4f}")
```

### Сравнение нескольких моделей

```python
import numpy as np
from ml.models.ensemble_model import EnsembleModel
from ml.training.model_validator import ModelValidator

# Создаем тестовые данные
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Создаем модели
model1 = EnsembleModel(name="ensemble_regressor_1", task_type="regression", aggregation_method="mean")
model2 = EnsembleModel(name="ensemble_regressor_2", task_type="regression", aggregation_method="median")

# Обучаем модели
model1.train(X, y)
model2.train(X, y)

# Создаем валидатор
validator = ModelValidator()

# Сравниваем модели
comparison_report = validator.compare_models([model1, model2], X, y, dataset_name="regression_comparison")

# Выводим сравнительные метрики
for model_name, metrics in comparison_report["metrics_comparison"].items():
    print(f"Model: {model_name}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2: {metrics['r2']:.4f}")
```

### Настройка визуализации

```python
# Создаем валидатор с настройками визуализации
config = {
    "save_reports": True,
    "visualization": True,
    "visualizer_config": {
        "plots_dir": "custom_plots",
        "style": "ggplot",
        "dpi": 300,
        "figsize": (12, 10)
    }
}
validator = ModelValidator(config)

# Валидируем модель
report = validator.validate(model, X, y)
```

### Отключение визуализации и сохранения отчетов

```python
# Создаем валидатор с отключенной визуализацией и сохранением отчетов
config = {
    "save_reports": False,
    "visualization": False
}
validator = ModelValidator(config)

# Валидируем модель
report = validator.validate(model, X, y)
```

### Обновление конфигурации

```python
# Создаем валидатор с базовой конфигурацией
validator = ModelValidator()

# Обновляем конфигурацию
new_config = {
    "reports_dir": "custom_reports",
    "save_reports": True,
    "visualization": False,
    "visualizer_config": {
        "dpi": 150,
        "figsize": (8, 6)
    }
}
validator.update_config(new_config)
``` 