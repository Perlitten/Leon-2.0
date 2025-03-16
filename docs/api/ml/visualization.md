# Модуль визуализации моделей

Модуль `ml.training.visualization` предоставляет инструменты для визуализации результатов работы моделей машинного обучения.

## Класс ModelVisualizer

Класс `ModelVisualizer` предназначен для создания различных графиков и визуализаций, которые помогают анализировать производительность моделей.

### Инициализация

```python
visualizer = ModelVisualizer(config=None)
```

**Параметры:**

- `config` (dict, optional): Словарь с конфигурационными параметрами. По умолчанию `None`.
  - `plots_dir` (str): Директория для сохранения графиков. По умолчанию `"reports/plots"`.
  - `style` (str): Стиль графиков matplotlib. По умолчанию `"seaborn-v0_8-whitegrid"`.
  - `dpi` (int): Разрешение сохраняемых изображений. По умолчанию `100`.
  - `figsize` (tuple): Размер графиков по умолчанию. По умолчанию `(10, 8)`.

### Методы

#### update_config

```python
visualizer.update_config(new_config)
```

Обновляет конфигурацию визуализатора.

**Параметры:**

- `new_config` (dict): Словарь с новыми конфигурационными параметрами.

#### create_model_plots_dir

```python
plots_dir = visualizer.create_model_plots_dir(model_name, dataset_name)
```

Создает директорию для графиков конкретной модели и набора данных.

**Параметры:**

- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `plots_dir` (str): Путь к созданной директории.

#### plot_confusion_matrix

```python
cm_path = visualizer.plot_confusion_matrix(y_true, y_pred, model_name, dataset_name, normalize=False)
```

Строит и сохраняет матрицу ошибок.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_pred` (numpy.ndarray): Предсказанные значения.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.
- `normalize` (bool, optional): Нормализовать ли матрицу. По умолчанию `False`.

**Возвращает:**

- `cm_path` (str): Путь к сохраненному графику.

#### plot_roc_curve

```python
roc_path = visualizer.plot_roc_curve(y_true, y_proba, model_name, dataset_name)
```

Строит и сохраняет ROC-кривую.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_proba` (numpy.ndarray): Вероятности классов.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `roc_path` (str): Путь к сохраненному графику.

#### plot_precision_recall_curve

```python
pr_path = visualizer.plot_precision_recall_curve(y_true, y_proba, model_name, dataset_name)
```

Строит и сохраняет кривую точность-полнота.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_proba` (numpy.ndarray): Вероятности классов.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `pr_path` (str): Путь к сохраненному графику.

#### plot_regression_predictions

```python
pred_path = visualizer.plot_regression_predictions(y_true, y_pred, model_name, dataset_name)
```

Строит и сохраняет график сравнения предсказанных и фактических значений.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_pred` (numpy.ndarray): Предсказанные значения.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `pred_path` (str): Путь к сохраненному графику.

#### plot_residuals

```python
resid_path = visualizer.plot_residuals(y_true, y_pred, model_name, dataset_name)
```

Строит и сохраняет график остатков.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_pred` (numpy.ndarray): Предсказанные значения.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `resid_path` (str): Путь к сохраненному графику.

#### plot_residuals_histogram

```python
resid_hist_path = visualizer.plot_residuals_histogram(y_true, y_pred, model_name, dataset_name)
```

Строит и сохраняет гистограмму остатков.

**Параметры:**

- `y_true` (numpy.ndarray): Истинные значения.
- `y_pred` (numpy.ndarray): Предсказанные значения.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `resid_hist_path` (str): Путь к сохраненному графику.

#### plot_feature_importance

```python
importance_path = visualizer.plot_feature_importance(feature_importance, feature_names, model_name, dataset_name, top_n=None)
```

Строит и сохраняет график важности признаков.

**Параметры:**

- `feature_importance` (numpy.ndarray): Важность признаков.
- `feature_names` (list): Названия признаков.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.
- `top_n` (int, optional): Количество самых важных признаков для отображения. По умолчанию `None` (отображаются все признаки).

**Возвращает:**

- `importance_path` (str): Путь к сохраненному графику.

#### plot_learning_curve

```python
learning_curve_path = visualizer.plot_learning_curve(train_sizes, train_scores, test_scores, model_name, dataset_name)
```

Строит и сохраняет кривую обучения.

**Параметры:**

- `train_sizes` (numpy.ndarray): Размеры обучающей выборки.
- `train_scores` (numpy.ndarray): Оценки на обучающей выборке.
- `test_scores` (numpy.ndarray): Оценки на тестовой выборке.
- `model_name` (str): Название модели.
- `dataset_name` (str): Название набора данных.

**Возвращает:**

- `learning_curve_path` (str): Путь к сохраненному графику.

#### plot_metrics_comparison

```python
comparison_path = visualizer.plot_metrics_comparison(metrics_dict, model_names, dataset_name, metric_name)
```

Строит и сохраняет график сравнения метрик для нескольких моделей.

**Параметры:**

- `metrics_dict` (dict): Словарь с метриками для каждой модели.
- `model_names` (list): Список названий моделей.
- `dataset_name` (str): Название набора данных.
- `metric_name` (str): Название метрики для сравнения.

**Возвращает:**

- `comparison_path` (str): Путь к сохраненному графику.

#### plot_multiple_metrics_comparison

```python
multi_comparison_path = visualizer.plot_multiple_metrics_comparison(metrics_dict, model_names, dataset_name, metric_names)
```

Строит и сохраняет график сравнения нескольких метрик для нескольких моделей.

**Параметры:**

- `metrics_dict` (dict): Словарь с метриками для каждой модели.
- `model_names` (list): Список названий моделей.
- `dataset_name` (str): Название набора данных.
- `metric_names` (list): Список названий метрик для сравнения.

**Возвращает:**

- `multi_comparison_path` (str): Путь к сохраненному графику.

## Примеры использования

### Визуализация результатов классификации

```python
import numpy as np
from ml.training.visualization import ModelVisualizer

# Создаем тестовые данные
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1],
                    [0.6, 0.4], [0.7, 0.3], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])

# Создаем визуализатор
visualizer = ModelVisualizer()

# Строим матрицу ошибок
cm_path = visualizer.plot_confusion_matrix(y_true, y_pred, "my_classifier", "test_data")
print(f"Матрица ошибок сохранена: {cm_path}")

# Строим ROC-кривую
roc_path = visualizer.plot_roc_curve(y_true, y_proba, "my_classifier", "test_data")
print(f"ROC-кривая сохранена: {roc_path}")

# Строим кривую точность-полнота
pr_path = visualizer.plot_precision_recall_curve(y_true, y_proba, "my_classifier", "test_data")
print(f"Кривая точность-полнота сохранена: {pr_path}")
```

### Визуализация результатов регрессии

```python
import numpy as np
from ml.training.visualization import ModelVisualizer

# Создаем тестовые данные
y_true = np.array([3.0, 1.5, 2.0, 3.5, 0.5, 2.5, 1.0, 3.0, 2.0, 1.5])
y_pred = np.array([2.8, 1.2, 2.5, 3.6, 0.8, 2.3, 1.1, 2.9, 1.8, 1.6])

# Создаем визуализатор
visualizer = ModelVisualizer()

# Строим график сравнения предсказаний
pred_path = visualizer.plot_regression_predictions(y_true, y_pred, "my_regressor", "test_data")
print(f"График сравнения предсказаний сохранен: {pred_path}")

# Строим график остатков
resid_path = visualizer.plot_residuals(y_true, y_pred, "my_regressor", "test_data")
print(f"График остатков сохранен: {resid_path}")

# Строим гистограмму остатков
resid_hist_path = visualizer.plot_residuals_histogram(y_true, y_pred, "my_regressor", "test_data")
print(f"Гистограмма остатков сохранена: {resid_hist_path}")
```

### Визуализация важности признаков

```python
import numpy as np
from ml.training.visualization import ModelVisualizer

# Создаем тестовые данные
feature_importance = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]

# Создаем визуализатор
visualizer = ModelVisualizer()

# Строим график важности признаков
importance_path = visualizer.plot_feature_importance(
    feature_importance, feature_names, "my_model", "test_data"
)
print(f"График важности признаков сохранен: {importance_path}")

# Строим график важности признаков (только топ-3)
importance_top_path = visualizer.plot_feature_importance(
    feature_importance, feature_names, "my_model", "test_data", top_n=3
)
print(f"График важности признаков (топ-3) сохранен: {importance_top_path}")
```

### Визуализация кривой обучения

```python
import numpy as np
from ml.training.visualization import ModelVisualizer

# Создаем тестовые данные
train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
train_scores = np.array([[0.8, 0.82], [0.85, 0.86], [0.88, 0.89], [0.9, 0.91], [0.92, 0.93]])
test_scores = np.array([[0.7, 0.71], [0.75, 0.76], [0.78, 0.79], [0.8, 0.81], [0.82, 0.83]])

# Создаем визуализатор
visualizer = ModelVisualizer()

# Строим кривую обучения
learning_curve_path = visualizer.plot_learning_curve(
    train_sizes, train_scores, test_scores, "my_model", "test_data"
)
print(f"Кривая обучения сохранена: {learning_curve_path}")
```

### Сравнение моделей

```python
from ml.training.visualization import ModelVisualizer

# Создаем тестовые данные
metrics_dict = {
    "model1": {"accuracy": 0.85, "precision": 0.8, "recall": 0.75, "f1": 0.77},
    "model2": {"accuracy": 0.88, "precision": 0.83, "recall": 0.79, "f1": 0.81},
    "model3": {"accuracy": 0.82, "precision": 0.78, "recall": 0.73, "f1": 0.75}
}
model_names = ["model1", "model2", "model3"]

# Создаем визуализатор
visualizer = ModelVisualizer()

# Строим график сравнения метрики accuracy
accuracy_path = visualizer.plot_metrics_comparison(
    metrics_dict, model_names, "comparison", "accuracy"
)
print(f"График сравнения accuracy сохранен: {accuracy_path}")

# Строим график сравнения нескольких метрик
multi_path = visualizer.plot_multiple_metrics_comparison(
    metrics_dict, model_names, "comparison", ["accuracy", "precision", "recall", "f1"]
)
print(f"График сравнения нескольких метрик сохранен: {multi_path}")
```

### Настройка визуализатора

```python
from ml.training.visualization import ModelVisualizer

# Создаем визуализатор с пользовательской конфигурацией
config = {
    "plots_dir": "custom_plots",
    "style": "ggplot",
    "dpi": 300,
    "figsize": (12, 10)
}
visualizer = ModelVisualizer(config)

# Обновляем конфигурацию
new_config = {
    "dpi": 150,
    "figsize": (8, 6)
}
visualizer.update_config(new_config)
``` 