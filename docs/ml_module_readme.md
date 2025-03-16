# Модуль машинного обучения

Модуль машинного обучения предоставляет инструменты для обучения, валидации и визуализации моделей машинного обучения.

## Компоненты модуля

### 1. Модуль визуализации (`ml/training/visualization.py`)
Класс `ModelVisualizer` предоставляет методы для визуализации результатов работы моделей машинного обучения, включая:
- Матрицу ошибок
- ROC-кривую
- Кривую точность-полнота
- График сравнения предсказаний
- График остатков
- Гистограмму остатков
- График важности признаков
- Кривую обучения
- Графики сравнения метрик

### 2. Модуль валидации (`ml/training/model_validator.py`)
Класс `ModelValidator` предоставляет методы для оценки производительности моделей машинного обучения, включая:
- Валидацию одной модели
- Сравнение нескольких моделей
- Расчет метрик
- Визуализацию результатов
- Сохранение отчетов

### 3. Модуль обучения моделей (`ml/training/trainer.py`)
Класс `ModelTrainer` предоставляет методы для обучения моделей машинного обучения, включая:
- Подготовку данных
- Создание моделей различных типов (LSTM, CNN, MLP, XGBoost, Random Forest)
- Обучение моделей
- Оценку моделей
- Сохранение моделей и метаданных

### 4. Модуль выбора признаков (`ml/models/feature_selector.py`)
Класс `FeatureSelector` предоставляет методы для выбора наиболее значимых признаков, включая:
- Фильтрацию на основе статистических тестов
- Встроенные методы (на основе моделей)
- Рекурсивное исключение признаков
- Удаление признаков с низкой дисперсией

### 5. Модуль фабрики моделей
Класс `ModelFactory` предоставляет унифицированный интерфейс для создания моделей разных типов с различными параметрами.

### 6. Модуль моделей
Включает базовый класс `BaseModel` и его наследников:
- `RegressionModel` - для задач регрессии
- `ClassificationModel` - для задач классификации
- `EnsembleModel` - для ансамблевых моделей
- `TimeSeriesModel` - для моделей временных рядов

## Примеры использования

### Полный пайплайн машинного обучения

Пример полного пайплайна машинного обучения можно найти в файле `examples/ml_pipeline_demo.py`. Этот скрипт демонстрирует:
1. Загрузку и подготовку данных
2. Выбор признаков
3. Обучение модели
4. Валидацию модели
5. Визуализацию результатов

### Выбор признаков

```python
from ml.models.feature_selector import FeatureSelector

# Создаем селектор признаков
feature_selector = FeatureSelector(
    method="k_best_f_regression",
    n_features=10,
    feature_names=feature_names
)

# Обучаем селектор и трансформируем данные
X_selected = feature_selector.fit_transform(X, y)

# Получаем выбранные признаки и их оценки
selected_features = feature_selector.get_selected_features()
feature_scores = feature_selector.get_feature_scores()
```

### Обучение модели

```python
from ml.training.trainer import ModelTrainer

# Создаем тренер
trainer = ModelTrainer({
    "models_dir": "models",
    "task_type": "classification",
    "test_size": 0.2,
    "random_state": 42
})

# Обучаем модель
results = trainer.train(
    model_type="random_forest",
    X=X_train,
    y=y_train,
    model_params={
        "n_estimators": 100,
        "max_depth": 6
    }
)

# Получаем путь к сохраненной модели и метрики
model_path = results["model_path"]
metrics = results["metrics"]
```

### Валидация модели

```python
from ml.training.model_validator import ModelValidator

# Создаем валидатор
validator = ModelValidator({
    "reports_dir": "reports",
    "save_reports": True,
    "visualization": True
})

# Валидируем модель
report = validator.validate(
    model=model,
    X=X_test,
    y=y_test,
    task_type="classification",
    dataset_name="test_dataset"
)

# Получаем метрики
metrics = report["metrics"]
```

### Визуализация результатов

```python
from ml.training.visualization import ModelVisualizer

# Создаем визуализатор
visualizer = ModelVisualizer({
    "plots_dir": "plots",
    "dpi": 150,
    "figsize": (10, 8)
})

# Визуализируем матрицу ошибок
cm_path = visualizer.plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    model_name="my_model",
    dataset_name="test_dataset"
)

# Визуализируем важность признаков
importance_path = visualizer.plot_feature_importance(
    feature_importance=feature_importance,
    feature_names=feature_names,
    model_name="my_model",
    dataset_name="test_dataset"
)
```

## Запуск демонстрационного скрипта

Для запуска демонстрационного скрипта выполните:

```bash
python examples/ml_pipeline_demo.py
```

Результаты будут сохранены в директории `examples/results`. 