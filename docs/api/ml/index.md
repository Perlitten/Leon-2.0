# Модуль машинного обучения (ML)

Модуль машинного обучения предоставляет классы и функции для интеграции моделей машинного обучения в торговую систему Leon.

## Основные компоненты

### ModelManager

`ModelManager` отвечает за управление моделями машинного обучения:
- Загрузка моделей из файлов
- Сохранение моделей в файлы
- Управление версиями моделей
- Предоставление моделей для использования

[Подробнее о ModelManager](./model_manager.md)

### FeatureExtractor

`FeatureExtractor` отвечает за извлечение признаков из исторических данных:
- Преобразование сырых данных в признаки для моделей
- Нормализация и масштабирование признаков
- Создание временных рядов для обучения

[Подробнее о FeatureExtractor](./feature_extractor.md)

### ModelEvaluator

`ModelEvaluator` отвечает за оценку производительности моделей:
- Оценка точности прогнозов модели
- Расчет метрик производительности
- Визуализация результатов
- Сохранение отчетов об оценке

[Подробнее о ModelEvaluator](./model_evaluator.md)

### ModelTrainer

`ModelTrainer` отвечает за обучение моделей:
- Подготовка данных для обучения
- Обучение моделей
- Валидация моделей
- Сохранение обученных моделей

[Подробнее о ModelTrainer](./model_trainer.md)

### ModelFactory

`ModelFactory` отвечает за создание моделей машинного обучения:
- Унифицированный интерфейс для создания моделей разных типов
- Создание моделей из конфигурации
- Поддержка различных типов моделей (регрессия, классификация, временные ряды, ансамбли)
- Упрощение процесса создания сложных моделей

[Подробнее о ModelFactory](./model_factory.md)

## Использование модуля

### Пример использования для прогнозирования

```python
from ml import ModelManager, FeatureExtractor
from core.config import ConfigManager

# Инициализация менеджера моделей
config = ConfigManager().get_config()
model_manager = ModelManager(config.get("ml", {}))

# Загрузка модели
model_id = "lstm_20230101_120000"
model = model_manager.load_model(model_id)

# Извлечение признаков из исторических данных
feature_extractor = FeatureExtractor(config.get("features", {}))
X, _ = feature_extractor.transform(historical_data)

# Получение прогноза
prediction = model.predict(X)
```

### Пример использования для обучения модели

```python
from ml import ModelTrainer, FeatureExtractor, ModelEvaluator, ModelFactory
from core.config import ConfigManager

# Инициализация компонентов
config = ConfigManager().get_config()
feature_extractor = FeatureExtractor(config.get("features", {}))
model_trainer = ModelTrainer(config.get("training", {}))
model_evaluator = ModelEvaluator(config.get("evaluation", {}))
model_factory = ModelFactory()

# Извлечение признаков из исторических данных
X, y = feature_extractor.transform(historical_data)

# Создание модели
model = model_factory.create_model(
    model_type="ensemble",
    name="price_ensemble",
    models=[
        model_factory.create_regression_model("model1", algorithm="random_forest"),
        model_factory.create_regression_model("model2", algorithm="gradient_boosting")
    ],
    aggregation_method="mean",
    weights=[0.6, 0.4]
)

# Обучение модели
model.train(X, y)

# Оценка модели
X_test, y_test = feature_extractor.transform(test_data)
evaluation_result = model_evaluator.evaluate(model, X_test, y_test)

print(f"Модель обучена с метриками: {evaluation_result}")
```

## Интеграция с торговой системой

Модуль машинного обучения интегрируется с торговой системой через `MLIntegrationManager` в модуле `core.orchestrator`. Это позволяет:

1. Использовать модели для генерации торговых сигналов
2. Обновлять модели на основе новых данных
3. Оценивать производительность моделей в реальном времени
4. Переключаться между различными моделями

[Подробнее об интеграции с торговой системой](../core/ml_integration.md) 