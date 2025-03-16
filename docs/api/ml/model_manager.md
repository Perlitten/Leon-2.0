# ModelManager

`ModelManager` - класс для управления моделями машинного обучения в системе Leon.

## Назначение

`ModelManager` отвечает за:
- Загрузку моделей из файлов
- Сохранение моделей в файлы
- Управление версиями моделей
- Предоставление моделей для использования

## API

### Инициализация

```python
from ml import ModelManager

# Инициализация с параметрами по умолчанию
model_manager = ModelManager()

# Инициализация с пользовательской конфигурацией
config = {
    "models_dir": "custom/models/path",
    "cache_size": 5
}
model_manager = ModelManager(config)
```

### Основные методы

#### get_available_models()

Возвращает список доступных моделей с их метаданными.

```python
models = model_manager.get_available_models()
for model_id, metadata in models.items():
    print(f"Модель: {model_id}, Тип: {metadata.get('model_type')}")
```

#### load_model(model_id)

Загружает модель по её идентификатору.

```python
model_id = "lstm_20230101_120000"
model = model_manager.load_model(model_id)
```

#### save_model(model, model_id, metadata)

Сохраняет модель и её метаданные.

```python
# Сохранение модели
metadata = {
    "model_type": "lstm",
    "description": "Модель для прогнозирования направления цены",
    "features": ["close", "volume", "ma_7", "rsi_14"]
}
model_manager.save_model(model, "my_model", metadata)
```

#### delete_model(model_id)

Удаляет модель и её метаданные.

```python
model_manager.delete_model("old_model")
```

#### get_model_metadata(model_id)

Получает метаданные модели.

```python
metadata = model_manager.get_model_metadata("lstm_20230101_120000")
print(f"Точность модели: {metadata.get('metrics', {}).get('accuracy')}")
```

#### update_model_metadata(model_id, new_metadata)

Обновляет метаданные модели.

```python
new_metadata = {
    "description": "Обновленное описание модели",
    "performance": {
        "live_accuracy": 0.78
    }
}
model_manager.update_model_metadata("lstm_20230101_120000", new_metadata)
```

#### clear_cache()

Очищает кэш загруженных моделей.

```python
model_manager.clear_cache()
```

#### reload_model_list()

Перезагружает список моделей из директории.

```python
model_manager.reload_model_list()
```

## Структура метаданных

Метаданные модели хранятся в формате JSON и содержат следующую информацию:

```json
{
    "model_id": "lstm_20230101_120000",
    "model_type": "lstm",
    "task_type": "classification",
    "created_at": "2023-01-01T12:00:00",
    "description": "Модель для прогнозирования направления цены",
    "features": ["close", "volume", "ma_7", "rsi_14"],
    "metrics": {
        "accuracy": 0.75,
        "precision": 0.72,
        "recall": 0.68,
        "f1_score": 0.70
    },
    "training_history": {
        "loss": [0.6, 0.5, 0.4, 0.3],
        "accuracy": [0.6, 0.65, 0.7, 0.75],
        "val_loss": [0.65, 0.55, 0.45, 0.35],
        "val_accuracy": [0.55, 0.6, 0.65, 0.75]
    },
    "parameters": {
        "units": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001
    }
}
```

## Пример использования

```python
from ml import ModelManager
import numpy as np

# Инициализация менеджера моделей
model_manager = ModelManager()

# Получение списка доступных моделей
available_models = model_manager.get_available_models()
print(f"Доступные модели: {list(available_models.keys())}")

# Загрузка модели
model_id = list(available_models.keys())[0]
model = model_manager.load_model(model_id)

# Использование модели для прогнозирования
X = np.random.random((1, 20, 10))  # Пример входных данных
prediction = model.predict(X)
print(f"Прогноз: {prediction}")

# Получение метаданных модели
metadata = model_manager.get_model_metadata(model_id)
print(f"Метрики модели: {metadata.get('metrics')}")

# Обновление метаданных
model_manager.update_model_metadata(model_id, {
    "last_used": "2023-05-15T10:30:00",
    "performance_notes": "Хорошо работает на волатильном рынке"
})
```

## Интеграция с другими компонентами

`ModelManager` интегрируется с другими компонентами системы Leon:

- `MLIntegrationManager` использует `ModelManager` для загрузки моделей и их использования в торговых стратегиях.
- `ModelTrainer` использует `ModelManager` для сохранения обученных моделей.
- `ModelEvaluator` использует `ModelManager` для загрузки моделей и сохранения результатов оценки в метаданных. 