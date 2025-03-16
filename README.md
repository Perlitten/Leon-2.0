# Leon 2.0

## Настройка проекта

### Конфигурация

1. Скопируйте файл `config.example.js` и переименуйте его в `config.js`:
   ```
   cp config.example.js config.js
   ```

2. Отредактируйте файл `config.js`, указав свои настройки:
   - Данные для подключения к базе данных
   - API ключи
   - Настройки сервера
   - Секретный ключ для авторизации

### Установка зависимостей

# Модуль машинного обучения

Модуль машинного обучения предоставляет набор инструментов для создания, обучения, оценки и управления моделями машинного обучения.

## Структура модуля

Модуль состоит из следующих компонентов:

### 1. Модели (`ml/models`)

- `BaseModel` - базовый класс для всех моделей
- `RegressionModel` - класс для регрессионных моделей
- `ClassificationModel` - класс для классификационных моделей
- `EnsembleModel` - класс для ансамблевых моделей
- `TimeSeriesModel` - класс для моделей временных рядов
- `ModelFactory` - фабрика для создания моделей
- `ModelManager` - менеджер для управления жизненным циклом моделей

### 2. Предобработка данных (`ml/preprocessing`)

- `DataPreprocessor` - класс для предобработки данных
- `FeatureSelector` - класс для выбора признаков
- `FeatureEngineer` - класс для инженерии признаков

### 3. Валидация и визуализация (`ml/validation`)

- `ModelValidator` - класс для валидации моделей
- `ModelVisualizer` - класс для визуализации результатов

### 4. Обучение (`ml/training`)

- `ModelTrainer` - класс для обучения моделей

## Примеры использования

### Базовый пример

```python
from ml.models import ModelFactory, ModelManager
from ml.preprocessing import DataPreprocessor, FeatureSelector
from ml.validation import ModelValidator
from ml.training import ModelTrainer

# Создание фабрики моделей
model_factory = ModelFactory()

# Создание модели
model = model_factory.create_regression_model(
    model_name="linear_regression",
    model_class="sklearn.linear_model.LinearRegression",
    params={},
    metadata={"description": "Линейная регрессия"}
)

# Предобработка данных
preprocessor = DataPreprocessor()
preprocessor.add_scaler("standard_scaler", scaler_type="standard")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Обучение модели
trainer = ModelTrainer()
trainer.train_model(model, X_train_processed, y_train)

# Валидация модели
validator = ModelValidator(report_dir="reports")
report = validator.validate_model(model, X_test_processed, y_test)

# Сохранение модели
model_manager = ModelManager(models_dir="models")
model_id = model_manager.save_model(model)
```

### Полный пример

Полный пример использования модуля можно найти в файле `examples/ml_pipeline_demo.py`.

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourusername/ml-module.git
cd ml-module

# Установка зависимостей
pip install -r requirements.txt
```

## Зависимости

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Лицензия

MIT