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

```bash
# Создание виртуального окружения Python
python -m venv venv

# Активация виртуального окружения
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Git и настройка репозитория

```bash
# Инициализация Git репозитория (если еще не сделано)
git init

# Добавление файлов в индекс
git add .

# Первый коммит
git commit -m "Initial commit"

# Подключение к удаленному репозиторию
git remote add origin https://github.com/Perlitten/Leon-2.0.git

# Отправка изменений на GitHub
git push -u origin master
```

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

# Модуль уведомлений

Модуль уведомлений предоставляет инструменты для отправки уведомлений и управления ботом через различные каналы связи.

## Структура модуля

### 1. Telegram (`notification/telegram`)

- `TelegramBot` - класс для отправки уведомлений и управления ботом через Telegram
- Функции управления ботом:
  - Приостановка работы бота
  - Возобновление работы бота
  - Остановка бота
  - Перезапуск бота
- Функции обучения модели:
  - Запуск обучения модели
  - Пропуск обучения и использование существующей модели

## Примеры использования

### Отправка уведомления через Telegram

```python
from notification.telegram import TelegramBot
from core.config_manager import ConfigManager
from core.localization import LocalizationManager

# Создание экземпляра бота
config_manager = ConfigManager("config.yaml")
localization = LocalizationManager("locales")
telegram_bot = TelegramBot(config_manager, localization)

# Запуск бота
await telegram_bot.start()

# Отправка уведомления
await telegram_bot.send_message("Привет! Это тестовое уведомление.")

# Отправка уведомления о торговой операции
await telegram_bot.send_trade_notification(
    symbol="BTCUSDT",
    direction="BUY",
    price=50000.0,
    size=0.1,
    pnl=None
)
```

# Модуль оркестрации

Модуль оркестрации отвечает за координацию работы всех подсистем бота.

## Структура модуля

### 1. Оркестратор (`core/orchestrator.py`)

- `LeonOrchestrator` - центральный класс для управления всеми подсистемами
- `EventBus` - система событий для коммуникации между компонентами
- `CommandProcessor` - обработчик команд
- `TradingModeManager` - менеджер режимов торговли
- `MLIntegrationManager` - менеджер интеграции с ML-моделями

### 2. Функции управления состоянием

- `pause()` - приостановка работы бота без полной остановки
- `resume()` - возобновление работы бота после паузы

## Примеры использования

### Управление ботом через оркестратор

```python
from core.orchestrator import LeonOrchestrator
from core.config_manager import ConfigManager
from core.localization import LocalizationManager

# Создание экземпляра оркестратора
config_manager = ConfigManager("config.yaml")
localization = LocalizationManager("locales")
orchestrator = LeonOrchestrator(config_manager, localization)

# Запуск бота
await orchestrator.start(mode="dry")

# Приостановка работы бота
await orchestrator.pause()

# Возобновление работы бота
await orchestrator.resume()

# Остановка бота
await orchestrator.stop()
```

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

# Планы развития

В рамках дальнейшего развития проекта Leon 2.0 планируется внедрение следующих улучшений:

## Улучшения модуля торговых стратегий
- Расширение библиотеки торговых стратегий с автоматическим выбором оптимальной стратегии
- Внедрение адаптивных параметров стратегий на основе машинного обучения
- Создание системы комбинирования стратегий для формирования мета-стратегий

## Улучшения модуля анализа рыночных данных
- Разработка продвинутых алгоритмов анализа рыночных данных
- Автоматическое определение оптимальных торговых пар
- Интеграция с внешними источниками данных (новости, социальные медиа)

## Улучшения визуализации данных
- Разработка интерактивного консольного интерфейса с графиками в реальном времени
- Добавление расширенной аналитической панели с ключевыми метриками
- Внедрение системы уведомлений о важных рыночных событиях

## Расширение функциональности модуля машинного обучения
- Внедрение автоматического определения размера позиции на основе обучения
- Разработка системы автоматического выбора оптимального интервала свечей
- Создание самообучающейся системы, адаптирующейся к изменениям рынка

Более подробную информацию о статусе разработки и планах можно найти в файле [module_status.md](module_status.md).