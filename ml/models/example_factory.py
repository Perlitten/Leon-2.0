"""
Пример использования фабрики моделей.

Этот скрипт демонстрирует, как использовать ModelFactory для создания
различных типов моделей машинного обучения.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml.model_factory import ModelFactory
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=100, n_features=5, random_state=42):
    """
    Создание тестовых данных для обучения моделей.
    
    Args:
        n_samples: Количество примеров
        n_features: Количество признаков
        random_state: Seed для генератора случайных чисел
        
    Returns:
        tuple: (X, y_reg, y_cls) - признаки, целевые значения для регрессии и классификации
    """
    np.random.seed(random_state)
    
    # Создание признаков
    X = np.random.rand(n_samples, n_features)
    
    # Создание целевых значений для регрессии
    y_reg = np.sin(X[:, 0] * 3) + 0.5 * X[:, 1] ** 2 + X[:, 2] + 0.1 * np.random.randn(n_samples)
    
    # Создание целевых значений для классификации
    y_cls = (y_reg > y_reg.mean()).astype(int)
    
    return X, y_reg, y_cls

def create_time_series_data(periods=100, freq='D', random_state=42):
    """
    Создание тестовых данных временного ряда.
    
    Args:
        periods: Количество периодов
        freq: Частота данных ('D' - дни, 'H' - часы и т.д.)
        random_state: Seed для генератора случайных чисел
        
    Returns:
        pd.Series: Временной ряд
    """
    np.random.seed(random_state)
    
    # Создаем даты
    dates = pd.date_range(start='2020-01-01', periods=periods, freq=freq)
    
    # Создаем базовый тренд
    trend = np.linspace(0, 10, periods)
    
    # Добавляем сезонность
    seasonality = 5 * np.sin(np.linspace(0, 10 * np.pi, periods))
    
    # Добавляем шум
    noise = np.random.normal(0, 1, periods)
    
    # Комбинируем компоненты
    values = trend + seasonality + noise
    
    # Создаем временной ряд
    time_series = pd.Series(values, index=dates)
    
    return time_series

def example_regression_models():
    """
    Пример создания и использования моделей регрессии.
    """
    logger.info("Запуск примера моделей регрессии")
    
    # Создание фабрики моделей
    factory = ModelFactory()
    
    # Создание тестовых данных
    X, y_reg, _ = create_sample_data()
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_reg[:train_size], y_reg[train_size:]
    
    # Создание моделей регрессии с разными алгоритмами
    models = {
        "Linear": factory.create_regression_model(
            name="linear_model",
            algorithm="linear"
        ),
        "Random Forest": factory.create_regression_model(
            name="rf_model",
            algorithm="random_forest",
            n_estimators=100,
            max_depth=5
        ),
        "Gradient Boosting": factory.create_regression_model(
            name="gb_model",
            algorithm="gradient_boosting",
            n_estimators=100,
            learning_rate=0.1
        )
    }
    
    # Обучение моделей и получение предсказаний
    predictions = {}
    metrics = {}
    
    for name, model in models.items():
        logger.info(f"Обучение модели: {name}")
        model.train(X_train, y_train)
        
        # Предсказание
        pred = model.predict(X_test)
        predictions[name] = pred
        
        # Оценка
        eval_metrics = model.evaluate(X_test, y_test)
        metrics[name] = eval_metrics
        logger.info(f"Метрики модели {name}: {eval_metrics}")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(y_test)), y_test, label='Истинные значения', color='black', alpha=0.5)
    
    for name, pred in predictions.items():
        plt.plot(range(len(pred)), pred, label=f'{name} (RMSE: {metrics[name]["rmse"]:.3f})')
    
    plt.title('Сравнение моделей регрессии')
    plt.xlabel('Индекс примера')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return models, predictions, metrics

def example_classification_models():
    """
    Пример создания и использования моделей классификации.
    """
    logger.info("Запуск примера моделей классификации")
    
    # Создание фабрики моделей
    factory = ModelFactory()
    
    # Создание тестовых данных
    X, _, y_cls = create_sample_data()
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_cls[:train_size], y_cls[train_size:]
    
    # Создание моделей классификации с разными алгоритмами
    models = {
        "Logistic": factory.create_classification_model(
            name="logistic_model",
            algorithm="logistic"
        ),
        "Random Forest": factory.create_classification_model(
            name="rf_model",
            algorithm="random_forest",
            n_estimators=100,
            max_depth=5
        ),
        "Gradient Boosting": factory.create_classification_model(
            name="gb_model",
            algorithm="gradient_boosting",
            n_estimators=100,
            learning_rate=0.1
        )
    }
    
    # Обучение моделей и получение предсказаний
    predictions = {}
    probabilities = {}
    metrics = {}
    
    for name, model in models.items():
        logger.info(f"Обучение модели: {name}")
        model.train(X_train, y_train)
        
        # Предсказание классов
        pred = model.predict(X_test)
        predictions[name] = pred
        
        # Предсказание вероятностей
        prob = model.predict_proba(X_test)
        probabilities[name] = prob
        
        # Оценка
        eval_metrics = model.evaluate(X_test, y_test)
        metrics[name] = eval_metrics
        logger.info(f"Метрики модели {name}: {eval_metrics}")
    
    # Визуализация результатов (ROC-кривые)
    plt.figure(figsize=(10, 8))
    
    for name, prob in probabilities.items():
        # Для бинарной классификации берем вероятность положительного класса
        pos_probs = prob[:, 1]
        
        # Расчет ROC-кривой
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, pos_probs)
        roc_auc = auc(fpr, tpr)
        
        # Построение ROC-кривой
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые для моделей классификации')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return models, predictions, probabilities, metrics

def example_time_series_models():
    """
    Пример создания и использования моделей временных рядов.
    """
    logger.info("Запуск примера моделей временных рядов")
    
    # Создание фабрики моделей
    factory = ModelFactory()
    
    # Создание тестовых данных
    time_series = create_time_series_data()
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(time_series) * 0.8)
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    # Создание моделей временных рядов с разными алгоритмами
    models = {
        "ARIMA": factory.create_time_series_model(
            name="arima_model",
            algorithm="arima",
            p=2, d=1, q=2
        ),
        "Exponential Smoothing": factory.create_time_series_model(
            name="exp_smoothing_model",
            algorithm="exp_smoothing",
            trend="add",
            seasonal="add",
            seasonal_periods=12
        )
    }
    
    # Обучение моделей и получение предсказаний
    forecasts = {}
    metrics = {}
    
    for name, model in models.items():
        logger.info(f"Обучение модели: {name}")
        model.train(train_data)
        
        # Прогнозирование
        forecast = model.predict(train_data, steps=len(test_data))
        forecasts[name] = forecast
        
        # Оценка
        eval_metrics = model.evaluate(test_data, test_data)
        metrics[name] = eval_metrics
        logger.info(f"Метрики модели {name}: {eval_metrics}")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    
    # Построение исходного ряда
    plt.plot(train_data.index, train_data.values, label='Обучающие данные', color='blue')
    plt.plot(test_data.index, test_data.values, label='Тестовые данные', color='green')
    
    # Построение прогнозов
    for name, forecast in forecasts.items():
        plt.plot(test_data.index, forecast, label=f'{name} (RMSE: {metrics[name]["rmse"]:.3f})')
    
    # Добавление вертикальной линии, разделяющей обучающие и тестовые данные
    plt.axvline(x=train_data.index[-1], color='black', linestyle='--')
    
    plt.title('Сравнение моделей временных рядов')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return models, forecasts, metrics

def example_ensemble_model():
    """
    Пример создания и использования ансамблевой модели.
    """
    logger.info("Запуск примера ансамблевой модели")
    
    # Создание фабрики моделей
    factory = ModelFactory()
    
    # Создание тестовых данных
    X, y_reg, _ = create_sample_data()
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_reg[:train_size], y_reg[train_size:]
    
    # Создание базовых моделей
    base_models = [
        factory.create_regression_model(
            name="linear_model",
            algorithm="linear"
        ),
        factory.create_regression_model(
            name="rf_model",
            algorithm="random_forest",
            n_estimators=100,
            max_depth=5
        ),
        factory.create_regression_model(
            name="gb_model",
            algorithm="gradient_boosting",
            n_estimators=100,
            learning_rate=0.1
        )
    ]
    
    # Создание ансамблевой модели
    ensemble = factory.create_ensemble_model(
        name="regression_ensemble",
        models=base_models,
        aggregation_method="mean",
        weights=[0.2, 0.5, 0.3]
    )
    
    # Обучение ансамбля
    logger.info("Обучение ансамблевой модели")
    ensemble_metrics = ensemble.train(X_train, y_train)
    
    # Предсказание
    ensemble_pred = ensemble.predict(X_test)
    
    # Оценка
    ensemble_eval = ensemble.evaluate(X_test, y_test)
    logger.info(f"Метрики ансамбля: {ensemble_eval}")
    
    # Обучение и оценка базовых моделей отдельно для сравнения
    base_predictions = {}
    base_metrics = {}
    
    for i, model in enumerate(base_models):
        name = f"Model {i+1}"
        model.train(X_train, y_train)
        pred = model.predict(X_test)
        base_predictions[name] = pred
        metrics = model.evaluate(X_test, y_test)
        base_metrics[name] = metrics
        logger.info(f"Метрики модели {name}: {metrics}")
    
    # Визуализация результатов
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(y_test)), y_test, label='Истинные значения', color='black', alpha=0.5)
    
    for name, pred in base_predictions.items():
        plt.plot(range(len(pred)), pred, label=f'{name} (RMSE: {base_metrics[name]["rmse"]:.3f})', alpha=0.5)
    
    plt.plot(range(len(ensemble_pred)), ensemble_pred, 
             label=f'Ансамбль (RMSE: {ensemble_eval["rmse"]:.3f})', 
             color='red', linewidth=2)
    
    plt.title('Сравнение ансамблевой модели с базовыми моделями')
    plt.xlabel('Индекс примера')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Получение важности признаков
    importance = ensemble.get_feature_importance()
    if importance is not None:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.title('Важность признаков (усредненная по ансамблю)')
        plt.xlabel('Индекс признака')
        plt.ylabel('Важность')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return ensemble, base_models, ensemble_pred, base_predictions, ensemble_eval, base_metrics

def example_model_from_config():
    """
    Пример создания моделей из конфигурации.
    """
    logger.info("Запуск примера создания моделей из конфигурации")
    
    # Создание фабрики моделей
    factory = ModelFactory()
    
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
    logger.info(f"Создана модель регрессии: {regression_model.name}")
    
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
                "algorithm": "random_forest",
                "max_depth": 5
            },
            {
                "type": "regression",
                "name": "model2",
                "algorithm": "gradient_boosting",
                "learning_rate": 0.1
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
    logger.info(f"Создана ансамблевая модель: {ensemble_model.name} с {len(ensemble_model.models)} базовыми моделями")
    
    # Конфигурация модели временных рядов
    time_series_config = {
        "type": "time_series",
        "name": "btc_price_predictor",
        "algorithm": "sarima",
        "p": 1, "d": 1, "q": 1,
        "P": 1, "D": 1, "Q": 1, "s": 7
    }
    
    # Создание модели временных рядов из конфигурации
    time_series_model = factory.create_model_from_config(time_series_config)
    logger.info(f"Создана модель временных рядов: {time_series_model.name} с алгоритмом {time_series_model.algorithm}")
    
    return regression_model, ensemble_model, time_series_model

if __name__ == "__main__":
    # Запуск примеров
    try:
        # Пример моделей регрессии
        reg_models, reg_predictions, reg_metrics = example_regression_models()
        
        # Пример моделей классификации
        cls_models, cls_predictions, cls_probabilities, cls_metrics = example_classification_models()
        
        # Пример моделей временных рядов
        ts_models, ts_forecasts, ts_metrics = example_time_series_models()
        
        # Пример ансамблевой модели
        ensemble, base_models, ensemble_pred, base_predictions, ensemble_eval, base_metrics = example_ensemble_model()
        
        # Пример создания моделей из конфигурации
        reg_model, ens_model, ts_model = example_model_from_config()
        
        logger.info("Все примеры успешно выполнены")
    except Exception as e:
        logger.error(f"Ошибка при выполнении примеров: {e}", exc_info=True) 