"""
Пример использования модели временных рядов.

Этот скрипт демонстрирует, как использовать TimeSeriesModel для прогнозирования
временных рядов с использованием различных алгоритмов.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml.models import TimeSeriesModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(periods=200, freq='D'):
    """
    Создание тестовых данных временного ряда.
    
    Args:
        periods: Количество периодов
        freq: Частота данных ('D' - дни, 'H' - часы и т.д.)
        
    Returns:
        pd.Series: Временной ряд
    """
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

def create_exog_data(time_series, n_features=2):
    """
    Создание экзогенных переменных для временного ряда.
    
    Args:
        time_series: Базовый временной ряд
        n_features: Количество экзогенных переменных
        
    Returns:
        pd.DataFrame: DataFrame с временным рядом и экзогенными переменными
    """
    # Создаем DataFrame с целевой переменной
    df = pd.DataFrame({'target': time_series.values}, index=time_series.index)
    
    # Добавляем экзогенные переменные
    for i in range(n_features):
        # Создаем переменную, коррелирующую с целевой, но с шумом
        exog = 0.7 * time_series.values + np.random.normal(0, 2, len(time_series))
        df[f'exog{i+1}'] = exog
    
    return df

def plot_forecast(time_series, forecast, title="Прогноз временного ряда"):
    """
    Визуализация прогноза временного ряда.
    
    Args:
        time_series: Исходный временной ряд
        forecast: Прогноз
        title: Заголовок графика
    """
    plt.figure(figsize=(12, 6))
    
    # Построение исходного ряда
    plt.plot(time_series.index, time_series.values, label='Исходные данные')
    
    # Построение прогноза
    forecast_dates = pd.date_range(
        start=time_series.index[-1] + pd.Timedelta(days=1),
        periods=len(forecast),
        freq=time_series.index.freq
    )
    plt.plot(forecast_dates, forecast, label='Прогноз', color='red')
    
    # Добавление вертикальной линии, разделяющей исходные данные и прогноз
    plt.axvline(x=time_series.index[-1], color='black', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def example_arima():
    """
    Пример использования модели ARIMA.
    """
    logger.info("Запуск примера ARIMA")
    
    # Создание данных
    time_series = create_sample_data(periods=150)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(time_series) * 0.8)
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    # Создание и обучение модели ARIMA
    model = TimeSeriesModel(
        name="arima_example",
        algorithm="arima",
        p=2,  # порядок авторегрессии
        d=1,  # порядок интегрирования
        q=2   # порядок скользящего среднего
    )
    
    # Обучение модели
    metrics = model.train(train_data)
    logger.info(f"Метрики обучения ARIMA: {metrics}")
    
    # Прогнозирование
    forecast_steps = len(test_data)
    forecast = model.predict(train_data, steps=forecast_steps)
    
    # Оценка модели
    eval_metrics = model.evaluate(test_data, test_data)
    logger.info(f"Метрики оценки ARIMA: {eval_metrics}")
    
    # Визуализация результатов
    plot_forecast(train_data, forecast, title="Прогноз ARIMA")
    
    # Получение и вывод сводки о модели
    summary = model.get_model_summary()
    logger.info(f"Сводка модели ARIMA:\n{summary}")
    
    # Получение информационных критериев
    aic = model.get_aic()
    bic = model.get_bic()
    logger.info(f"AIC: {aic}, BIC: {bic}")
    
    return model, train_data, test_data, forecast

def example_sarima():
    """
    Пример использования модели SARIMA с экзогенными переменными.
    """
    logger.info("Запуск примера SARIMA")
    
    # Создание данных с выраженной сезонностью
    time_series = create_sample_data(periods=200)
    
    # Создание экзогенных переменных
    exog_data = create_exog_data(time_series, n_features=2)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(exog_data) * 0.8)
    train_data = exog_data[:train_size]
    test_data = exog_data[train_size:]
    
    # Создание и обучение модели SARIMA
    model = TimeSeriesModel(
        name="sarima_example",
        algorithm="sarima",
        p=1, d=1, q=1,  # несезонные параметры
        P=1, D=1, Q=1,  # сезонные параметры
        s=12            # период сезонности
    )
    
    # Обучение модели
    metrics = model.train(train_data)
    logger.info(f"Метрики обучения SARIMA: {metrics}")
    
    # Создание данных для прогноза с экзогенными переменными
    forecast_steps = len(test_data)
    
    # Прогнозирование
    forecast = model.predict(test_data, steps=forecast_steps)
    
    # Оценка модели
    eval_metrics = model.evaluate(test_data, test_data['target'])
    logger.info(f"Метрики оценки SARIMA: {eval_metrics}")
    
    # Визуализация результатов
    plot_forecast(
        pd.Series(train_data['target'].values, index=train_data.index),
        forecast,
        title="Прогноз SARIMA с экзогенными переменными"
    )
    
    # Получение и вывод сводки о модели
    summary = model.get_model_summary()
    logger.info(f"Сводка модели SARIMA:\n{summary}")
    
    return model, train_data, test_data, forecast

def example_exp_smoothing():
    """
    Пример использования модели экспоненциального сглаживания.
    """
    logger.info("Запуск примера экспоненциального сглаживания")
    
    # Создание данных
    time_series = create_sample_data(periods=150)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(time_series) * 0.8)
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    # Создание и обучение модели экспоненциального сглаживания
    model = TimeSeriesModel(
        name="exp_smoothing_example",
        algorithm="exp_smoothing",
        trend='add',           # аддитивный тренд
        seasonal='add',        # аддитивная сезонность
        seasonal_periods=12    # период сезонности
    )
    
    # Обучение модели
    metrics = model.train(train_data)
    logger.info(f"Метрики обучения экспоненциального сглаживания: {metrics}")
    
    # Прогнозирование
    forecast_steps = len(test_data)
    forecast = model.predict(train_data, steps=forecast_steps)
    
    # Оценка модели
    eval_metrics = model.evaluate(test_data, test_data)
    logger.info(f"Метрики оценки экспоненциального сглаживания: {eval_metrics}")
    
    # Визуализация результатов
    plot_forecast(train_data, forecast, title="Прогноз экспоненциального сглаживания")
    
    return model, train_data, test_data, forecast

def compare_models():
    """
    Сравнение различных моделей временных рядов.
    """
    logger.info("Запуск сравнения моделей")
    
    # Создание данных
    time_series = create_sample_data(periods=150)
    
    # Разделение на обучающую и тестовую выборки
    train_size = int(len(time_series) * 0.8)
    train_data = time_series[:train_size]
    test_data = time_series[train_size:]
    
    # Список моделей для сравнения
    models = [
        TimeSeriesModel(name="arima", algorithm="arima", p=2, d=1, q=2),
        TimeSeriesModel(name="sarima", algorithm="sarima", p=1, d=1, q=1, P=1, D=1, Q=1, s=12),
        TimeSeriesModel(name="exp_smoothing", algorithm="exp_smoothing", trend='add', seasonal='add', seasonal_periods=12)
    ]
    
    # Обучение моделей и получение прогнозов
    forecasts = []
    metrics = []
    
    for model in models:
        # Обучение
        model.train(train_data)
        
        # Прогнозирование
        forecast = model.predict(train_data, steps=len(test_data))
        forecasts.append(forecast)
        
        # Оценка
        eval_metric = model.evaluate(test_data, test_data)
        metrics.append(eval_metric)
    
    # Визуализация сравнения
    plt.figure(figsize=(12, 6))
    
    # Построение исходного ряда
    plt.plot(train_data.index, train_data.values, label='Обучающие данные', color='blue')
    plt.plot(test_data.index, test_data.values, label='Тестовые данные', color='green')
    
    # Построение прогнозов
    forecast_dates = pd.date_range(
        start=train_data.index[-1] + pd.Timedelta(days=1),
        periods=len(test_data),
        freq=train_data.index.freq
    )
    
    colors = ['red', 'purple', 'orange']
    for i, (model, forecast) in enumerate(zip(models, forecasts)):
        plt.plot(forecast_dates, forecast, label=f'Прогноз {model.algorithm}', color=colors[i])
    
    # Добавление вертикальной линии, разделяющей обучающие и тестовые данные
    plt.axvline(x=train_data.index[-1], color='black', linestyle='--')
    
    plt.title("Сравнение моделей временных рядов")
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Вывод метрик
    for i, model in enumerate(models):
        logger.info(f"Метрики модели {model.algorithm}: {metrics[i]}")
    
    return models, forecasts, metrics

if __name__ == "__main__":
    # Запуск примеров
    try:
        # Пример ARIMA
        arima_model, arima_train, arima_test, arima_forecast = example_arima()
        
        # Пример SARIMA
        sarima_model, sarima_train, sarima_test, sarima_forecast = example_sarima()
        
        # Пример экспоненциального сглаживания
        exp_model, exp_train, exp_test, exp_forecast = example_exp_smoothing()
        
        # Сравнение моделей
        models, forecasts, metrics = compare_models()
        
        logger.info("Все примеры успешно выполнены")
    except Exception as e:
        logger.error(f"Ошибка при выполнении примеров: {e}", exc_info=True) 