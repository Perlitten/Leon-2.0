"""
Тесты для модуля моделей временных рядов.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

from ml.models.time_series_model import TimeSeriesModel


class TestTimeSeriesModel(unittest.TestCase):
    """
    Тесты для класса TimeSeriesModel.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем тестовые данные временного ряда
        np.random.seed(42)
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.values = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
        self.time_series = pd.Series(self.values, index=self.dates)
        
        # Создаем DataFrame с экзогенными переменными
        self.exog_data = pd.DataFrame({
            'target': self.values,
            'exog1': np.random.normal(0, 1, 100),
            'exog2': np.random.normal(0, 1, 100)
        }, index=self.dates)
        
        # Создаем модель с параметрами по умолчанию
        self.model = TimeSeriesModel(name="test_arima_model", algorithm="arima", p=1, d=1, q=1)
    
    def test_initialization(self):
        """
        Тест инициализации модели.
        """
        # Тест с правильными параметрами
        model = TimeSeriesModel(name="test_model", algorithm="arima", p=1, d=1, q=1)
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.algorithm, "arima")
        self.assertEqual(model.model_params, {'p': 1, 'd': 1, 'q': 1})
        
        # Тест с SARIMA
        model = TimeSeriesModel(name="test_sarima", algorithm="sarima", p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
        self.assertEqual(model.algorithm, "sarima")
        self.assertEqual(model.model_params, {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 's': 12})
        
        # Тест с экспоненциальным сглаживанием
        model = TimeSeriesModel(name="test_exp", algorithm="exp_smoothing", trend='add', seasonal='add', seasonal_periods=12)
        self.assertEqual(model.algorithm, "exp_smoothing")
        self.assertEqual(model.model_params, {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12})
        
        # Тест с неправильным алгоритмом
        with self.assertRaises(ValueError):
            TimeSeriesModel(name="test_invalid", algorithm="invalid_algorithm")
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_train_arima(self, mock_arima):
        """
        Тест обучения модели ARIMA.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        metrics = self.model.train(self.time_series)
        
        # Проверки
        mock_arima.assert_called_once()
        mock_model.fit.assert_called_once()
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.model_fit)
        
        # Проверка метрик
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
    
    @patch('ml.models.time_series_model.SARIMAX')
    def test_train_sarima(self, mock_sarimax):
        """
        Тест обучения модели SARIMA.
        """
        # Создаем модель SARIMA
        model = TimeSeriesModel(name="test_sarima", algorithm="sarima", p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
        
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_model.fit.return_value = mock_fit
        mock_sarimax.return_value = mock_model
        
        # Обучение модели
        metrics = model.train(self.exog_data)
        
        # Проверки
        mock_sarimax.assert_called_once()
        mock_model.fit.assert_called_once()
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.model_fit)
        
        # Проверка метрик
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
    
    @patch('ml.models.time_series_model.ExponentialSmoothing')
    def test_train_exp_smoothing(self, mock_exp):
        """
        Тест обучения модели экспоненциального сглаживания.
        """
        # Создаем модель экспоненциального сглаживания
        model = TimeSeriesModel(name="test_exp", algorithm="exp_smoothing", trend='add', seasonal='add', seasonal_periods=12)
        
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_model.fit.return_value = mock_fit
        mock_exp.return_value = mock_model
        
        # Обучение модели
        metrics = model.train(self.time_series)
        
        # Проверки
        mock_exp.assert_called_once()
        mock_model.fit.assert_called_once()
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.model_fit)
        
        # Проверка метрик
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
    
    def test_predict_without_training(self):
        """
        Тест прогнозирования без обучения.
        """
        with self.assertRaises(RuntimeError):
            self.model.predict(self.time_series, steps=5)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_predict_arima(self, mock_arima):
        """
        Тест прогнозирования модели ARIMA.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.forecast.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Прогнозирование
        forecast = self.model.predict(self.time_series, steps=5)
        
        # Проверки
        mock_fit.forecast.assert_called_once_with(steps=5, exog=None)
        self.assertEqual(len(forecast), 5)
        np.testing.assert_array_equal(forecast, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    
    @patch('ml.models.time_series_model.SARIMAX')
    def test_predict_sarima_with_exog(self, mock_sarimax):
        """
        Тест прогнозирования модели SARIMA с экзогенными переменными.
        """
        # Создаем модель SARIMA
        model = TimeSeriesModel(name="test_sarima", algorithm="sarima", p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
        
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.forecast.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.fit.return_value = mock_fit
        mock_sarimax.return_value = mock_model
        
        # Обучение модели
        model.train(self.exog_data)
        
        # Создаем тестовые данные для прогноза
        exog_future = pd.DataFrame({
            'target': [0] * 5,  # Не используется для прогноза
            'exog1': np.random.normal(0, 1, 5),
            'exog2': np.random.normal(0, 1, 5)
        })
        
        # Прогнозирование
        forecast = model.predict(exog_future, steps=5)
        
        # Проверки
        self.assertEqual(len(forecast), 5)
        np.testing.assert_array_equal(forecast, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    
    def test_evaluate_without_training(self):
        """
        Тест оценки без обучения.
        """
        with self.assertRaises(RuntimeError):
            self.model.evaluate(self.time_series)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_evaluate(self, mock_arima):
        """
        Тест оценки модели.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.forecast.return_value = self.values[:5]  # Используем первые 5 значений как прогноз
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Оценка модели
        metrics = self.model.evaluate(self.time_series[:5], self.time_series[:5])
        
        # Проверки
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_get_feature_importance(self, mock_arima):
        """
        Тест получения важности признаков.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Получение важности признаков
        importance = self.model.get_feature_importance()
        
        # Проверки
        self.assertIsNone(importance)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_get_model_summary(self, mock_arima):
        """
        Тест получения сводки о модели.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.summary.return_value = "Model Summary"
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Получение сводки
        summary = self.model.get_model_summary()
        
        # Проверки
        self.assertEqual(summary, "Model Summary")
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_get_residuals(self, mock_arima):
        """
        Тест получения остатков модели.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.resid = np.random.normal(0, 0.1, 100)
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Получение остатков
        residuals = self.model.get_residuals()
        
        # Проверки
        self.assertEqual(len(residuals), 100)
        np.testing.assert_array_equal(residuals, mock_fit.resid)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_get_aic(self, mock_arima):
        """
        Тест получения AIC.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.aic = 123.45
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Получение AIC
        aic = self.model.get_aic()
        
        # Проверки
        self.assertEqual(aic, 123.45)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_get_bic(self, mock_arima):
        """
        Тест получения BIC.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_fit.bic = 456.78
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Получение BIC
        bic = self.model.get_bic()
        
        # Проверки
        self.assertEqual(bic, 456.78)
    
    @patch('ml.models.time_series_model.ARIMA')
    def test_calculate_metrics(self, mock_arima):
        """
        Тест расчета метрик.
        """
        # Настройка мока
        mock_model = MagicMock()
        mock_fit = MagicMock()
        mock_fit.fittedvalues = self.values
        mock_model.fit.return_value = mock_fit
        mock_arima.return_value = mock_model
        
        # Обучение модели
        self.model.train(self.time_series)
        
        # Тестовые данные
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Расчет метрик
        metrics = self.model._calculate_metrics(actual, predictions)
        
        # Проверки
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('mape', metrics)
        
        # Проверка значений метрик
        self.assertAlmostEqual(metrics['mse'], 0.01, places=2)
        self.assertAlmostEqual(metrics['rmse'], 0.1, places=2)
        self.assertAlmostEqual(metrics['mae'], 0.1, places=2)
        
        # Тест с нулевыми значениями (для MAPE)
        actual_with_zero = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
        metrics_with_zero = self.model._calculate_metrics(actual_with_zero, predictions)
        self.assertTrue(np.isnan(metrics_with_zero['mape']))


if __name__ == '__main__':
    unittest.main() 