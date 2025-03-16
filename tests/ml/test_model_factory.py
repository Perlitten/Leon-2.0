"""
Тесты для модуля фабрики моделей.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ml.model_factory import ModelFactory
from ml.models.base_model import BaseModel
from ml.models.regression_model import RegressionModel
from ml.models.classification_model import ClassificationModel
from ml.models.ensemble_model import EnsembleModel
from ml.models.time_series_model import TimeSeriesModel


class TestModelFactory(unittest.TestCase):
    """
    Тесты для класса ModelFactory.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        self.factory = ModelFactory()
    
    def test_initialization(self):
        """
        Тест инициализации фабрики моделей.
        """
        self.assertIsInstance(self.factory, ModelFactory)
        self.assertEqual(len(self.factory.MODEL_TYPES), 4)
        self.assertIn('regression', self.factory.MODEL_TYPES)
        self.assertIn('classification', self.factory.MODEL_TYPES)
        self.assertIn('ensemble', self.factory.MODEL_TYPES)
        self.assertIn('time_series', self.factory.MODEL_TYPES)
    
    @patch('ml.models.regression_model.RegressionModel')
    def test_create_regression_model(self, mock_regression):
        """
        Тест создания модели регрессии.
        """
        # Настройка мока
        mock_instance = MagicMock()
        mock_regression.return_value = mock_instance
        
        # Создание модели
        model = self.factory.create_regression_model(
            name="test_regression",
            algorithm="random_forest",
            max_depth=5
        )
        
        # Проверка вызова конструктора
        mock_regression.assert_called_once_with(
            name="test_regression",
            algorithm="random_forest",
            max_depth=5
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_instance)
    
    @patch('ml.models.classification_model.ClassificationModel')
    def test_create_classification_model(self, mock_classification):
        """
        Тест создания модели классификации.
        """
        # Настройка мока
        mock_instance = MagicMock()
        mock_classification.return_value = mock_instance
        
        # Создание модели
        model = self.factory.create_classification_model(
            name="test_classification",
            algorithm="svc",
            kernel="rbf"
        )
        
        # Проверка вызова конструктора
        mock_classification.assert_called_once_with(
            name="test_classification",
            algorithm="svc",
            kernel="rbf"
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_instance)
    
    @patch('ml.models.time_series_model.TimeSeriesModel')
    def test_create_time_series_model(self, mock_time_series):
        """
        Тест создания модели временных рядов.
        """
        # Настройка мока
        mock_instance = MagicMock()
        mock_time_series.return_value = mock_instance
        
        # Создание модели
        model = self.factory.create_time_series_model(
            name="test_time_series",
            algorithm="sarima",
            p=1, d=1, q=1,
            P=1, D=1, Q=1, s=12
        )
        
        # Проверка вызова конструктора
        mock_time_series.assert_called_once_with(
            name="test_time_series",
            algorithm="sarima",
            p=1, d=1, q=1,
            P=1, D=1, Q=1, s=12
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_instance)
    
    @patch('ml.models.ensemble_model.EnsembleModel')
    def test_create_ensemble_model(self, mock_ensemble):
        """
        Тест создания ансамблевой модели.
        """
        # Создание моделей для ансамбля
        model1 = MagicMock(spec=BaseModel)
        model2 = MagicMock(spec=BaseModel)
        models = [model1, model2]
        
        # Настройка мока
        mock_instance = MagicMock()
        mock_ensemble.return_value = mock_instance
        
        # Создание модели
        model = self.factory.create_ensemble_model(
            name="test_ensemble",
            models=models,
            aggregation_method="voting",
            weights=[0.7, 0.3]
        )
        
        # Проверка вызова конструктора
        mock_ensemble.assert_called_once_with(
            name="test_ensemble",
            models=models,
            aggregation_method="voting",
            weights=[0.7, 0.3]
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_instance)
    
    def test_create_model_invalid_type(self):
        """
        Тест создания модели с неподдерживаемым типом.
        """
        with self.assertRaises(ValueError):
            self.factory.create_model("invalid_type", "test_model")
    
    @patch('ml.model_factory.ModelFactory.create_model')
    def test_create_model_from_config_basic(self, mock_create_model):
        """
        Тест создания модели из конфигурации.
        """
        # Настройка мока
        mock_instance = MagicMock()
        mock_create_model.return_value = mock_instance
        
        # Конфигурация модели
        config = {
            "type": "regression",
            "name": "test_from_config",
            "algorithm": "random_forest",
            "max_depth": 5
        }
        
        # Создание модели
        model = self.factory.create_model_from_config(config)
        
        # Проверка вызова метода create_model
        mock_create_model.assert_called_once_with(
            "regression",
            "test_from_config",
            algorithm="random_forest",
            max_depth=5
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_instance)
    
    def test_create_model_from_config_missing_type(self):
        """
        Тест создания модели из конфигурации без указания типа.
        """
        config = {
            "name": "test_from_config",
            "algorithm": "random_forest"
        }
        
        with self.assertRaises(ValueError):
            self.factory.create_model_from_config(config)
    
    def test_create_model_from_config_missing_name(self):
        """
        Тест создания модели из конфигурации без указания имени.
        """
        config = {
            "type": "regression",
            "algorithm": "random_forest"
        }
        
        with self.assertRaises(ValueError):
            self.factory.create_model_from_config(config)
    
    @patch('ml.model_factory.ModelFactory.create_ensemble_model')
    @patch('ml.model_factory.ModelFactory.create_model_from_config')
    def test_create_model_from_config_ensemble(self, mock_create_from_config, mock_create_ensemble):
        """
        Тест создания ансамблевой модели из конфигурации.
        """
        # Настройка моков
        model1 = MagicMock(spec=BaseModel)
        model2 = MagicMock(spec=BaseModel)
        mock_create_from_config.side_effect = [model1, model2]
        
        mock_ensemble = MagicMock()
        mock_create_ensemble.return_value = mock_ensemble
        
        # Конфигурация ансамблевой модели
        config = {
            "type": "ensemble",
            "name": "test_ensemble_config",
            "aggregation_method": "mean",
            "weights": [0.6, 0.4],
            "models_config": [
                {
                    "type": "regression",
                    "name": "model1",
                    "algorithm": "random_forest"
                },
                {
                    "type": "regression",
                    "name": "model2",
                    "algorithm": "gradient_boosting"
                }
            ]
        }
        
        # Создание модели
        model = self.factory.create_model_from_config(config)
        
        # Проверка вызовов create_model_from_config для каждой подмодели
        self.assertEqual(mock_create_from_config.call_count, 2)
        
        # Проверка вызова create_ensemble_model
        mock_create_ensemble.assert_called_once_with(
            "test_ensemble_config",
            [model1, model2],
            aggregation_method="mean",
            weights=[0.6, 0.4]
        )
        
        # Проверка возвращаемого значения
        self.assertEqual(model, mock_ensemble)
    
    def test_create_model_from_config_ensemble_missing_models(self):
        """
        Тест создания ансамблевой модели из конфигурации без указания моделей.
        """
        config = {
            "type": "ensemble",
            "name": "test_ensemble_config",
            "aggregation_method": "mean"
        }
        
        with self.assertRaises(ValueError):
            self.factory.create_model_from_config(config)
    
    @patch('ml.models.regression_model.RegressionModel')
    def test_create_model_exception(self, mock_regression):
        """
        Тест обработки исключений при создании модели.
        """
        # Настройка мока для вызова исключения
        mock_regression.side_effect = ValueError("Test error")
        
        # Проверка, что исключение пробрасывается дальше
        with self.assertRaises(ValueError):
            self.factory.create_regression_model("test_model")


if __name__ == "__main__":
    unittest.main() 