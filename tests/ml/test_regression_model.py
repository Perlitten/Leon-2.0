"""
Тесты для модуля regression_model.

Тестирование функциональности класса RegressionModel.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from sklearn.datasets import make_regression
import logging
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression, Ridge

# Отключение логирования для тестов
logging.disable(logging.CRITICAL)

# Добавление корневой директории проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.regression_model import RegressionModel


class TestRegressionModel(unittest.TestCase):
    """Тесты для класса RegressionModel."""

    def setUp(self):
        """Настройка тестового окружения."""
        # Создание временной директории для тестов
        self.test_dir = tempfile.mkdtemp()
        
        # Создание тестовых данных
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        
        # Преобразование в DataFrame и Series
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        # Создание модели
        self.model = RegressionModel(name="test_model", algorithm="linear")

    def tearDown(self):
        """Очистка после тестов."""
        # Удаление временной директории
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Тест инициализации модели."""
        # Тест с параметрами по умолчанию
        model = RegressionModel(name="test_model")
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.algorithm, "linear")
        self.assertEqual(model.version, "1.0.0")
        self.assertFalse(model.is_trained)
        self.assertIsInstance(model.model, LinearRegression)
        
        # Тест с указанием алгоритма
        model = RegressionModel(name="test_model", algorithm="ridge", alpha=0.5)
        self.assertEqual(model.algorithm, "ridge")
        self.assertIsInstance(model.model, Ridge)
        self.assertEqual(model.model.alpha, 0.5)
        
        # Тест с неподдерживаемым алгоритмом
        with self.assertRaises(ValueError):
            RegressionModel(name="test_model", algorithm="unknown_algorithm")

    def test_train_with_numpy(self):
        """Тест обучения модели с numpy массивами."""
        metrics = self.model.train(self.X, self.y)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_train_with_pandas(self):
        """Тест обучения модели с pandas DataFrame и Series."""
        metrics = self.model.train(self.X, self.y)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_predict_without_training(self):
        """Тест предсказания без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X)

    def test_predict_with_numpy(self):
        """Тест предсказания с numpy массивом после обучения."""
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X))

    def test_predict_with_pandas(self):
        """Тест предсказания с pandas DataFrame после обучения."""
        self.model.train(self.X, self.y)
        predictions = self.model.predict(self.X)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X))

    def test_evaluate_without_training(self):
        """Тест оценки без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.evaluate(self.X, self.y)

    def test_evaluate_with_numpy(self):
        """Тест оценки с numpy массивами после обучения."""
        self.model.train(self.X, self.y)
        metrics = self.model.evaluate(self.X, self.y)
        
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_evaluate_with_pandas(self):
        """Тест оценки с pandas DataFrame и Series после обучения."""
        self.model.train(self.X, self.y)
        metrics = self.model.evaluate(self.X, self.y)
        
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)

    def test_get_feature_importance_linear(self):
        """Тест получения важности признаков для линейной модели."""
        self.model.train(self.X, self.y)
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X.shape[1])

    def test_get_feature_importance_random_forest(self):
        """Тест получения важности признаков для модели случайного леса."""
        model = RegressionModel(name="rf_model", algorithm="random_forest")
        model.train(self.X, self.y)
        importance = model.get_feature_importance()
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X.shape[1])

    def test_get_feature_importance_without_training(self):
        """Тест получения важности признаков без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.get_feature_importance()

    def test_save_load(self):
        """Тест сохранения и загрузки модели."""
        # Обучение модели
        self.model.train(self.X, self.y)
        
        # Сохранение модели
        model_path = os.path.join(self.test_dir, "test_model.pkl")
        saved_path = self.model.save(model_path)
        
        # Проверка, что файлы созданы
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(os.path.exists(f"{os.path.splitext(saved_path)[0]}_metadata.json"))
        
        # Создание новой модели
        loaded_model = RegressionModel(name="loaded_model")
        
        # Загрузка модели
        result = loaded_model.load(saved_path)
        
        # Проверка, что загрузка успешна
        self.assertTrue(result)
        
        # Проверка метаданных
        self.assertEqual(loaded_model.metadata["name"], "test_model")
        self.assertEqual(loaded_model.metadata["type"], "regression")
        self.assertEqual(loaded_model.metadata["algorithm"], "linear")
        
        # Предсказание с помощью загруженной модели
        predictions_original = self.model.predict(self.X)
        predictions_loaded = loaded_model.predict(self.X)
        
        # Проверка, что предсказания совпадают
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)


if __name__ == '__main__':
    unittest.main() 