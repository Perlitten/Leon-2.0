"""
Тесты для модуля classification_model.

Тестирование функциональности класса ClassificationModel.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from sklearn.datasets import make_classification
import logging
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Отключение логирования для тестов
logging.disable(logging.CRITICAL)

# Добавление корневой директории проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.models.classification_model import ClassificationModel


class TestClassificationModel(unittest.TestCase):
    """Тесты для класса ClassificationModel."""

    def setUp(self):
        """Настройка тестового окружения."""
        # Создание временной директории для тестов
        self.test_dir = tempfile.mkdtemp()
        
        # Создание тестовых данных для бинарной классификации
        self.X_binary = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        self.y_binary = np.array([0, 1, 0, 1, 0, 1])
        
        # Создание тестовых данных для многоклассовой классификации
        self.X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        self.y_multi = np.array([0, 1, 2, 0, 1, 2])
        
        # Создание DataFrame и Series для тестирования с pandas
        self.X_df = pd.DataFrame(self.X_binary, columns=['feature1', 'feature2'])
        self.y_series = pd.Series(self.y_binary, name='target')
        
        # Создание модели
        self.model = ClassificationModel(name="test_classification", algorithm="logistic")

    def tearDown(self):
        """Очистка после тестов."""
        # Удаление временной директории
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Тест инициализации модели."""
        # Тест с параметрами по умолчанию
        model = ClassificationModel(name="test_model")
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.algorithm, "logistic")
        self.assertEqual(model.version, "1.0.0")
        self.assertFalse(model.is_trained)
        self.assertIsInstance(model.model, LogisticRegression)
        
        # Тест с указанием алгоритма
        model = ClassificationModel(name="test_model", algorithm="random_forest", n_estimators=50)
        self.assertEqual(model.algorithm, "random_forest")
        self.assertIsInstance(model.model, RandomForestClassifier)
        self.assertEqual(model.model.n_estimators, 50)
        
        # Тест с неподдерживаемым алгоритмом
        with self.assertRaises(ValueError):
            ClassificationModel(name="test_model", algorithm="unknown_algorithm")

    def test_train_binary_with_numpy(self):
        """Тест обучения бинарной классификационной модели с numpy массивами."""
        metrics = self.model.train(self.X_binary, self.y_binary)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.classes_)
        self.assertEqual(len(self.model.classes_), 2)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)

    def test_train_multi_with_numpy(self):
        """Тест обучения многоклассовой классификационной модели с numpy массивами."""
        metrics = self.model.train(self.X_multi, self.y_multi)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.classes_)
        self.assertEqual(len(self.model.classes_), 3)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)

    def test_train_with_pandas(self):
        """Тест обучения модели с pandas DataFrame и Series."""
        metrics = self.model.train(self.X_df, self.y_series)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)

    def test_predict_without_training(self):
        """Тест предсказания без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X_binary)

    def test_predict_with_numpy(self):
        """Тест предсказания с numpy массивом после обучения."""
        self.model.train(self.X_binary, self.y_binary)
        predictions = self.model.predict(self.X_binary)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_binary))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_predict_with_pandas(self):
        """Тест предсказания с pandas DataFrame после обучения."""
        self.model.train(self.X_binary, self.y_binary)
        predictions = self.model.predict(self.X_df)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X_df))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_predict_proba_without_training(self):
        """Тест предсказания вероятностей без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.predict_proba(self.X_binary)

    def test_predict_proba_with_numpy(self):
        """Тест предсказания вероятностей с numpy массивом после обучения."""
        self.model.train(self.X_binary, self.y_binary)
        probabilities = self.model.predict_proba(self.X_binary)
        
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, (len(self.X_binary), 2))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        self.assertTrue(np.allclose(np.sum(probabilities, axis=1), 1.0))

    def test_evaluate_without_training(self):
        """Тест оценки без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.evaluate(self.X_binary, self.y_binary)

    def test_evaluate_binary_with_numpy(self):
        """Тест оценки бинарной классификационной модели с numpy массивами."""
        self.model.train(self.X_binary, self.y_binary)
        metrics = self.model.evaluate(self.X_binary, self.y_binary)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('precision_class_0', metrics)
        self.assertIn('recall_class_0', metrics)
        self.assertIn('f1_class_0', metrics)
        self.assertIn('precision_class_1', metrics)
        self.assertIn('recall_class_1', metrics)
        self.assertIn('f1_class_1', metrics)

    def test_evaluate_multi_with_numpy(self):
        """Тест оценки многоклассовой классификационной модели с numpy массивами."""
        self.model.train(self.X_multi, self.y_multi)
        metrics = self.model.evaluate(self.X_multi, self.y_multi)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)

    def test_get_feature_importance_logistic(self):
        """Тест получения важности признаков для логистической регрессии."""
        self.model.train(self.X_binary, self.y_binary)
        importance = self.model.get_feature_importance()
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X_binary.shape[1])

    def test_get_feature_importance_random_forest(self):
        """Тест получения важности признаков для модели случайного леса."""
        model = ClassificationModel(name="rf_model", algorithm="random_forest")
        model.train(self.X_binary, self.y_binary)
        importance = model.get_feature_importance()
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), self.X_binary.shape[1])

    def test_get_feature_importance_without_training(self):
        """Тест получения важности признаков без предварительного обучения."""
        with self.assertRaises(RuntimeError):
            self.model.get_feature_importance()


if __name__ == '__main__':
    unittest.main() 