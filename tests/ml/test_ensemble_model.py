"""
Тесты для класса ансамблевой модели.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from ml.models.base_model import BaseModel
from ml.models.ensemble_model import EnsembleModel


class TestEnsembleModel(unittest.TestCase):
    """
    Тесты для класса ансамблевой модели.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем мок-модели для тестирования
        self.model1 = MagicMock(spec=BaseModel)
        self.model1.name = "model1"
        self.model1.is_trained = True
        self.model1.predict.return_value = np.array([1, 0, 1, 0])
        self.model1.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]])
        self.model1.evaluate.return_value = {"accuracy": 0.75, "f1": 0.8}
        self.model1.train.return_value = {"accuracy": 0.8, "f1": 0.85}
        self.model1.get_feature_importance.return_value = np.array([0.7, 0.3])
        
        self.model2 = MagicMock(spec=BaseModel)
        self.model2.name = "model2"
        self.model2.is_trained = True
        self.model2.predict.return_value = np.array([1, 1, 0, 0])
        self.model2.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
        self.model2.evaluate.return_value = {"accuracy": 0.7, "f1": 0.75}
        self.model2.train.return_value = {"accuracy": 0.75, "f1": 0.8}
        self.model2.get_feature_importance.return_value = np.array([0.4, 0.6])
        
        self.model3 = MagicMock(spec=BaseModel)
        self.model3.name = "model3"
        self.model3.is_trained = True
        self.model3.predict.return_value = np.array([0, 1, 1, 0])
        self.model3.predict_proba.return_value = np.array([[0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]])
        self.model3.evaluate.return_value = {"accuracy": 0.8, "f1": 0.85}
        self.model3.train.return_value = {"accuracy": 0.85, "f1": 0.9}
        self.model3.get_feature_importance.return_value = np.array([0.5, 0.5])
        
        # Создаем тестовые данные
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([1, 1, 0, 0])
        
        # Создаем ансамблевую модель
        self.ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2, self.model3],
            aggregation_method="voting"
        )
    
    def test_initialization(self):
        """
        Тест инициализации ансамблевой модели.
        """
        # Тест с параметрами по умолчанию
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2]
        )
        self.assertEqual(ensemble.name, "test_ensemble")
        self.assertEqual(ensemble.model_count, 2)
        self.assertEqual(ensemble.aggregation_method, "mean")
        self.assertIsNone(ensemble.weights)
        
        # Тест с указанием метода агрегации и весов
        weights = [0.7, 0.3]
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2],
            aggregation_method="voting",
            weights=weights
        )
        self.assertEqual(ensemble.aggregation_method, "voting")
        np.testing.assert_array_almost_equal(ensemble.weights, np.array([0.7, 0.3]))
        
        # Тест с пользовательской функцией агрегации
        custom_func = lambda predictions, weights: np.mean(predictions, axis=0)
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2],
            custom_aggregation_func=custom_func
        )
        self.assertEqual(ensemble.aggregation_method, "custom")
        self.assertEqual(ensemble.aggregation_func, custom_func)
        
        # Тест с неподдерживаемым методом агрегации
        with self.assertRaises(ValueError):
            EnsembleModel(
                name="test_ensemble",
                models=[self.model1, self.model2],
                aggregation_method="unknown"
            )
        
        # Тест с неправильным количеством весов
        with self.assertRaises(ValueError):
            EnsembleModel(
                name="test_ensemble",
                models=[self.model1, self.model2],
                weights=[0.5, 0.3, 0.2]
            )
        
        # Тест с пустым списком моделей
        with self.assertRaises(ValueError):
            EnsembleModel(
                name="test_ensemble",
                models=[]
            )
    
    def test_train(self):
        """
        Тест метода обучения ансамблевой модели.
        """
        # Мокаем метод evaluate, чтобы он не вызывал predict
        self.ensemble.evaluate = MagicMock(return_value={"accuracy": 0.8, "f1": 0.85})
        
        # Обучение ансамбля
        metrics = self.ensemble.train(self.X, self.y)
        
        # Проверка, что все модели были обучены
        self.model1.train.assert_called_once_with(self.X, self.y)
        self.model2.train.assert_called_once_with(self.X, self.y)
        self.model3.train.assert_called_once_with(self.X, self.y)
        
        # Проверка, что метрики содержат результаты для всех моделей и ансамбля
        self.assertIn("model_1", metrics)
        self.assertIn("model_2", metrics)
        self.assertIn("model_3", metrics)
        self.assertIn("ensemble", metrics)
        
        # Проверка, что ансамбль помечен как обученный
        self.assertTrue(self.ensemble.is_trained)
    
    def test_predict_voting(self):
        """
        Тест метода предсказания с методом агрегации 'voting'.
        """
        # Предсказания моделей:
        # model1: [1, 0, 1, 0]
        # model2: [1, 1, 0, 0]
        # model3: [0, 1, 1, 0]
        # Результат голосования: [1, 1, 1, 0]
        
        predictions = self.ensemble.predict(self.X)
        
        # Проверка, что все модели были вызваны для предсказания
        self.model1.predict.assert_called_once_with(self.X)
        self.model2.predict.assert_called_once_with(self.X)
        self.model3.predict.assert_called_once_with(self.X)
        
        # Проверка результата голосования
        np.testing.assert_array_equal(predictions, np.array([1, 1, 1, 0]))
    
    def test_predict_mean(self):
        """
        Тест метода предсказания с методом агрегации 'mean'.
        """
        # Предсказания моделей:
        # model1: [1, 0, 1, 0]
        # model2: [1, 1, 0, 0]
        # model3: [0, 1, 1, 0]
        # Среднее: [0.67, 0.67, 0.67, 0]
        # Округленное среднее: [1, 1, 1, 0]
        
        # Создаем ансамбль с методом агрегации 'mean'
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2, self.model3],
            aggregation_method="mean"
        )
        
        predictions = ensemble.predict(self.X)
        
        # Проверка, что все модели были вызваны для предсказания
        self.model1.predict.assert_called_with(self.X)
        self.model2.predict.assert_called_with(self.X)
        self.model3.predict.assert_called_with(self.X)
        
        # Проверка результата усреднения
        expected = np.array([2/3, 2/3, 2/3, 0])
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_predict_weighted(self):
        """
        Тест метода предсказания с весами.
        """
        # Предсказания моделей:
        # model1: [1, 0, 1, 0] с весом 0.5
        # model2: [1, 1, 0, 0] с весом 0.3
        # model3: [0, 1, 1, 0] с весом 0.2
        # Взвешенное среднее: [0.8, 0.5, 0.7, 0]
        # Округленное взвешенное среднее: [1, 1, 1, 0]
        
        # Создаем ансамбль с весами
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[self.model1, self.model2, self.model3],
            aggregation_method="mean",
            weights=[0.5, 0.3, 0.2]
        )
        
        predictions = ensemble.predict(self.X)
        
        # Проверка, что все модели были вызваны для предсказания
        self.model1.predict.assert_called_with(self.X)
        self.model2.predict.assert_called_with(self.X)
        self.model3.predict.assert_called_with(self.X)
        
        # Проверка результата взвешенного усреднения
        expected = np.array([0.8, 0.5, 0.7, 0])
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_predict_proba(self):
        """
        Тест метода предсказания вероятностей.
        """
        # Вероятности от моделей:
        # model1: [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4]]
        # model2: [[0.3, 0.7], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]
        # model3: [[0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.7, 0.3]]
        # Среднее: [[0.37, 0.63], [0.4, 0.6], [0.37, 0.63], [0.73, 0.27]]
        
        probabilities = self.ensemble.predict_proba(self.X)
        
        # Проверка, что все модели были вызваны для предсказания вероятностей
        self.model1.predict_proba.assert_called_once_with(self.X)
        self.model2.predict_proba.assert_called_once_with(self.X)
        self.model3.predict_proba.assert_called_once_with(self.X)
        
        # Проверка результата усреднения вероятностей
        expected = np.array([
            [0.37, 0.63],
            [0.4, 0.6],
            [0.37, 0.63],
            [0.73, 0.27]
        ])
        np.testing.assert_array_almost_equal(probabilities, expected, decimal=2)
    
    def test_evaluate(self):
        """
        Тест метода оценки ансамблевой модели.
        """
        # Мокаем метод _calculate_metrics первой модели
        self.model1._calculate_metrics = MagicMock(return_value={"accuracy": 0.85, "f1": 0.9})
        
        # Оценка ансамбля
        metrics = self.ensemble.evaluate(self.X, self.y)
        
        # Проверка, что метод _calculate_metrics был вызван
        self.model1._calculate_metrics.assert_called_once()
        
        # Проверка результатов оценки
        self.assertEqual(metrics["accuracy"], 0.85)
        self.assertEqual(metrics["f1"], 0.9)
    
    def test_get_feature_importance(self):
        """
        Тест метода получения важности признаков.
        """
        # Важности признаков от моделей:
        # model1: [0.7, 0.3]
        # model2: [0.4, 0.6]
        # model3: [0.5, 0.5]
        # Среднее: [0.53, 0.47]
        
        importance = self.ensemble.get_feature_importance()
        
        # Проверка, что все модели были вызваны для получения важности признаков
        self.model1.get_feature_importance.assert_called_once()
        self.model2.get_feature_importance.assert_called_once()
        self.model3.get_feature_importance.assert_called_once()
        
        # Проверка результата усреднения важностей
        expected = np.array([0.53, 0.47])
        np.testing.assert_array_almost_equal(importance, expected, decimal=2)
    
    @patch("os.makedirs")
    @patch("builtins.open", create=True)
    @patch("pickle.dump")
    @patch("json.dump")
    def test_save(self, mock_json_dump, mock_pickle_dump, mock_open, mock_makedirs):
        """
        Тест метода сохранения ансамблевой модели.
        """
        # Мокаем метод save базовых моделей
        self.model1.save.return_value = "path/to/model1.pkl"
        self.model2.save.return_value = "path/to/model2.pkl"
        self.model3.save.return_value = "path/to/model3.pkl"
        
        # Мокаем метод save базового класса
        with patch.object(BaseModel, 'save', return_value="path/to/ensemble.pkl"):
            # Сохранение ансамбля
            path = "path/to/ensemble.pkl"
            result = self.ensemble.save(path)
            
            # Проверка, что директория была создана
            mock_makedirs.assert_called_once()
            
            # Проверка, что все модели были сохранены
            self.model1.save.assert_called_once()
            self.model2.save.assert_called_once()
            self.model3.save.assert_called_once()
            
            # Проверка, что метаданные были обновлены
            self.assertIn("model_paths", self.ensemble.metadata)
            self.assertIn("aggregation_method", self.ensemble.metadata)
            
            # Проверка результата
            self.assertEqual(result, "path/to/ensemble.pkl")
    
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load(self, mock_json_load, mock_open, mock_exists):
        """
        Тест метода загрузки ансамблевой модели.
        """
        # Мокаем метод load базового класса
        with patch.object(BaseModel, 'load', return_value=True):
            # Мокаем существование файлов
            mock_exists.return_value = True
            
            # Мокаем загрузку метаданных
            mock_json_load.side_effect = [
                # Метаданные ансамбля
                {
                    "model_paths": ["path/to/model1.pkl", "path/to/model2.pkl"],
                    "aggregation_method": "voting",
                    "weights": [0.6, 0.4]
                },
                # Метаданные первой модели
                {"type": "RegressionModel", "name": "model1"},
                # Метаданные второй модели
                {"type": "ClassificationModel", "name": "model2"}
            ]
            
            # Мокаем импорт и создание моделей
            with patch("ml.models.regression_model.RegressionModel") as mock_regression:
                with patch("ml.models.classification_model.ClassificationModel") as mock_classification:
                    # Мокаем загрузку моделей
                    mock_regression.return_value.load.return_value = True
                    mock_classification.return_value.load.return_value = True
                    
                    # Загрузка ансамбля
                    path = "path/to/ensemble.pkl"
                    result = self.ensemble.load(path)
                    
                    # Проверка, что базовый метод load был вызван
                    # Проверка, что модели были созданы и загружены
                    mock_regression.assert_called_once_with(name="model1")
                    mock_classification.assert_called_once_with(name="model2")
                    mock_regression.return_value.load.assert_called_once_with("path/to/model1.pkl")
                    mock_classification.return_value.load.assert_called_once_with("path/to/model2.pkl")
                    
                    # Проверка, что метод агрегации и веса были установлены
                    self.assertEqual(self.ensemble.aggregation_method, "voting")
                    np.testing.assert_array_equal(self.ensemble.weights, np.array([0.6, 0.4]))
                    
                    # Проверка результата
                    self.assertTrue(result)


if __name__ == "__main__":
    unittest.main() 