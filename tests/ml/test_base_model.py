"""
Тесты для базового класса моделей машинного обучения.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from ml.models.base_model import BaseModel


class TestBaseModel(unittest.TestCase):
    """
    Тесты для базового класса моделей машинного обучения.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем конкретную реализацию абстрактного класса для тестирования
        class ConcreteModel(BaseModel):
            def train(self, X, y, **kwargs):
                self.is_trained = True
                self.model = MagicMock()
                return {"metric1": 0.95, "metric2": 0.85}
            
            def predict(self, X):
                return np.array([1, 0, 1])
            
            def evaluate(self, X, y):
                return {"metric1": 0.9, "metric2": 0.8}
        
        self.model_class = ConcreteModel
        self.model = self.model_class(name="test_model", version="1.0.0")
    
    def test_initialization(self):
        """
        Тест инициализации модели.
        """
        model = self.model_class(name="test_model", version="2.0.0", param1=10, param2="value")
        
        self.assertEqual(model.name, "test_model")
        self.assertEqual(model.version, "2.0.0")
        self.assertEqual(model.params, {"param1": 10, "param2": "value"})
        self.assertFalse(model.is_trained)
        self.assertIsNone(model.model)
    
    def test_get_params(self):
        """
        Тест получения параметров модели.
        """
        model = self.model_class(name="test_model", param1=10, param2="value")
        
        params = model.get_params()
        self.assertEqual(params, {"param1": 10, "param2": "value"})
    
    def test_set_params(self):
        """
        Тест установки параметров модели.
        """
        model = self.model_class(name="test_model", param1=10)
        
        model.set_params(param1=20, param2="new_value")
        
        params = model.get_params()
        self.assertEqual(params, {"param1": 20, "param2": "new_value"})
    
    def test_get_metadata(self):
        """
        Тест получения метаданных модели.
        """
        model = self.model_class(name="test_model", version="1.0.0", param1=10)
        
        metadata = model.get_metadata()
        
        self.assertIn("name", metadata)
        self.assertIn("version", metadata)
        self.assertIn("is_trained", metadata)
        self.assertIn("params", metadata)
        self.assertIn("model_type", metadata)
        
        self.assertEqual(metadata["name"], "test_model")
        self.assertEqual(metadata["version"], "1.0.0")
        self.assertEqual(metadata["is_trained"], False)
        self.assertEqual(metadata["params"], {"param1": 10})
        self.assertEqual(metadata["model_type"], "ConcreteModel")
    
    def test_str_representation(self):
        """
        Тест строкового представления модели.
        """
        model = self.model_class(name="test_model", version="1.0.0")
        
        str_repr = str(model)
        
        self.assertIn("ConcreteModel", str_repr)
        self.assertIn("test_model", str_repr)
        self.assertIn("1.0.0", str_repr)
        self.assertIn("trained=False", str_repr)
    
    def test_train_method(self):
        """
        Тест метода обучения модели.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        metrics = self.model.train(X, y)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
        self.assertEqual(metrics, {"metric1": 0.95, "metric2": 0.85})
    
    def test_predict_method(self):
        """
        Тест метода предсказания.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        predictions = self.model.predict(X)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.tolist(), [1, 0, 1])
    
    def test_evaluate_method(self):
        """
        Тест метода оценки модели.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        metrics = self.model.evaluate(X, y)
        
        self.assertEqual(metrics, {"metric1": 0.9, "metric2": 0.8})
    
    @patch("pickle.dump")
    @patch("json.dump")
    @patch("builtins.open", create=True)
    def test_save_method(self, mock_open, mock_json_dump, mock_pickle_dump):
        """
        Тест метода сохранения модели.
        """
        # Настройка мока для имитации обученной модели
        self.model.is_trained = True
        self.model.model = MagicMock()
        
        # Вызов метода сохранения
        path = "models/test_model.pkl"
        result = self.model.save(path)
        
        # Проверка результата
        self.assertEqual(result, path)
        
        # Проверка вызовов функций
        self.assertEqual(mock_open.call_count, 2)  # Один раз для модели, один раз для метаданных
        self.assertEqual(mock_pickle_dump.call_count, 1)
        self.assertEqual(mock_json_dump.call_count, 1)
    
    @patch("pickle.load")
    @patch("json.load")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    def test_load_method(self, mock_open, mock_exists, mock_json_load, mock_pickle_load):
        """
        Тест метода загрузки модели.
        """
        # Настройка моков
        mock_exists.return_value = True
        mock_pickle_load.return_value = MagicMock()
        mock_json_load.return_value = {"name": "test_model", "version": "1.0.0"}
        
        # Вызов метода загрузки
        path = "models/test_model.pkl"
        result = self.model.load(path)
        
        # Проверка результата
        self.assertTrue(result)
        self.assertIsNotNone(self.model.model)
        
        # Проверка вызовов функций
        self.assertEqual(mock_open.call_count, 2)  # Один раз для модели, один раз для метаданных
        self.assertEqual(mock_pickle_load.call_count, 1)
        self.assertEqual(mock_json_load.call_count, 1)


if __name__ == "__main__":
    unittest.main() 