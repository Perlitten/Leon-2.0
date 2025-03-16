"""
Тесты для модуля обучения моделей.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import os
import json
import pickle
import tensorflow as tf
from datetime import datetime

from ml.training.trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """
    Тесты для класса ModelTrainer.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем временную директорию для моделей
        self.test_models_dir = "test_models"
        
        # Конфигурация для тестов
        self.config = {
            "models_dir": self.test_models_dir,
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
            "batch_size": 32,
            "epochs": 2,  # Уменьшаем для ускорения тестов
            "early_stopping": False,  # Отключаем для тестов
            "patience": 5
        }
        
        # Создаем тренер
        self.trainer = ModelTrainer(self.config)
        
        # Создаем тестовые данные
        np.random.seed(42)
        
        # Данные для классификации
        self.X_cls = np.random.rand(100, 5)
        self.y_cls = np.random.randint(0, 2, 100)
        
        # Данные для регрессии
        self.X_reg = np.random.rand(100, 5)
        self.y_reg = np.random.rand(100)
        
        # Данные для LSTM
        self.X_lstm = np.random.rand(100, 10, 5)
        self.y_lstm = np.random.randint(0, 2, 100)
    
    def tearDown(self):
        """
        Очистка после тестов.
        """
        # Удаляем временную директорию, если она существует
        if os.path.exists(self.test_models_dir):
            import shutil
            shutil.rmtree(self.test_models_dir)
    
    def test_initialization(self):
        """
        Тест инициализации тренера моделей.
        """
        # Проверка параметров по умолчанию
        trainer = ModelTrainer()
        self.assertEqual(trainer.models_dir, "ml/models")
        self.assertEqual(trainer.task_type, "classification")
        self.assertEqual(trainer.test_size, 0.2)
        self.assertEqual(trainer.random_state, 42)
        self.assertEqual(trainer.batch_size, 32)
        self.assertEqual(trainer.epochs, 50)
        self.assertTrue(trainer.early_stopping)
        self.assertEqual(trainer.patience, 10)
        
        # Проверка пользовательской конфигурации
        self.assertEqual(self.trainer.models_dir, self.test_models_dir)
        self.assertEqual(self.trainer.task_type, "classification")
        self.assertEqual(self.trainer.test_size, 0.2)
        self.assertEqual(self.trainer.random_state, 42)
        self.assertEqual(self.trainer.batch_size, 32)
        self.assertEqual(self.trainer.epochs, 2)
        self.assertFalse(self.trainer.early_stopping)
        self.assertEqual(self.trainer.patience, 5)
    
    @patch('os.makedirs')
    def test_initialization_creates_directory(self, mock_makedirs):
        """
        Тест создания директории при инициализации.
        """
        # Сбрасываем счетчик вызовов перед тестом
        mock_makedirs.reset_mock()
        
        config = {
            "models_dir": "test_dir"
        }
        
        trainer = ModelTrainer(config)
        
        # Проверяем, что вызов был с правильными параметрами
        mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
    
    def test_prepare_data(self):
        """
        Тест подготовки данных для обучения.
        """
        X_train, X_val, y_train, y_val = self.trainer._prepare_data(self.X_cls, self.y_cls)
        
        # Проверка размеров выборок
        self.assertEqual(X_train.shape[0], 80)  # 80% данных для обучения
        self.assertEqual(X_val.shape[0], 20)    # 20% данных для валидации
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_val.shape[0], 20)
        
        # Проверка типов данных
        self.assertEqual(X_train.dtype, np.float64)
        # Тип может быть int32 или int64 в зависимости от реализации
        self.assertTrue(np.issubdtype(y_train.dtype, np.integer))
    
    @patch('ml.training.trainer.ModelTrainer._create_lstm_model')
    @patch('ml.training.trainer.ModelTrainer._create_cnn_model')
    @patch('ml.training.trainer.ModelTrainer._create_mlp_model')
    @patch('ml.training.trainer.ModelTrainer._create_xgboost_model')
    @patch('ml.training.trainer.ModelTrainer._create_random_forest_model')
    def test_create_model(self, mock_rf, mock_xgb, mock_mlp, mock_cnn, mock_lstm):
        """
        Тест создания модели.
        """
        # Настраиваем моки
        mock_lstm.return_value = MagicMock()
        mock_cnn.return_value = MagicMock()
        mock_mlp.return_value = MagicMock()
        mock_xgb.return_value = MagicMock()
        mock_rf.return_value = MagicMock()
        
        # Параметры модели
        model_params = {"param1": "value1", "param2": "value2"}
        
        # Тестируем создание разных типов моделей
        self.trainer._create_model("lstm", self.X_lstm.shape, model_params)
        mock_lstm.assert_called_once_with(self.X_lstm.shape, model_params)
        
        self.trainer._create_model("cnn", self.X_cls.shape, model_params)
        mock_cnn.assert_called_once_with(self.X_cls.shape, model_params)
        
        self.trainer._create_model("mlp", self.X_cls.shape, model_params)
        mock_mlp.assert_called_once_with(self.X_cls.shape, model_params)
        
        self.trainer._create_model("xgboost", self.X_cls.shape, model_params)
        mock_xgb.assert_called_once_with(model_params)
        
        self.trainer._create_model("random_forest", self.X_cls.shape, model_params)
        mock_rf.assert_called_once_with(model_params)
        
        # Тестируем неизвестный тип модели
        with self.assertRaises(ValueError):
            self.trainer._create_model("unknown_model", self.X_cls.shape, model_params)
    
    @patch('tensorflow.keras.callbacks.EarlyStopping')
    def test_train_tf_model(self, mock_early_stopping):
        """
        Тест обучения TensorFlow модели.
        """
        # Мокаем модель
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value=MagicMock())
        
        # Мокаем EarlyStopping
        mock_early_stopping.return_value = MagicMock()
        
        # Включаем раннюю остановку
        self.trainer.early_stopping = True
        
        # Обучаем модель
        history = self.trainer._train_tf_model(
            mock_model, self.X_cls, self.y_cls, self.X_cls, self.y_cls
        )
        
        # Проверяем, что fit был вызван
        mock_model.fit.assert_called_once()
        
        # Проверяем, что EarlyStopping был вызван
        mock_early_stopping.assert_called_once()
    
    def test_evaluate_model_classification(self):
        """
        Тест оценки модели для задачи классификации.
        """
        # Создаем мок-модель
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([0, 1, 0, 1, 0]))
        
        # Истинные значения
        y_true = np.array([0, 1, 1, 1, 0])
        
        # Оцениваем модель
        self.trainer.task_type = "classification"
        metrics = self.trainer._evaluate_model(mock_model, np.array([[0], [1], [2], [3], [4]]), y_true)
        
        # Проверяем, что predict был вызван
        mock_model.predict.assert_called_once()
        
        # Проверяем наличие основных метрик
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)
    
    def test_evaluate_model_regression(self):
        """
        Тест оценки модели для задачи регрессии.
        """
        # Создаем мок-модель
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        
        # Истинные значения
        y_true = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        
        # Оцениваем модель
        self.trainer.task_type = "regression"
        metrics = self.trainer._evaluate_model(mock_model, np.array([[0], [1], [2], [3], [4]]), y_true)
        
        # Проверяем, что predict был вызван
        mock_model.predict.assert_called_once()
        
        # Проверяем наличие основных метрик
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)
    
    def test_create_metadata(self):
        """
        Тест создания метаданных модели.
        """
        # Метрики
        metrics = {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.75,
            "f1_score": 0.77
        }
        
        # История обучения
        training_history = {
            "loss": [0.5, 0.4, 0.3],
            "accuracy": [0.7, 0.8, 0.85],
            "val_loss": [0.6, 0.5, 0.4],
            "val_accuracy": [0.65, 0.75, 0.8]
        }
        
        # Дополнительные метаданные
        additional_metadata = {
            "feature_names": ["feature1", "feature2", "feature3"],
            "target_name": "target",
            "preprocessing": {
                "scaling": "standard",
                "encoding": "one-hot"
            }
        }
        
        # Создаем метаданные
        metadata = self.trainer._create_metadata(
            "test_model_123", "mlp", metrics, training_history, additional_metadata
        )
        
        # Проверяем основные поля метаданных
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["model_id"], "test_model_123")
        self.assertEqual(metadata["model_type"], "mlp")
        self.assertEqual(metadata["task_type"], "classification")
        self.assertIn("metrics", metadata)
        self.assertIn("training_history", metadata)
    
    @patch('pickle.dump')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_model(self, mock_file_open, mock_json_dump, mock_pickle_dump):
        """
        Тест сохранения модели.
        """
        # Создаем мок-модель
        mock_model = MagicMock()
        
        # Метаданные
        metadata = {
            "model_id": "test_model_123",
            "model_type": "mlp",
            "task_type": "classification",
            "metrics": {
                "accuracy": 0.85
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Сохраняем модель
        model_path = self.trainer._save_model(mock_model, "test_model_123", metadata)
        
        # Проверяем, что open был вызван дважды (для модели и метаданных)
        self.assertEqual(mock_file_open.call_count, 2)
        
        # Проверяем, что pickle.dump был вызван
        mock_pickle_dump.assert_called_once()
        
        # Проверяем, что json.dump был вызван
        mock_json_dump.assert_called_once()
        
        # Проверяем, что путь к модели содержит имя модели
        self.assertIn("test_model_123", model_path)
    
    def test_update_config(self):
        """
        Тест обновления конфигурации.
        """
        # Исходная конфигурация
        self.assertEqual(self.trainer.models_dir, self.test_models_dir)
        self.assertEqual(self.trainer.batch_size, 32)
        self.assertEqual(self.trainer.epochs, 2)
        
        # Обновляем конфигурацию
        new_config = {
            "models_dir": "new_models_dir",
            "batch_size": 64,
            "epochs": 100
        }
        
        with patch('os.makedirs'):
            self.trainer.update_config(new_config)
        
        # Проверка обновления параметров
        self.assertEqual(self.trainer.models_dir, "new_models_dir")
        self.assertEqual(self.trainer.batch_size, 64)
        self.assertEqual(self.trainer.epochs, 100)
    
    @patch('ml.training.trainer.ModelTrainer._create_model')
    @patch('ml.training.trainer.ModelTrainer._prepare_data')
    @patch('ml.training.trainer.ModelTrainer._evaluate_model')
    @patch('ml.training.trainer.ModelTrainer._create_metadata')
    @patch('ml.training.trainer.ModelTrainer._save_model')
    def test_train(self, mock_save_model, mock_create_metadata, mock_evaluate_model, 
                  mock_prepare_data, mock_create_model):
        """
        Тест метода train.
        """
        # Мокаем внутренние методы
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_prepare_data.return_value = (self.X_cls, self.X_cls, self.y_cls, self.y_cls)
        mock_evaluate_model.return_value = {"accuracy": 0.85}
        mock_create_metadata.return_value = {"model_id": "test_model_123"}
        mock_save_model.return_value = "path/to/model.pkl"
        
        # Обучаем модель
        results = self.trainer.train("mlp", self.X_cls, self.y_cls)
        
        # Проверяем, что внутренние методы были вызваны
        mock_prepare_data.assert_called_once()
        mock_create_model.assert_called_once()
        mock_evaluate_model.assert_called_once()
        mock_create_metadata.assert_called_once()
        mock_save_model.assert_called_once()
        
        # Проверяем структуру результатов
        self.assertIn("model_id", results)
        self.assertIn("model_path", results)
        self.assertIn("metrics", results)
        self.assertIn("metadata", results)


if __name__ == "__main__":
    unittest.main() 