"""
Тесты для модуля валидации моделей.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

from ml.training.model_validator import ModelValidator
from ml.models.base_model import BaseModel


class TestModelValidator(unittest.TestCase):
    """
    Тесты для класса ModelValidator.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем временную директорию для отчетов
        self.test_reports_dir = "test_reports"
        
        # Конфигурация для тестов
        self.config = {
            "reports_dir": self.test_reports_dir,
            "save_reports": False,  # Отключаем сохранение отчетов для тестов
            "visualization": False,  # Отключаем визуализацию для тестов
            "visualizer_config": {
                "plots_dir": "test_plots"
            }
        }
        
        # Создаем валидатор
        self.validator = ModelValidator(self.config)
        
        # Создаем тестовые данные
        np.random.seed(42)
        self.X = np.random.rand(100, 5)
        
        # Данные для классификации
        self.y_cls = np.random.randint(0, 2, 100)
        
        # Данные для регрессии
        self.y_reg = np.random.rand(100)
        
        # Мок-модель для классификации
        self.cls_model = MagicMock(spec=BaseModel)
        self.cls_model.name = "test_classification_model"
        self.cls_model.predict = MagicMock(return_value=np.random.randint(0, 2, 100))
        self.cls_model.predict_proba = MagicMock(return_value=np.random.rand(100, 2))
        
        # Мок-модель для регрессии
        self.reg_model = MagicMock(spec=BaseModel)
        self.reg_model.name = "test_regression_model"
        self.reg_model.predict = MagicMock(return_value=np.random.rand(100))
    
    def tearDown(self):
        """
        Очистка после тестов.
        """
        # Удаляем временную директорию, если она существует
        if os.path.exists(self.test_reports_dir):
            import shutil
            shutil.rmtree(self.test_reports_dir)
        
        if os.path.exists("test_plots"):
            import shutil
            shutil.rmtree("test_plots")
    
    def test_initialization(self):
        """
        Тест инициализации валидатора моделей.
        """
        # Проверка параметров по умолчанию
        validator = ModelValidator()
        self.assertEqual(validator.reports_dir, "reports/model_validation")
        self.assertTrue(validator.save_reports)
        self.assertTrue(validator.visualization)
        
        # Проверка пользовательской конфигурации
        self.assertEqual(self.validator.reports_dir, self.test_reports_dir)
        self.assertFalse(self.validator.save_reports)
        self.assertFalse(self.validator.visualization)
    
    @patch('os.makedirs')
    def test_initialization_creates_directory(self, mock_makedirs):
        """
        Тест создания директории при инициализации.
        """
        # Сбрасываем счетчик вызовов перед тестом
        mock_makedirs.reset_mock()
        
        config = {
            "reports_dir": "test_dir",
            "save_reports": True
        }
        
        validator = ModelValidator(config)
        
        # Проверяем, что вызов был хотя бы один раз с правильными параметрами
        mock_makedirs.assert_any_call("test_dir", exist_ok=True)
        self.assertTrue(mock_makedirs.called)
    
    def test_validate_classification(self):
        """
        Тест валидации модели классификации.
        """
        # Запуск валидации
        report = self.validator.validate(
            self.cls_model, 
            self.X, 
            self.y_cls, 
            task_type="classification", 
            dataset_name="test_cls"
        )
        
        # Проверка вызова методов модели
        self.cls_model.predict.assert_called_once()
        self.cls_model.predict_proba.assert_called_once()
        
        # Проверка структуры отчета
        self.assertEqual(report["model_name"], "test_classification_model")
        self.assertEqual(report["dataset_name"], "test_cls")
        self.assertEqual(report["task_type"], "classification")
        self.assertIn("metrics", report)
        self.assertIn("timestamp", report)
        self.assertEqual(report["data_shape"], self.X.shape)
        
        # Проверка метрик
        metrics = report["metrics"]
        self.assertIn("accuracy", metrics)
        
        # Для бинарной классификации
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
    
    def test_validate_regression(self):
        """
        Тест валидации модели регрессии.
        """
        # Запуск валидации
        report = self.validator.validate(
            self.reg_model, 
            self.X, 
            self.y_reg, 
            task_type="regression", 
            dataset_name="test_reg"
        )
        
        # Проверка вызова методов модели
        self.reg_model.predict.assert_called_once()
        
        # Проверка структуры отчета
        self.assertEqual(report["model_name"], "test_regression_model")
        self.assertEqual(report["dataset_name"], "test_reg")
        self.assertEqual(report["task_type"], "regression")
        self.assertIn("metrics", report)
        self.assertIn("timestamp", report)
        self.assertEqual(report["data_shape"], self.X.shape)
        
        # Проверка метрик
        metrics = report["metrics"]
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
    
    def test_validate_auto_task_type(self):
        """
        Тест автоматического определения типа задачи.
        """
        # Для классификации
        self.cls_model.task_type = None
        report_cls = self.validator.validate(
            self.cls_model, 
            self.X, 
            self.y_cls, 
            task_type=None, 
            dataset_name="auto_cls"
        )
        self.assertEqual(report_cls["task_type"], "classification")
        
        # Для регрессии
        self.reg_model.task_type = None
        report_reg = self.validator.validate(
            self.reg_model, 
            self.X, 
            self.y_reg, 
            task_type=None, 
            dataset_name="auto_reg"
        )
        self.assertEqual(report_reg["task_type"], "regression")
        
        # Если модель имеет атрибут task_type
        self.cls_model.task_type = "classification"
        report_with_attr = self.validator.validate(
            self.cls_model, 
            self.X, 
            self.y_cls, 
            task_type=None, 
            dataset_name="with_attr"
        )
        self.assertEqual(report_with_attr["task_type"], "classification")
    
    def test_compare_models(self):
        """
        Тест сравнения моделей.
        """
        # Создаем несколько моделей для сравнения
        models = [self.cls_model, MagicMock(spec=BaseModel)]
        models[1].name = "test_classification_model_2"
        models[1].predict = MagicMock(return_value=np.random.randint(0, 2, 100))
        models[1].predict_proba = MagicMock(return_value=np.random.rand(100, 2))
        
        # Запуск сравнения
        comparison_report = self.validator.compare_models(
            models, 
            self.X, 
            self.y_cls, 
            task_type="classification", 
            dataset_name="comparison_test"
        )
        
        # Проверка структуры отчета
        self.assertEqual(comparison_report["dataset_name"], "comparison_test")
        self.assertEqual(comparison_report["task_type"], "classification")
        self.assertEqual(len(comparison_report["models"]), 2)
        self.assertIn("metrics_comparison", comparison_report)
        self.assertIn("timestamp", comparison_report)
        
        # Проверка метрик сравнения
        metrics_comparison = comparison_report["metrics_comparison"]
        self.assertIn("test_classification_model", metrics_comparison)
        self.assertIn("test_classification_model_2", metrics_comparison)
        
        # Проверка метрик для каждой модели
        for model_name in metrics_comparison:
            self.assertIn("accuracy", metrics_comparison[model_name])
            self.assertIn("precision", metrics_comparison[model_name])
            self.assertIn("recall", metrics_comparison[model_name])
            self.assertIn("f1", metrics_comparison[model_name])
    
    def test_calculate_metrics_classification(self):
        """
        Тест расчета метрик для задачи классификации.
        """
        # Бинарная классификация
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
        
        metrics = self.validator._calculate_metrics(y_true, y_pred, y_proba, "classification")
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        
        # Многоклассовая классификация
        y_true_multi = np.array([0, 1, 2, 1, 0])
        y_pred_multi = np.array([0, 1, 1, 1, 0])
        
        metrics_multi = self.validator._calculate_metrics(y_true_multi, y_pred_multi, None, "classification")
        
        self.assertIn("accuracy", metrics_multi)
        self.assertIn("precision_macro", metrics_multi)
        self.assertIn("recall_macro", metrics_multi)
        self.assertIn("f1_macro", metrics_multi)
        self.assertIn("precision_weighted", metrics_multi)
        self.assertIn("recall_weighted", metrics_multi)
        self.assertIn("f1_weighted", metrics_multi)
    
    def test_calculate_metrics_regression(self):
        """
        Тест расчета метрик для задачи регрессии.
        """
        y_true = np.array([3.0, 1.5, 2.0, 3.5, 0.5])
        y_pred = np.array([2.8, 1.2, 2.5, 3.6, 0.8])
        
        metrics = self.validator._calculate_metrics(y_true, y_pred, None, "regression")
        
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("mape", metrics)
        
        # Проверка расчета MAPE с нулевыми значениями
        y_true_with_zeros = np.array([3.0, 0.0, 2.0, 0.0, 0.5])
        y_pred_with_zeros = np.array([2.8, 0.2, 2.5, 0.1, 0.8])
        
        metrics_with_zeros = self.validator._calculate_metrics(y_true_with_zeros, y_pred_with_zeros, None, "regression")
        
        self.assertIn("mape", metrics_with_zeros)
    
    @patch('ml.training.visualization.ModelVisualizer.plot_confusion_matrix')
    @patch('ml.training.visualization.ModelVisualizer.plot_roc_curve')
    @patch('ml.training.visualization.ModelVisualizer.plot_precision_recall_curve')
    def test_visualize_results_classification(self, mock_pr_curve, mock_roc_curve, mock_confusion_matrix):
        """
        Тест визуализации результатов для задачи классификации.
        """
        # Включаем визуализацию
        self.validator.visualization = True
        
        # Устанавливаем возвращаемые значения для моков
        mock_confusion_matrix.return_value = "path/to/confusion_matrix.png"
        mock_roc_curve.return_value = "path/to/roc_curve.png"
        mock_pr_curve.return_value = "path/to/pr_curve.png"
        
        # Бинарная классификация
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
        
        plots = self.validator._visualize_results(
            y_true, y_pred, y_proba, "classification", "test_model", "test_dataset"
        )
        
        # Проверка вызовов методов визуализатора
        mock_confusion_matrix.assert_called_once_with(
            y_true, y_pred, "test_model", "test_dataset"
        )
        mock_roc_curve.assert_called_once_with(
            y_true, y_proba, "test_model", "test_dataset"
        )
        mock_pr_curve.assert_called_once_with(
            y_true, y_proba, "test_model", "test_dataset"
        )
        
        # Проверка структуры результата
        self.assertIn("confusion_matrix", plots)
        self.assertIn("roc_curve", plots)
        self.assertIn("pr_curve", plots)
        
        # Проверка путей к графикам
        self.assertEqual(plots["confusion_matrix"], "path/to/confusion_matrix.png")
        self.assertEqual(plots["roc_curve"], "path/to/roc_curve.png")
        self.assertEqual(plots["pr_curve"], "path/to/pr_curve.png")
    
    @patch('ml.training.visualization.ModelVisualizer.plot_regression_predictions')
    @patch('ml.training.visualization.ModelVisualizer.plot_residuals')
    @patch('ml.training.visualization.ModelVisualizer.plot_residuals_histogram')
    def test_visualize_results_regression(self, mock_residuals_hist, mock_residuals, mock_predictions):
        """
        Тест визуализации результатов для задачи регрессии.
        """
        # Включаем визуализацию
        self.validator.visualization = True
        
        # Устанавливаем возвращаемые значения для моков
        mock_predictions.return_value = "path/to/prediction_comparison.png"
        mock_residuals.return_value = "path/to/residuals.png"
        mock_residuals_hist.return_value = "path/to/residuals_hist.png"
        
        # Регрессия
        y_true = np.array([3.0, 1.5, 2.0, 3.5, 0.5])
        y_pred = np.array([2.8, 1.2, 2.5, 3.6, 0.8])
        
        plots = self.validator._visualize_results(
            y_true, y_pred, None, "regression", "test_model", "test_dataset"
        )
        
        # Проверка вызовов методов визуализатора
        mock_predictions.assert_called_once_with(
            y_true, y_pred, "test_model", "test_dataset"
        )
        mock_residuals.assert_called_once_with(
            y_true, y_pred, "test_model", "test_dataset"
        )
        mock_residuals_hist.assert_called_once_with(
            y_true, y_pred, "test_model", "test_dataset"
        )
        
        # Проверка структуры результата
        self.assertIn("prediction_comparison", plots)
        self.assertIn("residuals", plots)
        self.assertIn("residuals_hist", plots)
        
        # Проверка путей к графикам
        self.assertEqual(plots["prediction_comparison"], "path/to/prediction_comparison.png")
        self.assertEqual(plots["residuals"], "path/to/residuals.png")
        self.assertEqual(plots["residuals_hist"], "path/to/residuals_hist.png")
    
    @patch('ml.training.visualization.ModelVisualizer.plot_metrics_comparison')
    @patch('ml.training.visualization.ModelVisualizer.plot_multiple_metrics_comparison')
    def test_visualize_comparison(self, mock_multiple_metrics, mock_metrics_comparison):
        """
        Тест визуализации сравнения моделей.
        """
        # Включаем визуализацию
        self.validator.visualization = True
        
        # Устанавливаем возвращаемые значения для моков
        mock_metrics_comparison.return_value = "path/to/metric_comparison.png"
        mock_multiple_metrics.return_value = "path/to/multiple_metrics_comparison.png"
        
        # Создаем тестовые отчеты
        model_reports = [
            {
                "model_name": "model1",
                "dataset_name": "test_dataset",
                "task_type": "classification",
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.8,
                    "recall": 0.75,
                    "f1": 0.77
                }
            },
            {
                "model_name": "model2",
                "dataset_name": "test_dataset",
                "task_type": "classification",
                "metrics": {
                    "accuracy": 0.88,
                    "precision": 0.83,
                    "recall": 0.79,
                    "f1": 0.81
                }
            }
        ]
        
        plots = self.validator._visualize_comparison(
            model_reports, "classification", "test_dataset"
        )
        
        # Проверка вызовов методов визуализатора
        self.assertEqual(mock_metrics_comparison.call_count, 4)  # 4 метрики
        mock_multiple_metrics.assert_called_once()
        
        # Проверка структуры результата
        self.assertIn("accuracy_comparison", plots)
        self.assertIn("precision_comparison", plots)
        self.assertIn("recall_comparison", plots)
        self.assertIn("f1_comparison", plots)
        self.assertIn("multiple_metrics_comparison", plots)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_report(self, mock_json_dump, mock_file_open):
        """
        Тест сохранения отчета.
        """
        # Включаем сохранение отчетов
        self.validator.save_reports = True
        
        # Создаем тестовый отчет
        report = {
            "model_name": "test_model",
            "dataset_name": "test_dataset",
            "task_type": "classification",
            "metrics": {"accuracy": 0.8},
            "timestamp": datetime.now().isoformat(),
            "plots": {"confusion_matrix": "path/to/cm.png"}
        }
        
        # Сохраняем отчет
        filepath = self.validator._save_report(report, "test_model", "test_dataset")
        
        # Проверка вызова open
        mock_file_open.assert_called_once()
        
        # Проверка вызова json.dump
        mock_json_dump.assert_called_once()
        
        # Проверка, что plots не включены в сохраненный отчет
        args, kwargs = mock_json_dump.call_args
        saved_report = args[0]
        self.assertNotIn("plots", saved_report)
    
    def test_update_config(self):
        """
        Тест обновления конфигурации.
        """
        # Исходная конфигурация
        self.assertFalse(self.validator.save_reports)
        self.assertFalse(self.validator.visualization)
        self.assertEqual(self.validator.reports_dir, self.test_reports_dir)
        
        # Обновляем конфигурацию
        new_config = {
            "save_reports": True,
            "visualization": True,
            "reports_dir": "new_reports_dir",
            "visualizer_config": {
                "dpi": 300,
                "figsize": (12, 10)
            }
        }
        
        with patch('os.makedirs'):
            self.validator.update_config(new_config)
        
        # Проверка обновления параметров
        self.assertTrue(self.validator.save_reports)
        self.assertTrue(self.validator.visualization)
        self.assertEqual(self.validator.reports_dir, "new_reports_dir")
        
        # Проверка обновления параметров визуализатора
        self.assertEqual(self.validator.visualizer.dpi, 300)
        self.assertEqual(self.validator.visualizer.figsize, (12, 10))


if __name__ == "__main__":
    unittest.main() 