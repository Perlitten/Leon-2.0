"""
Тесты для модуля визуализации моделей.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from ml.training.visualization import ModelVisualizer


class TestModelVisualizer(unittest.TestCase):
    """
    Тесты для класса ModelVisualizer.
    """
    
    def setUp(self):
        """
        Настройка тестового окружения.
        """
        # Создаем временную директорию для графиков
        self.test_plots_dir = "test_plots"
        
        # Конфигурация для тестов
        self.config = {
            "plots_dir": self.test_plots_dir,
            "dpi": 72,
            "figsize": (8, 6)
        }
        
        # Создаем визуализатор
        self.visualizer = ModelVisualizer(self.config)
        
        # Создаем тестовые данные
        np.random.seed(42)
        
        # Данные для классификации
        self.y_true_cls = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred_cls = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        self.y_proba_cls = np.random.rand(10, 2)
        self.y_proba_cls = self.y_proba_cls / self.y_proba_cls.sum(axis=1)[:, np.newaxis]
        
        # Данные для регрессии
        self.y_true_reg = np.array([3.0, 1.5, 2.0, 3.5, 0.5, 2.5, 1.0, 3.0, 2.0, 1.5])
        self.y_pred_reg = np.array([2.8, 1.2, 2.5, 3.6, 0.8, 2.3, 1.1, 2.9, 1.8, 1.6])
        
        # Данные для важности признаков
        self.feature_importance = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
        self.feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
        
        # Данные для кривой обучения
        self.train_sizes = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.train_scores = np.array([[0.8, 0.82], [0.85, 0.86], [0.88, 0.89], [0.9, 0.91], [0.92, 0.93]])
        self.test_scores = np.array([[0.7, 0.71], [0.75, 0.76], [0.78, 0.79], [0.8, 0.81], [0.82, 0.83]])
        
        # Данные для сравнения метрик
        self.metrics_dict = {
            "model1": {"accuracy": 0.85, "precision": 0.8, "recall": 0.75, "f1": 0.77},
            "model2": {"accuracy": 0.88, "precision": 0.83, "recall": 0.79, "f1": 0.81},
            "model3": {"accuracy": 0.82, "precision": 0.78, "recall": 0.73, "f1": 0.75}
        }
        self.model_names = ["model1", "model2", "model3"]
        self.metric_names = ["accuracy", "precision", "recall", "f1"]
    
    def tearDown(self):
        """
        Очистка после тестов.
        """
        # Удаляем временную директорию, если она существует
        if os.path.exists(self.test_plots_dir):
            import shutil
            shutil.rmtree(self.test_plots_dir)
    
    def test_initialization(self):
        """
        Тест инициализации визуализатора моделей.
        """
        # Проверка параметров по умолчанию
        visualizer = ModelVisualizer()
        self.assertEqual(visualizer.plots_dir, "reports/plots")
        self.assertEqual(visualizer.dpi, 100)
        self.assertEqual(visualizer.figsize, (10, 8))
        
        # Проверка пользовательской конфигурации
        self.assertEqual(self.visualizer.plots_dir, self.test_plots_dir)
        self.assertEqual(self.visualizer.dpi, 72)
        self.assertEqual(self.visualizer.figsize, (8, 6))
    
    @patch('os.makedirs')
    def test_initialization_creates_directory(self, mock_makedirs):
        """
        Тест создания директории при инициализации.
        """
        # Сбрасываем счетчик вызовов перед тестом
        mock_makedirs.reset_mock()
        
        config = {
            "plots_dir": "test_dir"
        }
        
        visualizer = ModelVisualizer(config)
        mock_makedirs.assert_called_with("test_dir", exist_ok=True)
        # Проверяем, что вызов был хотя бы один раз
        self.assertTrue(mock_makedirs.called)
    
    def test_update_config(self):
        """
        Тест обновления конфигурации.
        """
        # Исходная конфигурация
        self.assertEqual(self.visualizer.dpi, 72)
        self.assertEqual(self.visualizer.figsize, (8, 6))
        
        # Обновляем конфигурацию
        new_config = {
            "dpi": 150,
            "figsize": (12, 10)
        }
        
        self.visualizer.update_config(new_config)
        
        # Проверка обновления параметров
        self.assertEqual(self.visualizer.dpi, 150)
        self.assertEqual(self.visualizer.figsize, (12, 10))
    
    @patch('os.makedirs')
    def test_create_model_plots_dir(self, mock_makedirs):
        """
        Тест создания директории для графиков модели.
        """
        # Сбрасываем счетчик вызовов перед тестом
        mock_makedirs.reset_mock()
        
        plots_dir = self.visualizer.create_model_plots_dir("test_model", "test_dataset")
        
        expected_dir = os.path.join(self.test_plots_dir, "test_model", "test_dataset")
        self.assertEqual(plots_dir, expected_dir)
        
        mock_makedirs.assert_called_with(expected_dir, exist_ok=True)
        self.assertTrue(mock_makedirs.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.heatmap')
    def test_plot_confusion_matrix(self, mock_heatmap, mock_close, mock_figure, mock_savefig):
        """
        Тест построения матрицы ошибок.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        mock_heatmap.reset_mock()
        
        with patch('os.makedirs'):
            cm_path = self.visualizer.plot_confusion_matrix(
                self.y_true_cls, self.y_pred_cls, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "confusion_matrix.png"
        )
        self.assertEqual(cm_path, expected_path)
        
        # Проверка вызовов matplotlib и seaborn
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_heatmap.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
        
        # Проверка нормализованной матрицы ошибок
        with patch('os.makedirs'):
            cm_norm_path = self.visualizer.plot_confusion_matrix(
                self.y_true_cls, self.y_pred_cls, "test_model", "test_dataset", normalize=True
            )
        
        self.assertEqual(cm_norm_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.legend')
    def test_plot_roc_curve(self, mock_legend, mock_plot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения ROC-кривой.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_plot.reset_mock()
        mock_legend.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            roc_path = self.visualizer.plot_roc_curve(
                self.y_true_cls, self.y_proba_cls, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "roc_curve.png"
        )
        self.assertEqual(roc_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_plot.called)
        self.assertTrue(mock_legend.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
        
        # Проверка многоклассовой ROC-кривой
        y_proba_multi = np.random.rand(10, 3)
        y_proba_multi = y_proba_multi / y_proba_multi.sum(axis=1)[:, np.newaxis]
        
        with patch('os.makedirs'):
            roc_multi_path = self.visualizer.plot_roc_curve(
                np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), y_proba_multi, "test_model", "test_dataset"
            )
        
        self.assertEqual(roc_multi_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.legend')
    def test_plot_precision_recall_curve(self, mock_legend, mock_plot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения кривой точность-полнота.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_plot.reset_mock()
        mock_legend.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            pr_path = self.visualizer.plot_precision_recall_curve(
                self.y_true_cls, self.y_proba_cls, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "precision_recall_curve.png"
        )
        self.assertEqual(pr_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_plot.called)
        self.assertTrue(mock_legend.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
        
        # Проверка многоклассовой PR-кривой
        y_proba_multi = np.random.rand(10, 3)
        y_proba_multi = y_proba_multi / y_proba_multi.sum(axis=1)[:, np.newaxis]
        
        with patch('os.makedirs'):
            pr_multi_path = self.visualizer.plot_precision_recall_curve(
                np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]), y_proba_multi, "test_model", "test_dataset"
            )
        
        self.assertEqual(pr_multi_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.plot')
    def test_plot_regression_predictions(self, mock_plot, mock_scatter, mock_close, mock_figure, mock_savefig):
        """
        Тест построения графика сравнения предсказаний для регрессии.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_scatter.reset_mock()
        mock_plot.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            pred_path = self.visualizer.plot_regression_predictions(
                self.y_true_reg, self.y_pred_reg, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "prediction_comparison.png"
        )
        self.assertEqual(pred_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_scatter.called)
        self.assertTrue(mock_plot.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.scatter')
    @patch('matplotlib.pyplot.hlines')
    def test_plot_residuals(self, mock_hlines, mock_scatter, mock_close, mock_figure, mock_savefig):
        """
        Тест построения графика остатков.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_scatter.reset_mock()
        mock_hlines.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            resid_path = self.visualizer.plot_residuals(
                self.y_true_reg, self.y_pred_reg, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "residuals.png"
        )
        self.assertEqual(resid_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_scatter.called)
        self.assertTrue(mock_hlines.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.histplot')
    def test_plot_residuals_histogram(self, mock_histplot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения гистограммы остатков.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_histplot.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            resid_hist_path = self.visualizer.plot_residuals_histogram(
                self.y_true_reg, self.y_pred_reg, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "residuals_hist.png"
        )
        self.assertEqual(resid_hist_path, expected_path)
        
        # Проверка вызовов matplotlib и seaborn
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_histplot.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.barplot')
    def test_plot_feature_importance(self, mock_barplot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения графика важности признаков.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_barplot.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            importance_path = self.visualizer.plot_feature_importance(
                self.feature_importance, self.feature_names, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "feature_importance.png"
        )
        self.assertEqual(importance_path, expected_path)
        
        # Проверка вызовов matplotlib и seaborn
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_barplot.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
        
        # Проверка с ограничением количества признаков
        with patch('os.makedirs'):
            importance_top_path = self.visualizer.plot_feature_importance(
                self.feature_importance, self.feature_names, "test_model", "test_dataset", top_n=3
            )
        
        self.assertEqual(importance_top_path, expected_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.fill_between')
    @patch('matplotlib.pyplot.legend')
    def test_plot_learning_curve(self, mock_legend, mock_fill_between, mock_plot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения кривой обучения.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_plot.reset_mock()
        mock_fill_between.reset_mock()
        mock_legend.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            learning_curve_path = self.visualizer.plot_learning_curve(
                self.train_sizes, self.train_scores, self.test_scores, "test_model", "test_dataset"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "test_model", "test_dataset", "learning_curve.png"
        )
        self.assertEqual(learning_curve_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_plot.called)
        self.assertTrue(mock_fill_between.called)
        self.assertTrue(mock_legend.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.barplot')
    def test_plot_metrics_comparison(self, mock_barplot, mock_close, mock_figure, mock_savefig):
        """
        Тест построения графика сравнения метрик.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_barplot.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            comparison_path = self.visualizer.plot_metrics_comparison(
                self.metrics_dict, self.model_names, "test_dataset", "accuracy"
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "comparison", "test_dataset", "accuracy_comparison.png"
        )
        self.assertEqual(comparison_path, expected_path)
        
        # Проверка вызовов matplotlib и seaborn
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_barplot.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.bar')
    @patch('matplotlib.pyplot.legend')
    def test_plot_multiple_metrics_comparison(self, mock_legend, mock_bar, mock_close, mock_figure, mock_savefig):
        """
        Тест построения графика сравнения нескольких метрик.
        """
        # Сбрасываем счетчики вызовов перед тестом
        mock_figure.reset_mock()
        mock_bar.reset_mock()
        mock_legend.reset_mock()
        mock_savefig.reset_mock()
        mock_close.reset_mock()
        
        with patch('os.makedirs'):
            multi_comparison_path = self.visualizer.plot_multiple_metrics_comparison(
                self.metrics_dict, self.model_names, "test_dataset", ["accuracy", "precision"]
            )
        
        # Проверка пути к сохраненному графику
        expected_path = os.path.join(
            self.test_plots_dir, "comparison", "test_dataset", "multiple_metrics_comparison.png"
        )
        self.assertEqual(multi_comparison_path, expected_path)
        
        # Проверка вызовов matplotlib
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_bar.called)
        self.assertTrue(mock_legend.called)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)


if __name__ == "__main__":
    unittest.main() 