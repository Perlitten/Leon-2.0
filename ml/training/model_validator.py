"""
Модуль для валидации моделей машинного обучения.

Предоставляет инструменты для оценки производительности моделей,
расчета метрик, визуализации результатов и сравнения нескольких моделей.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support
)

from ml.training.visualization import ModelVisualizer

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Класс для валидации моделей машинного обучения.
    
    Позволяет оценивать производительность моделей, рассчитывать метрики,
    визуализировать результаты и сравнивать несколько моделей между собой.
    
    Attributes:
        reports_dir (str): Директория для сохранения отчетов.
        save_reports (bool): Флаг, указывающий, нужно ли сохранять отчеты.
        visualization (bool): Флаг, указывающий, нужно ли создавать визуализации.
        visualizer (ModelVisualizer): Объект для визуализации результатов.
    """
    
    def __init__(self, config=None):
        """
        Инициализирует валидатор моделей.
        
        Args:
            config (dict, optional): Словарь с конфигурационными параметрами.
                Может содержать следующие ключи:
                - reports_dir (str): Директория для сохранения отчетов.
                - save_reports (bool): Флаг, указывающий, нужно ли сохранять отчеты.
                - visualization (bool): Флаг, указывающий, нужно ли создавать визуализации.
                - visualizer_config (dict): Конфигурация для визуализатора.
        """
        # Параметры по умолчанию
        self.reports_dir = "reports/model_validation"
        self.save_reports = True
        self.visualization = True
        
        # Обновляем параметры из конфигурации, если она предоставлена
        if config:
            self.update_config(config)
        
        # Создаем директорию для отчетов, если она не существует и сохранение отчетов включено
        if self.save_reports:
            os.makedirs(self.reports_dir, exist_ok=True)
            logger.info(f"Директория для отчетов создана: {self.reports_dir}")
        
        # Создаем визуализатор
        visualizer_config = config.get("visualizer_config", {}) if config else {}
        if "plots_dir" not in visualizer_config:
            visualizer_config["plots_dir"] = os.path.join(self.reports_dir, "plots")
        
        self.visualizer = ModelVisualizer(visualizer_config)
    
    def update_config(self, new_config):
        """
        Обновляет конфигурацию валидатора.
        
        Args:
            new_config (dict): Словарь с новыми конфигурационными параметрами.
        """
        if "reports_dir" in new_config:
            self.reports_dir = new_config["reports_dir"]
        
        if "save_reports" in new_config:
            self.save_reports = new_config["save_reports"]
        
        if "visualization" in new_config:
            self.visualization = new_config["visualization"]
        
        # Создаем директорию для отчетов, если она не существует и сохранение отчетов включено
        if self.save_reports:
            os.makedirs(self.reports_dir, exist_ok=True)
            logger.info(f"Директория для отчетов обновлена: {self.reports_dir}")
        
        # Обновляем конфигурацию визуализатора, если он уже создан
        if hasattr(self, "visualizer") and "visualizer_config" in new_config:
            visualizer_config = new_config["visualizer_config"]
            if "plots_dir" not in visualizer_config:
                visualizer_config["plots_dir"] = os.path.join(self.reports_dir, "plots")
            
            self.visualizer.update_config(visualizer_config)
    
    def validate(self, model, X, y, task_type=None, dataset_name="dataset"):
        """
        Валидирует модель на заданных данных и возвращает отчет с метриками.
        
        Args:
            model: Модель для валидации.
            X (numpy.ndarray или pandas.DataFrame): Признаки для валидации.
            y (numpy.ndarray или pandas.Series): Целевые значения для валидации.
            task_type (str, optional): Тип задачи ("classification" или "regression").
                Если не указан, определяется автоматически.
            dataset_name (str, optional): Название набора данных. По умолчанию "dataset".
        
        Returns:
            dict: Словарь с результатами валидации, включающий:
                - model_name (str): Название модели.
                - dataset_name (str): Название набора данных.
                - task_type (str): Тип задачи.
                - metrics (dict): Словарь с метриками.
                - timestamp (str): Временная метка.
                - data_shape (tuple): Форма входных данных.
                - plots (dict, optional): Словарь с путями к графикам (если включена визуализация).
        """
        # Определяем тип задачи, если не указан
        if task_type is None:
            # Проверяем, есть ли у модели атрибут task_type
            if hasattr(model, "task_type") and model.task_type is not None:
                task_type = model.task_type
            else:
                # Определяем тип задачи по целевой переменной
                if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 10:
                    task_type = "classification"
                else:
                    task_type = "regression"
                logger.info(f"Тип задачи определен автоматически: {task_type}")
        
        # Получаем предсказания модели
        logger.info(f"Получение предсказаний для модели {model.name}")
        y_pred = model.predict(X)
        
        # Для задачи классификации получаем вероятности классов, если метод доступен
        y_proba = None
        if task_type == "classification" and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)
            except (AttributeError, NotImplementedError):
                logger.warning(f"Модель {model.name} не поддерживает predict_proba")
        
        # Рассчитываем метрики
        logger.info(f"Расчет метрик для модели {model.name}")
        metrics = self._calculate_metrics(y, y_pred, y_proba, task_type)
        
        # Создаем отчет
        report = {
            "model_name": model.name,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "data_shape": X.shape
        }
        
        # Визуализируем результаты, если включено
        if self.visualization:
            logger.info(f"Визуализация результатов для модели {model.name}")
            plots = self._visualize_results(y, y_pred, y_proba, task_type, model.name, dataset_name)
            report["plots"] = plots
        
        # Сохраняем отчет, если включено
        if self.save_reports:
            logger.info(f"Сохранение отчета для модели {model.name}")
            report_path = self._save_report(report, model.name, dataset_name)
            logger.info(f"Отчет сохранен: {report_path}")
        
        return report
    
    def compare_models(self, models, X, y, task_type=None, dataset_name="comparison"):
        """
        Сравнивает несколько моделей на одном наборе данных и возвращает отчет с сравнительными метриками.
        
        Args:
            models (list): Список моделей для сравнения.
            X (numpy.ndarray или pandas.DataFrame): Признаки для валидации.
            y (numpy.ndarray или pandas.Series): Целевые значения для валидации.
            task_type (str, optional): Тип задачи ("classification" или "regression").
                Если не указан, определяется автоматически.
            dataset_name (str, optional): Название набора данных. По умолчанию "comparison".
        
        Returns:
            dict: Словарь с результатами сравнения, включающий:
                - dataset_name (str): Название набора данных.
                - task_type (str): Тип задачи.
                - models (list): Список названий моделей.
                - metrics_comparison (dict): Словарь с метриками для каждой модели.
                - timestamp (str): Временная метка.
                - plots (dict, optional): Словарь с путями к сравнительным графикам (если включена визуализация).
        """
        # Определяем тип задачи, если не указан
        if task_type is None:
            # Проверяем, есть ли у первой модели атрибут task_type
            if hasattr(models[0], "task_type") and models[0].task_type is not None:
                task_type = models[0].task_type
            else:
                # Определяем тип задачи по целевой переменной
                if np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) < 10:
                    task_type = "classification"
                else:
                    task_type = "regression"
                logger.info(f"Тип задачи определен автоматически: {task_type}")
        
        # Валидируем каждую модель
        logger.info(f"Сравнение {len(models)} моделей на наборе данных {dataset_name}")
        model_reports = []
        model_names = []
        metrics_comparison = {}
        
        for model in models:
            report = self.validate(model, X, y, task_type, dataset_name)
            model_reports.append(report)
            model_names.append(model.name)
            metrics_comparison[model.name] = report["metrics"]
        
        # Создаем сравнительный отчет
        comparison_report = {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "models": model_names,
            "metrics_comparison": metrics_comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        # Визуализируем сравнительные результаты, если включено
        if self.visualization:
            logger.info(f"Визуализация сравнительных результатов для {len(models)} моделей")
            comparison_plots = self._visualize_comparison(model_reports, task_type, dataset_name)
            comparison_report["plots"] = comparison_plots
        
        # Сохраняем сравнительный отчет, если включено
        if self.save_reports:
            logger.info(f"Сохранение сравнительного отчета для {len(models)} моделей")
            report_path = self._save_comparison_report(comparison_report, dataset_name)
            logger.info(f"Сравнительный отчет сохранен: {report_path}")
        
        return comparison_report
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None, task_type="classification"):
        """
        Рассчитывает метрики для заданного типа задачи.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            y_proba (numpy.ndarray, optional): Вероятности классов для задачи классификации.
            task_type (str): Тип задачи ("classification" или "regression").
        
        Returns:
            dict: Словарь с метриками.
        """
        metrics = {}
        
        if task_type == "classification":
            # Определяем, бинарная или многоклассовая классификация
            unique_classes = np.unique(y_true)
            is_binary = len(unique_classes) == 2
            
            # Общие метрики для любой классификации
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            
            if is_binary:
                # Метрики для бинарной классификации
                metrics["precision"] = precision_score(y_true, y_pred, average="binary")
                metrics["recall"] = recall_score(y_true, y_pred, average="binary")
                metrics["f1"] = f1_score(y_true, y_pred, average="binary")
                
                # ROC AUC, если доступны вероятности
                if y_proba is not None and y_proba.shape[1] == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Метрики для многоклассовой классификации
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average=None
                )
                
                # Макро-усреднение
                metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
                metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")
                metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
                
                # Взвешенное усреднение
                metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
                metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
                metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
                
                # Для каждого класса
                for i, cls in enumerate(unique_classes):
                    metrics[f"precision_class_{cls}"] = precision[i]
                    metrics[f"recall_class_{cls}"] = recall[i]
                    metrics[f"f1_class_{cls}"] = f1[i]
        
        elif task_type == "regression":
            # Метрики для регрессии
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            # Избегаем деления на ноль
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics["mape"] = mape
            else:
                metrics["mape"] = np.nan
        
        return metrics
    
    def _visualize_results(self, y_true, y_pred, y_proba=None, task_type="classification", model_name="model", dataset_name="dataset"):
        """
        Визуализирует результаты валидации с использованием ModelVisualizer.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            y_proba (numpy.ndarray, optional): Вероятности классов для задачи классификации.
            task_type (str): Тип задачи ("classification" или "regression").
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            dict: Словарь с путями к сохраненным графикам.
        """
        plots = {}
        
        if task_type == "classification":
            # Матрица ошибок
            cm_path = self.visualizer.plot_confusion_matrix(
                y_true, y_pred, model_name, dataset_name
            )
            plots["confusion_matrix"] = cm_path
            
            # Для бинарной классификации с вероятностями
            if y_proba is not None and y_proba.shape[1] == 2:
                # ROC кривая
                roc_path = self.visualizer.plot_roc_curve(
                    y_true, y_proba, model_name, dataset_name
                )
                plots["roc_curve"] = roc_path
                
                # PR кривая
                pr_path = self.visualizer.plot_precision_recall_curve(
                    y_true, y_proba, model_name, dataset_name
                )
                plots["pr_curve"] = pr_path
        
        elif task_type == "regression":
            # График сравнения предсказанных и фактических значений
            pred_path = self.visualizer.plot_regression_predictions(
                y_true, y_pred, model_name, dataset_name
            )
            plots["prediction_comparison"] = pred_path
            
            # График остатков
            resid_path = self.visualizer.plot_residuals(
                y_true, y_pred, model_name, dataset_name
            )
            plots["residuals"] = resid_path
            
            # Гистограмма остатков
            resid_hist_path = self.visualizer.plot_residuals_histogram(
                y_true, y_pred, model_name, dataset_name
            )
            plots["residuals_hist"] = resid_hist_path
        
        return plots
    
    def _visualize_comparison(self, model_reports, task_type, dataset_name):
        """
        Визуализирует сравнение нескольких моделей с использованием ModelVisualizer.
        
        Args:
            model_reports (list): Список отчетов по моделям.
            task_type (str): Тип задачи ("classification" или "regression").
            dataset_name (str): Название набора данных.
        
        Returns:
            dict: Словарь с путями к сохраненным графикам.
        """
        plots = {}
        
        # Извлекаем названия моделей и метрики
        model_names = [report["model_name"] for report in model_reports]
        metrics_dict = {report["model_name"]: report["metrics"] for report in model_reports}
        
        if task_type == "classification":
            # Сравнение метрик классификации
            if "precision" in model_reports[0]["metrics"]:
                # Для бинарной классификации
                metrics = ["accuracy", "precision", "recall", "f1"]
                
                for metric in metrics:
                    metric_path = self.visualizer.plot_metrics_comparison(
                        metrics_dict, model_names, dataset_name, metric
                    )
                    plots[f"{metric}_comparison"] = metric_path
                
                # Сравнение всех метрик
                multi_path = self.visualizer.plot_multiple_metrics_comparison(
                    metrics_dict, model_names, dataset_name, metrics
                )
                plots["multiple_metrics_comparison"] = multi_path
            
            elif "precision_macro" in model_reports[0]["metrics"]:
                # Для многоклассовой классификации
                metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                
                for metric in metrics:
                    metric_path = self.visualizer.plot_metrics_comparison(
                        metrics_dict, model_names, dataset_name, metric
                    )
                    plots[f"{metric}_comparison"] = metric_path
                
                # Сравнение всех метрик
                multi_path = self.visualizer.plot_multiple_metrics_comparison(
                    metrics_dict, model_names, dataset_name, metrics
                )
                plots["multiple_metrics_comparison"] = multi_path
        
        elif task_type == "regression":
            # Сравнение метрик регрессии
            metrics = ["rmse", "mae", "r2"]
            
            for metric in metrics:
                metric_path = self.visualizer.plot_metrics_comparison(
                    metrics_dict, model_names, dataset_name, metric
                )
                plots[f"{metric}_comparison"] = metric_path
            
            # Сравнение всех метрик
            multi_path = self.visualizer.plot_multiple_metrics_comparison(
                metrics_dict, model_names, dataset_name, metrics
            )
            plots["multiple_metrics_comparison"] = multi_path
        
        return plots
    
    def _save_report(self, report, model_name, dataset_name):
        """
        Сохраняет отчет в JSON-файл.
        
        Args:
            report (dict): Отчет для сохранения.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному файлу.
        """
        # Создаем копию отчета без графиков (они сохраняются отдельно)
        report_copy = report.copy()
        if "plots" in report_copy:
            del report_copy["plots"]
        
        # Формируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_name}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Сохраняем отчет
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_copy, f, indent=4, ensure_ascii=False)
        
        return filepath
    
    def _save_comparison_report(self, report, dataset_name):
        """
        Сохраняет сравнительный отчет в JSON-файл.
        
        Args:
            report (dict): Сравнительный отчет для сохранения.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному файлу.
        """
        # Создаем копию отчета без графиков (они сохраняются отдельно)
        report_copy = report.copy()
        if "plots" in report_copy:
            del report_copy["plots"]
        
        # Формируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{dataset_name}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Сохраняем отчет
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_copy, f, indent=4, ensure_ascii=False)
        
        return filepath 