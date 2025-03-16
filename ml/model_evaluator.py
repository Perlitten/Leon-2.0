"""
Модуль оценки моделей машинного обучения.

Предоставляет класс для оценки производительности моделей на исторических данных.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import os
import json
from datetime import datetime

class ModelEvaluator:
    """
    Класс для оценки производительности моделей машинного обучения.
    
    Отвечает за:
    - Оценку точности прогнозов модели
    - Расчет метрик производительности
    - Визуализацию результатов
    - Сохранение отчетов об оценке
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация оценщика моделей.
        
        Args:
            config: Конфигурация оценщика моделей
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Параметры по умолчанию
        self.reports_dir = self.config.get("reports_dir", "ml/reports")
        self.task_type = self.config.get("task_type", "classification")  # classification или regression
        self.threshold = self.config.get("threshold", 0.5)  # для классификации
        self.save_plots = self.config.get("save_plots", True)
        
        # Создание директории для отчетов, если она не существует
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def evaluate(self, model_id: str, y_true: np.ndarray, y_pred: np.ndarray, 
                 additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Оценка производительности модели.
        
        Args:
            model_id: Идентификатор модели
            y_true: Истинные значения
            y_pred: Предсказанные значения
            additional_data: Дополнительные данные для оценки (например, цены, даты)
            
        Returns:
            Словарь с метриками производительности
        """
        try:
            # Расчет метрик в зависимости от типа задачи
            if self.task_type == "classification":
                metrics = self._evaluate_classification(y_true, y_pred)
            else:  # regression
                metrics = self._evaluate_regression(y_true, y_pred)
            
            # Добавление информации о модели и времени оценки
            metrics["model_id"] = model_id
            metrics["evaluation_time"] = datetime.now().isoformat()
            
            # Создание визуализаций
            if self.save_plots:
                plots_path = self._create_plots(model_id, y_true, y_pred, additional_data)
                metrics["plots_path"] = plots_path
            
            # Сохранение отчета
            report_path = self._save_report(model_id, metrics, additional_data)
            metrics["report_path"] = report_path
            
            self.logger.info(f"Оценка модели {model_id} завершена. Метрики: {metrics}")
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Ошибка при оценке модели {model_id}: {e}")
            return {"error": str(e)}
    
    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели классификации.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения (вероятности)
            
        Returns:
            Словарь с метриками производительности
        """
        # Преобразование вероятностей в классы
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Многоклассовая классификация
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            # Бинарная классификация
            y_pred_class = (y_pred > self.threshold).astype(int)
        
        # Расчет метрик
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred_class)),
            "precision": float(precision_score(y_true, y_pred_class, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_class, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred_class, average="weighted", zero_division=0))
        }
        
        return metrics
    
    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели регрессии.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            Словарь с метриками производительности
        """
        # Расчет метрик
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mae": float(mae)
        }
        
        return metrics
    
    def _create_plots(self, model_id: str, y_true: np.ndarray, y_pred: np.ndarray, 
                     additional_data: Dict[str, Any] = None) -> str:
        """
        Создание визуализаций для оценки модели.
        
        Args:
            model_id: Идентификатор модели
            y_true: Истинные значения
            y_pred: Предсказанные значения
            additional_data: Дополнительные данные для визуализации
            
        Returns:
            Путь к сохраненным визуализациям
        """
        # Создание директории для визуализаций
        plots_dir = os.path.join(self.reports_dir, model_id, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Создание визуализаций в зависимости от типа задачи
        if self.task_type == "classification":
            self._create_classification_plots(y_true, y_pred, plots_dir)
        else:  # regression
            self._create_regression_plots(y_true, y_pred, plots_dir, additional_data)
        
        return plots_dir
    
    def _create_classification_plots(self, y_true: np.ndarray, y_pred: np.ndarray, plots_dir: str) -> None:
        """
        Создание визуализаций для модели классификации.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            plots_dir: Директория для сохранения визуализаций
        """
        # Преобразование вероятностей в классы
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Многоклассовая классификация
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            # Бинарная классификация
            y_pred_class = (y_pred > self.threshold).astype(int)
        
        # Создание графика распределения классов
        plt.figure(figsize=(10, 6))
        plt.hist([y_true, y_pred_class], bins=np.arange(-0.5, max(np.max(y_true), np.max(y_pred_class)) + 1.5, 1), 
                 label=["Истинные", "Предсказанные"])
        plt.xlabel("Класс")
        plt.ylabel("Количество")
        plt.title("Распределение классов")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "class_distribution.png"))
        plt.close()
        
        # Создание графика точности по классам
        if len(np.unique(y_true)) <= 10:  # Ограничение на количество классов для визуализации
            classes = np.unique(np.concatenate([y_true, y_pred_class]))
            class_accuracy = []
            
            for cls in classes:
                mask = (y_true == cls)
                if np.sum(mask) > 0:
                    acc = np.mean(y_pred_class[mask] == cls)
                    class_accuracy.append(acc)
                else:
                    class_accuracy.append(0)
            
            plt.figure(figsize=(10, 6))
            plt.bar(classes, class_accuracy)
            plt.xlabel("Класс")
            plt.ylabel("Точность")
            plt.title("Точность по классам")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "class_accuracy.png"))
            plt.close()
    
    def _create_regression_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               plots_dir: str, additional_data: Dict[str, Any] = None) -> None:
        """
        Создание визуализаций для модели регрессии.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            plots_dir: Директория для сохранения визуализаций
            additional_data: Дополнительные данные для визуализации
        """
        # Создание графика сравнения истинных и предсказанных значений
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Добавление линии идеального предсказания
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Истинные значения")
        plt.ylabel("Предсказанные значения")
        plt.title("Сравнение истинных и предсказанных значений")
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "true_vs_pred.png"))
        plt.close()
        
        # Создание графика ошибок
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30)
        plt.xlabel("Ошибка")
        plt.ylabel("Количество")
        plt.title("Распределение ошибок")
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "error_distribution.png"))
        plt.close()
        
        # Создание графика временного ряда, если есть данные о времени
        if additional_data and "timestamps" in additional_data:
            timestamps = additional_data["timestamps"]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, y_true, label="Истинные значения")
            plt.plot(timestamps, y_pred, label="Предсказанные значения")
            plt.xlabel("Время")
            plt.ylabel("Значение")
            plt.title("Временной ряд истинных и предсказанных значений")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "time_series.png"))
            plt.close()
    
    def _save_report(self, model_id: str, metrics: Dict[str, Any], 
                    additional_data: Dict[str, Any] = None) -> str:
        """
        Сохранение отчета об оценке модели.
        
        Args:
            model_id: Идентификатор модели
            metrics: Метрики производительности
            additional_data: Дополнительные данные для отчета
            
        Returns:
            Путь к сохраненному отчету
        """
        # Создание директории для отчета
        report_dir = os.path.join(self.reports_dir, model_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # Создание отчета
        report = {
            "model_id": model_id,
            "evaluation_time": metrics.get("evaluation_time", datetime.now().isoformat()),
            "metrics": {k: v for k, v in metrics.items() if k not in ["model_id", "evaluation_time", "plots_path", "report_path"]},
            "task_type": self.task_type
        }
        
        # Добавление дополнительных данных
        if additional_data:
            # Исключение больших массивов данных
            safe_additional_data = {}
            for k, v in additional_data.items():
                if isinstance(v, (np.ndarray, list)) and len(v) > 100:
                    safe_additional_data[k] = f"Array of length {len(v)}"
                elif not isinstance(v, (str, int, float, bool, dict, list)):
                    safe_additional_data[k] = str(v)
                else:
                    safe_additional_data[k] = v
            
            report["additional_data"] = safe_additional_data
        
        # Сохранение отчета в JSON
        report_path = os.path.join(report_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        
        return report_path
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Сравнение нескольких моделей.
        
        Args:
            model_ids: Список идентификаторов моделей для сравнения
            
        Returns:
            Словарь с результатами сравнения
        """
        try:
            # Загрузка отчетов для каждой модели
            models_data = []
            for model_id in model_ids:
                report_path = os.path.join(self.reports_dir, model_id, "evaluation_report.json")
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report = json.load(f)
                    models_data.append(report)
                else:
                    self.logger.warning(f"Отчет для модели {model_id} не найден")
            
            if not models_data:
                return {"error": "Не найдено отчетов для сравнения"}
            
            # Создание сравнительной таблицы
            comparison = {
                "models": [data["model_id"] for data in models_data],
                "evaluation_times": [data["evaluation_time"] for data in models_data],
                "metrics": {}
            }
            
            # Сбор метрик для всех моделей
            all_metrics = set()
            for data in models_data:
                all_metrics.update(data["metrics"].keys())
            
            # Заполнение таблицы метрик
            for metric in all_metrics:
                comparison["metrics"][metric] = [data["metrics"].get(metric, None) for data in models_data]
            
            # Определение лучшей модели по каждой метрике
            best_models = {}
            for metric in all_metrics:
                values = comparison["metrics"][metric]
                if all(v is not None for v in values):
                    # Определение лучшего значения (больше или меньше в зависимости от метрики)
                    if metric in ["accuracy", "precision", "recall", "f1_score", "r2"]:
                        best_idx = np.argmax(values)
                    else:  # mse, rmse, mae - меньше лучше
                        best_idx = np.argmin(values)
                    
                    best_models[metric] = comparison["models"][best_idx]
            
            comparison["best_models"] = best_models
            
            # Создание визуализации сравнения
            if self.save_plots:
                self._create_comparison_plots(comparison)
            
            return comparison
        
        except Exception as e:
            self.logger.error(f"Ошибка при сравнении моделей: {e}")
            return {"error": str(e)}
    
    def _create_comparison_plots(self, comparison: Dict[str, Any]) -> None:
        """
        Создание визуализаций для сравнения моделей.
        
        Args:
            comparison: Данные сравнения моделей
        """
        # Создание директории для визуализаций
        plots_dir = os.path.join(self.reports_dir, "comparisons")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Создание графиков для каждой метрики
        for metric, values in comparison["metrics"].items():
            if all(v is not None for v in values):
                plt.figure(figsize=(10, 6))
                plt.bar(comparison["models"], values)
                plt.xlabel("Модель")
                plt.ylabel(metric)
                plt.title(f"Сравнение моделей по метрике {metric}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, f"comparison_{metric}.png"))
                plt.close()
        
        # Сохранение сравнения в JSON
        comparison_path = os.path.join(plots_dir, "comparison_report.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=4)
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновление конфигурации оценщика моделей.
        
        Args:
            config: Новая конфигурация
        """
        self.config.update(config)
        
        # Обновление параметров
        self.reports_dir = self.config.get("reports_dir", self.reports_dir)
        self.task_type = self.config.get("task_type", self.task_type)
        self.threshold = self.config.get("threshold", self.threshold)
        self.save_plots = self.config.get("save_plots", self.save_plots)
        
        # Создание директории для отчетов, если она не существует
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.logger.info("Конфигурация оценщика моделей обновлена") 