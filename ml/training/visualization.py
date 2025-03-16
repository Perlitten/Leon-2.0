"""
Модуль для визуализации результатов валидации моделей машинного обучения.

Предоставляет функции для создания различных графиков и визуализаций,
которые помогают анализировать производительность моделей.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve,
    auc, precision_recall_fscore_support
)
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Класс для визуализации результатов работы моделей машинного обучения.
    
    Предоставляет методы для создания различных графиков и визуализаций,
    которые помогают анализировать производительность моделей.
    
    Attributes:
        plots_dir (str): Директория для сохранения графиков.
        style (str): Стиль графиков matplotlib.
        dpi (int): Разрешение сохраняемых изображений.
        figsize (tuple): Размер графиков по умолчанию.
    """
    
    def __init__(self, config=None):
        """
        Инициализирует визуализатор моделей.
        
        Args:
            config (dict, optional): Словарь с конфигурационными параметрами.
                Может содержать следующие ключи:
                - plots_dir (str): Директория для сохранения графиков.
                - style (str): Стиль графиков matplotlib.
                - dpi (int): Разрешение сохраняемых изображений.
                - figsize (tuple): Размер графиков по умолчанию.
        """
        # Параметры по умолчанию
        self.plots_dir = "reports/plots"
        self.style = "seaborn-v0_8-whitegrid"
        self.dpi = 100
        self.figsize = (10, 8)
        
        # Обновляем параметры из конфигурации, если она предоставлена
        if config:
            self.update_config(config)
        
        # Устанавливаем стиль для matplotlib
        plt.style.use(self.style)
        
        # Создаем директорию для графиков, если она не существует
        os.makedirs(self.plots_dir, exist_ok=True)
        logger.info(f"Директория для графиков создана: {self.plots_dir}")
    
    def update_config(self, new_config):
        """
        Обновляет конфигурацию визуализатора.
        
        Args:
            new_config (dict): Словарь с новыми конфигурационными параметрами.
        """
        if "plots_dir" in new_config:
            self.plots_dir = new_config["plots_dir"]
        
        if "style" in new_config:
            self.style = new_config["style"]
            plt.style.use(self.style)
        
        if "dpi" in new_config:
            self.dpi = new_config["dpi"]
        
        if "figsize" in new_config:
            self.figsize = new_config["figsize"]
        
        # Создаем директорию для графиков, если она не существует
        os.makedirs(self.plots_dir, exist_ok=True)
        logger.info(f"Директория для графиков обновлена: {self.plots_dir}")
    
    def create_model_plots_dir(self, model_name, dataset_name):
        """
        Создает директорию для графиков конкретной модели и набора данных.
        
        Args:
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к созданной директории.
        """
        plots_dir = os.path.join(self.plots_dir, model_name, dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, dataset_name, normalize=False):
        """
        Строит и сохраняет матрицу ошибок.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
            normalize (bool, optional): Нормализовать ли матрицу. По умолчанию False.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        plt.figure(figsize=self.figsize)
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f"Нормализованная матрица ошибок - {model_name}"
        else:
            title = f"Матрица ошибок - {model_name}"
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', cbar=True)
        
        plt.title(title)
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        tick_marks = np.arange(len(unique_classes))
        plt.xticks(tick_marks + 0.5, unique_classes)
        plt.yticks(tick_marks + 0.5, unique_classes)
        
        plt.tight_layout()
        cm_path = os.path.join(plots_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Матрица ошибок сохранена: {cm_path}")
        return cm_path
    
    def plot_roc_curve(self, y_true, y_proba, model_name, dataset_name):
        """
        Строит и сохраняет ROC-кривую.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_proba (numpy.ndarray): Вероятности классов.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        plt.figure(figsize=self.figsize)
        
        # Для бинарной классификации
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC кривая (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC-кривая - {model_name}')
            plt.legend(loc="lower right")
        
        # Для многоклассовой классификации
        else:
            n_classes = y_proba.shape[1]
            
            # Вычисляем ROC-кривую и AUC для каждого класса
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                # Преобразуем в формат one-vs-rest
                y_true_binary = (y_true == i).astype(int)
                fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                plt.plot(fpr[i], tpr[i],
                         label=f'Класс {i} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC-кривая для каждого класса - {model_name}')
            plt.legend(loc="lower right")
        
        plt.tight_layout()
        roc_path = os.path.join(plots_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"ROC-кривая сохранена: {roc_path}")
        return roc_path
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name, dataset_name):
        """
        Строит и сохраняет кривую точность-полнота.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_proba (numpy.ndarray): Вероятности классов.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        plt.figure(figsize=self.figsize)
        
        # Для бинарной классификации
        if y_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'PR кривая (AUC = {pr_auc:.2f})')
            plt.xlabel('Полнота (Recall)')
            plt.ylabel('Точность (Precision)')
            plt.title(f'Кривая точность-полнота - {model_name}')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc="lower left")
        
        # Для многоклассовой классификации
        else:
            n_classes = y_proba.shape[1]
            
            # Вычисляем PR-кривую для каждого класса
            precision = dict()
            recall = dict()
            pr_auc = dict()
            
            for i in range(n_classes):
                # Преобразуем в формат one-vs-rest
                y_true_binary = (y_true == i).astype(int)
                precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_proba[:, i])
                pr_auc[i] = auc(recall[i], precision[i])
                
                plt.plot(recall[i], precision[i],
                         label=f'Класс {i} (AUC = {pr_auc[i]:.2f})')
            
            plt.xlabel('Полнота (Recall)')
            plt.ylabel('Точность (Precision)')
            plt.title(f'Кривая точность-полнота для каждого класса - {model_name}')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc="lower left")
        
        plt.tight_layout()
        pr_path = os.path.join(plots_dir, "precision_recall_curve.png")
        plt.savefig(pr_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Кривая точность-полнота сохранена: {pr_path}")
        return pr_path
    
    def plot_regression_predictions(self, y_true, y_pred, model_name, dataset_name):
        """
        Строит и сохраняет график сравнения предсказанных и фактических значений.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        plt.figure(figsize=self.figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Добавляем линию идеального предсказания
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.xlabel('Истинные значения')
        plt.ylabel('Предсказанные значения')
        plt.title(f'Сравнение предсказаний - {model_name}')
        
        plt.tight_layout()
        pred_path = os.path.join(plots_dir, "prediction_comparison.png")
        plt.savefig(pred_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"График сравнения предсказаний сохранен: {pred_path}")
        return pred_path
    
    def plot_residuals(self, y_true, y_pred, model_name, dataset_name):
        """
        Строит и сохраняет график остатков.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        residuals = y_true - y_pred
        
        plt.figure(figsize=self.figsize)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
        
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title(f'Остатки - {model_name}')
        
        plt.tight_layout()
        resid_path = os.path.join(plots_dir, "residuals.png")
        plt.savefig(resid_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"График остатков сохранен: {resid_path}")
        return resid_path
    
    def plot_residuals_histogram(self, y_true, y_pred, model_name, dataset_name):
        """
        Строит и сохраняет гистограмму остатков.
        
        Args:
            y_true (numpy.ndarray): Истинные значения.
            y_pred (numpy.ndarray): Предсказанные значения.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        residuals = y_true - y_pred
        
        plt.figure(figsize=self.figsize)
        sns.histplot(residuals, kde=True)
        
        plt.xlabel('Значение остатка')
        plt.ylabel('Частота')
        plt.title(f'Гистограмма остатков - {model_name}')
        
        plt.tight_layout()
        resid_hist_path = os.path.join(plots_dir, "residuals_hist.png")
        plt.savefig(resid_hist_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Гистограмма остатков сохранена: {resid_hist_path}")
        return resid_hist_path
    
    def plot_feature_importance(self, feature_importance, feature_names, model_name, dataset_name, top_n=None):
        """
        Строит и сохраняет график важности признаков.
        
        Args:
            feature_importance (numpy.ndarray): Важность признаков.
            feature_names (list): Названия признаков.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
            top_n (int, optional): Количество самых важных признаков для отображения.
                По умолчанию None (отображаются все признаки).
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        # Создаем DataFrame для удобства сортировки
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        
        # Сортируем по важности
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Ограничиваем количество признаков, если указано
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=self.figsize)
        sns.barplot(x='importance', y='feature', data=importance_df)
        
        plt.title(f'Важность признаков - {model_name}')
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        
        plt.tight_layout()
        importance_path = os.path.join(plots_dir, "feature_importance.png")
        plt.savefig(importance_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"График важности признаков сохранен: {importance_path}")
        return importance_path
    
    def plot_learning_curve(self, train_sizes, train_scores, test_scores, model_name, dataset_name):
        """
        Строит и сохраняет кривую обучения.
        
        Args:
            train_sizes (numpy.ndarray): Размеры обучающей выборки.
            train_scores (numpy.ndarray): Оценки на обучающей выборке.
            test_scores (numpy.ndarray): Оценки на тестовой выборке.
            model_name (str): Название модели.
            dataset_name (str): Название набора данных.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = self.create_model_plots_dir(model_name, dataset_name)
        
        plt.figure(figsize=self.figsize)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Обучающая выборка")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Тестовая выборка")
        
        plt.title(f'Кривая обучения - {model_name}')
        plt.xlabel('Размер обучающей выборки')
        plt.ylabel('Оценка')
        plt.legend(loc="best")
        
        plt.tight_layout()
        learning_curve_path = os.path.join(plots_dir, "learning_curve.png")
        plt.savefig(learning_curve_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Кривая обучения сохранена: {learning_curve_path}")
        return learning_curve_path
    
    def plot_metrics_comparison(self, metrics_dict, model_names, dataset_name, metric_name):
        """
        Строит и сохраняет график сравнения метрик для нескольких моделей.
        
        Args:
            metrics_dict (dict): Словарь с метриками для каждой модели.
            model_names (list): Список названий моделей.
            dataset_name (str): Название набора данных.
            metric_name (str): Название метрики для сравнения.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = os.path.join(self.plots_dir, "comparison", dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=self.figsize)
        
        # Извлекаем значения метрики для каждой модели
        metric_values = [metrics_dict[model][metric_name] for model in model_names]
        
        # Создаем DataFrame для удобства визуализации
        df = pd.DataFrame({
            'model': model_names,
            'value': metric_values
        })
        
        # Сортируем по значению метрики (в зависимости от метрики, может быть по возрастанию или убыванию)
        if metric_name in ['mse', 'rmse', 'mae', 'mape']:
            # Для этих метрик меньше - лучше
            df = df.sort_values('value', ascending=True)
        else:
            # Для остальных метрик больше - лучше
            df = df.sort_values('value', ascending=False)
        
        sns.barplot(x='model', y='value', data=df)
        
        plt.title(f'Сравнение {metric_name} - {dataset_name}')
        plt.xlabel('Модель')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        comparison_path = os.path.join(plots_dir, f"{metric_name}_comparison.png")
        plt.savefig(comparison_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"График сравнения {metric_name} сохранен: {comparison_path}")
        return comparison_path
    
    def plot_multiple_metrics_comparison(self, metrics_dict, model_names, dataset_name, metric_names):
        """
        Строит и сохраняет график сравнения нескольких метрик для нескольких моделей.
        
        Args:
            metrics_dict (dict): Словарь с метриками для каждой модели.
            model_names (list): Список названий моделей.
            dataset_name (str): Название набора данных.
            metric_names (list): Список названий метрик для сравнения.
        
        Returns:
            str: Путь к сохраненному графику.
        """
        plots_dir = os.path.join(self.plots_dir, "comparison", dataset_name)
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=self.figsize)
        
        # Создаем данные для графика
        x = np.arange(len(model_names))
        width = 0.8 / len(metric_names)
        
        for i, metric in enumerate(metric_names):
            metric_values = [metrics_dict[model][metric] for model in model_names]
            plt.bar(x + i * width, metric_values, width, label=metric)
        
        plt.xlabel('Модели')
        plt.ylabel('Значение')
        plt.title(f'Сравнение метрик - {dataset_name}')
        plt.xticks(x + width * (len(metric_names) - 1) / 2, model_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        multi_comparison_path = os.path.join(plots_dir, "multiple_metrics_comparison.png")
        plt.savefig(multi_comparison_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"График сравнения нескольких метрик сохранен: {multi_comparison_path}")
        return multi_comparison_path 