"""
Модуль моделей машинного обучения.

Предоставляет базовые классы для различных типов моделей
и их реализации.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
import os
import json
from datetime import datetime
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseModel(ABC):
    """
    Базовый абстрактный класс для всех моделей машинного обучения.
    
    Определяет общий интерфейс для всех типов моделей, включая
    методы обучения, предсказания, сохранения и загрузки.
    
    Attributes:
        name (str): Имя модели
        model: Объект модели
        params (Dict[str, Any]): Параметры модели
        metadata (Dict[str, Any]): Метаданные модели
        is_fitted (bool): Флаг, указывающий, обучена ли модель
    """
    
    def __init__(self, name: str, model: Any = None, 
                params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация базовой модели.
        
        Args:
            name: Имя модели
            model: Объект модели (если уже создан)
            params: Параметры модели
            metadata: Метаданные модели
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.name = name
        self.model = model
        self.params = params or {}
        self.metadata = metadata or {}
        self.is_fitted = False
        
        self.logger.info(f"Инициализирована модель {name}")
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Обучение модели.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            self: Обученная модель
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание целевой переменной.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные значения
        """
        pass
    
    def save(self, path: str) -> str:
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения модели
            
        Returns:
            str: Путь к сохраненной модели
        """
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Добавление метаданных
        self.metadata.update({
            "name": self.name,
            "params": self.params,
            "is_fitted": self.is_fitted,
            "saved_at": datetime.now().isoformat()
        })
        
        # Сохранение модели
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Сохранение метаданных отдельно для удобства
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        self.logger.info(f"Модель сохранена в {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Загрузка модели.
        
        Args:
            path: Путь к модели
            
        Returns:
            BaseModel: Загруженная модель
            
        Raises:
            FileNotFoundError: Если файл модели не найден
        """
        # Проверка существования файла
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели {path} не найден")
        
        # Загрузка модели
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logging.getLogger(cls.__name__).info(f"Модель загружена из {path}")
        
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров модели.
        
        Returns:
            Dict[str, Any]: Параметры модели
        """
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Установка параметров модели.
        
        Args:
            **params: Параметры модели
            
        Returns:
            self: Модель с обновленными параметрами
        """
        self.params.update(params)
        
        # Если модель уже создана, обновляем ее параметры
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Получение метаданных модели.
        
        Returns:
            Dict[str, Any]: Метаданные модели
        """
        return self.metadata.copy()
    
    def update_metadata(self, **metadata) -> 'BaseModel':
        """
        Обновление метаданных модели.
        
        Args:
            **metadata: Метаданные модели
            
        Returns:
            self: Модель с обновленными метаданными
        """
        self.metadata.update(metadata)
        return self 

class RegressionModel(BaseModel):
    """
    Класс для регрессионных моделей.
    
    Предоставляет интерфейс для работы с моделями регрессии,
    включая методы обучения, предсказания и оценки качества.
    
    Attributes:
        name (str): Имя модели
        model: Объект модели
        params (Dict[str, Any]): Параметры модели
        metadata (Dict[str, Any]): Метаданные модели
        is_fitted (bool): Флаг, указывающий, обучена ли модель
        metrics (Dict[str, float]): Метрики качества модели
    """
    
    def __init__(self, name: str, model: Any = None, 
                params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация регрессионной модели.
        
        Args:
            name: Имя модели
            model: Объект модели (если уже создан)
            params: Параметры модели
            metadata: Метаданные модели
        """
        super().__init__(name, model, params, metadata)
        
        # Метрики качества модели
        self.metrics = {}
        
        # Обновление метаданных
        self.metadata.update({
            "task_type": "regression"
        })
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series],
           eval_set: Optional[List[Tuple[Union[np.ndarray, pd.DataFrame], 
                                        Union[np.ndarray, pd.Series]]]] = None,
           **kwargs) -> 'RegressionModel':
        """
        Обучение модели.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            eval_set: Набор данных для оценки качества модели во время обучения
            **kwargs: Дополнительные параметры для передачи в метод fit модели
            
        Returns:
            self: Обученная модель
            
        Raises:
            ValueError: Если модель не инициализирована
        """
        if self.model is None:
            raise ValueError("Модель не инициализирована")
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Обучение модели
        if eval_set is not None and hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
            # Если модель поддерживает eval_set (например, XGBoost, LightGBM)
            self.model.fit(X_np, y_np, eval_set=eval_set, **kwargs)
        else:
            # Стандартное обучение
            self.model.fit(X_np, y_np, **kwargs)
        
        # Обновление флага обучения
        self.is_fitted = True
        
        # Расчет метрик качества на обучающей выборке
        self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Модель {self.name} обучена на данных размером {X_np.shape}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание целевой переменной.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Преобразование входных данных в numpy массив
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Предсказание
        y_pred = self.model.predict(X_np)
        
        return y_pred
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества модели.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Предсказание
        y_pred = self.predict(X)
        
        # Расчет метрик
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Расчет MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-10))) * 100
        
        # Сохранение метрик
        self.metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        }
        
        # Обновление метаданных
        self.metadata.update({
            "metrics": self.metrics
        })
        
        return self.metrics
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка качества модели на новых данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Расчет метрик
        metrics = self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Оценка модели {self.name} на данных размером {X_np.shape}")
        self.logger.info(f"Метрики: {metrics}")
        
        return metrics
    
    def plot_residuals(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика остатков.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Предсказание
        y_pred = self.predict(X_np)
        
        # Расчет остатков
        residuals = y_np - y_pred
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # Диаграмма рассеяния остатков
        ax.scatter(y_pred, residuals, alpha=0.5)
        
        # Горизонтальная линия на нуле
        ax.axhline(y=0, color='r', linestyle='-')
        
        # Настройка графика
        ax.set_title(f'Остатки модели {self.name}')
        ax.set_xlabel('Предсказанные значения')
        ax.set_ylabel('Остатки')
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"График остатков сохранен в {save_path}")
        
        return fig
    
    def plot_predictions(self, X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series],
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика предсказаний.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Предсказание
        y_pred = self.predict(X_np)
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # Диаграмма рассеяния фактических и предсказанных значений
        ax.scatter(y_np, y_pred, alpha=0.5)
        
        # Линия идеального предсказания
        min_val = min(np.min(y_np), np.min(y_pred))
        max_val = max(np.max(y_np), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Настройка графика
        ax.set_title(f'Предсказания модели {self.name}')
        ax.set_xlabel('Фактические значения')
        ax.set_ylabel('Предсказанные значения')
        
        # Добавление метрик на график
        metrics_text = f"MSE: {self.metrics.get('mse', 0):.4f}\n"
        metrics_text += f"RMSE: {self.metrics.get('rmse', 0):.4f}\n"
        metrics_text += f"MAE: {self.metrics.get('mae', 0):.4f}\n"
        metrics_text += f"R²: {self.metrics.get('r2', 0):.4f}\n"
        metrics_text += f"MAPE: {self.metrics.get('mape', 0):.4f}%"
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"График предсказаний сохранен в {save_path}")
        
        return fig 

class ClassificationModel(BaseModel):
    """
    Класс для классификационных моделей.
    
    Предоставляет интерфейс для работы с моделями классификации,
    включая методы обучения, предсказания и оценки качества.
    
    Attributes:
        name (str): Имя модели
        model: Объект модели
        params (Dict[str, Any]): Параметры модели
        metadata (Dict[str, Any]): Метаданные модели
        is_fitted (bool): Флаг, указывающий, обучена ли модель
        metrics (Dict[str, float]): Метрики качества модели
        classes (np.ndarray): Классы, которые может предсказывать модель
    """
    
    def __init__(self, name: str, model: Any = None, 
                params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация классификационной модели.
        
        Args:
            name: Имя модели
            model: Объект модели (если уже создан)
            params: Параметры модели
            metadata: Метаданные модели
        """
        super().__init__(name, model, params, metadata)
        
        # Метрики качества модели
        self.metrics = {}
        
        # Классы, которые может предсказывать модель
        self.classes = None
        
        # Обновление метаданных
        self.metadata.update({
            "task_type": "classification"
        })
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series],
           eval_set: Optional[List[Tuple[Union[np.ndarray, pd.DataFrame], 
                                        Union[np.ndarray, pd.Series]]]] = None,
           **kwargs) -> 'ClassificationModel':
        """
        Обучение модели.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            eval_set: Набор данных для оценки качества модели во время обучения
            **kwargs: Дополнительные параметры для передачи в метод fit модели
            
        Returns:
            self: Обученная модель
            
        Raises:
            ValueError: Если модель не инициализирована
        """
        if self.model is None:
            raise ValueError("Модель не инициализирована")
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Обучение модели
        if eval_set is not None and hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
            # Если модель поддерживает eval_set (например, XGBoost, LightGBM)
            self.model.fit(X_np, y_np, eval_set=eval_set, **kwargs)
        else:
            # Стандартное обучение
            self.model.fit(X_np, y_np, **kwargs)
        
        # Обновление флага обучения
        self.is_fitted = True
        
        # Сохранение классов
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_
        else:
            self.classes = np.unique(y_np)
        
        # Расчет метрик качества на обучающей выборке
        self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Модель {self.name} обучена на данных размером {X_np.shape}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание классов.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные классы
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Преобразование входных данных в numpy массив
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Предсказание
        y_pred = self.model.predict(X_np)
        
        return y_pred
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные вероятности классов
            
        Raises:
            ValueError: Если модель не обучена
            ValueError: Если модель не поддерживает предсказание вероятностей
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Модель не поддерживает предсказание вероятностей")
        
        # Преобразование входных данных в numpy массив
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Предсказание вероятностей
        y_proba = self.model.predict_proba(X_np)
        
        return y_proba
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества модели.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Предсказание
        y_pred = self.predict(X)
        
        # Расчет метрик
        accuracy = accuracy_score(y, y_pred)
        
        # Для бинарной классификации
        if len(self.classes) == 2:
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            # Расчет ROC AUC, если модель поддерживает предсказание вероятностей
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.predict_proba(X)
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = None
            
            # Сохранение метрик
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            if roc_auc is not None:
                self.metrics["roc_auc"] = roc_auc
        
        # Для многоклассовой классификации
        else:
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Сохранение метрик
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        # Обновление метаданных
        self.metadata.update({
            "metrics": self.metrics,
            "classes": self.classes.tolist() if self.classes is not None else None
        })
        
        return self.metrics
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка качества модели на новых данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Расчет метрик
        metrics = self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Оценка модели {self.name} на данных размером {X_np.shape}")
        self.logger.info(f"Метрики: {metrics}")
        
        return metrics
    
    def plot_confusion_matrix(self, X: Union[np.ndarray, pd.DataFrame], 
                             y: Union[np.ndarray, pd.Series],
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение матрицы ошибок.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Предсказание
        y_pred = self.predict(X_np)
        
        # Расчет матрицы ошибок
        cm = confusion_matrix(y_np, y_pred)
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # Тепловая карта матрицы ошибок
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Настройка графика
        ax.set_title(f'Матрица ошибок модели {self.name}')
        ax.set_xlabel('Предсказанные классы')
        ax.set_ylabel('Фактические классы')
        
        # Установка меток классов
        if self.classes is not None:
            ax.set_xticklabels(self.classes)
            ax.set_yticklabels(self.classes)
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Матрица ошибок сохранена в {save_path}")
        
        return fig
    
    def plot_roc_curve(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение ROC-кривой.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
            
        Raises:
            ValueError: Если модель не поддерживает предсказание вероятностей
            ValueError: Если задача не является бинарной классификацией
        """
        from sklearn.metrics import roc_curve, auc
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Модель не поддерживает предсказание вероятностей")
        
        if len(self.classes) != 2:
            raise ValueError("ROC-кривая поддерживается только для бинарной классификации")
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Предсказание вероятностей
        y_proba = self.predict_proba(X_np)
        
        # Расчет ROC-кривой
        fpr, tpr, _ = roc_curve(y_np, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # ROC-кривая
        ax.plot(fpr, tpr, label=f'ROC кривая (AUC = {roc_auc:.4f})')
        
        # Линия случайного классификатора
        ax.plot([0, 1], [0, 1], 'k--')
        
        # Настройка графика
        ax.set_title(f'ROC-кривая модели {self.name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"ROC-кривая сохранена в {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, X: Union[np.ndarray, pd.DataFrame], 
                                   y: Union[np.ndarray, pd.Series],
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение кривой точности-полноты.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
            
        Raises:
            ValueError: Если модель не поддерживает предсказание вероятностей
            ValueError: Если задача не является бинарной классификацией
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Модель не поддерживает предсказание вероятностей")
        
        if len(self.classes) != 2:
            raise ValueError("Кривая точности-полноты поддерживается только для бинарной классификации")
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Предсказание вероятностей
        y_proba = self.predict_proba(X_np)
        
        # Расчет кривой точности-полноты
        precision, recall, _ = precision_recall_curve(y_np, y_proba[:, 1])
        avg_precision = average_precision_score(y_np, y_proba[:, 1])
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # Кривая точности-полноты
        ax.plot(recall, precision, label=f'Средняя точность = {avg_precision:.4f}')
        
        # Настройка графика
        ax.set_title(f'Кривая точности-полноты модели {self.name}')
        ax.set_xlabel('Полнота')
        ax.set_ylabel('Точность')
        ax.legend(loc='lower left')
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Кривая точности-полноты сохранена в {save_path}")
        
        return fig 

class EnsembleModel(BaseModel):
    """
    Класс для ансамблевых моделей.
    
    Предоставляет интерфейс для работы с ансамблями моделей,
    включая методы обучения, предсказания и оценки качества.
    
    Attributes:
        name (str): Имя модели
        models (List[BaseModel]): Список моделей в ансамбле
        weights (List[float]): Веса моделей в ансамбле
        task_type (str): Тип задачи ('classification' или 'regression')
        voting (str): Тип голосования ('hard', 'soft' или 'weighted')
        params (Dict[str, Any]): Параметры модели
        metadata (Dict[str, Any]): Метаданные модели
        is_fitted (bool): Флаг, указывающий, обучена ли модель
        metrics (Dict[str, float]): Метрики качества модели
    """
    
    def __init__(self, name: str, models: List[BaseModel] = None, 
                weights: List[float] = None, task_type: str = 'classification',
                voting: str = 'hard', params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация ансамблевой модели.
        
        Args:
            name: Имя модели
            models: Список моделей в ансамбле
            weights: Веса моделей в ансамбле
            task_type: Тип задачи ('classification' или 'regression')
            voting: Тип голосования ('hard', 'soft' или 'weighted')
            params: Параметры модели
            metadata: Метаданные модели
        """
        super().__init__(name, None, params, metadata)
        
        self.models = models or []
        self.weights = weights or [1.0] * len(self.models)
        self.task_type = task_type
        self.voting = voting
        
        # Проверка соответствия типов моделей
        for model in self.models:
            if task_type == 'classification' and not isinstance(model, ClassificationModel):
                self.logger.warning(f"Модель {model.name} не является классификационной")
            elif task_type == 'regression' and not isinstance(model, RegressionModel):
                self.logger.warning(f"Модель {model.name} не является регрессионной")
        
        # Проверка соответствия длины весов
        if len(self.weights) != len(self.models):
            self.logger.warning(f"Количество весов ({len(self.weights)}) не соответствует количеству моделей ({len(self.models)})")
            self.weights = [1.0] * len(self.models)
        
        # Нормализация весов
        if self.weights:
            sum_weights = sum(self.weights)
            if sum_weights > 0:
                self.weights = [w / sum_weights for w in self.weights]
        
        # Метрики качества модели
        self.metrics = {}
        
        # Обновление метаданных
        self.metadata.update({
            "task_type": task_type,
            "voting": voting,
            "models": [model.name for model in self.models],
            "weights": self.weights
        })
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> 'EnsembleModel':
        """
        Добавление модели в ансамбль.
        
        Args:
            model: Модель для добавления
            weight: Вес модели
            
        Returns:
            self: Ансамблевая модель с добавленной моделью
        """
        # Проверка соответствия типа модели
        if self.task_type == 'classification' and not isinstance(model, ClassificationModel):
            self.logger.warning(f"Модель {model.name} не является классификационной")
        elif self.task_type == 'regression' and not isinstance(model, RegressionModel):
            self.logger.warning(f"Модель {model.name} не является регрессионной")
        
        # Добавление модели и веса
        self.models.append(model)
        self.weights.append(weight)
        
        # Нормализация весов
        sum_weights = sum(self.weights)
        if sum_weights > 0:
            self.weights = [w / sum_weights for w in self.weights]
        
        # Обновление метаданных
        self.metadata.update({
            "models": [model.name for model in self.models],
            "weights": self.weights
        })
        
        return self
    
    def remove_model(self, model_name: str) -> 'EnsembleModel':
        """
        Удаление модели из ансамбля.
        
        Args:
            model_name: Имя модели для удаления
            
        Returns:
            self: Ансамблевая модель без удаленной модели
            
        Raises:
            ValueError: Если модель с указанным именем не найдена
        """
        # Поиск модели по имени
        for i, model in enumerate(self.models):
            if model.name == model_name:
                # Удаление модели и веса
                self.models.pop(i)
                self.weights.pop(i)
                
                # Нормализация весов
                sum_weights = sum(self.weights)
                if sum_weights > 0:
                    self.weights = [w / sum_weights for w in self.weights]
                
                # Обновление метаданных
                self.metadata.update({
                    "models": [model.name for model in self.models],
                    "weights": self.weights
                })
                
                return self
        
        raise ValueError(f"Модель с именем {model_name} не найдена в ансамбле")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series],
           **kwargs) -> 'EnsembleModel':
        """
        Обучение ансамбля моделей.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            **kwargs: Дополнительные параметры для передачи в метод fit моделей
            
        Returns:
            self: Обученный ансамбль моделей
            
        Raises:
            ValueError: Если ансамбль не содержит моделей
        """
        if not self.models:
            raise ValueError("Ансамбль не содержит моделей")
        
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Обучение каждой модели в ансамбле
        for model in self.models:
            model.fit(X_np, y_np, **kwargs)
        
        # Обновление флага обучения
        self.is_fitted = True
        
        # Расчет метрик качества на обучающей выборке
        self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Ансамбль {self.name} обучен на данных размером {X_np.shape}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание целевой переменной.
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            ValueError: Если ансамбль не обучен
            ValueError: Если ансамбль не содержит моделей
        """
        if not self.is_fitted:
            raise ValueError("Ансамбль не обучен")
        
        if not self.models:
            raise ValueError("Ансамбль не содержит моделей")
        
        # Преобразование входных данных в numpy массив
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Предсказания каждой модели
        predictions = [model.predict(X_np) for model in self.models]
        
        # Агрегация предсказаний в зависимости от типа задачи и голосования
        if self.task_type == 'classification':
            if self.voting == 'hard':
                # Жесткое голосование (выбор класса по большинству голосов)
                predictions_array = np.array(predictions)
                # Для каждого примера выбираем класс, который предсказало большинство моделей
                final_predictions = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(), 
                    axis=0, 
                    arr=predictions_array
                )
            elif self.voting == 'soft':
                # Мягкое голосование (усреднение вероятностей)
                probas = [model.predict_proba(X_np) for model in self.models]
                # Усреднение вероятностей
                avg_probas = np.average(probas, axis=0, weights=self.weights)
                # Выбор класса с наибольшей вероятностью
                final_predictions = np.argmax(avg_probas, axis=1)
            else:  # weighted
                # Взвешенное голосование (взвешенное усреднение предсказаний)
                final_predictions = np.average(predictions, axis=0, weights=self.weights)
                # Округление до ближайшего целого для классификации
                final_predictions = np.round(final_predictions).astype(int)
        else:  # regression
            # Взвешенное усреднение предсказаний
            final_predictions = np.average(predictions, axis=0, weights=self.weights)
        
        return final_predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Предсказание вероятностей классов (только для классификации).
        
        Args:
            X: Матрица признаков
            
        Returns:
            np.ndarray: Предсказанные вероятности классов
            
        Raises:
            ValueError: Если ансамбль не обучен
            ValueError: Если ансамбль не содержит моделей
            ValueError: Если тип задачи не является классификацией
        """
        if not self.is_fitted:
            raise ValueError("Ансамбль не обучен")
        
        if not self.models:
            raise ValueError("Ансамбль не содержит моделей")
        
        if self.task_type != 'classification':
            raise ValueError("Метод predict_proba доступен только для задач классификации")
        
        # Преобразование входных данных в numpy массив
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # Предсказания вероятностей каждой модели
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X_np))
            else:
                self.logger.warning(f"Модель {model.name} не поддерживает предсказание вероятностей")
        
        if not probas:
            raise ValueError("Ни одна из моделей не поддерживает предсказание вероятностей")
        
        # Усреднение вероятностей
        avg_probas = np.average(probas, axis=0, weights=self.weights[:len(probas)])
        
        return avg_probas
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества ансамбля.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества ансамбля
        """
        # Предсказание
        y_pred = self.predict(X)
        
        # Расчет метрик в зависимости от типа задачи
        if self.task_type == 'classification':
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score
            )
            
            # Расчет метрик
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # Сохранение метрик
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            # Расчет ROC AUC, если возможно
            if hasattr(self, 'predict_proba') and len(np.unique(y)) == 2:
                try:
                    y_proba = self.predict_proba(X)
                    roc_auc = roc_auc_score(y, y_proba[:, 1])
                    self.metrics["roc_auc"] = roc_auc
                except:
                    self.logger.warning("Не удалось рассчитать ROC AUC")
        else:  # regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Расчет метрик
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Расчет MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-10))) * 100
            
            # Сохранение метрик
            self.metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape
            }
        
        # Обновление метаданных
        self.metadata.update({
            "metrics": self.metrics
        })
        
        return self.metrics
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка качества ансамбля на новых данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Метрики качества ансамбля
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Расчет метрик
        metrics = self._calculate_metrics(X_np, y_np)
        
        self.logger.info(f"Оценка ансамбля {self.name} на данных размером {X_np.shape}")
        self.logger.info(f"Метрики: {metrics}")
        
        return metrics
    
    def get_models(self) -> List[BaseModel]:
        """
        Получение списка моделей в ансамбле.
        
        Returns:
            List[BaseModel]: Список моделей
        """
        return self.models.copy()
    
    def get_weights(self) -> List[float]:
        """
        Получение весов моделей в ансамбле.
        
        Returns:
            List[float]: Список весов
        """
        return self.weights.copy()
    
    def set_weights(self, weights: List[float]) -> 'EnsembleModel':
        """
        Установка весов моделей в ансамбле.
        
        Args:
            weights: Список весов
            
        Returns:
            self: Ансамблевая модель с обновленными весами
            
        Raises:
            ValueError: Если количество весов не соответствует количеству моделей
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Количество весов ({len(weights)}) не соответствует количеству моделей ({len(self.models)})")
        
        # Установка весов
        self.weights = weights
        
        # Нормализация весов
        sum_weights = sum(self.weights)
        if sum_weights > 0:
            self.weights = [w / sum_weights for w in self.weights]
        
        # Обновление метаданных
        self.metadata.update({
            "weights": self.weights
        })
        
        return self 

class TimeSeriesModel(BaseModel):
    """
    Класс для моделей временных рядов.
    
    Предоставляет интерфейс для работы с моделями временных рядов,
    включая методы обучения, предсказания и оценки качества.
    
    Attributes:
        name (str): Имя модели
        model: Объект модели
        params (Dict[str, Any]): Параметры модели
        metadata (Dict[str, Any]): Метаданные модели
        is_fitted (bool): Флаг, указывающий, обучена ли модель
        metrics (Dict[str, float]): Метрики качества модели
        sequence_length (int): Длина последовательности для предсказания
        forecast_horizon (int): Горизонт прогнозирования
        seasonality (int): Период сезонности
    """
    
    def __init__(self, name: str, model: Any = None, 
                params: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None,
                sequence_length: int = 10,
                forecast_horizon: int = 1,
                seasonality: int = 1):
        """
        Инициализация модели временных рядов.
        
        Args:
            name: Имя модели
            model: Объект модели (если уже создан)
            params: Параметры модели
            metadata: Метаданные модели
            sequence_length: Длина последовательности для предсказания
            forecast_horizon: Горизонт прогнозирования
            seasonality: Период сезонности
        """
        super().__init__(name, model, params, metadata)
        
        # Параметры временного ряда
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.seasonality = seasonality
        
        # Метрики качества модели
        self.metrics = {}
        
        # Обновление метаданных
        self.metadata.update({
            "task_type": "time_series",
            "sequence_length": sequence_length,
            "forecast_horizon": forecast_horizon,
            "seasonality": seasonality
        })
    
    def _prepare_sequences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка последовательностей для обучения модели.
        
        Args:
            X: Временной ряд
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Входные последовательности и целевые значения
        """
        # Проверка размерности входных данных
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Создание последовательностей
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length - self.forecast_horizon + 1):
            # Входная последовательность
            seq = X[i:i+self.sequence_length]
            # Целевое значение (или последовательность)
            target = X[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
           y: Optional[Union[np.ndarray, pd.Series]] = None,
           **kwargs) -> 'TimeSeriesModel':
        """
        Обучение модели.
        
        Args:
            X: Временной ряд
            y: Целевая переменная (не используется для одномерных временных рядов)
            **kwargs: Дополнительные параметры для передачи в метод fit модели
            
        Returns:
            self: Обученная модель
            
        Raises:
            ValueError: Если модель не инициализирована
        """
        if self.model is None:
            raise ValueError("Модель не инициализирована")
        
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Подготовка последовательностей
        sequences, targets = self._prepare_sequences(X_np)
        
        # Обучение модели
        if hasattr(self.model, 'fit'):
            self.model.fit(sequences, targets.reshape(targets.shape[0], -1), **kwargs)
        
        # Обновление флага обучения
        self.is_fitted = True
        
        # Расчет метрик качества на обучающей выборке
        self._calculate_metrics(X_np)
        
        self.logger.info(f"Модель {self.name} обучена на данных размером {X_np.shape}")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Предсказание временного ряда.
        
        Args:
            X: Временной ряд
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Проверка размерности входных данных
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Если длина входных данных меньше sequence_length, дополняем нулями
        if len(X_np) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(X_np), X_np.shape[1]))
            X_np = np.vstack([padding, X_np])
        
        # Получение последней последовательности
        last_sequence = X_np[-self.sequence_length:]
        
        # Предсказание
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(last_sequence.reshape(1, self.sequence_length, -1))
            # Преобразование предсказания в нужную форму
            prediction = prediction.reshape(self.forecast_horizon, -1)
        else:
            raise ValueError("Модель не поддерживает метод predict")
        
        return prediction
    
    def forecast(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
                    steps: int = 1, recursive: bool = True) -> np.ndarray:
        """
        Прогнозирование временного ряда на несколько шагов вперед.
        
        Args:
            X: Временной ряд
            steps: Количество шагов для прогнозирования
            recursive: Использовать ли рекурсивное прогнозирование
            
        Returns:
            np.ndarray: Прогноз временного ряда
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Проверка размерности входных данных
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Если длина входных данных меньше sequence_length, дополняем нулями
        if len(X_np) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(X_np), X_np.shape[1]))
            X_np = np.vstack([padding, X_np])
        
        # Получение последней последовательности
        sequence = X_np[-self.sequence_length:].copy()
        
        # Прогнозирование
        forecasts = []
        
        for _ in range(steps):
            # Предсказание следующего шага
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(sequence.reshape(1, self.sequence_length, -1))
                # Преобразование предсказания в нужную форму
                prediction = prediction.reshape(self.forecast_horizon, -1)
                
                # Добавление предсказания в результат
                forecasts.append(prediction[0])
                
                # Обновление последовательности для рекурсивного прогнозирования
                if recursive and self.forecast_horizon == 1:
                    sequence = np.vstack([sequence[1:], prediction[0]])
            else:
                raise ValueError("Модель не поддерживает метод predict")
        
        return np.array(forecasts)
    
    def _calculate_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества модели.
        
        Args:
            X: Временной ряд
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Подготовка последовательностей
        sequences, targets = self._prepare_sequences(X)
        
        # Предсказание
        predictions = []
        for seq in sequences:
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(seq.reshape(1, self.sequence_length, -1))
                predictions.append(pred.reshape(self.forecast_horizon, -1))
            else:
                raise ValueError("Модель не поддерживает метод predict")
        
        predictions = np.array(predictions)
        
        # Расчет метрик
        mse = mean_squared_error(targets.reshape(-1), predictions.reshape(-1))
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets.reshape(-1), predictions.reshape(-1))
        
        # Расчет MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targets.reshape(-1) - predictions.reshape(-1)) / np.maximum(np.abs(targets.reshape(-1)), 1e-10))) * 100
        
        # Сохранение метрик
        self.metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        }
        
        # Обновление метаданных
        self.metadata.update({
            "metrics": self.metrics
        })
        
        return self.metrics
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Dict[str, float]:
        """
        Оценка качества модели на новых данных.
        
        Args:
            X: Временной ряд
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Расчет метрик
        metrics = self._calculate_metrics(X_np)
        
        self.logger.info(f"Оценка модели {self.name} на данных размером {X_np.shape}")
        self.logger.info(f"Метрики: {metrics}")
        
        return metrics
    
    def plot_forecast(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
                         steps: int = 10, recursive: bool = True,
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика прогноза временного ряда.
        
        Args:
            X: Временной ряд
            steps: Количество шагов для прогнозирования
            recursive: Использовать ли рекурсивное прогнозирование
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
        """
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Прогнозирование
        forecast = self.forecast(X_np, steps=steps, recursive=recursive)
        
        # Построение графика
        fig, ax = plt.subplots(figsize=figsize)
        
        # Исторические данные
        ax.plot(range(len(X_np)), X_np[:, 0], label='Исторические данные')
        
        # Прогноз
        ax.plot(range(len(X_np), len(X_np) + len(forecast)), forecast[:, 0], 'r--', label='Прогноз')
        
        # Настройка графика
        ax.set_title(f'Прогноз временного ряда моделью {self.name}')
        ax.set_xlabel('Время')
        ax.set_ylabel('Значение')
        ax.legend()
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"График прогноза сохранен в {save_path}")
        
        return fig
    
    def plot_components(self, X: Union[np.ndarray, pd.DataFrame, pd.Series],
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Построение графика компонент временного ряда (тренд, сезонность, остатки).
        
        Args:
            X: Временной ряд
            figsize: Размер графика
            save_path: Путь для сохранения графика
            
        Returns:
            plt.Figure: Объект графика
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Преобразование входных данных в numpy массивы
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_np = X.values
        else:
            X_np = X
        
        # Проверка размерности входных данных
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        
        # Преобразование в pandas Series для декомпозиции
        series = pd.Series(X_np[:, 0])
        
        # Декомпозиция временного ряда
        try:
            result = seasonal_decompose(series, model='additive', period=self.seasonality)
            
            # Построение графика
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            # Исходный ряд
            axes[0].plot(series.values)
            axes[0].set_title('Исходный временной ряд')
            
            # Тренд
            axes[1].plot(result.trend.values)
            axes[1].set_title('Тренд')
            
            # Сезонность
            axes[2].plot(result.seasonal.values)
            axes[2].set_title('Сезонность')
            
            # Остатки
            axes[3].plot(result.resid.values)
            axes[3].set_title('Остатки')
            
            # Настройка графика
            plt.tight_layout()
            
            # Сохранение графика, если указан путь
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"График компонент сохранен в {save_path}")
            
            return fig
        except:
            self.logger.warning("Не удалось выполнить декомпозицию временного ряда")
            return None 