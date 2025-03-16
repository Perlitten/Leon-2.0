"""
Модуль классификационных моделей машинного обучения.

Предоставляет реализацию классификационных моделей на основе базового класса.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from ml.models.base_model import BaseModel


class ClassificationModel(BaseModel):
    """
    Класс для классификационных моделей машинного обучения.
    
    Поддерживает различные алгоритмы классификации из scikit-learn.
    """
    
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'svm': SVC,
        'naive_bayes': GaussianNB,
        'knn': KNeighborsClassifier
    }
    
    def __init__(self, name: str, algorithm: str = 'logistic', version: str = "1.0.0", **kwargs):
        """
        Инициализация классификационной модели.
        
        Args:
            name: Название модели
            algorithm: Алгоритм классификации ('logistic', 'random_forest', 'gradient_boosting', 
                      'svm', 'naive_bayes', 'knn')
            version: Версия модели
            **kwargs: Параметры для выбранного алгоритма
        
        Raises:
            ValueError: Если указан неподдерживаемый алгоритм
        """
        super().__init__(name=name, version=version, **kwargs)
        
        if algorithm not in self.SUPPORTED_MODELS:
            raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}. "
                            f"Доступные алгоритмы: {', '.join(self.SUPPORTED_MODELS.keys())}")
        
        self.algorithm = algorithm
        # Для SVM и некоторых других моделей добавляем probability=True для поддержки predict_proba
        if algorithm == 'svm':
            kwargs['probability'] = True
            
        self.model = self.SUPPORTED_MODELS[algorithm](**kwargs)
        self.logger.info(f"Создана классификационная модель с алгоритмом {algorithm}")
        self.classes_ = None
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Обучение классификационной модели на данных.
        
        Args:
            X: Признаки для обучения
            y: Целевые значения (метки классов)
            **kwargs: Дополнительные параметры обучения
            
        Returns:
            Dict[str, Any]: Метрики обучения
        """
        self.logger.info(f"Начало обучения модели {self.name}")
        
        # Преобразование входных данных в numpy массивы, если необходимо
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = y.values if isinstance(y, pd.Series) else y
        
        # Обучение модели
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Оценка на обучающих данных
        y_pred = self.model.predict(X_train)
        
        # Расчет метрик
        metrics = self._calculate_metrics(y_train, y_pred)
        
        self.logger.info(f"Модель {self.name} обучена. Метрики на обучающих данных: {metrics}")
        return metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение предсказаний классификационной модели.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказанные метки классов
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Преобразование входных данных в numpy массивы, если необходимо
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        
        return self.model.predict(X_pred)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение вероятностей классов.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Вероятности принадлежности к каждому классу
            
        Raises:
            RuntimeError: Если модель не обучена
            AttributeError: Если модель не поддерживает вероятностные предсказания
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Преобразование входных данных в numpy массивы, если необходимо
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        
        # Проверка, поддерживает ли модель вероятностные предсказания
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_pred)
        else:
            self.logger.warning(f"Модель {self.algorithm} не поддерживает вероятностные предсказания")
            # Возвращаем one-hot encoding предсказаний
            y_pred = self.model.predict(X_pred)
            n_samples = len(y_pred)
            n_classes = len(self.classes_)
            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(y_pred):
                class_idx = np.where(self.classes_ == pred)[0][0]
                probas[i, class_idx] = 1.0
            return probas
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка производительности классификационной модели на данных.
        
        Args:
            X: Признаки для оценки
            y: Истинные метки классов
            
        Returns:
            Dict[str, float]: Метрики оценки
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Преобразование входных данных в numpy массивы, если необходимо
        X_eval = X.values if isinstance(X, pd.DataFrame) else X
        y_true = y.values if isinstance(y, pd.Series) else y
        
        # Получение предсказаний
        y_pred = self.model.predict(X_eval)
        
        # Расчет метрик
        metrics = self._calculate_metrics(y_true, y_pred)
        
        self.logger.info(f"Оценка модели {self.name}. Метрики: {metrics}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик для оценки классификационной модели.
        
        Args:
            y_true: Истинные метки классов
            y_pred: Предсказанные метки классов
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Добавление метрик для бинарной классификации, если применимо
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            # Для бинарной классификации добавляем ROC AUC
            try:
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X_eval)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                self.logger.warning(f"Не удалось рассчитать ROC AUC: {str(e)}")
            
            # Добавляем метрики для каждого класса
            metrics['precision_class_0'] = precision_score(y_true, y_pred, pos_label=unique_classes[0])
            metrics['recall_class_0'] = recall_score(y_true, y_pred, pos_label=unique_classes[0])
            metrics['f1_class_0'] = f1_score(y_true, y_pred, pos_label=unique_classes[0])
            
            metrics['precision_class_1'] = precision_score(y_true, y_pred, pos_label=unique_classes[1])
            metrics['recall_class_1'] = recall_score(y_true, y_pred, pos_label=unique_classes[1])
            metrics['f1_class_1'] = f1_score(y_true, y_pred, pos_label=unique_classes[1])
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Получение важности признаков, если поддерживается моделью.
        
        Returns:
            Optional[np.ndarray]: Массив важности признаков или None, если не поддерживается
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Проверка, поддерживает ли модель получение важности признаков
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Для моделей с коэффициентами (например, логистическая регрессия)
            if len(self.model.coef_.shape) == 2:
                # Для многоклассовой классификации берем среднее по модулю
                return np.mean(np.abs(self.model.coef_), axis=0)
            else:
                return np.abs(self.model.coef_)
        else:
            self.logger.warning(f"Модель {self.algorithm} не поддерживает получение важности признаков")
            return None 