"""
Модуль для выбора признаков в моделях машинного обучения.

Предоставляет классы и функции для отбора наиболее информативных признаков
с использованием различных методов.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class FeatureSelector:
    """
    Класс для выбора признаков в моделях машинного обучения.
    
    Предоставляет различные методы отбора признаков:
    - Фильтрация на основе статистических тестов (ANOVA, корреляция и т.д.)
    - Встроенные методы (на основе моделей, например, важность признаков в деревьях)
    - Рекурсивное исключение признаков
    - Удаление признаков с низкой дисперсией
    
    Attributes:
        method (str): Метод отбора признаков
        n_features (int): Количество признаков для отбора
        percentile (int): Процент признаков для отбора
        threshold (float): Порог для отбора признаков
        estimator: Модель для встроенных методов отбора
        feature_names (List[str]): Имена признаков
        selected_features (List[str]): Имена отобранных признаков
        feature_scores (Dict[str, float]): Оценки важности признаков
        selector: Объект для отбора признаков
    """
    
    # Словарь доступных методов отбора признаков
    AVAILABLE_METHODS = {
        # Фильтрация
        'k_best_anova': 'SelectKBest с f_classif',
        'k_best_regression': 'SelectKBest с f_regression',
        'k_best_mutual_info_classif': 'SelectKBest с mutual_info_classif',
        'k_best_mutual_info_regression': 'SelectKBest с mutual_info_regression',
        'percentile_anova': 'SelectPercentile с f_classif',
        'percentile_regression': 'SelectPercentile с f_regression',
        'percentile_mutual_info_classif': 'SelectPercentile с mutual_info_classif',
        'percentile_mutual_info_regression': 'SelectPercentile с mutual_info_regression',
        'variance_threshold': 'VarianceThreshold',
        
        # Встроенные методы
        'random_forest': 'SelectFromModel с RandomForest',
        'lasso': 'SelectFromModel с Lasso',
        'logistic_regression': 'SelectFromModel с LogisticRegression',
        
        # Рекурсивное исключение признаков
        'rfe': 'RFE (Recursive Feature Elimination)',
        'rfecv': 'RFECV (Recursive Feature Elimination with Cross-Validation)'
    }
    
    def __init__(self, method: str = 'k_best_anova', n_features: int = 10, 
                 percentile: int = 20, threshold: float = 0.01,
                 estimator = None, feature_names: List[str] = None):
        """
        Инициализация селектора признаков.
        
        Args:
            method: Метод отбора признаков
            n_features: Количество признаков для отбора (для методов k_best и rfe)
            percentile: Процент признаков для отбора (для методов percentile)
            threshold: Порог для отбора признаков (для методов variance_threshold и встроенных)
            estimator: Модель для встроенных методов отбора
            feature_names: Имена признаков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Метод {method} не поддерживается. Доступные методы: {list(self.AVAILABLE_METHODS.keys())}")
        
        self.method = method
        self.n_features = n_features
        self.percentile = percentile
        self.threshold = threshold
        self.estimator = estimator
        self.feature_names = feature_names
        self.selected_features = None
        self.feature_scores = None
        self.selector = None
        
        self.logger.info(f"Инициализирован селектор признаков с методом {method}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'FeatureSelector':
        """
        Обучение селектора признаков.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            self: Обученный селектор признаков
        """
        # Преобразование входных данных в numpy массивы
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Сохранение имен признаков, если они не были переданы при инициализации
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]
        
        # Создание и обучение селектора в зависимости от выбранного метода
        if self.method.startswith('k_best'):
            if self.method == 'k_best_anova':
                self.selector = SelectKBest(f_classif, k=self.n_features)
            elif self.method == 'k_best_regression':
                self.selector = SelectKBest(f_regression, k=self.n_features)
            elif self.method == 'k_best_mutual_info_classif':
                self.selector = SelectKBest(mutual_info_classif, k=self.n_features)
            elif self.method == 'k_best_mutual_info_regression':
                self.selector = SelectKBest(mutual_info_regression, k=self.n_features)
        
        elif self.method.startswith('percentile'):
            if self.method == 'percentile_anova':
                self.selector = SelectPercentile(f_classif, percentile=self.percentile)
            elif self.method == 'percentile_regression':
                self.selector = SelectPercentile(f_regression, percentile=self.percentile)
            elif self.method == 'percentile_mutual_info_classif':
                self.selector = SelectPercentile(mutual_info_classif, percentile=self.percentile)
            elif self.method == 'percentile_mutual_info_regression':
                self.selector = SelectPercentile(mutual_info_regression, percentile=self.percentile)
        
        elif self.method == 'variance_threshold':
            self.selector = VarianceThreshold(threshold=self.threshold)
        
        elif self.method.startswith('random_forest'):
            if self.estimator is None:
                if len(np.unique(y_np)) <= 10:  # Эвристика для определения задачи классификации
                    self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector = SelectFromModel(self.estimator, threshold=self.threshold)
        
        elif self.method == 'lasso':
            if self.estimator is None:
                self.estimator = Lasso(alpha=0.01)
            self.selector = SelectFromModel(self.estimator, threshold=self.threshold)
        
        elif self.method == 'logistic_regression':
            if self.estimator is None:
                self.estimator = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=42)
            self.selector = SelectFromModel(self.estimator, threshold=self.threshold)
        
        elif self.method == 'rfe':
            if self.estimator is None:
                if len(np.unique(y_np)) <= 10:  # Эвристика для определения задачи классификации
                    self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector = RFE(self.estimator, n_features_to_select=self.n_features)
        
        elif self.method == 'rfecv':
            if self.estimator is None:
                if len(np.unique(y_np)) <= 10:  # Эвристика для определения задачи классификации
                    self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            self.selector = RFECV(self.estimator, min_features_to_select=1, cv=5)
        
        # Обучение селектора
        self.logger.info(f"Обучение селектора признаков на данных размером {X_np.shape}")
        self.selector.fit(X_np, y_np)
        
        # Получение масок выбранных признаков
        if hasattr(self.selector, 'get_support'):
            mask = self.selector.get_support()
            self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
        else:
            self.selected_features = self.feature_names
        
        # Получение оценок важности признаков
        self.feature_scores = self._get_feature_scores(X_np, y_np)
        
        self.logger.info(f"Выбрано {len(self.selected_features)} признаков из {len(self.feature_names)}")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Преобразование данных с использованием выбранных признаков.
        
        Args:
            X: Матрица признаков
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Преобразованные данные
        """
        if self.selector is None:
            raise ValueError("Селектор не обучен. Сначала вызовите метод fit.")
        
        # Преобразование входных данных
        is_dataframe = isinstance(X, pd.DataFrame)
        X_np = X.values if is_dataframe else X
        
        # Применение селектора
        X_transformed = self.selector.transform(X_np)
        
        # Возвращение результата в том же формате, что и входные данные
        if is_dataframe:
            return pd.DataFrame(X_transformed, index=X.index, columns=self.selected_features)
        else:
            return X_transformed
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Обучение селектора и преобразование данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Преобразованные данные
        """
        return self.fit(X, y).transform(X)
    
    def _get_feature_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Получение оценок важности признаков.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            
        Returns:
            Dict[str, float]: Словарь с оценками важности признаков
        """
        scores = {}
        
        # Для методов на основе статистических тестов
        if hasattr(self.selector, 'scores_'):
            for i, feature_name in enumerate(self.feature_names):
                scores[feature_name] = float(self.selector.scores_[i]) if i < len(self.selector.scores_) else 0.0
        
        # Для методов на основе моделей
        elif hasattr(self.selector, 'estimator_') and hasattr(self.selector.estimator_, 'feature_importances_'):
            for i, feature_name in enumerate(self.feature_names):
                scores[feature_name] = float(self.selector.estimator_.feature_importances_[i]) if i < len(self.selector.estimator_.feature_importances_) else 0.0
        
        # Для методов на основе коэффициентов
        elif hasattr(self.selector, 'estimator_') and hasattr(self.selector.estimator_, 'coef_'):
            coefs = self.selector.estimator_.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)
            for i, feature_name in enumerate(self.feature_names):
                scores[feature_name] = float(coefs[i]) if i < len(coefs) else 0.0
        
        # Для RFE и RFECV
        elif hasattr(self.selector, 'ranking_'):
            for i, feature_name in enumerate(self.feature_names):
                # Инвертируем ранги, чтобы более важные признаки имели более высокие оценки
                scores[feature_name] = 1.0 / float(self.selector.ranking_[i]) if i < len(self.selector.ranking_) else 0.0
        
        # Если не удалось получить оценки, используем маску выбранных признаков
        elif hasattr(self.selector, 'get_support'):
            mask = self.selector.get_support()
            for i, feature_name in enumerate(self.feature_names):
                scores[feature_name] = 1.0 if i < len(mask) and mask[i] else 0.0
        
        return scores
    
    def get_feature_ranking(self) -> List[Tuple[str, float]]:
        """
        Получение ранжированного списка признаков по их важности.
        
        Returns:
            List[Tuple[str, float]]: Список кортежей (имя признака, оценка важности)
        """
        if self.feature_scores is None:
            raise ValueError("Селектор не обучен. Сначала вызовите метод fit.")
        
        # Сортировка признаков по убыванию оценок важности
        return sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8), 
                               save_path: Optional[str] = None) -> None:
        """
        Визуализация важности признаков.
        
        Args:
            top_n: Количество наиболее важных признаков для отображения
            figsize: Размер графика
            save_path: Путь для сохранения графика
        """
        if self.feature_scores is None:
            raise ValueError("Селектор не обучен. Сначала вызовите метод fit.")
        
        # Получение ранжированного списка признаков
        feature_ranking = self.get_feature_ranking()
        
        # Ограничение количества отображаемых признаков
        if top_n > 0:
            feature_ranking = feature_ranking[:top_n]
        
        # Создание графика
        plt.figure(figsize=figsize)
        
        # Построение горизонтальной столбчатой диаграммы
        features = [f[0] for f in feature_ranking]
        scores = [f[1] for f in feature_ranking]
        
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Важность признака')
        plt.title(f'Топ-{len(features)} важных признаков ({self.method})')
        plt.tight_layout()
        
        # Сохранение графика, если указан путь
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"График важности признаков сохранен в {save_path}")
        
        plt.show()
    
    def save(self, path: str) -> str:
        """
        Сохранение селектора признаков.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        import pickle
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение селектора
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Селектор признаков сохранен в {path}")
        return path
    
    def load(self, path: str) -> 'FeatureSelector':
        """
        Загрузка селектора признаков.
        
        Args:
            path: Путь к сохраненному селектору
            
        Returns:
            FeatureSelector: Загруженный селектор признаков
        """
        import pickle
        
        # Загрузка селектора
        with open(path, 'rb') as f:
            loaded_selector = pickle.load(f)
        
        # Копирование атрибутов
        self.__dict__.update(loaded_selector.__dict__)
        
        self.logger.info(f"Селектор признаков загружен из {path}")
        return self
    
    def get_support_mask(self) -> np.ndarray:
        """
        Получение маски выбранных признаков.
        
        Returns:
            np.ndarray: Булева маска выбранных признаков
        """
        if self.selector is None:
            raise ValueError("Селектор не обучен. Сначала вызовите метод fit.")
        
        if hasattr(self.selector, 'get_support'):
            return self.selector.get_support()
        else:
            # Если селектор не имеет метода get_support, возвращаем маску на основе selected_features
            mask = np.zeros(len(self.feature_names), dtype=bool)
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in self.selected_features:
                    mask[i] = True
            return mask
    
    def get_params(self) -> Dict[str, Any]:
        """
        Получение параметров селектора.
        
        Returns:
            Dict[str, Any]: Словарь с параметрами селектора
        """
        return {
            'method': self.method,
            'n_features': self.n_features,
            'percentile': self.percentile,
            'threshold': self.threshold,
            'estimator': self.estimator,
            'feature_names': self.feature_names
        }
    
    def set_params(self, **params) -> 'FeatureSelector':
        """
        Установка параметров селектора.
        
        Args:
            **params: Параметры для установки
            
        Returns:
            FeatureSelector: Селектор с обновленными параметрами
        """
        for key, value in params.items():
            if key in ['method', 'n_features', 'percentile', 'threshold', 'estimator', 'feature_names']:
                setattr(self, key, value)
            else:
                raise ValueError(f"Неизвестный параметр: {key}")
        
        # Сброс селектора, так как параметры изменились
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
        
        return self
    
    def __str__(self) -> str:
        """
        Строковое представление селектора.
        
        Returns:
            str: Строковое представление
        """
        return (f"FeatureSelector(method='{self.method}', "
                f"n_features={self.n_features}, "
                f"percentile={self.percentile}, "
                f"threshold={self.threshold})")
    
    def __repr__(self) -> str:
        """
        Представление селектора для отладки.
        
        Returns:
            str: Представление для отладки
        """
        return self.__str__() 