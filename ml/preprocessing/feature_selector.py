"""
Модуль выбора признаков для машинного обучения.

Предоставляет классы и функции для отбора наиболее информативных признаков
для моделей машинного обучения.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import pickle
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    RFE, RFECV, VarianceThreshold,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector:
    """
    Класс для выбора признаков для машинного обучения.
    
    Предоставляет методы для отбора наиболее информативных признаков
    с использованием различных алгоритмов.
    
    Attributes:
        config (Dict[str, Any]): Конфигурация выбора признаков
        selectors (Dict[str, Any]): Словарь селекторов признаков
        feature_names (List[str]): Список имен признаков
        selected_features (Dict[str, List[str]]): Словарь выбранных признаков для каждого селектора
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация селектора признаков.
        
        Args:
            config: Конфигурация выбора признаков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.config = config or {}
        self.selectors = {}
        self.feature_names = []
        self.selected_features = {}
        
        self.logger.info("Инициализирован селектор признаков")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Union[np.ndarray, pd.Series],
           feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Обучение селектора признаков.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            feature_names: Список имен признаков
            
        Returns:
            FeatureSelector: Обученный селектор признаков
        """
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = X_df.columns.tolist()
        
        # Сохранение имен признаков
        self.feature_names = feature_names
        
        # Обучение селекторов
        for selector_name, selector in self.selectors.items():
            if hasattr(selector, 'fit'):
                selector.fit(X_df, y)
                
                # Сохранение выбранных признаков
                if hasattr(selector, 'get_support'):
                    mask = selector.get_support()
                    self.selected_features[selector_name] = [feature for feature, selected in zip(feature_names, mask) if selected]
                
                self.logger.info(f"Обучен селектор {selector_name}, выбрано {len(self.selected_features.get(selector_name, []))} признаков")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                selector_name: Optional[str] = None) -> pd.DataFrame:
        """
        Преобразование данных с выбором признаков.
        
        Args:
            X: Матрица признаков
            selector_name: Имя селектора для применения (если None, используется первый селектор)
            
        Returns:
            pd.DataFrame: Преобразованные данные с выбранными признаками
            
        Raises:
            ValueError: Если селектор с указанным именем не найден или нет селекторов
        """
        # Проверка наличия селекторов
        if not self.selectors:
            raise ValueError("Нет доступных селекторов")
        
        # Определение селектора для применения
        if selector_name is None:
            selector_name = next(iter(self.selectors))
        elif selector_name not in self.selectors:
            raise ValueError(f"Селектор с именем {selector_name} не найден")
        
        selector = self.selectors[selector_name]
        
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            if len(self.feature_names) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Применение селектора
        if hasattr(selector, 'transform'):
            X_transformed = selector.transform(X_df)
            
            # Если результат - массив, создаем DataFrame
            if isinstance(X_transformed, np.ndarray):
                X_transformed = pd.DataFrame(
                    X_transformed, 
                    index=X_df.index, 
                    columns=self.selected_features.get(selector_name, [f"feature_{i}" for i in range(X_transformed.shape[1])])
                )
            
            self.logger.info(f"Применен селектор {selector_name}, выбрано {X_transformed.shape[1]} признаков")
            
            return X_transformed
        else:
            # Если селектор не имеет метода transform, возвращаем только выбранные признаки
            selected = self.selected_features.get(selector_name, [])
            
            if not selected:
                self.logger.warning(f"Селектор {selector_name} не имеет выбранных признаков")
                return X_df
            
            return X_df[selected]
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]] = None,
                     selector_name: Optional[str] = None) -> pd.DataFrame:
        """
        Обучение и преобразование данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная
            feature_names: Список имен признаков
            selector_name: Имя селектора для применения (если None, используется первый селектор)
            
        Returns:
            pd.DataFrame: Преобразованные данные с выбранными признаками
        """
        return self.fit(X, y, feature_names).transform(X, selector_name)
    
    def add_selector(self, name: str, selector: Any) -> 'FeatureSelector':
        """
        Добавление селектора признаков.
        
        Args:
            name: Имя селектора
            selector: Селектор признаков
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        self.selectors[name] = selector
        self.logger.info(f"Добавлен селектор {name}")
        
        return self
    
    def remove_selector(self, name: str) -> 'FeatureSelector':
        """
        Удаление селектора признаков.
        
        Args:
            name: Имя селектора
            
        Returns:
            FeatureSelector: Селектор признаков
            
        Raises:
            ValueError: Если селектор с указанным именем не найден
        """
        if name not in self.selectors:
            raise ValueError(f"Селектор с именем {name} не найден")
        
        del self.selectors[name]
        
        if name in self.selected_features:
            del self.selected_features[name]
        
        self.logger.info(f"Удален селектор {name}")
        
        return self
    
    def add_variance_threshold(self, name: str, threshold: float = 0.0) -> 'FeatureSelector':
        """
        Добавление селектора на основе порога дисперсии.
        
        Args:
            name: Имя селектора
            threshold: Порог дисперсии
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        selector = VarianceThreshold(threshold=threshold)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_k_best(self, name: str, k: int = 10, 
                 score_func: Optional[Callable] = None,
                 task_type: str = 'classification') -> 'FeatureSelector':
        """
        Добавление селектора k лучших признаков.
        
        Args:
            name: Имя селектора
            k: Количество признаков для выбора
            score_func: Функция оценки признаков
            task_type: Тип задачи ('classification' или 'regression')
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение функции оценки
        if score_func is None:
            if task_type == 'classification':
                score_func = f_classif
            else:  # 'regression'
                score_func = f_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_percentile(self, name: str, percentile: int = 10, 
                     score_func: Optional[Callable] = None,
                     task_type: str = 'classification') -> 'FeatureSelector':
        """
        Добавление селектора на основе процентиля.
        
        Args:
            name: Имя селектора
            percentile: Процентиль признаков для выбора
            score_func: Функция оценки признаков
            task_type: Тип задачи ('classification' или 'regression')
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение функции оценки
        if score_func is None:
            if task_type == 'classification':
                score_func = f_classif
            else:  # 'regression'
                score_func = f_regression
        
        selector = SelectPercentile(score_func=score_func, percentile=percentile)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_mutual_info(self, name: str, k: int = 10, 
                      task_type: str = 'classification') -> 'FeatureSelector':
        """
        Добавление селектора на основе взаимной информации.
        
        Args:
            name: Имя селектора
            k: Количество признаков для выбора
            task_type: Тип задачи ('classification' или 'regression')
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение функции оценки
        if task_type == 'classification':
            score_func = mutual_info_classif
        else:  # 'regression'
            score_func = mutual_info_regression
        
        selector = SelectKBest(score_func=score_func, k=k)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_from_model(self, name: str, estimator: Optional[BaseEstimator] = None,
                     task_type: str = 'classification',
                     threshold: Optional[Union[str, float]] = None,
                     max_features: Optional[int] = None) -> 'FeatureSelector':
        """
        Добавление селектора на основе модели.
        
        Args:
            name: Имя селектора
            estimator: Модель для выбора признаков
            task_type: Тип задачи ('classification' или 'regression')
            threshold: Порог для выбора признаков
            max_features: Максимальное количество признаков
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение модели
        if estimator is None:
            if task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # 'regression'
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = SelectFromModel(estimator=estimator, threshold=threshold, max_features=max_features)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_rfe(self, name: str, estimator: Optional[BaseEstimator] = None,
              task_type: str = 'classification',
              n_features_to_select: Optional[int] = None,
              step: Union[int, float] = 1) -> 'FeatureSelector':
        """
        Добавление селектора на основе рекурсивного исключения признаков.
        
        Args:
            name: Имя селектора
            estimator: Модель для выбора признаков
            task_type: Тип задачи ('classification' или 'regression')
            n_features_to_select: Количество признаков для выбора
            step: Шаг исключения признаков
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение модели
        if estimator is None:
            if task_type == 'classification':
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            else:  # 'regression'
                estimator = Lasso(random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        
        self.add_selector(name, selector)
        
        return self
    
    def add_rfecv(self, name: str, estimator: Optional[BaseEstimator] = None,
                task_type: str = 'classification',
                min_features_to_select: int = 1,
                step: Union[int, float] = 1,
                cv: int = 5) -> 'FeatureSelector':
        """
        Добавление селектора на основе рекурсивного исключения признаков с кросс-валидацией.
        
        Args:
            name: Имя селектора
            estimator: Модель для выбора признаков
            task_type: Тип задачи ('classification' или 'regression')
            min_features_to_select: Минимальное количество признаков для выбора
            step: Шаг исключения признаков
            cv: Количество фолдов для кросс-валидации
            
        Returns:
            FeatureSelector: Селектор признаков
        """
        # Определение модели
        if estimator is None:
            if task_type == 'classification':
                estimator = LogisticRegression(max_iter=1000, random_state=42)
            else:  # 'regression'
                estimator = Lasso(random_state=42)
        
        selector = RFECV(estimator=estimator, min_features_to_select=min_features_to_select, step=step, cv=cv)
        
        self.add_selector(name, selector)
        
        return self
    
    def get_feature_importances(self, selector_name: Optional[str] = None) -> pd.DataFrame:
        """
        Получение важности признаков.
        
        Args:
            selector_name: Имя селектора (если None, используется первый селектор)
            
        Returns:
            pd.DataFrame: DataFrame с важностью признаков
            
        Raises:
            ValueError: Если селектор с указанным именем не найден или нет селекторов
        """
        # Проверка наличия селекторов
        if not self.selectors:
            raise ValueError("Нет доступных селекторов")
        
        # Определение селектора
        if selector_name is None:
            selector_name = next(iter(self.selectors))
        elif selector_name not in self.selectors:
            raise ValueError(f"Селектор с именем {selector_name} не найден")
        
        selector = self.selectors[selector_name]
        
        # Получение важности признаков
        importances = None
        
        if hasattr(selector, 'scores_'):
            importances = selector.scores_
        elif hasattr(selector, 'feature_importances_'):
            importances = selector.feature_importances_
        elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
            importances = selector.estimator_.feature_importances_
        elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_).ravel()
        
        if importances is None:
            self.logger.warning(f"Селектор {selector_name} не имеет информации о важности признаков")
            return pd.DataFrame()
        
        # Создание DataFrame с важностью признаков
        if len(importances) == len(self.feature_names):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
        else:
            importance_df = pd.DataFrame({
                'feature': [f"feature_{i}" for i in range(len(importances))],
                'importance': importances
            })
        
        # Сортировка по убыванию важности
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_selected_features(self, selector_name: Optional[str] = None) -> List[str]:
        """
        Получение списка выбранных признаков.
        
        Args:
            selector_name: Имя селектора (если None, используется первый селектор)
            
        Returns:
            List[str]: Список выбранных признаков
            
        Raises:
            ValueError: Если селектор с указанным именем не найден или нет селекторов
        """
        # Проверка наличия селекторов
        if not self.selectors:
            raise ValueError("Нет доступных селекторов")
        
        # Определение селектора
        if selector_name is None:
            selector_name = next(iter(self.selectors))
        elif selector_name not in self.selectors:
            raise ValueError(f"Селектор с именем {selector_name} не найден")
        
        # Получение выбранных признаков
        return self.selected_features.get(selector_name, [])
    
    def save(self, path: str) -> str:
        """
        Сохранение селектора признаков.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            str: Путь к сохраненному селектору признаков
        """
        import pickle
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение селектора признаков
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Селектор признаков сохранен в {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'FeatureSelector':
        """
        Загрузка селектора признаков.
        
        Args:
            path: Путь к сохраненному селектору признаков
            
        Returns:
            FeatureSelector: Загруженный селектор признаков
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        import pickle
        
        # Проверка существования файла
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")
        
        # Загрузка селектора признаков
        with open(path, 'rb') as f:
            feature_selector = pickle.load(f)
        
        feature_selector.logger.info(f"Селектор признаков загружен из {path}")
        
        return feature_selector 