"""
Модуль регрессионных моделей машинного обучения.

Предоставляет реализацию регрессионных моделей на основе базового класса.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from ml.models.base_model import BaseModel


class RegressionModel(BaseModel):
    """
    Класс для регрессионных моделей машинного обучения.
    
    Поддерживает различные алгоритмы регрессии из scikit-learn.
    """
    
    SUPPORTED_MODELS = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor
    }
    
    def __init__(self, name: str, algorithm: str = 'linear', version: str = "1.0.0", **kwargs):
        """
        Инициализация регрессионной модели.
        
        Args:
            name: Название модели
            algorithm: Алгоритм регрессии ('linear', 'ridge', 'lasso', 'elastic_net', 
                      'random_forest', 'gradient_boosting')
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
        self.model = self.SUPPORTED_MODELS[algorithm](**kwargs)
        self.logger.info(f"Создана регрессионная модель с алгоритмом {algorithm}")
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Обучение регрессионной модели на данных.
        
        Args:
            X: Признаки для обучения
            y: Целевые значения
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
        
        # Оценка на обучающих данных
        y_pred = self.model.predict(X_train)
        metrics = {
            'mse': mean_squared_error(y_train, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
            'mae': mean_absolute_error(y_train, y_pred),
            'r2': r2_score(y_train, y_pred)
        }
        
        self.logger.info(f"Модель {self.name} обучена. Метрики на обучающих данных: {metrics}")
        return metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение предсказаний регрессионной модели.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Преобразование входных данных в numpy массивы, если необходимо
        X_pred = X.values if isinstance(X, pd.DataFrame) else X
        
        return self.model.predict(X_pred)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка производительности регрессионной модели на данных.
        
        Args:
            X: Признаки для оценки
            y: Истинные значения
            
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
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        self.logger.info(f"Оценка модели {self.name}. Метрики: {metrics}")
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
            return self.model.coef_
        else:
            self.logger.warning(f"Модель {self.algorithm} не поддерживает получение важности признаков")
            return None
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Для регрессионных моделей этот метод не применим, но реализован для совместимости.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Пустой массив
        """
        self.logger.warning("Метод predict_proba не применим для регрессионных моделей")
        return np.array([])
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """
        Получение параметров по умолчанию для выбранного алгоритма.
        
        Returns:
            Параметры по умолчанию
        """
        if self.algorithm == "random_forest":
            return {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42
            }
        elif self.algorithm == "linear":
            return {
                "fit_intercept": True,
                "normalize": False,
                "copy_X": True,
                "n_jobs": None
            }
        # Здесь можно добавить параметры для других алгоритмов
        return {}
    
    def _create_model(self) -> Any:
        """
        Создание модели на основе выбранного алгоритма и параметров.
        
        Returns:
            Экземпляр модели
        """
        parameters = self.get_parameters()
        
        if self.algorithm == "random_forest":
            return RandomForestRegressor(**parameters)
        elif self.algorithm == "linear":
            return LinearRegression(**parameters)
        # Здесь можно добавить создание других алгоритмов
        
        self.logger.error(f"Неизвестный алгоритм: {self.algorithm}")
        return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Обучение модели.
        
        Args:
            X: Признаки для обучения
            y: Целевая переменная
            **kwargs: Дополнительные параметры для обучения
            
        Returns:
            Метрики обучения
        """
        # Создание модели
        self.model = self._create_model()
        if self.model is None:
            return {"error": f"Неизвестный алгоритм: {self.algorithm}"}
        
        # Сохранение информации о признаках и целевой переменной
        self.metadata["features"] = X.columns.tolist()
        self.metadata["target"] = y.name
        
        # Обучение модели
        self.logger.info(f"Начало обучения модели {self.name}")
        self.model.fit(X, y)
        self.logger.info(f"Модель {self.name} обучена")
        
        # Оценка на обучающих данных
        metrics = self.evaluate(X, y)
        self.metadata["metrics"] = metrics
        
        return metrics 