"""
Модуль моделей для временных рядов.

Предоставляет реализацию моделей для прогнозирования временных рядов на основе базового класса.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ml.models.base_model import BaseModel


class TimeSeriesModel(BaseModel):
    """
    Класс для моделей прогнозирования временных рядов.
    
    Поддерживает различные алгоритмы прогнозирования временных рядов:
    ARIMA, SARIMA, экспоненциальное сглаживание и др.
    """
    
    SUPPORTED_MODELS = {
        'arima': ARIMA,
        'sarima': SARIMAX,
        'exp_smoothing': ExponentialSmoothing
    }
    
    def __init__(self, name: str, algorithm: str = 'arima', version: str = "1.0.0", **kwargs):
        """
        Инициализация модели временных рядов.
        
        Args:
            name: Название модели
            algorithm: Алгоритм прогнозирования ('arima', 'sarima', 'exp_smoothing')
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
        self.model_params = kwargs
        self.model = None
        self.model_fit = None
        self.logger.info(f"Создана модель временных рядов с алгоритмом {algorithm}")
    
    def train(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
             y: Optional[Union[np.ndarray, pd.Series]] = None, **kwargs) -> Dict[str, Any]:
        """
        Обучение модели временных рядов на данных.
        
        Для моделей временных рядов параметр y обычно не используется, 
        так как X уже содержит временной ряд.
        
        Args:
            X: Временной ряд для обучения
            y: Не используется для большинства моделей временных рядов
            **kwargs: Дополнительные параметры обучения
            
        Returns:
            Dict[str, Any]: Метрики обучения
        """
        self.logger.info(f"Начало обучения модели {self.name}")
        
        # Преобразование входных данных в подходящий формат
        if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            time_series = X.iloc[:, 0]
        elif isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            # Если есть экзогенные переменные, используем первый столбец как целевую переменную
            time_series = X.iloc[:, 0]
            exog = X.iloc[:, 1:]
        else:
            time_series = X
            exog = None
        
        # Создание и обучение модели в зависимости от алгоритма
        if self.algorithm == 'arima':
            # Параметры ARIMA: p, d, q
            p = self.model_params.get('p', 1)
            d = self.model_params.get('d', 1)
            q = self.model_params.get('q', 1)
            
            self.model = ARIMA(time_series, order=(p, d, q), **kwargs)
            self.model_fit = self.model.fit()
            
        elif self.algorithm == 'sarima':
            # Параметры SARIMA: p, d, q, P, D, Q, s
            p = self.model_params.get('p', 1)
            d = self.model_params.get('d', 1)
            q = self.model_params.get('q', 1)
            P = self.model_params.get('P', 0)
            D = self.model_params.get('D', 0)
            Q = self.model_params.get('Q', 0)
            s = self.model_params.get('s', 12)  # Сезонность (например, 12 для месячных данных)
            
            self.model = SARIMAX(
                time_series, 
                exog=exog,
                order=(p, d, q), 
                seasonal_order=(P, D, Q, s),
                **kwargs
            )
            self.model_fit = self.model.fit(disp=False)
            
        elif self.algorithm == 'exp_smoothing':
            # Параметры экспоненциального сглаживания
            trend = self.model_params.get('trend', None)
            seasonal = self.model_params.get('seasonal', None)
            seasonal_periods = self.model_params.get('seasonal_periods', None)
            
            self.model = ExponentialSmoothing(
                time_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                **kwargs
            )
            self.model_fit = self.model.fit()
        
        self.is_trained = True
        
        # Получение предсказаний на обучающих данных
        in_sample_predictions = self.model_fit.fittedvalues
        
        # Расчет метрик на обучающих данных
        metrics = self._calculate_metrics(time_series, in_sample_predictions)
        
        self.logger.info(f"Модель {self.name} обучена. Метрики на обучающих данных: {metrics}")
        return metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
               steps: int = 1, **kwargs) -> np.ndarray:
        """
        Получение прогноза временного ряда.
        
        Args:
            X: Данные для прогноза (может быть None для некоторых моделей)
            steps: Количество шагов для прогноза
            **kwargs: Дополнительные параметры прогнозирования
            
        Returns:
            np.ndarray: Прогнозные значения
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Подготовка экзогенных переменных, если они есть
        exog = None
        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            exog = X.iloc[:, 1:]
        
        # Прогнозирование в зависимости от алгоритма
        if self.algorithm in ['arima', 'sarima']:
            forecast = self.model_fit.forecast(steps=steps, exog=exog)
        elif self.algorithm == 'exp_smoothing':
            forecast = self.model_fit.forecast(steps=steps)
        
        return np.array(forecast)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame, pd.Series], 
                y: Optional[Union[np.ndarray, pd.Series]] = None, 
                steps: int = 1) -> Dict[str, float]:
        """
        Оценка производительности модели временных рядов.
        
        Args:
            X: Временной ряд для оценки
            y: Истинные значения (если None, используется X)
            steps: Количество шагов для прогноза
            
        Returns:
            Dict[str, float]: Метрики оценки
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        # Если y не указан, используем X как временной ряд
        if y is None:
            if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
                time_series = X.iloc[:, 0]
            elif isinstance(X, pd.DataFrame) and X.shape[1] > 1:
                time_series = X.iloc[:, 0]
            else:
                time_series = X
        else:
            time_series = y
        
        # Получение прогноза
        predictions = self.predict(X, steps=steps)
        
        # Если длина прогноза меньше длины временного ряда, используем только последние значения
        if len(predictions) < len(time_series):
            actual = time_series[-len(predictions):]
        else:
            actual = time_series
            predictions = predictions[:len(actual)]
        
        # Расчет метрик
        metrics = self._calculate_metrics(actual, predictions)
        
        self.logger.info(f"Оценка модели {self.name}. Метрики: {metrics}")
        return metrics
    
    def _calculate_metrics(self, actual: Union[np.ndarray, pd.Series], 
                          predictions: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Расчет метрик для оценки модели временных рядов.
        
        Args:
            actual: Истинные значения
            predictions: Предсказанные значения
            
        Returns:
            Dict[str, float]: Словарь с метриками
        """
        # Преобразование в numpy массивы
        actual_array = np.array(actual)
        pred_array = np.array(predictions)
        
        # Расчет метрик
        metrics = {
            'mse': mean_squared_error(actual_array, pred_array),
            'rmse': np.sqrt(mean_squared_error(actual_array, pred_array)),
            'mae': mean_absolute_error(actual_array, pred_array)
        }
        
        # Расчет R2, если возможно
        try:
            metrics['r2'] = r2_score(actual_array, pred_array)
        except:
            metrics['r2'] = np.nan
        
        # Расчет MAPE, если нет нулевых значений
        if not np.any(actual_array == 0):
            metrics['mape'] = np.mean(np.abs((actual_array - pred_array) / actual_array)) * 100
        else:
            metrics['mape'] = np.nan
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Получение важности признаков.
        
        Для моделей временных рядов этот метод может не иметь смысла,
        но реализован для совместимости с интерфейсом BaseModel.
        
        Returns:
            Optional[np.ndarray]: Массив важности признаков или None, если не поддерживается
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        self.logger.warning("Метод get_feature_importance не имеет смысла для большинства моделей временных рядов")
        return None
    
    def get_model_summary(self) -> str:
        """
        Получение сводки о модели.
        
        Returns:
            str: Сводка о модели
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if hasattr(self.model_fit, 'summary'):
            return str(self.model_fit.summary())
        else:
            return f"Сводка недоступна для модели {self.algorithm}"
    
    def get_residuals(self) -> np.ndarray:
        """
        Получение остатков модели.
        
        Returns:
            np.ndarray: Остатки модели
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if hasattr(self.model_fit, 'resid'):
            return self.model_fit.resid
        else:
            return np.array([])
    
    def get_aic(self) -> float:
        """
        Получение информационного критерия Акаике (AIC).
        
        Returns:
            float: Значение AIC
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if hasattr(self.model_fit, 'aic'):
            return self.model_fit.aic
        else:
            return np.nan
    
    def get_bic(self) -> float:
        """
        Получение байесовского информационного критерия (BIC).
        
        Returns:
            float: Значение BIC
            
        Raises:
            RuntimeError: Если модель не обучена
        """
        if not self.is_trained or self.model_fit is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")
        
        if hasattr(self.model_fit, 'bic'):
            return self.model_fit.bic
        else:
            return np.nan 