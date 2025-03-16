"""
Модуль фабрики моделей машинного обучения.

Предоставляет интерфейс для создания различных типов моделей машинного обучения.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type

from ml.models.base_model import BaseModel
from ml.models.regression_model import RegressionModel
from ml.models.classification_model import ClassificationModel
from ml.models.ensemble_model import EnsembleModel
from ml.models.time_series_model import TimeSeriesModel


class ModelFactory:
    """
    Фабрика для создания моделей машинного обучения различных типов.
    
    Предоставляет унифицированный интерфейс для создания моделей разных типов
    с различными параметрами.
    """
    
    # Словарь соответствия типов моделей и их классов
    MODEL_TYPES = {
        'regression': RegressionModel,
        'classification': ClassificationModel,
        'ensemble': EnsembleModel,
        'time_series': TimeSeriesModel
    }
    
    def __init__(self):
        """
        Инициализация фабрики моделей.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Инициализация фабрики моделей")
    
    def create_model(self, model_type: str, name: str, **kwargs) -> BaseModel:
        """
        Создание модели указанного типа с заданными параметрами.
        
        Args:
            model_type: Тип модели ('regression', 'classification', 'ensemble', 'time_series')
            name: Название модели
            **kwargs: Дополнительные параметры для конкретного типа модели
            
        Returns:
            BaseModel: Созданная модель
            
        Raises:
            ValueError: Если указан неподдерживаемый тип модели
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}. "
                            f"Доступные типы: {', '.join(self.MODEL_TYPES.keys())}")
        
        model_class = self.MODEL_TYPES[model_type]
        
        try:
            model = model_class(name=name, **kwargs)
            self.logger.info(f"Создана модель типа '{model_type}' с именем '{name}'")
            return model
        except Exception as e:
            self.logger.error(f"Ошибка при создании модели типа '{model_type}': {str(e)}")
            raise
    
    def create_regression_model(self, name: str, algorithm: str = "linear", **kwargs) -> RegressionModel:
        """
        Создание модели регрессии.
        
        Args:
            name: Название модели
            algorithm: Алгоритм регрессии
            **kwargs: Дополнительные параметры
            
        Returns:
            RegressionModel: Созданная модель регрессии
        """
        return self.create_model('regression', name, algorithm=algorithm, **kwargs)
    
    def create_classification_model(self, name: str, algorithm: str = "logistic", **kwargs) -> ClassificationModel:
        """
        Создание модели классификации.
        
        Args:
            name: Название модели
            algorithm: Алгоритм классификации
            **kwargs: Дополнительные параметры
            
        Returns:
            ClassificationModel: Созданная модель классификации
        """
        return self.create_model('classification', name, algorithm=algorithm, **kwargs)
    
    def create_time_series_model(self, name: str, algorithm: str = "arima", **kwargs) -> TimeSeriesModel:
        """
        Создание модели временных рядов.
        
        Args:
            name: Название модели
            algorithm: Алгоритм прогнозирования временных рядов
            **kwargs: Дополнительные параметры
            
        Returns:
            TimeSeriesModel: Созданная модель временных рядов
        """
        return self.create_model('time_series', name, algorithm=algorithm, **kwargs)
    
    def create_ensemble_model(self, name: str, models: List[BaseModel], 
                             aggregation_method: str = "mean", 
                             weights: Optional[List[float]] = None,
                             **kwargs) -> EnsembleModel:
        """
        Создание ансамблевой модели.
        
        Args:
            name: Название модели
            models: Список моделей для ансамбля
            aggregation_method: Метод агрегации результатов
            weights: Веса моделей
            **kwargs: Дополнительные параметры
            
        Returns:
            EnsembleModel: Созданная ансамблевая модель
        """
        return self.create_model('ensemble', name, models=models, 
                               aggregation_method=aggregation_method, 
                               weights=weights, **kwargs)
    
    def create_model_from_config(self, config: Dict[str, Any]) -> BaseModel:
        """
        Создание модели на основе конфигурации.
        
        Args:
            config: Словарь с конфигурацией модели
            
        Returns:
            BaseModel: Созданная модель
            
        Raises:
            ValueError: Если в конфигурации отсутствуют обязательные параметры
        """
        if 'type' not in config:
            raise ValueError("В конфигурации модели отсутствует параметр 'type'")
        
        if 'name' not in config:
            raise ValueError("В конфигурации модели отсутствует параметр 'name'")
        
        model_type = config.pop('type')
        name = config.pop('name')
        
        # Особая обработка для ансамблевой модели
        if model_type == 'ensemble':
            if 'models_config' not in config:
                raise ValueError("Для ансамблевой модели необходим параметр 'models_config'")
            
            # Создание моделей для ансамбля
            models = []
            for model_config in config.pop('models_config'):
                models.append(self.create_model_from_config(model_config))
            
            # Создание ансамблевой модели
            return self.create_ensemble_model(name, models, **config)
        
        # Создание обычной модели
        return self.create_model(model_type, name, **config) 