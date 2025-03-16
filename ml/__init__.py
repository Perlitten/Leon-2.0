"""
Модуль машинного обучения.

Предоставляет классы и функции для работы с моделями машинного обучения,
включая создание, обучение, оценку, визуализацию и управление моделями.
"""

from .models import (
    BaseModel,
    RegressionModel,
    ClassificationModel,
    EnsembleModel,
    TimeSeriesModel,
    ModelFactory,
    ModelManager
)

from .preprocessing import (
    DataPreprocessor,
    FeatureSelector,
    FeatureEngineer
)

from .validation import (
    ModelValidator,
    ModelVisualizer
)

from .training import (
    ModelTrainer
)

__all__ = [
    # Модели
    'BaseModel',
    'RegressionModel',
    'ClassificationModel',
    'EnsembleModel',
    'TimeSeriesModel',
    'ModelFactory',
    'ModelManager',
    
    # Предобработка данных
    'DataPreprocessor',
    'FeatureSelector',
    'FeatureEngineer',
    
    # Валидация и визуализация
    'ModelValidator',
    'ModelVisualizer',
    
    # Обучение
    'ModelTrainer'
] 