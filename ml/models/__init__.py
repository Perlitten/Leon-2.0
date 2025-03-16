"""
Модуль моделей машинного обучения.

Предоставляет классы и функции для работы с моделями машинного обучения,
включая создание, обучение, оценку и управление моделями.
"""

from .models import (
    BaseModel,
    RegressionModel,
    ClassificationModel,
    EnsembleModel,
    TimeSeriesModel
)

from .model_factory import ModelFactory
from .model_manager import ModelManager

__all__ = [
    'BaseModel',
    'RegressionModel',
    'ClassificationModel',
    'EnsembleModel',
    'TimeSeriesModel',
    'ModelFactory',
    'ModelManager'
] 