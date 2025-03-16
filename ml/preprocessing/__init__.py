"""
Модуль предобработки данных для машинного обучения.

Предоставляет классы и функции для очистки, трансформации, выбора признаков
и инженерии признаков для подготовки данных к обучению моделей.
"""

from .data_preprocessor import DataPreprocessor
from .feature_selector import FeatureSelector
from .feature_engineer import FeatureEngineer

__all__ = [
    'DataPreprocessor',
    'FeatureSelector',
    'FeatureEngineer'
] 