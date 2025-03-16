"""
Модуль бэктестинга.

Предоставляет классы и функции для проведения бэктестинга торговых стратегий.
"""

from backtesting.backtester import Backtester
from backtesting.performance_metrics import PerformanceMetrics

__all__ = [
    'Backtester',
    'PerformanceMetrics'
] 