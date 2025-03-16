"""
Пакет стратегий для Leon Trading Bot.

Предоставляет различные торговые стратегии для использования в боте.
"""

from trading.strategies.base import StrategyBase
from trading.strategies.simple_ma import SimpleMAStrategy

__all__ = ['StrategyBase', 'SimpleMAStrategy'] 