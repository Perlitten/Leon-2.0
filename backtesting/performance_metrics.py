"""
Метрики производительности торговых стратегий.

Предоставляет класс для расчета и анализа метрик производительности торговых стратегий.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging


class PerformanceMetrics:
    """
    Класс для расчета и анализа метрик производительности торговых стратегий.
    
    Предоставляет методы для расчета различных метрик, таких как доходность,
    просадка, коэффициент Шарпа, коэффициент Сортино и другие.
    """
    
    def __init__(self):
        """Инициализация класса метрик производительности."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Расчет доходности на основе кривой капитала.
        
        Args:
            equity_curve: Кривая капитала (временной ряд с балансом счета)
            
        Returns:
            Временной ряд с доходностью
        """
        returns = equity_curve.pct_change().fillna(0)
        return returns
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Расчет кумулятивной доходности.
        
        Args:
            returns: Временной ряд с доходностью
            
        Returns:
            Временной ряд с кумулятивной доходностью
        """
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        Расчет просадки.
        
        Args:
            equity_curve: Кривая капитала
            
        Returns:
            Временной ряд с просадкой
        """
        # Расчет максимума на текущий момент
        rolling_max = equity_curve.cummax()
        # Расчет просадки
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Расчет максимальной просадки.
        
        Args:
            equity_curve: Кривая капитала
            
        Returns:
            Максимальная просадка (в процентах)
        """
        drawdown = self.calculate_drawdown(equity_curve)
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Расчет коэффициента Шарпа.
        
        Args:
            returns: Временной ряд с доходностью
            risk_free_rate: Безрисковая ставка (годовая)
            periods_per_year: Количество периодов в году
            
        Returns:
            Коэффициент Шарпа
        """
        # Расчет избыточной доходности
        excess_returns = returns - risk_free_rate / periods_per_year
        # Расчет коэффициента Шарпа
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Расчет коэффициента Сортино.
        
        Args:
            returns: Временной ряд с доходностью
            risk_free_rate: Безрисковая ставка (годовая)
            periods_per_year: Количество периодов в году
            
        Returns:
            Коэффициент Сортино
        """
        # Расчет избыточной доходности
        excess_returns = returns - risk_free_rate / periods_per_year
        # Расчет отрицательной полудисперсии
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt(np.sum(negative_returns ** 2) / len(returns))
        # Расчет коэффициента Сортино
        sortino_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan
        return sortino_ratio
    
    def calculate_calmar_ratio(self, returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """
        Расчет коэффициента Кальмара.
        
        Args:
            returns: Временной ряд с доходностью
            equity_curve: Кривая капитала
            periods_per_year: Количество периодов в году
            
        Returns:
            Коэффициент Кальмара
        """
        # Расчет годовой доходности
        annual_return = returns.mean() * periods_per_year
        # Расчет максимальной просадки
        max_drawdown = abs(self.calculate_max_drawdown(equity_curve))
        # Расчет коэффициента Кальмара
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.nan
        return calmar_ratio
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Расчет коэффициента Омега.
        
        Args:
            returns: Временной ряд с доходностью
            threshold: Пороговое значение доходности
            periods_per_year: Количество периодов в году
            
        Returns:
            Коэффициент Омега
        """
        # Расчет избыточной доходности относительно порога
        excess_returns = returns - threshold / periods_per_year
        # Разделение на положительные и отрицательные доходности
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        # Расчет коэффициента Омега
        omega_ratio = positive_returns.sum() / abs(negative_returns.sum()) if len(negative_returns) > 0 and negative_returns.sum() != 0 else np.nan
        return omega_ratio
    
    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """
        Расчет процента прибыльных сделок.
        
        Args:
            trades: DataFrame с информацией о сделках
            
        Returns:
            Процент прибыльных сделок
        """
        if len(trades) == 0:
            return 0.0
        
        profitable_trades = trades[trades['profit'] > 0]
        win_rate = len(profitable_trades) / len(trades)
        return win_rate
    
    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """
        Расчет фактора прибыли.
        
        Args:
            trades: DataFrame с информацией о сделках
            
        Returns:
            Фактор прибыли
        """
        if len(trades) == 0:
            return 0.0
        
        profitable_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] < 0]
        
        total_profit = profitable_trades['profit'].sum() if len(profitable_trades) > 0 else 0
        total_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss != 0 else np.inf
        return profit_factor
    
    def calculate_average_trade(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет средних показателей по сделкам.
        
        Args:
            trades: DataFrame с информацией о сделках
            
        Returns:
            Словарь со средними показателями
        """
        if len(trades) == 0:
            return {
                'avg_profit': 0.0,
                'avg_profit_percent': 0.0,
                'avg_duration': timedelta(0),
                'avg_profit_win': 0.0,
                'avg_loss_loss': 0.0
            }
        
        profitable_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] < 0]
        
        avg_profit = trades['profit'].mean()
        avg_profit_percent = trades['profit_percent'].mean() if 'profit_percent' in trades.columns else 0.0
        
        if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
            trades['duration'] = trades['exit_time'] - trades['entry_time']
            avg_duration = trades['duration'].mean()
        else:
            avg_duration = timedelta(0)
        
        avg_profit_win = profitable_trades['profit'].mean() if len(profitable_trades) > 0 else 0.0
        avg_loss_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0.0
        
        return {
            'avg_profit': avg_profit,
            'avg_profit_percent': avg_profit_percent,
            'avg_duration': avg_duration,
            'avg_profit_win': avg_profit_win,
            'avg_loss_loss': avg_loss_loss
        }
    
    def calculate_expectancy(self, trades: pd.DataFrame) -> float:
        """
        Расчет математического ожидания прибыли на сделку.
        
        Args:
            trades: DataFrame с информацией о сделках
            
        Returns:
            Математическое ожидание прибыли на сделку
        """
        if len(trades) == 0:
            return 0.0
        
        win_rate = self.calculate_win_rate(trades)
        
        profitable_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] < 0]
        
        avg_win = profitable_trades['profit'].mean() if len(profitable_trades) > 0 else 0.0
        avg_loss = losing_trades['profit'].mean() if len(losing_trades) > 0 else 0.0
        
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        return expectancy
    
    def calculate_all_metrics(self, equity_curve: pd.Series, trades: pd.DataFrame, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, Any]:
        """
        Расчет всех метрик производительности.
        
        Args:
            equity_curve: Кривая капитала
            trades: DataFrame с информацией о сделках
            risk_free_rate: Безрисковая ставка (годовая)
            periods_per_year: Количество периодов в году
            
        Returns:
            Словарь со всеми метриками
        """
        returns = self.calculate_returns(equity_curve)
        cumulative_returns = self.calculate_cumulative_returns(returns)
        drawdown = self.calculate_drawdown(equity_curve)
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0.0,
            'annual_return': returns.mean() * periods_per_year,
            'volatility': returns.std() * np.sqrt(periods_per_year),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'sortino_ratio': self.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
            'calmar_ratio': self.calculate_calmar_ratio(returns, equity_curve, periods_per_year),
            'omega_ratio': self.calculate_omega_ratio(returns, risk_free_rate, periods_per_year),
            'win_rate': self.calculate_win_rate(trades),
            'profit_factor': self.calculate_profit_factor(trades),
            'expectancy': self.calculate_expectancy(trades),
            'total_trades': len(trades),
            'avg_trade_metrics': self.calculate_average_trade(trades)
        }
        
        return metrics 