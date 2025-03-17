"""
Модуль для расчета размера позиции.

Предоставляет класс для расчета размера позиции на основе различных методов
управления рисками.
"""

import logging
from typing import Dict, Any, Optional

class PositionSizer:
    """
    Класс для расчета размера позиции.
    
    Предоставляет методы для расчета размера позиции на основе различных
    подходов к управлению рисками, таких как фиксированный риск,
    фиксированный процент от баланса, фиксированный размер позиции и др.
    """
    
    def __init__(self, default_method: str = "fixed_risk", default_params: Dict[str, Any] = None):
        """
        Инициализация калькулятора размера позиции.
        
        Args:
            default_method: Метод расчета по умолчанию
            default_params: Параметры метода по умолчанию
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.default_method = default_method
        self.default_params = default_params or {"risk_percent": 1.0}
        
        # Словарь доступных методов расчета
        self.methods = {
            "fixed_risk": self._calculate_fixed_risk,
            "fixed_percent": self._calculate_fixed_percent,
            "fixed_size": self._calculate_fixed_size,
            "kelly": self._calculate_kelly
        }
        
        self.logger.info(f"Инициализирован калькулятор размера позиции (метод: {default_method})")
    
    def calculate(self, balance: float, entry_price: float, stop_loss: float, 
                 method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
        """
        Расчет размера позиции.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            method: Метод расчета (если None, используется метод по умолчанию)
            params: Параметры метода (если None, используются параметры по умолчанию)
            
        Returns:
            float: Размер позиции
            
        Raises:
            ValueError: Если указан неизвестный метод расчета
        """
        # Определение метода расчета
        method = method or self.default_method
        
        # Проверка наличия метода
        if method not in self.methods:
            self.logger.error(f"Неизвестный метод расчета размера позиции: {method}")
            raise ValueError(f"Неизвестный метод расчета размера позиции: {method}")
        
        # Определение параметров
        params = params or self.default_params
        
        # Расчет размера позиции
        position_size = self.methods[method](balance, entry_price, stop_loss, params)
        
        # Проверка на отрицательный размер позиции
        if position_size < 0:
            self.logger.warning("Рассчитан отрицательный размер позиции, установлено значение 0")
            position_size = 0
        
        return position_size
    
    def _calculate_fixed_risk(self, balance: float, entry_price: float, stop_loss: float, 
                             params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции на основе фиксированного риска.
        
        Размер позиции рассчитывается таким образом, чтобы при срабатывании
        стоп-лосса убыток составил заданный процент от баланса.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            params: Параметры метода
            
        Returns:
            float: Размер позиции
        """
        # Получение процента риска
        risk_percent = params.get("risk_percent", 1.0)
        
        # Расчет риска в абсолютных единицах
        risk_amount = balance * (risk_percent / 100)
        
        # Расчет размера позиции
        price_diff = abs(entry_price - stop_loss)
        
        # Проверка на нулевую разницу цен
        if price_diff == 0:
            self.logger.warning("Нулевая разница между ценой входа и стоп-лоссом")
            return 0
        
        position_size = risk_amount / price_diff
        
        self.logger.debug(f"Расчет размера позиции (fixed_risk): баланс={balance}, "
                         f"риск={risk_percent}%, размер={position_size}")
        
        return position_size
    
    def _calculate_fixed_percent(self, balance: float, entry_price: float, stop_loss: float, 
                                params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции на основе фиксированного процента от баланса.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса (не используется)
            params: Параметры метода
            
        Returns:
            float: Размер позиции
        """
        # Получение процента от баланса
        percent = params.get("percent", 5.0)
        
        # Расчет размера позиции
        position_value = balance * (percent / 100)
        position_size = position_value / entry_price
        
        self.logger.debug(f"Расчет размера позиции (fixed_percent): баланс={balance}, "
                         f"процент={percent}%, размер={position_size}")
        
        return position_size
    
    def _calculate_fixed_size(self, balance: float, entry_price: float, stop_loss: float, 
                             params: Dict[str, Any]) -> float:
        """
        Расчет фиксированного размера позиции.
        
        Args:
            balance: Доступный баланс (не используется)
            entry_price: Цена входа (не используется)
            stop_loss: Цена стоп-лосса (не используется)
            params: Параметры метода
            
        Returns:
            float: Размер позиции
        """
        # Получение фиксированного размера
        size = params.get("size", 0.01)
        
        self.logger.debug(f"Расчет размера позиции (fixed_size): размер={size}")
        
        return size
    
    def _calculate_kelly(self, balance: float, entry_price: float, stop_loss: float, 
                        params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции по формуле Келли.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            params: Параметры метода
            
        Returns:
            float: Размер позиции
        """
        # Получение параметров
        win_rate = params.get("win_rate", 0.5)
        win_loss_ratio = params.get("win_loss_ratio", 2.0)
        max_risk = params.get("max_risk", 5.0)  # Максимальный риск в процентах
        
        # Расчет доли по формуле Келли
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Ограничение доли
        kelly_fraction = max(0, min(kelly_fraction, max_risk / 100))
        
        # Расчет риска в абсолютных единицах
        risk_amount = balance * kelly_fraction
        
        # Расчет размера позиции
        price_diff = abs(entry_price - stop_loss)
        
        # Проверка на нулевую разницу цен
        if price_diff == 0:
            self.logger.warning("Нулевая разница между ценой входа и стоп-лоссом")
            return 0
        
        position_size = risk_amount / price_diff
        
        self.logger.debug(f"Расчет размера позиции (kelly): баланс={balance}, "
                         f"доля Келли={kelly_fraction}, размер={position_size}")
        
        return position_size 