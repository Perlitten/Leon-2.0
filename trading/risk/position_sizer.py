"""
Модуль для расчета размера позиции.

Предоставляет класс для расчета размера позиции на основе различных методов
управления рисками.
"""

import logging
from typing import Dict, Any, Optional

class PositionSizer:
    """
    Класс для расчета размера позиции на основе различных методов управления рисками.
    
    Поддерживает следующие методы:
    - Фиксированный процент от баланса
    - Фиксированный риск на сделку
    - Фиксированный размер позиции
    - Метод Келли
    """
    
    def __init__(self, default_method: str = "fixed_risk", default_params: Dict[str, Any] = None):
        """
        Инициализация калькулятора размера позиции.
        
        Args:
            default_method: Метод расчета по умолчанию
                ("fixed_risk", "fixed_percent", "fixed_size", "kelly")
            default_params: Параметры метода по умолчанию
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.default_method = default_method
        self.default_params = default_params or {}
        
        # Проверка метода по умолчанию
        if default_method not in ["fixed_risk", "fixed_percent", "fixed_size", "kelly"]:
            self.logger.warning(f"Неизвестный метод расчета размера позиции: {default_method}. Используется fixed_risk.")
            self.default_method = "fixed_risk"
        
        # Установка параметров по умолчанию, если не указаны
        if "risk_per_trade" not in self.default_params:
            self.default_params["risk_per_trade"] = 2.0  # 2% риска на сделку
        
        if "position_size" not in self.default_params:
            self.default_params["position_size"] = 0.001  # 0.001 BTC
        
        if "percent" not in self.default_params:
            self.default_params["percent"] = 5.0  # 5% от баланса
        
        if "win_rate" not in self.default_params:
            self.default_params["win_rate"] = 0.5  # 50% вероятность выигрыша
        
        if "win_loss_ratio" not in self.default_params:
            self.default_params["win_loss_ratio"] = 1.5  # Соотношение выигрыша к проигрышу
    
    def calculate(self, balance: float, entry_price: float, stop_loss: float, 
                 method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
        """
        Расчет размера позиции.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            method: Метод расчета
                ("fixed_risk", "fixed_percent", "fixed_size", "kelly")
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        # Использование метода и параметров по умолчанию, если не указаны
        method = method or self.default_method
        
        # Объединение параметров по умолчанию и переданных параметров
        all_params = self.default_params.copy()
        if params:
            all_params.update(params)
        
        # Расчет размера позиции в зависимости от метода
        if method == "fixed_risk":
            return self._calculate_fixed_risk(balance, entry_price, stop_loss, all_params)
        elif method == "fixed_percent":
            return self._calculate_fixed_percent(balance, entry_price, all_params)
        elif method == "fixed_size":
            return self._calculate_fixed_size(all_params)
        elif method == "kelly":
            return self._calculate_kelly(balance, entry_price, stop_loss, all_params)
        else:
            self.logger.warning(f"Неизвестный метод расчета размера позиции: {method}. Используется fixed_risk.")
            return self._calculate_fixed_risk(balance, entry_price, stop_loss, all_params)
    
    def _calculate_fixed_risk(self, balance: float, entry_price: float, stop_loss: float, 
                             params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции на основе фиксированного риска на сделку.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        risk_per_trade = params.get("risk_per_trade", 2.0)  # Процент риска на сделку
        
        # Расчет риска в абсолютных единицах
        risk_amount = balance * (risk_per_trade / 100)
        
        # Расчет размера позиции на основе риска и стоп-лосса
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            self.logger.warning("Разница между ценой входа и стоп-лоссом равна 0. Используется минимальный размер позиции.")
            return 0.001  # Минимальный размер позиции
        
        position_size = risk_amount / price_diff
        
        return position_size
    
    def _calculate_fixed_percent(self, balance: float, entry_price: float, 
                                params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции на основе фиксированного процента от баланса.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        percent = params.get("percent", 5.0)  # Процент от баланса
        
        # Расчет размера позиции
        position_value = balance * (percent / 100)
        position_size = position_value / entry_price
        
        return position_size
    
    def _calculate_fixed_size(self, params: Dict[str, Any]) -> float:
        """
        Возвращает фиксированный размер позиции.
        
        Args:
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        return params.get("position_size", 0.001)
    
    def _calculate_kelly(self, balance: float, entry_price: float, stop_loss: float, 
                        params: Dict[str, Any]) -> float:
        """
        Расчет размера позиции на основе критерия Келли.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        win_rate = params.get("win_rate", 0.5)  # Вероятность выигрыша
        win_loss_ratio = params.get("win_loss_ratio", 1.5)  # Соотношение выигрыша к проигрышу
        
        # Расчет доли капитала по формуле Келли
        kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Ограничение доли капитала
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Не более 25% капитала
        
        # Расчет размера позиции
        position_value = balance * kelly_fraction
        position_size = position_value / entry_price
        
        return position_size 