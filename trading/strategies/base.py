"""
Базовый класс для торговых стратегий.

Определяет интерфейс для всех торговых стратегий в системе.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple

class StrategyBase(ABC):
    """
    Базовый абстрактный класс для торговых стратегий.
    
    Определяет общий интерфейс для всех стратегий в системе.
    Конкретные стратегии должны наследоваться от этого класса и
    реализовывать все абстрактные методы.
    """
    
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any] = None):
        """
        Инициализация базового класса стратегии.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            timeframe: Таймфрейм для анализа (например, "1m", "5m", "1h", "1d")
            params: Дополнительные параметры стратегии
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self, historical_data: List[List[Union[int, str, float]]]) -> bool:
        """
        Инициализация стратегии с историческими данными.
        
        Args:
            historical_data: Исторические данные для инициализации стратегии
            
        Returns:
            Успешность инициализации
        """
        pass
    
    @abstractmethod
    async def update(self, candle: List[Union[int, str, float]]) -> None:
        """
        Обновление стратегии новыми данными.
        
        Args:
            candle: Новая свеча для обновления стратегии
        """
        pass
    
    @abstractmethod
    async def generate_signal(self) -> Dict[str, Any]:
        """
        Генерация торгового сигнала на основе текущего состояния стратегии.
        
        Returns:
            Торговый сигнал в виде словаря с ключами:
                - "action": Действие ("BUY", "SELL", "HOLD")
                - "price": Цена входа (опционально)
                - "stop_loss": Уровень стоп-лосса (опционально)
                - "take_profit": Уровень тейк-профита (опционально)
                - "reason": Причина сигнала (опционально)
                - "confidence": Уверенность в сигнале от 0 до 1 (опционально)
        """
        pass
    
    @abstractmethod
    async def calculate_position_size(self, balance: float, risk_per_trade: float) -> float:
        """
        Расчет размера позиции на основе баланса и риска.
        
        Args:
            balance: Доступный баланс
            risk_per_trade: Процент риска на сделку (от 0 до 100)
            
        Returns:
            Размер позиции
        """
        pass
    
    @abstractmethod
    async def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня стоп-лосса.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень стоп-лосса
        """
        pass
    
    @abstractmethod
    async def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня тейк-профита.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень тейк-профита
        """
        pass
    
    @abstractmethod
    async def should_exit(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """
        Проверка необходимости выхода из позиции.
        
        Args:
            position: Информация о текущей позиции
            current_price: Текущая цена
            
        Returns:
            Кортеж (нужно ли выходить, причина выхода)
        """
        pass
    
    @abstractmethod
    async def get_indicators(self) -> Dict[str, Any]:
        """
        Получение текущих значений индикаторов стратегии.
        
        Returns:
            Словарь с текущими значениями индикаторов
        """
        pass
    
    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния стратегии.
        
        Returns:
            Словарь с текущим состоянием стратегии
        """
        pass
    
    @abstractmethod
    async def set_state(self, state: Dict[str, Any]) -> None:
        """
        Установка состояния стратегии.
        
        Args:
            state: Состояние стратегии
        """
        pass 