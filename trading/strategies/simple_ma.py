"""
Простая стратегия на основе скользящих средних.

Стратегия использует пересечение быстрой и медленной скользящих средних
для генерации торговых сигналов.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from trading.strategies.base import StrategyBase

class SimpleMAStrategy(StrategyBase):
    """
    Простая стратегия на основе скользящих средних.
    
    Стратегия использует пересечение быстрой и медленной скользящих средних
    для генерации торговых сигналов.
    """
    
    def __init__(self, symbol: str, timeframe: str, params: Dict[str, Any] = None):
        """
        Инициализация стратегии.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            timeframe: Таймфрейм для анализа (например, "1m", "5m", "1h", "1d")
            params: Дополнительные параметры стратегии:
                - fast_ma_period: Период быстрой скользящей средней (по умолчанию 9)
                - slow_ma_period: Период медленной скользящей средней (по умолчанию 21)
                - stop_loss_percent: Процент стоп-лосса (по умолчанию 2.0)
                - take_profit_percent: Процент тейк-профита (по умолчанию 3.0)
        """
        default_params = {
            "fast_ma_period": 9,
            "slow_ma_period": 21,
            "stop_loss_percent": 2.0,
            "take_profit_percent": 3.0
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(symbol, timeframe, default_params)
        
        self.prices = []
        self.fast_ma = []
        self.slow_ma = []
        self.last_signal = "HOLD"
        self.last_signal_price = 0.0
        self.last_signal_time = 0
        
    async def initialize(self, historical_data: List[List[Union[int, str, float]]]) -> bool:
        """
        Инициализация стратегии с историческими данными.
        
        Args:
            historical_data: Исторические данные для инициализации стратегии
            
        Returns:
            Успешность инициализации
        """
        try:
            # Извлечение цен закрытия из исторических данных
            # Формат Binance: [timestamp, open, high, low, close, volume, ...]
            self.prices = [float(candle[4]) for candle in historical_data]
            
            # Расчет скользящих средних
            self._calculate_indicators()
            
            self.is_initialized = True
            self.logger.info(f"Стратегия {self.__class__.__name__} инициализирована с {len(historical_data)} свечами")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации стратегии: {e}")
            return False
    
    async def update(self, candle: List[Union[int, str, float]]) -> None:
        """
        Обновление стратегии новыми данными.
        
        Args:
            candle: Новая свеча для обновления стратегии
        """
        try:
            # Добавление новой цены закрытия
            self.prices.append(float(candle[4]))
            
            # Ограничение размера массива цен
            max_period = max(self.params["fast_ma_period"], self.params["slow_ma_period"])
            if len(self.prices) > max_period * 3:
                self.prices = self.prices[-max_period * 3:]
            
            # Пересчет индикаторов
            self._calculate_indicators()
            
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении стратегии: {e}")
    
    async def generate_signal(self) -> Dict[str, Any]:
        """
        Генерация торгового сигнала на основе текущего состояния стратегии.
        
        Returns:
            Торговый сигнал в виде словаря
        """
        if not self.is_initialized or len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return {"action": "HOLD", "reason": "Недостаточно данных"}
        
        # Получение текущих и предыдущих значений индикаторов
        current_fast_ma = self.fast_ma[-1]
        previous_fast_ma = self.fast_ma[-2]
        current_slow_ma = self.slow_ma[-1]
        previous_slow_ma = self.slow_ma[-2]
        current_price = self.prices[-1]
        
        # Проверка пересечения скользящих средних
        if previous_fast_ma <= previous_slow_ma and current_fast_ma > current_slow_ma:
            # Пересечение снизу вверх - сигнал на покупку
            self.last_signal = "BUY"
            self.last_signal_price = current_price
            
            # Расчет уровней стоп-лосса и тейк-профита
            stop_loss = await self.calculate_stop_loss(current_price, "BUY")
            take_profit = await self.calculate_take_profit(current_price, "BUY")
            
            return {
                "action": "BUY",
                "price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": "Пересечение MA снизу вверх",
                "confidence": self._calculate_confidence("BUY")
            }
            
        elif previous_fast_ma >= previous_slow_ma and current_fast_ma < current_slow_ma:
            # Пересечение сверху вниз - сигнал на продажу
            self.last_signal = "SELL"
            self.last_signal_price = current_price
            
            # Расчет уровней стоп-лосса и тейк-профита
            stop_loss = await self.calculate_stop_loss(current_price, "SELL")
            take_profit = await self.calculate_take_profit(current_price, "SELL")
            
            return {
                "action": "SELL",
                "price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "reason": "Пересечение MA сверху вниз",
                "confidence": self._calculate_confidence("SELL")
            }
        
        # Нет сигнала
        return {"action": "HOLD", "reason": "Нет пересечения MA"}
    
    async def calculate_position_size(self, balance: float, risk_per_trade: float) -> float:
        """
        Расчет размера позиции на основе баланса и риска.
        
        Args:
            balance: Доступный баланс
            risk_per_trade: Процент риска на сделку (от 0 до 100)
            
        Returns:
            Размер позиции
        """
        if not self.is_initialized:
            return 0.0
        
        current_price = self.prices[-1]
        
        # Расчет риска в абсолютных единицах
        risk_amount = balance * (risk_per_trade / 100)
        
        # Расчет размера позиции на основе риска и стоп-лосса
        stop_loss_percent = self.params["stop_loss_percent"]
        position_size = risk_amount / (current_price * (stop_loss_percent / 100))
        
        return position_size
    
    async def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня стоп-лосса.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень стоп-лосса
        """
        stop_loss_percent = self.params["stop_loss_percent"]
        
        if direction == "BUY":
            return entry_price * (1 - stop_loss_percent / 100)
        else:  # SELL
            return entry_price * (1 + stop_loss_percent / 100)
    
    async def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня тейк-профита.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень тейк-профита
        """
        take_profit_percent = self.params["take_profit_percent"]
        
        if direction == "BUY":
            return entry_price * (1 + take_profit_percent / 100)
        else:  # SELL
            return entry_price * (1 - take_profit_percent / 100)
    
    async def should_exit(self, position: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """
        Проверка необходимости выхода из позиции.
        
        Args:
            position: Информация о текущей позиции
            current_price: Текущая цена
            
        Returns:
            Кортеж (нужно ли выходить, причина выхода)
        """
        if not position:
            return False, ""
        
        direction = position.get("direction", "")
        entry_price = position.get("entry_price", 0.0)
        stop_loss = position.get("stop_loss", 0.0)
        take_profit = position.get("take_profit", 0.0)
        
        # Проверка достижения стоп-лосса
        if direction == "BUY" and current_price <= stop_loss:
            return True, "Достигнут стоп-лосс"
        elif direction == "SELL" and current_price >= stop_loss:
            return True, "Достигнут стоп-лосс"
        
        # Проверка достижения тейк-профита
        if direction == "BUY" and current_price >= take_profit:
            return True, "Достигнут тейк-профит"
        elif direction == "SELL" and current_price <= take_profit:
            return True, "Достигнут тейк-профит"
        
        # Проверка противоположного сигнала
        signal = await self.generate_signal()
        if (direction == "BUY" and signal["action"] == "SELL") or (direction == "SELL" and signal["action"] == "BUY"):
            return True, "Получен противоположный сигнал"
        
        return False, ""
    
    async def get_indicators(self) -> Dict[str, Any]:
        """
        Получение текущих значений индикаторов стратегии.
        
        Returns:
            Словарь с текущими значениями индикаторов
        """
        if not self.is_initialized:
            return {}
        
        return {
            "fast_ma": self.fast_ma[-1] if self.fast_ma else None,
            "slow_ma": self.slow_ma[-1] if self.slow_ma else None,
            "fast_ma_period": self.params["fast_ma_period"],
            "slow_ma_period": self.params["slow_ma_period"],
            "current_price": self.prices[-1] if self.prices else None
        }
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния стратегии.
        
        Returns:
            Словарь с текущим состоянием стратегии
        """
        return {
            "prices": self.prices,
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "last_signal": self.last_signal,
            "last_signal_price": self.last_signal_price,
            "last_signal_time": self.last_signal_time,
            "params": self.params,
            "is_initialized": self.is_initialized
        }
    
    async def set_state(self, state: Dict[str, Any]) -> None:
        """
        Установка состояния стратегии.
        
        Args:
            state: Состояние стратегии
        """
        self.prices = state.get("prices", [])
        self.fast_ma = state.get("fast_ma", [])
        self.slow_ma = state.get("slow_ma", [])
        self.last_signal = state.get("last_signal", "HOLD")
        self.last_signal_price = state.get("last_signal_price", 0.0)
        self.last_signal_time = state.get("last_signal_time", 0)
        self.params = state.get("params", self.params)
        self.is_initialized = state.get("is_initialized", False)
        
        # Пересчет индикаторов, если есть цены
        if self.prices:
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """
        Расчет индикаторов стратегии.
        """
        if len(self.prices) < self.params["slow_ma_period"]:
            return
        
        # Расчет быстрой скользящей средней
        fast_period = self.params["fast_ma_period"]
        self.fast_ma = self._calculate_simple_ma(self.prices, fast_period)
        
        # Расчет медленной скользящей средней
        slow_period = self.params["slow_ma_period"]
        self.slow_ma = self._calculate_simple_ma(self.prices, slow_period)
    
    def _calculate_simple_ma(self, data: List[float], period: int) -> List[float]:
        """
        Расчет простой скользящей средней.
        
        Args:
            data: Массив данных
            period: Период скользящей средней
            
        Returns:
            Массив значений скользящей средней
        """
        if len(data) < period:
            return []
        
        result = []
        for i in range(len(data) - period + 1):
            window = data[i:i+period]
            result.append(sum(window) / period)
        
        return result
    
    def _calculate_confidence(self, signal_type: str) -> float:
        """
        Расчет уверенности в сигнале.
        
        Args:
            signal_type: Тип сигнала ("BUY" или "SELL")
            
        Returns:
            Уверенность в сигнале от 0 до 1
        """
        if not self.is_initialized or len(self.fast_ma) < 2 or len(self.slow_ma) < 2:
            return 0.0
        
        # Расчет разницы между скользящими средними
        diff = abs(self.fast_ma[-1] - self.slow_ma[-1])
        price = self.prices[-1]
        
        # Нормализация разницы относительно цены
        normalized_diff = diff / price
        
        # Расчет уверенности на основе нормализованной разницы
        # Чем больше разница, тем выше уверенность
        confidence = min(normalized_diff * 100, 1.0)
        
        return confidence 