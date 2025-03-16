"""
Модуль управления рисками.

Предоставляет класс для управления рисками при торговле.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from trading.risk.position_sizer import PositionSizer

class RiskManager:
    """
    Класс для управления рисками при торговле.
    
    Отвечает за:
    - Расчет размера позиции
    - Установку стоп-лоссов и тейк-профитов
    - Контроль максимальных убытков
    - Управление риском на портфель
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера рисков.
        
        Args:
            config: Конфигурация менеджера рисков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Создание калькулятора размера позиции
        position_sizer_config = config.get("position_sizer", {})
        self.position_sizer = PositionSizer(
            default_method=position_sizer_config.get("method", "fixed_risk"),
            default_params=position_sizer_config.get("params", {})
        )
        
        # Параметры управления рисками
        self.max_open_positions = config.get("max_open_positions", 5)
        self.max_daily_loss = config.get("max_daily_loss", 5.0)  # Процент от баланса
        self.max_position_size = config.get("max_position_size", 10.0)  # Процент от баланса
        self.default_stop_loss = config.get("default_stop_loss", 2.0)  # Процент от цены входа
        self.default_take_profit = config.get("default_take_profit", 3.0)  # Процент от цены входа
        
        # Статистика
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.open_positions = []
        self.closed_positions = []
    
    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float,
                               method: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> float:
        """
        Расчет размера позиции с учетом ограничений.
        
        Args:
            balance: Доступный баланс
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            method: Метод расчета
            params: Параметры метода
            
        Returns:
            Размер позиции
        """
        # Расчет размера позиции
        position_size = self.position_sizer.calculate(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss,
            method=method,
            params=params
        )
        
        # Проверка ограничений
        max_position_value = balance * (self.max_position_size / 100)
        max_position_size = max_position_value / entry_price
        
        # Ограничение размера позиции
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня стоп-лосса.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень стоп-лосса
        """
        stop_loss_percent = self.default_stop_loss
        
        if direction == "BUY":
            return entry_price * (1 - stop_loss_percent / 100)
        else:  # SELL
            return entry_price * (1 + stop_loss_percent / 100)
    
    def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """
        Расчет уровня тейк-профита.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ("BUY" или "SELL")
            
        Returns:
            Уровень тейк-профита
        """
        take_profit_percent = self.default_take_profit
        
        if direction == "BUY":
            return entry_price * (1 + take_profit_percent / 100)
        else:  # SELL
            return entry_price * (1 - take_profit_percent / 100)
    
    def can_open_position(self, balance: float) -> Tuple[bool, str]:
        """
        Проверка возможности открытия новой позиции.
        
        Args:
            balance: Текущий баланс
            
        Returns:
            Кортеж (можно ли открыть позицию, причина)
        """
        # Проверка количества открытых позиций
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Достигнуто максимальное количество открытых позиций ({self.max_open_positions})"
        
        # Проверка дневного убытка
        max_loss = balance * (self.max_daily_loss / 100)
        if self.daily_pnl < -max_loss:
            return False, f"Достигнут максимальный дневной убыток ({self.max_daily_loss}%)"
        
        return True, ""
    
    def add_position(self, position: Dict[str, Any]) -> None:
        """
        Добавление новой позиции.
        
        Args:
            position: Информация о позиции
        """
        self.open_positions.append(position)
        self.logger.info(f"Открыта новая позиция: {position}")
    
    def close_position(self, position: Dict[str, Any], exit_price: float, pnl: float) -> None:
        """
        Закрытие позиции.
        
        Args:
            position: Информация о позиции
            exit_price: Цена выхода
            pnl: Прибыль/убыток
        """
        # Обновление позиции
        position["exit_price"] = exit_price
        position["pnl"] = pnl
        position["status"] = "closed"
        
        # Перемещение позиции из открытых в закрытые
        self.open_positions.remove(position)
        self.closed_positions.append(position)
        
        # Обновление статистики
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        self.logger.info(f"Закрыта позиция: {position}")
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о позиции по ID.
        
        Args:
            position_id: ID позиции
            
        Returns:
            Информация о позиции или None, если позиция не найдена
        """
        for position in self.open_positions:
            if position.get("id") == position_id:
                return position
        
        return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Получение списка открытых позиций.
        
        Returns:
            Список открытых позиций
        """
        return self.open_positions
    
    def get_closed_positions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получение списка закрытых позиций.
        
        Args:
            limit: Максимальное количество позиций
            
        Returns:
            Список закрытых позиций
        """
        return self.closed_positions[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики торговли.
        
        Returns:
            Статистика торговли
        """
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        return {
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "open_positions_count": len(self.open_positions),
            "closed_positions_count": len(self.closed_positions)
        }
    
    def reset_daily_statistics(self) -> None:
        """
        Сброс дневной статистики.
        """
        self.daily_pnl = 0.0
        self.logger.info("Дневная статистика сброшена")
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновление конфигурации менеджера рисков.
        
        Args:
            config: Новая конфигурация
        """
        self.config.update(config)
        
        # Обновление параметров
        self.max_open_positions = self.config.get("max_open_positions", self.max_open_positions)
        self.max_daily_loss = self.config.get("max_daily_loss", self.max_daily_loss)
        self.max_position_size = self.config.get("max_position_size", self.max_position_size)
        self.default_stop_loss = self.config.get("default_stop_loss", self.default_stop_loss)
        self.default_take_profit = self.config.get("default_take_profit", self.default_take_profit)
        
        # Обновление калькулятора размера позиции
        position_sizer_config = self.config.get("position_sizer", {})
        self.position_sizer = PositionSizer(
            default_method=position_sizer_config.get("method", self.position_sizer.default_method),
            default_params=position_sizer_config.get("params", self.position_sizer.default_params)
        )
        
        self.logger.info("Конфигурация менеджера рисков обновлена") 