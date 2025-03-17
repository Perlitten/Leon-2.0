"""
Модуль управления рисками.

Предоставляет класс для управления рисками при торговле.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading.risk.position_sizer import PositionSizer


class RiskManager:
    """
    Класс для управления рисками при торговле.
    
    Отвечает за контроль рисков, расчет размеров позиций,
    установку стоп-лоссов и тейк-профитов, а также за
    соблюдение ограничений по убыткам и количеству сделок.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера рисков.
        
        Args:
            config: Конфигурация менеджера рисков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Основные параметры риска
        self.max_daily_loss = config.get("max_daily_loss", 5.0)  # Максимальный дневной убыток в процентах
        self.max_daily_trades = config.get("max_daily_trades", 10)  # Максимальное количество сделок в день
        self.max_open_positions = config.get("max_open_positions", 5)  # Максимальное количество открытых позиций
        self.default_stop_loss = config.get("default_stop_loss", 2.0)  # Стоп-лосс по умолчанию в процентах
        self.default_take_profit = config.get("default_take_profit", 3.0)  # Тейк-профит по умолчанию в процентах
        
        # Инициализация калькулятора размера позиции
        position_sizer_config = config.get("position_sizer", {})
        self.position_sizer = PositionSizer(
            default_method=position_sizer_config.get("method", "fixed_risk"),
            default_params=position_sizer_config.get("params", {"risk_percent": 1.0})
        )
        
        # Статистика торговли
        self.daily_trades = []  # Список сделок за день
        self.daily_pnl = 0.0  # Прибыль/убыток за день
        self.open_positions = []  # Список открытых позиций
        self.last_reset = datetime.now()  # Время последнего сброса статистики
        
        self.logger.info(f"Инициализирован менеджер рисков (max_daily_loss={self.max_daily_loss}%, "
                        f"max_daily_trades={self.max_daily_trades}, "
                        f"max_open_positions={self.max_open_positions})")
    
    async def assess_risk(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценка риска для списка сигналов.
        
        Args:
            signals: Список торговых сигналов
            market_data: Рыночные данные
            
        Returns:
            Dict[str, Any]: Результат оценки риска
        """
        # Проверяем, нужно ли сбросить дневную статистику
        self._check_daily_reset()
        
        # Если нет сигналов, возвращаем разрешение на торговлю
        if not signals:
            return {
                "can_trade": True,
                "reason": "Нет сигналов для оценки",
                "position_size": 0.0
            }
        
        # Проверяем ограничения по количеству сделок
        if len(self.daily_trades) >= self.max_daily_trades:
            return {
                "can_trade": False,
                "reason": f"Достигнут лимит дневных сделок ({self.max_daily_trades})",
                "position_size": 0.0
            }
        
        # Проверяем ограничения по дневному убытку
        if self.daily_pnl <= -self.max_daily_loss:
            return {
                "can_trade": False,
                "reason": f"Достигнут лимит дневного убытка ({self.max_daily_loss}%)",
                "position_size": 0.0
            }
        
        # Проверяем ограничения по количеству открытых позиций
        if len(self.open_positions) >= self.max_open_positions:
            return {
                "can_trade": False,
                "reason": f"Достигнут лимит открытых позиций ({self.max_open_positions})",
                "position_size": 0.0
            }
        
        # Находим самый сильный сигнал
        strongest_signal = max(signals, key=lambda x: x.get("strength", 0))
        
        # Получаем текущий баланс (заглушка)
        balance = market_data.get("balance", 1000.0)
        
        # Получаем текущую цену
        symbol = strongest_signal.get("symbol", "BTCUSDT")
        current_price = market_data.get("price", {}).get(symbol, 0.0)
        
        if current_price <= 0:
            return {
                "can_trade": False,
                "reason": "Некорректная цена актива",
                "position_size": 0.0
            }
        
        # Рассчитываем стоп-лосс и тейк-профит
        stop_loss_percent = strongest_signal.get("stop_loss", self.default_stop_loss)
        take_profit_percent = strongest_signal.get("take_profit", self.default_take_profit)
        
        # Рассчитываем цены стоп-лосса и тейк-профита
        direction = strongest_signal.get("type", "buy")
        
        if direction == "buy":
            stop_loss_price = current_price * (1 - stop_loss_percent / 100)
            take_profit_price = current_price * (1 + take_profit_percent / 100)
        else:  # sell
            stop_loss_price = current_price * (1 + stop_loss_percent / 100)
            take_profit_price = current_price * (1 - take_profit_percent / 100)
        
        # Рассчитываем размер позиции
        position_size = self.position_sizer.calculate(
            balance=balance,
            entry_price=current_price,
            stop_loss=stop_loss_price,
            method="fixed_risk",
            params={"risk_percent": 1.0}
        )
        
        # Возвращаем результат оценки риска
        return {
            "can_trade": True,
            "reason": "Торговля разрешена",
            "position_size": position_size,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "risk_reward_ratio": take_profit_percent / stop_loss_percent
        }
    
    def register_trade(self, trade: Dict[str, Any]) -> None:
        """
        Регистрация новой сделки.
        
        Args:
            trade: Информация о сделке
        """
        # Проверяем, нужно ли сбросить дневную статистику
        self._check_daily_reset()
        
        # Добавляем сделку в список дневных сделок
        self.daily_trades.append(trade)
        
        # Если это открытие позиции, добавляем ее в список открытых позиций
        if trade.get("action") in ["buy", "sell"]:
            self.open_positions.append(trade)
            self.logger.info(f"Открыта новая позиция: {trade.get('symbol')} {trade.get('action')} "
                            f"(размер: {trade.get('amount')})")
        
        # Если это закрытие позиции, обновляем список открытых позиций и дневной P&L
        elif trade.get("action") in ["close_buy", "close_sell"]:
            # Находим соответствующую открытую позицию
            position_id = trade.get("position_id")
            for i, position in enumerate(self.open_positions):
                if position.get("id") == position_id:
                    # Удаляем позицию из списка открытых
                    closed_position = self.open_positions.pop(i)
                    
                    # Рассчитываем P&L
                    pnl = trade.get("pnl", 0.0)
                    self.daily_pnl += pnl
                    
                    self.logger.info(f"Закрыта позиция: {closed_position.get('symbol')} "
                                    f"(P&L: {pnl:.2f}, дневной P&L: {self.daily_pnl:.2f})")
                    break
    
    def get_position_size(self, balance: float, entry_price: float, stop_loss_price: float,
                         method: str = "fixed_risk", params: Optional[Dict[str, Any]] = None) -> float:
        """
        Расчет размера позиции.
        
        Args:
            balance: Текущий баланс
            entry_price: Цена входа
            stop_loss_price: Цена стоп-лосса
            method: Метод расчета размера позиции
            params: Параметры для метода расчета
            
        Returns:
            float: Размер позиции
        """
        return self.position_sizer.calculate(
            balance=balance,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            method=method,
            params=params
        )
    
    def get_stop_loss_price(self, entry_price: float, direction: str, 
                          percent: Optional[float] = None) -> float:
        """
        Расчет цены стоп-лосса.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ('buy' или 'sell')
            percent: Процент стоп-лосса (если None, используется значение по умолчанию)
            
        Returns:
            float: Цена стоп-лосса
        """
        stop_loss_percent = percent if percent is not None else self.default_stop_loss
        
        if direction == "buy":
            return entry_price * (1 - stop_loss_percent / 100)
        else:  # sell
            return entry_price * (1 + stop_loss_percent / 100)
    
    def get_take_profit_price(self, entry_price: float, direction: str, 
                            percent: Optional[float] = None) -> float:
        """
        Расчет цены тейк-профита.
        
        Args:
            entry_price: Цена входа
            direction: Направление сделки ('buy' или 'sell')
            percent: Процент тейк-профита (если None, используется значение по умолчанию)
            
        Returns:
            float: Цена тейк-профита
        """
        take_profit_percent = percent if percent is not None else self.default_take_profit
        
        if direction == "buy":
            return entry_price * (1 + take_profit_percent / 100)
        else:  # sell
            return entry_price * (1 - take_profit_percent / 100)
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по рискам.
        
        Returns:
            Dict[str, Any]: Статистика по рискам
        """
        return {
            "daily_trades_count": len(self.daily_trades),
            "max_daily_trades": self.max_daily_trades,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "open_positions_count": len(self.open_positions),
            "max_open_positions": self.max_open_positions,
            "last_reset": self.last_reset.isoformat()
        }
    
    def _check_daily_reset(self) -> None:
        """
        Проверка необходимости сброса дневной статистики.
        """
        now = datetime.now()
        
        # Если прошел день с момента последнего сброса, сбрасываем статистику
        if now.date() > self.last_reset.date():
            self.daily_trades = []
            self.daily_pnl = 0.0
            self.last_reset = now
            
            self.logger.info("Сброс дневной статистики торговли")
    
    def reset_stats(self) -> None:
        """
        Сброс всей статистики.
        """
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()
        
        self.logger.info("Сброс всей статистики торговли") 