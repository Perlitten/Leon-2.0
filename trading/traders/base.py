"""
Базовый класс трейдера, определяющий общий интерфейс для всех типов трейдеров.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio


class TraderBase:
    """
    Базовый класс для всех типов трейдеров.
    Определяет общий интерфейс и базовую логику управления позициями.
    """
    
    def __init__(self, 
                 symbol: str, 
                 exchange_client,
                 strategy, 
                 notification_service=None,
                 risk_controller=None,
                 initial_balance: float = 1000.0,
                 leverage: int = 1):
        """
        Инициализация базового трейдера.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            exchange_client: Клиент биржи для получения данных и выполнения операций
            strategy: Торговая стратегия
            notification_service: Сервис уведомлений (опционально)
            risk_controller: Контроллер рисков (опционально)
            initial_balance: Начальный баланс
            leverage: Кредитное плечо
        """
        self.symbol = symbol
        self.exchange_client = exchange_client
        self.strategy = strategy
        self.notification_service = notification_service
        self.risk_controller = risk_controller
        self.initial_balance = initial_balance
        self.leverage = leverage
        
        # Состояние позиции
        self.in_position = False
        self.current_position = None
        
        # Состояние трейдера
        self.is_running = False
        
        # Логгер
        self.logger = logging.getLogger(f"TraderBase-{self.symbol}")
    
    async def initialize(self) -> bool:
        """
        Инициализация трейдера.
        
        Returns:
            bool: Успешность инициализации
        """
        self.logger.info(f"Инициализация трейдера для {self.symbol}")
        return True
    
    async def start(self) -> bool:
        """
        Запуск трейдера.
        
        Returns:
            bool: Успешность запуска
        """
        self.logger.info(f"Запуск трейдера для {self.symbol}")
        self.is_running = True
        return True
    
    async def stop(self) -> bool:
        """
        Остановка трейдера.
        
        Returns:
            bool: Успешность остановки
        """
        self.logger.info(f"Остановка трейдера для {self.symbol}")
        self.is_running = False
        return True
    
    async def enter_position(self, 
                            direction: str, 
                            size: float, 
                            price: Optional[float] = None) -> Dict[str, Any]:
        """
        Вход в позицию.
        
        Args:
            direction: Направление ("LONG" или "SHORT")
            size: Размер позиции
            price: Цена входа (опционально)
            
        Returns:
            Dict[str, Any]: Информация о созданной позиции
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    async def exit_position(self, 
                           position_id: str, 
                           price: Optional[float] = None) -> Dict[str, Any]:
        """
        Выход из позиции.
        
        Args:
            position_id: Идентификатор позиции
            price: Цена выхода (опционально)
            
        Returns:
            Dict[str, Any]: Информация о закрытой позиции
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    async def update_position(self, 
                             position_id: str, 
                             stop_loss: Optional[float] = None, 
                             take_profit: Optional[float] = None) -> bool:
        """
        Обновление параметров позиции.
        
        Args:
            position_id: Идентификатор позиции
            stop_loss: Новый уровень стоп-лосса (опционально)
            take_profit: Новый уровень тейк-профита (опционально)
            
        Returns:
            bool: Успешность обновления
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Получение списка открытых позиций.
        
        Returns:
            List[Dict[str, Any]]: Список открытых позиций
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    async def get_balance(self) -> float:
        """
        Получение текущего баланса.
        
        Returns:
            float: Текущий баланс
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе")
    
    async def get_current_price(self) -> float:
        """
        Получение текущей цены торговой пары.
        
        Returns:
            float: Текущая цена
        """
        raise NotImplementedError("Метод должен быть реализован в дочернем классе") 