"""
Базовый класс для работы с биржами.

Определяет интерфейс для взаимодействия с различными биржами.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class ExchangeBase(ABC):
    """
    Базовый абстрактный класс для работы с биржами.
    
    Определяет общий интерфейс для всех классов, работающих с биржами.
    Конкретные реализации должны наследоваться от этого класса и
    реализовывать все абстрактные методы.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Инициализация базового класса биржи.
        
        Args:
            api_key: API ключ биржи
            api_secret: API секрет биржи
            testnet: Использовать тестовую сеть биржи (по умолчанию False)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
    
    @abstractmethod
    async def initialize(self):
        """
        Инициализация соединения с биржей.
        
        Должна быть вызвана перед использованием других методов.
        """
        pass
    
    @abstractmethod
    async def close(self):
        """
        Закрытие соединения с биржей.
        
        Должна быть вызвана при завершении работы с биржей.
        """
        pass
    
    @abstractmethod
    async def ping(self) -> Dict[str, Any]:
        """
        Проверка соединения с биржей.
        
        Returns:
            Результат проверки соединения
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Получение текущей цены для указанного символа.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            
        Returns:
            Информация о текущей цене
        """
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, 
                        start_time: Optional[int] = None, 
                        end_time: Optional[int] = None,
                        limit: int = 500) -> List[List[Union[int, str, float]]]:
        """
        Получение исторических данных (свечей) для указанного символа.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            interval: Интервал свечей (например, "1m", "5m", "1h", "1d")
            start_time: Начальное время в миллисекундах
            end_time: Конечное время в миллисекундах
            limit: Максимальное количество свечей
            
        Returns:
            Список свечей
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Получение информации об аккаунте.
        
        Returns:
            Информация об аккаунте
        """
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         quantity: float, price: Optional[float] = None,
                         time_in_force: str = "GTC", **kwargs) -> Dict[str, Any]:
        """
        Размещение ордера.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            side: Сторона ордера ("BUY" или "SELL")
            order_type: Тип ордера ("LIMIT", "MARKET", и т.д.)
            quantity: Количество
            price: Цена (для LIMIT ордеров)
            time_in_force: Время действия ордера (для LIMIT ордеров)
            **kwargs: Дополнительные параметры
            
        Returns:
            Информация о размещенном ордере
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                          orig_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Отмена ордера.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            order_id: ID ордера
            orig_client_order_id: ID клиентского ордера
            
        Returns:
            Информация об отмененном ордере
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение открытых ордеров.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            
        Returns:
            Список открытых ордеров
        """
        pass
    
    @abstractmethod
    async def subscribe_to_klines(self, symbol: str, interval: str, callback):
        """
        Подписка на обновления свечей.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            interval: Интервал свечей (например, "1m", "5m", "1h", "1d")
            callback: Функция обратного вызова для обработки обновлений
        """
        pass
    
    @abstractmethod
    async def subscribe_to_ticker(self, symbol: str, callback):
        """
        Подписка на обновления тикера.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            callback: Функция обратного вызова для обработки обновлений
        """
        pass 