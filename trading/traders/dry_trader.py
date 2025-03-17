"""
Модуль трейдера для работы в режиме симуляции (dry mode).
Позволяет тестировать стратегии без реальных сделок с имитацией реального рынка.
"""

import asyncio
import logging
import traceback
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import random
import time
import uuid
import math
from enum import Enum, auto

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False


class MarketCondition(Enum):
    """Типы рыночных состояний для симуляции"""
    NORMAL = auto()
    TREND_UP = auto()
    TREND_DOWN = auto()
    VOLATILE = auto()
    FLAT = auto()
    FLASH_CRASH = auto()
    FLASH_RALLY = auto()
    LOW_LIQUIDITY = auto()


class TimeSeriesBuffer:
    """Эффективное хранение временных рядов (цены, объемы) в памяти"""
    
    def __init__(self, max_size=10000):
        """
        Инициализация буфера временных рядов.
        
        Args:
            max_size: Максимальный размер буфера
        """
        self.data = np.zeros(max_size, dtype=[
            ('timestamp', 'i8'),
            ('price', 'f8'),
            ('volume', 'f8')
        ])
        self.position = 0
        self.is_full = False
        self._max_size = max_size
    
    def add(self, timestamp: int, price: float, volume: float) -> None:
        """
        Добавить новые данные в буфер.
        
        Args:
            timestamp: Временная метка в формате timestamp
            price: Цена
            volume: Объем торгов
        """
        self.data[self.position] = (timestamp, price, volume)
        self.position = (self.position + 1) % self._max_size
        
        if self.position == 0:
            self.is_full = True
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Получить данные в формате DataFrame.
        
        Returns:
            pd.DataFrame: Данные временного ряда
        """
        if self.is_full:
            # Если буфер заполнен, возвращаем все данные, начиная с текущей позиции
            return pd.DataFrame(
                np.concatenate([self.data[self.position:], self.data[:self.position]]),
                columns=['timestamp', 'price', 'volume']
            )
        else:
            # Если буфер не заполнен, возвращаем только заполненную часть
            return pd.DataFrame(
                self.data[:self.position],
                columns=['timestamp', 'price', 'volume']
            )
    
    def get_latest_price(self) -> Optional[float]:
        """
        Получить последнюю цену из буфера.
        
        Returns:
            Optional[float]: Последняя цена или None, если буфер пуст
        """
        if self.position > 0:
            return self.data[self.position - 1]['price']
        elif self.is_full:
            return self.data[self._max_size - 1]['price']
        else:
            return None
    
    def get_price_volatility(self, window: int = 20) -> float:
        """
        Рассчитать волатильность цены.
        
        Args:
            window: Размер окна для расчета волатильности
            
        Returns:
            float: Волатильность цены в процентах
        """
        df = self.get_data_as_dataframe()
        
        if len(df) < window:
            return 0.0
        
        # Используем последние window значений
        prices = df['price'].iloc[-window:].values
        
        # Рассчитываем логарифмические доходности
        log_returns = np.diff(np.log(prices))
        
        # Волатильность - стандартное отклонение логарифмических доходностей
        volatility = np.std(log_returns) * 100
        
        return volatility


class OrderBook:
    """Симуляция стакана заявок (Order Book)"""
    
    def __init__(self, base_price: float, depth: int = 10, volatility: float = 0.005):
        """
        Инициализация стакана заявок.
        
        Args:
            base_price: Базовая цена
            depth: Глубина стакана
            volatility: Волатильность для генерации вариаций цен
        """
        self.base_price = base_price
        self.depth = depth
        self.volatility = volatility
        
        # Инициализация стакана заявок
        self.bids = []  # Покупатели (цена, объем)
        self.asks = []  # Продавцы (цена, объем)
        
        # Генерация начального стакана заявок
        self.generate_orders()
    
    def generate_orders(self) -> None:
        """Генерация стакана заявок на основе базовой цены."""
        self.bids = []
        self.asks = []
        
        # Параметры распределения объемов
        min_volume = 0.1
        max_volume = 10.0
        
        # Генерация заявок на покупку (bids)
        for i in range(self.depth):
            # Цена уменьшается с удалением от базовой цены
            price_decrease = self.base_price * self.volatility * (i + 1) * (1 + random.uniform(-0.5, 0.5))
            price = self.base_price - price_decrease
            
            # Объем увеличивается с удалением от базовой цены (реальные рынки имеют больший объем на уровнях дальше от середины)
            volume = random.uniform(min_volume, max_volume) * (1 + i * 0.2)
            
            self.bids.append((price, volume))
        
        # Сортировка заявок на покупку по убыванию цены
        self.bids.sort(key=lambda x: x[0], reverse=True)
        
        # Генерация заявок на продажу (asks)
        for i in range(self.depth):
            # Цена увеличивается с удалением от базовой цены
            price_increase = self.base_price * self.volatility * (i + 1) * (1 + random.uniform(-0.5, 0.5))
            price = self.base_price + price_increase
            
            # Объем увеличивается с удалением от базовой цены
            volume = random.uniform(min_volume, max_volume) * (1 + i * 0.2)
            
            self.asks.append((price, volume))
        
        # Сортировка заявок на продажу по возрастанию цены
        self.asks.sort(key=lambda x: x[0])
    
    def update(self, new_base_price: float = None, market_condition: MarketCondition = MarketCondition.NORMAL) -> None:
        """
        Обновление стакана заявок с учетом рыночных условий.
        
        Args:
            new_base_price: Новая базовая цена (если None, используется текущая)
            market_condition: Состояние рынка
        """
        if new_base_price is not None:
            self.base_price = new_base_price
        
        # Модифицируем волатильность и распределение объемов в зависимости от состояния рынка
        original_volatility = self.volatility
        volume_skew = 0  # Смещение объема (положительное - в сторону покупок, отрицательное - в сторону продаж)
        
        if market_condition == MarketCondition.VOLATILE:
            self.volatility *= 2.5
        elif market_condition == MarketCondition.FLAT:
            self.volatility *= 0.5
        elif market_condition == MarketCondition.TREND_UP:
            volume_skew = 0.3  # Больше объема на покупках
        elif market_condition == MarketCondition.TREND_DOWN:
            volume_skew = -0.3  # Больше объема на продажах
        elif market_condition == MarketCondition.FLASH_CRASH:
            self.volatility *= 3.0
            volume_skew = -0.6
        elif market_condition == MarketCondition.FLASH_RALLY:
            self.volatility *= 3.0
            volume_skew = 0.6
        elif market_condition == MarketCondition.LOW_LIQUIDITY:
            # Уменьшаем общий объем, но увеличиваем спреды
            self.volatility *= 1.5
            
        # Генерация заявок
        self.generate_orders()
        
        # Модифицируем объемы с учетом смещения
        if volume_skew != 0:
            # Для покупок (bids)
            for i in range(len(self.bids)):
                price, volume = self.bids[i]
                if volume_skew > 0:
                    # Увеличиваем объем покупок
                    self.bids[i] = (price, volume * (1 + volume_skew))
                else:
                    # Уменьшаем объем покупок
                    self.bids[i] = (price, volume * (1 + volume_skew/2))
            
            # Для продаж (asks)
            for i in range(len(self.asks)):
                price, volume = self.asks[i]
                if volume_skew < 0:
                    # Увеличиваем объем продаж
                    self.asks[i] = (price, volume * (1 - volume_skew))
                else:
                    # Уменьшаем объем продаж
                    self.asks[i] = (price, volume * (1 - volume_skew/2))
        
        # Восстанавливаем исходную волатильность для следующего обновления
        self.volatility = original_volatility
    
    def get_market_price(self) -> float:
        """
        Получить рыночную цену (середина спреда).
        
        Returns:
            float: Рыночная цена
        """
        if self.asks and self.bids:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        else:
            return self.base_price
    
    def get_best_bid(self) -> Tuple[float, float]:
        """
        Получить лучшую заявку на покупку.
        
        Returns:
            Tuple[float, float]: (цена, объем)
        """
        if self.bids:
            return self.bids[0]
        else:
            return (self.base_price * 0.99, 1.0)
    
    def get_best_ask(self) -> Tuple[float, float]:
        """
        Получить лучшую заявку на продажу.
        
        Returns:
            Tuple[float, float]: (цена, объем)
        """
        if self.asks:
            return self.asks[0]
        else:
            return (self.base_price * 1.01, 1.0)
    
    def get_bid_ladder(self) -> List[Tuple[float, float]]:
        """
        Получить уровни цен на покупку.
        
        Returns:
            List[Tuple[float, float]]: Список пар (цена, объем)
        """
        return self.bids
    
    def get_ask_ladder(self) -> List[Tuple[float, float]]:
        """
        Получить уровни цен на продажу.
        
        Returns:
            List[Tuple[float, float]]: Список пар (цена, объем)
        """
        return self.asks
    
    def execute_market_order(self, side: str, size: float) -> Tuple[float, float, float]:
        """
        Исполнить рыночный ордер с учетом проскальзывания.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            
        Returns:
            Tuple[float, float, float]: (средняя цена исполнения, исполненный объем, проскальзывание в %)
        """
        executed_price = 0.0
        executed_volume = 0.0
        orders = self.asks if side.lower() == 'buy' else self.bids
        
        # Клонируем массив ордеров, чтобы не изменять оригинал
        orders_copy = orders.copy()
        
        total_price = 0.0
        remaining_size = size
        
        # Получаем лучшую цену для оценки проскальзывания
        best_price = orders_copy[0][0] if orders_copy else self.base_price
        
        # Проходим по ордерам, пока не наберем нужный объем
        for price, volume in orders_copy:
            if remaining_size <= 0:
                break
                
            used_volume = min(volume, remaining_size)
            total_price += price * used_volume
            executed_volume += used_volume
            remaining_size -= used_volume
        
        # Если нам удалось исполнить ордер хотя бы частично
        if executed_volume > 0:
            executed_price = total_price / executed_volume
            
            # Вычисляем проскальзывание в процентах
            slippage_percent = abs(executed_price - best_price) / best_price * 100
        else:
            # Если не нашлось ордеров для исполнения
            executed_price = best_price
            slippage_percent = 0.0
        
        # После исполнения обновляем стакан заявок
        if side.lower() == 'buy':
            # Убираем использованный объем из лучших заявок на продажу
            remaining_to_execute = size
            for i in range(len(self.asks)):
                if remaining_to_execute <= 0:
                    break
                    
                if remaining_to_execute >= self.asks[i][1]:
                    # Полностью используем этот уровень
                    remaining_to_execute -= self.asks[i][1]
                    self.asks[i] = (self.asks[i][0], 0.01)  # Оставляем минимальный объем для непрерывности
                else:
                    # Используем часть этого уровня
                    self.asks[i] = (self.asks[i][0], self.asks[i][1] - remaining_to_execute)
                    remaining_to_execute = 0
        else:
            # Убираем использованный объем из лучших заявок на покупку
            remaining_to_execute = size
            for i in range(len(self.bids)):
                if remaining_to_execute <= 0:
                    break
                    
                if remaining_to_execute >= self.bids[i][1]:
                    # Полностью используем этот уровень
                    remaining_to_execute -= self.bids[i][1]
                    self.bids[i] = (self.bids[i][0], 0.01)  # Оставляем минимальный объем для непрерывности
                else:
                    # Используем часть этого уровня
                    self.bids[i] = (self.bids[i][0], self.bids[i][1] - remaining_to_execute)
                    remaining_to_execute = 0
        
        return executed_price, executed_volume, slippage_percent


class MarketSimulator:
    """Симулятор рыночных условий и движений цены"""
    
    def __init__(self, 
                 symbol: str,
                 initial_price: float,
                 volatility: float = 0.1,
                 trend: float = 0.0,
                 min_price: float = None,
                 max_price: float = None,
                 market_condition: MarketCondition = MarketCondition.NORMAL):
        """
        Инициализация симулятора рынка.
        
        Args:
            symbol: Торговая пара
            initial_price: Начальная цена
            volatility: Волатильность (стандартное отклонение изменений цены)
            trend: Тренд (положительное значение - восходящий, отрицательное - нисходящий)
            min_price: Минимальная цена (если None, минимальная цена не ограничена)
            max_price: Максимальная цена (если None, максимальная цена не ограничена)
            market_condition: Начальное состояние рынка
        """
        self.symbol = symbol
        self.current_price = initial_price
        self.base_volatility = volatility
        self.current_volatility = volatility
        self.base_trend = trend
        self.current_trend = trend
        self.min_price = min_price if min_price is not None else initial_price * 0.01
        self.max_price = max_price if max_price is not None else initial_price * 100
        self.market_condition = market_condition
        
        # Создаем стакан заявок
        self.order_book = OrderBook(initial_price)
        
        # Хранение данных цены и объема
        self.price_buffer = TimeSeriesBuffer(max_size=10000)
        
        # Время последней генерации цены
        self.last_time = time.time()
        
        # Логгер
        self.logger = logging.getLogger(f"MarketSimulator-{self.symbol}")
        self.logger.info(f"Инициализирован симулятор рынка для {self.symbol} с начальной ценой {initial_price}")
        
        # Добавляем первую точку данных
        self.price_buffer.add(
            int(time.time()), 
            initial_price, 
            random.uniform(10, 100)
        )
    
    def generate_next_price(self, time_delta: float = 1.0) -> float:
        """
        Генерирует следующую цену на основе текущего состояния рынка.
        
        Args:
            time_delta: Временной интервал для генерации цены (в секундах)
            
        Returns:
            float: Новая цена
        """
        # Рассчитываем изменение цены на основе волатильности и тренда
        price_change_ratio = random.normalvariate(
            self.current_trend * time_delta, 
            self.current_volatility * math.sqrt(time_delta)
        )
        
        # Применяем изменение цены
        new_price = self.current_price * (1 + price_change_ratio)
        
        # Проверяем, не вышла ли цена за пределы
        if new_price < self.min_price:
            new_price = self.min_price
        elif new_price > self.max_price:
            new_price = self.max_price
        
        # Обновляем текущую цену
        self.current_price = new_price
        
        # Обновляем стакан заявок
        self.order_book.update(new_price, self.market_condition)
        
        # Сохраняем новую цену в буфере
        current_time = int(time.time())
        
        # Генерируем объем на основе волатильности
        # При высокой волатильности обычно выше объемы торгов
        base_volume = random.lognormvariate(3, 0.5)  # Логнормальное распределение для объемов
        volume_multiplier = 1.0 + abs(price_change_ratio) * 10  # Увеличиваем объем при больших движениях цены
        volume = base_volume * volume_multiplier
        
        self.price_buffer.add(current_time, new_price, volume)
        
        return new_price
    
    def update_market_condition(self, new_condition: MarketCondition) -> None:
        """
        Обновляет состояние рынка.
        
        Args:
            new_condition: Новое состояние рынка
        """
        self.market_condition = new_condition
        
        # Обновляем параметры в зависимости от состояния рынка
        if new_condition == MarketCondition.NORMAL:
            self.current_volatility = self.base_volatility
            self.current_trend = self.base_trend
        elif new_condition == MarketCondition.TREND_UP:
            self.current_volatility = self.base_volatility * 1.2
            self.current_trend = abs(self.base_trend) + 0.001  # Положительный тренд
        elif new_condition == MarketCondition.TREND_DOWN:
            self.current_volatility = self.base_volatility * 1.2
            self.current_trend = -abs(self.base_trend) - 0.001  # Отрицательный тренд
        elif new_condition == MarketCondition.VOLATILE:
            self.current_volatility = self.base_volatility * 2.5
            self.current_trend = self.base_trend
        elif new_condition == MarketCondition.FLAT:
            self.current_volatility = self.base_volatility * 0.5
            self.current_trend = 0
        elif new_condition == MarketCondition.FLASH_CRASH:
            self.current_volatility = self.base_volatility * 3.0
            self.current_trend = -0.01  # Резкое снижение цены
        elif new_condition == MarketCondition.FLASH_RALLY:
            self.current_volatility = self.base_volatility * 3.0
            self.current_trend = 0.01  # Резкий рост цены
        elif new_condition == MarketCondition.LOW_LIQUIDITY:
            self.current_volatility = self.base_volatility * 1.5
            self.current_trend = self.base_trend
        
        self.logger.info(f"Состояние рынка изменено на {new_condition.name}")
    
    def simulate_special_condition(self, condition: MarketCondition, duration: float = 10.0) -> None:
        """
        Запускает специальное рыночное событие (асинхронно).
        
        Args:
            condition: Состояние рынка для симуляции
            duration: Продолжительность события в секундах
        """
        # Сохраняем предыдущее состояние рынка
        previous_condition = self.market_condition
        
        # Применяем новое состояние
        self.update_market_condition(condition)
        
        # Запускаем асинхронную задачу для возврата в предыдущее состояние через указанное время
        async def revert_condition():
            await asyncio.sleep(duration)
            self.update_market_condition(previous_condition)
            self.logger.info(f"Возврат к предыдущему состоянию рынка {previous_condition.name} после специального события")
        
        # Запускаем задачу (необходимо быть в асинхронном контексте)
        asyncio.create_task(revert_condition())
    
    def get_current_price(self) -> float:
        """
        Получить текущую рыночную цену.
        
        Returns:
            float: Текущая цена
        """
        # Обновляем цену, если прошло достаточно времени с последней генерации
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        if time_delta > 0.1:  # Обновляем цену, если прошло более 0.1 секунды
            self.generate_next_price(time_delta)
            self.last_time = current_time
        
        return self.current_price
    
    def get_market_info(self) -> Dict[str, Any]:
        """
        Получить информацию о текущем состоянии рынка.
        
        Returns:
            Dict[str, Any]: Информация о рынке
        """
        # Получаем текущую волатильность
        volatility = self.price_buffer.get_price_volatility()
        
        # Определяем направление тренда
        df = self.price_buffer.get_data_as_dataframe()
        if len(df) < 20:
            trend_direction = 0.0
        else:
            # Используем простую линейную регрессию для определения направления тренда
            prices = df['price'].iloc[-20:].values
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Нормализуем наклон для получения значения от -1 до 1
            trend_direction = np.clip(slope / (prices.mean() * 0.01), -1, 1)
        
        return {
            'symbol': self.symbol,
            'price': self.current_price,
            'market_condition': self.market_condition,
            'volatility': volatility,
            'trend_direction': trend_direction,
            'bid_price': self.order_book.get_best_bid()[0],
            'ask_price': self.order_book.get_best_ask()[0],
            'spread': self.order_book.get_best_ask()[0] - self.order_book.get_best_bid()[0],
            'spread_percent': (self.order_book.get_best_ask()[0] - self.order_book.get_best_bid()[0]) / self.current_price * 100
        }
    
    def execute_market_order(self, side: str, size: float) -> Dict[str, Any]:
        """
        Исполнить рыночный ордер с учетом проскальзывания.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        executed_price, executed_volume, slippage = self.order_book.execute_market_order(side, size)
        
        return {
            'symbol': self.symbol,
            'side': side,
            'requested_size': size,
            'executed_volume': executed_volume,
            'executed_price': executed_price,
            'slippage': slippage,
            'order_book_impact': True if slippage > 0.1 else False,
            'timestamp': datetime.now().isoformat()
        }


class DryModeData:
    """Встроенный менеджер для виртуального баланса и истории сделок"""
    
    def __init__(self, symbol: str, storage_file: Optional[str] = None):
        """
        Инициализация менеджера виртуального баланса.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            storage_file: Файл для хранения данных
        """
        self.symbol = symbol
        if storage_file is None:
            self.storage_file = f'data/dry_mode_{symbol.lower().replace("/", "_")}.json'
        else:
            self.storage_file = storage_file
            
        self.data = {
            "balance": 1000.0,  # Начальный баланс по умолчанию
            "initial_balance": 1000.0,
            "trades": [],
            "open_positions": []
        }
        
        self.logger = logging.getLogger(f"DryModeData-{self.symbol}")
        
        # Загрузка данных, если файл существует
        self.load_data()
    
    async def load_data_async(self) -> bool:
        """
        Асинхронная загрузка данных из файла.
        
        Returns:
            bool: Успешность загрузки
        """
        if not AIOFILES_AVAILABLE:
            self.logger.warning("Библиотека aiofiles не установлена, используется синхронная загрузка")
            return self.load_data()
            
        try:
            if os.path.exists(self.storage_file):
                async with aiofiles.open(self.storage_file, 'r') as f:
                    content = await f.read()
                    self.data = json.loads(content)
                self.logger.info(f"Данные dry mode асинхронно загружены из {self.storage_file}")
                return True
            else:
                self.logger.info(f"Файл данных {self.storage_file} не найден, используются значения по умолчанию")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка асинхронной загрузки данных dry mode: {str(e)}")
            return False
    
    def load_data(self) -> bool:
        """
        Загрузка данных из файла.
        
        Returns:
            bool: Успешность загрузки
        """
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.data = json.load(f)
                self.logger.info(f"Данные dry mode загружены из {self.storage_file}")
                return True
            else:
                self.logger.info(f"Файл данных {self.storage_file} не найден, используются значения по умолчанию")
                return False
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных dry mode: {str(e)}")
            return False
    
    def initialize(self, initial_balance: float) -> None:
        """
        Инициализация данных с заданным начальным балансом.
        
        Args:
            initial_balance: Начальный баланс
        """
        self.data["balance"] = initial_balance
        self.data["initial_balance"] = initial_balance
        self.data["trades"] = []
        self.data["open_positions"] = []
        self.save_data()
    
    async def save_data_async(self) -> bool:
        """
        Асинхронное сохранение данных в файл.
        
        Returns:
            bool: Успешность сохранения
        """
        if not AIOFILES_AVAILABLE:
            self.logger.warning("Библиотека aiofiles не установлена, используется синхронное сохранение")
            return self.save_data()
            
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            async with aiofiles.open(self.storage_file, 'w') as f:
                await f.write(json.dumps(self.data, indent=4))
            self.logger.debug(f"Данные dry mode асинхронно сохранены в {self.storage_file}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка асинхронного сохранения данных dry mode: {str(e)}")
            return False
    
    def save_data(self) -> bool:
        """
        Сохранение данных в файл.
        
        Returns:
            bool: Успешность сохранения
        """
        try:
            # Создаем директорию, если она не существует
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            with open(self.storage_file, 'w') as f:
                json.dump(self.data, f, indent=4)
            self.logger.debug(f"Данные dry mode сохранены в {self.storage_file}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных dry mode: {str(e)}")
            return False
    
    def get_balance(self) -> float:
        """
        Получение текущего баланса.
        
        Returns:
            float: Текущий баланс
        """
        return self.data["balance"]
    
    def update_balance(self, amount: float) -> float:
        """
        Обновление баланса.
        
        Args:
            amount: Сумма изменения баланса (положительная или отрицательная)
            
        Returns:
            float: Новый баланс
        """
        self.data["balance"] += amount
        self.logger.debug(f"Баланс обновлен: {self.data['balance']:.2f} ({amount:+.2f})")
        
        # Сохраняем данные после каждого обновления баланса
        self.save_data()
        
        return self.data["balance"]
    
    async def update_balance_async(self, amount: float) -> float:
        """
        Асинхронное обновление баланса.
        
        Args:
            amount: Сумма изменения баланса (положительная или отрицательная)
            
        Returns:
            float: Новый баланс
        """
        self.data["balance"] += amount
        self.logger.debug(f"Баланс обновлен: {self.data['balance']:.2f} ({amount:+.2f})")
        
        # Асинхронно сохраняем данные
        await self.save_data_async()
        
        return self.data["balance"]
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Добавление сделки в историю.
        
        Args:
            trade_data: Данные о сделке
        """
        self.data["trades"].append(trade_data)
        self.save_data()
    
    async def add_trade_async(self, trade_data: Dict[str, Any]) -> None:
        """
        Асинхронное добавление сделки в историю.
        
        Args:
            trade_data: Данные о сделке
        """
        self.data["trades"].append(trade_data)
        await self.save_data_async()
    
    def add_position(self, position_data: Dict[str, Any]) -> None:
        """
        Добавление открытой позиции.
        
        Args:
            position_data: Данные о позиции
        """
        self.data["open_positions"].append(position_data)
        self.save_data()
    
    async def add_position_async(self, position_data: Dict[str, Any]) -> None:
        """
        Асинхронное добавление открытой позиции.
        
        Args:
            position_data: Данные о позиции
        """
        self.data["open_positions"].append(position_data)
        await self.save_data_async()
    
    def remove_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Удаление позиции из списка открытых.
        
        Args:
            position_id: Идентификатор позиции
            
        Returns:
            Optional[Dict[str, Any]]: Данные удаленной позиции или None
        """
        for i, position in enumerate(self.data["open_positions"]):
            if position["id"] == position_id:
                removed = self.data["open_positions"].pop(i)
                self.save_data()
                return removed
        return None
    
    async def remove_position_async(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Асинхронное удаление позиции из списка открытых.
        
        Args:
            position_id: Идентификатор позиции
            
        Returns:
            Optional[Dict[str, Any]]: Данные удаленной позиции или None
        """
        for i, position in enumerate(self.data["open_positions"]):
            if position["id"] == position_id:
                removed = self.data["open_positions"].pop(i)
                await self.save_data_async()
                return removed
        return None
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Получение списка открытых позиций.
        
        Returns:
            List[Dict[str, Any]]: Список открытых позиций
        """
        return self.data["open_positions"]
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Получение истории сделок.
        
        Returns:
            List[Dict[str, Any]]: История сделок
        """
        return self.data["trades"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности.
        
        Returns:
            Dict[str, Any]: Статистика производительности
        """
        stats = {
            "initial_balance": self.data["initial_balance"],
            "current_balance": self.data["balance"],
            "profit_loss": self.data["balance"] - self.data["initial_balance"],
            "profit_loss_percent": ((self.data["balance"] / self.data["initial_balance"]) - 1) * 100,
            "total_trades": len(self.data["trades"]),
            "open_positions": len(self.data["open_positions"])
        }
        
        # Расчет дополнительной статистики, если есть сделки
        if stats["total_trades"] > 0:
            trades = self.data["trades"]
            
            # Прибыльные и убыточные сделки
            profitable_trades = [t for t in trades if t.get("profit", 0) > 0]
            losing_trades = [t for t in trades if t.get("profit", 0) <= 0]
            
            stats["profitable_trades"] = len(profitable_trades)
            stats["losing_trades"] = len(losing_trades)
            stats["win_rate"] = (len(profitable_trades) / len(trades)) * 100 if trades else 0
            
            # Средняя прибыль/убыток
            if profitable_trades:
                stats["avg_profit"] = sum(t.get("profit", 0) for t in profitable_trades) / len(profitable_trades)
            else:
                stats["avg_profit"] = 0
                
            if losing_trades:
                stats["avg_loss"] = sum(t.get("profit", 0) for t in losing_trades) / len(losing_trades)
            else:
                stats["avg_loss"] = 0
                
            # Максимальная просадка
            balances = [self.data["initial_balance"]]
            for trade in trades:
                balances.append(balances[-1] + trade.get("profit", 0))
            
            peak = self.data["initial_balance"]
            max_drawdown = 0
            
            for balance in balances:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            stats["max_drawdown_percent"] = max_drawdown
            
            # Дополнительные метрики для оценки эффективности
            
            # Коэффициент Шарпа
            returns = []
            for i in range(1, len(balances)):
                returns.append((balances[i] - balances[i-1]) / balances[i-1])
            
            if returns:
                mean_return = sum(returns) / len(returns)
                std_dev = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))
                
                if std_dev > 0:
                    stats["sharpe_ratio"] = mean_return / std_dev * math.sqrt(252)  # Годовой коэффициент Шарпа
                else:
                    stats["sharpe_ratio"] = 0
            else:
                stats["sharpe_ratio"] = 0
            
            # Profit Factor (отношение валовой прибыли к валовому убытку)
            total_profit = sum(t.get("profit", 0) for t in profitable_trades)
            total_loss = abs(sum(t.get("profit", 0) for t in losing_trades))
            
            if total_loss > 0:
                stats["profit_factor"] = total_profit / total_loss
            else:
                stats["profit_factor"] = float('inf') if total_profit > 0 else 0
        
        return stats


class AdvancedOrderTypes:
    """
    Расширенные типы ордеров для симуляции реальной торговли.
    """
    
    @staticmethod
    async def limit_order(
        market_simulator: MarketSimulator, 
        symbol: str, 
        side: str,
        price: float,
        size: float,
        time_in_force: str = 'GTC',
        post_only: bool = False,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Симуляция лимитного ордера.
        
        Args:
            market_simulator: Симулятор рынка
            symbol: Торговая пара
            side: Сторона ордера ('buy' или 'sell')
            price: Цена ордера
            size: Размер ордера
            time_in_force: Время жизни ордера ('GTC', 'IOC', 'FOK')
            post_only: Только добавление в стакан (не исполнять как маркет)
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        # Проверяем, может ли ордер быть исполнен сразу
        side_lower = side.lower()
        can_execute = False
        
        if side_lower == 'buy':
            # Для покупки - если цена выше или равна текущей цене продажи
            ask_price = market_simulator.order_book.get_best_ask()[0]
            can_execute = price >= ask_price
        else:
            # Для продажи - если цена ниже или равна текущей цене покупки
            bid_price = market_simulator.order_book.get_best_bid()[0]
            can_execute = price <= bid_price
        
        # Если ордер может быть исполнен как маркет и не post_only
        if can_execute and not post_only:
            # Исполняем как маркет ордер
            execution = market_simulator.execute_market_order(side_lower, size)
            
            return {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'price': price,
                'requested_size': size,
                'executed_price': execution['executed_price'],
                'executed_volume': execution['executed_volume'],
                'status': 'FILLED',
                'time_in_force': time_in_force,
                'post_only': post_only,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Для IOC и FOK, которые требуют немедленного исполнения
            if time_in_force == 'IOC' or time_in_force == 'FOK':
                return {
                    'symbol': symbol,
                    'side': side,
                    'type': 'LIMIT',
                    'price': price,
                    'requested_size': size,
                    'executed_volume': 0,
                    'status': 'EXPIRED',
                    'time_in_force': time_in_force,
                    'post_only': post_only,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Для остальных - ставим в стакан и ждем исполнения
            order_id = str(uuid.uuid4())
            
            # Функция для проверки исполнения ордера
            async def check_execution():
                start_time = time.time()
                
                while True:
                    # Проверяем, может ли ордер быть исполнен сейчас
                    current_price = market_simulator.get_current_price()
                    
                    if side_lower == 'buy':
                        # Для покупки - если цена упала до нашего уровня
                        is_executed = current_price <= price
                    else:
                        # Для продажи - если цена выросла до нашего уровня
                        is_executed = current_price >= price
                    
                    if is_executed:
                        # Исполняем ордер
                        execution = market_simulator.execute_market_order(side_lower, size)
                        
                        return {
                            'symbol': symbol,
                            'side': side,
                            'type': 'LIMIT',
                            'price': price,
                            'requested_size': size,
                            'executed_price': execution['executed_price'],
                            'executed_volume': execution['executed_volume'],
                            'status': 'FILLED',
                            'time_in_force': time_in_force,
                            'post_only': post_only,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Проверяем таймаут
                    if timeout is not None and time.time() - start_time > timeout:
                        return {
                            'symbol': symbol,
                            'side': side,
                            'type': 'LIMIT',
                            'price': price,
                            'requested_size': size,
                            'executed_volume': 0,
                            'status': 'EXPIRED',
                            'time_in_force': time_in_force,
                            'post_only': post_only,
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Ждем перед следующей проверкой
                    await asyncio.sleep(0.1)
            
            # Запускаем проверку исполнения асинхронно
            return await check_execution()
    
    @staticmethod
    async def stop_order(
        market_simulator: MarketSimulator, 
        symbol: str, 
        side: str,
        stop_price: float,
        size: float,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Симуляция стоп-ордера.
        
        Args:
            market_simulator: Симулятор рынка
            symbol: Торговая пара
            side: Сторона ордера ('buy' или 'sell')
            stop_price: Цена активации ордера
            size: Размер ордера
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        order_id = str(uuid.uuid4())
        side_lower = side.lower()
        
        # Функция для проверки активации стоп-ордера
        async def check_activation():
            start_time = time.time()
            
            while True:
                # Проверяем, должен ли стоп-ордер быть активирован
                current_price = market_simulator.get_current_price()
                
                if side_lower == 'buy':
                    # Для покупки - если цена выросла до стоп-цены
                    is_activated = current_price >= stop_price
                else:
                    # Для продажи - если цена упала до стоп-цены
                    is_activated = current_price <= stop_price
                
                if is_activated:
                    # Исполняем как маркет ордер
                    execution = market_simulator.execute_market_order(side_lower, size)
                    
                    return {
                        'symbol': symbol,
                        'side': side,
                        'type': 'STOP',
                        'stop_price': stop_price,
                        'requested_size': size,
                        'executed_price': execution['executed_price'],
                        'executed_volume': execution['executed_volume'],
                        'status': 'FILLED',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Проверяем таймаут
                if timeout is not None and time.time() - start_time > timeout:
                    return {
                        'symbol': symbol,
                        'side': side,
                        'type': 'STOP',
                        'stop_price': stop_price,
                        'requested_size': size,
                        'executed_volume': 0,
                        'status': 'EXPIRED',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Ждем перед следующей проверкой
                await asyncio.sleep(0.1)
        
        # Запускаем проверку активации асинхронно
        return await check_activation()
    
    @staticmethod
    async def trailing_stop(
        market_simulator: MarketSimulator, 
        symbol: str, 
        side: str,
        activation_price: float,
        trailing_offset: float,
        size: float,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Симуляция трейлинг-стоп ордера.
        
        Args:
            market_simulator: Симулятор рынка
            symbol: Торговая пара
            side: Сторона ордера ('buy' или 'sell')
            activation_price: Цена активации трейлинг-стопа
            trailing_offset: Отступ от максимальной/минимальной цены
            size: Размер ордера
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        order_id = str(uuid.uuid4())
        side_lower = side.lower()
        
        # Функция для реализации трейлинг-стопа
        async def track_price():
            start_time = time.time()
            
            # Флаг активации и пиковые значения
            is_activated = False
            peak_price = None
            stop_price = None
            
            while True:
                # Получаем текущую цену
                current_price = market_simulator.get_current_price()
                
                # Проверяем активацию ордера
                if not is_activated:
                    if side_lower == 'buy':
                        # Для покупки - если цена упала до активационной
                        is_activated = current_price <= activation_price
                    else:
                        # Для продажи - если цена выросла до активационной
                        is_activated = current_price >= activation_price
                    
                    if is_activated:
                        # Инициализируем пик при активации
                        peak_price = current_price
                        
                        # Начальная стоп-цена
                        if side_lower == 'buy':
                            stop_price = peak_price - trailing_offset
                        else:
                            stop_price = peak_price + trailing_offset
                else:
                    # Обновление пика и стоп-цены
                    if side_lower == 'buy':
                        # Для покупки - фиксируем минимальную цену
                        if current_price < peak_price:
                            peak_price = current_price
                            stop_price = peak_price - trailing_offset
                        
                        # Проверяем, не пора ли исполнить ордер
                        if current_price >= stop_price:
                            # Исполняем как маркет ордер
                            execution = market_simulator.execute_market_order(side_lower, size)
                            
                            return {
                                'symbol': symbol,
                                'side': side,
                                'type': 'TRAILING_STOP',
                                'activation_price': activation_price,
                                'trailing_offset': trailing_offset,
                                'stop_price': stop_price,
                                'requested_size': size,
                                'executed_price': execution['executed_price'],
                                'executed_volume': execution['executed_volume'],
                                'status': 'FILLED',
                                'timestamp': datetime.now().isoformat()
                            }
                    else:
                        # Для продажи - фиксируем максимальную цену
                        if current_price > peak_price:
                            peak_price = current_price
                            stop_price = peak_price - trailing_offset
                        
                        # Проверяем, не пора ли исполнить ордер
                        if current_price <= stop_price:
                            # Исполняем как маркет ордер
                            execution = market_simulator.execute_market_order(side_lower, size)
                            
                            return {
                                'symbol': symbol,
                                'side': side,
                                'type': 'TRAILING_STOP',
                                'activation_price': activation_price,
                                'trailing_offset': trailing_offset,
                                'stop_price': stop_price,
                                'requested_size': size,
                                'executed_price': execution['executed_price'],
                                'executed_volume': execution['executed_volume'],
                                'status': 'FILLED',
                                'timestamp': datetime.now().isoformat()
                            }
                
                # Проверяем таймаут
                if timeout is not None and time.time() - start_time > timeout:
                    return {
                        'symbol': symbol,
                        'side': side,
                        'type': 'TRAILING_STOP',
                        'activation_price': activation_price,
                        'trailing_offset': trailing_offset,
                        'stop_price': stop_price,
                        'requested_size': size,
                        'executed_volume': 0,
                        'status': 'EXPIRED',
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Ждем перед следующей проверкой
                await asyncio.sleep(0.1)
        
        # Запускаем слежение за ценой асинхронно
        return await track_price()


class TradingEnvironment:
    """
    Среда для обучения с подкреплением, совместимая с API Gym.
    """
    
    def __init__(self, 
                symbol: str, 
                initial_balance: float = 10000.0,
                leverage: int = 1,
                fee_rate: float = 0.001,  # 0.1%
                window_size: int = 100,  # Размер окна наблюдения
                reward_scaling: float = 1.0):  # Масштабирование вознаграждения
        """
        Инициализация среды для обучения с подкреплением.
        
        Args:
            symbol: Торговая пара
            initial_balance: Начальный баланс
            leverage: Кредитное плечо
            fee_rate: Комиссия за сделку
            window_size: Размер окна наблюдения (количество предыдущих шагов)
            reward_scaling: Масштабирование вознаграждения
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Создаем симулятор рынка
        initial_price = 100.0  # Начальная цена
        self.market_simulator = MarketSimulator(symbol, initial_price)
        
        # Хранение исторических данных
        self.price_history = []
        self.action_history = []
        self.reward_history = []
        self.balance_history = [initial_balance]
        self.position_history = []
        
        # Текущая позиция
        self.in_position = False
        self.position_type = None  # 'long' или 'short'
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Счетчик шагов
        self.steps = 0
        self.max_steps = 10000  # Максимальное количество шагов в эпизоде
        
        # Логгер
        self.logger = logging.getLogger(f"TradingEnvironment-{self.symbol}")
        self.logger.info(f"Инициализирована среда для {self.symbol} с начальным балансом {initial_balance}")
    
    def reset(self) -> np.ndarray:
        """
        Сброс среды в начальное состояние.
        
        Returns:
            np.ndarray: Наблюдение
        """
        # Сбрасываем счетчик шагов
        self.steps = 0
        
        # Сбрасываем баланс
        self.balance = self.initial_balance
        self.balance_history = [self.initial_balance]
        
        # Сбрасываем позицию
        self.in_position = False
        self.position_type = None
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Сбрасываем историю
        self.price_history = []
        self.action_history = []
        self.reward_history = []
        self.position_history = []
        
        # Сбрасываем симулятор рынка (инициализация с новой ценой)
        current_price = self.market_simulator.get_current_price()
        self.market_simulator = MarketSimulator(self.symbol, current_price)
        
        # Заполняем историю цен начальными значениями
        for _ in range(self.window_size):
            price = self.market_simulator.get_current_price()
            self.price_history.append(price)
        
        # Возвращаем наблюдение
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Выполнение шага в среде.
        
        Args:
            action: Действие (0 - держать, 1 - купить, 2 - продать)
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: (наблюдение, вознаграждение, флаг завершения, информация)
        """
        # Увеличиваем счетчик шагов
        self.steps += 1
        
        # Получаем текущую цену
        current_price = self.market_simulator.get_current_price()
        self.price_history.append(current_price)
        
        # Вычисляем вознаграждение и обновляем состояние среды
        reward = 0.0
        info = {}
        
        # Обрабатываем действие
        if action == 0:  # Держать
            self.action_history.append(0)
            
            # Если есть открытая позиция, вознаграждение зависит от изменения цены
            if self.in_position:
                if self.position_type == 'long':
                    # Для длинной позиции вознаграждение положительное при росте цены
                    reward = (current_price - self.price_history[-2]) / self.price_history[-2] * self.leverage
                else:  # short
                    # Для короткой позиции вознаграждение положительное при падении цены
                    reward = (self.price_history[-2] - current_price) / self.price_history[-2] * self.leverage
        
        elif action == 1:  # Купить
            self.action_history.append(1)
            
            if not self.in_position:
                # Открываем длинную позицию
                self._open_position('long', current_price)
                
                # Вознаграждение за открытие позиции - отрицательное из-за комиссии
                reward = -self.fee_rate
                
                # Информация о позиции
                info['position_opened'] = True
                info['position_type'] = 'long'
                info['entry_price'] = current_price
            elif self.position_type == 'short':
                # Закрываем короткую позицию
                profit_percent = (self.entry_price - current_price) / self.entry_price * self.leverage
                profit = self.position_size * profit_percent
                
                # Вознаграждение за закрытие позиции - прибыль минус комиссия
                reward = profit_percent - self.fee_rate
                
                # Обновляем баланс
                self.balance += profit - (self.position_size * self.fee_rate)
                self.balance_history.append(self.balance)
                
                # Очищаем позицию
                self.in_position = False
                self.position_type = None
                self.position_size = 0.0
                self.entry_price = 0.0
                
                # Информация о позиции
                info['position_closed'] = True
                info['position_type'] = 'short'
                info['exit_price'] = current_price
                info['profit'] = profit
                info['profit_percent'] = profit_percent * 100
            else:
                # Если уже есть длинная позиция, ничего не делаем (или можно штрафовать)
                reward = -0.01  # Небольшой штраф за бесполезное действие
        
        elif action == 2:  # Продать
            self.action_history.append(2)
            
            if not self.in_position:
                # Открываем короткую позицию
                self._open_position('short', current_price)
                
                # Вознаграждение за открытие позиции - отрицательное из-за комиссии
                reward = -self.fee_rate
                
                # Информация о позиции
                info['position_opened'] = True
                info['position_type'] = 'short'
                info['entry_price'] = current_price
            elif self.position_type == 'long':
                # Закрываем длинную позицию
                profit_percent = (current_price - self.entry_price) / self.entry_price * self.leverage
                profit = self.position_size * profit_percent
                
                # Вознаграждение за закрытие позиции - прибыль минус комиссия
                reward = profit_percent - self.fee_rate
                
                # Обновляем баланс
                self.balance += profit - (self.position_size * self.fee_rate)
                self.balance_history.append(self.balance)
                
                # Очищаем позицию
                self.in_position = False
                self.position_type = None
                self.position_size = 0.0
                self.entry_price = 0.0
                
                # Информация о позиции
                info['position_closed'] = True
                info['position_type'] = 'long'
                info['exit_price'] = current_price
                info['profit'] = profit
                info['profit_percent'] = profit_percent * 100
            else:
                # Если уже есть короткая позиция, ничего не делаем (или можно штрафовать)
                reward = -0.01  # Небольшой штраф за бесполезное действие
        
        # Масштабируем вознаграждение
        reward *= self.reward_scaling
        
        # Сохраняем историю вознаграждений
        self.reward_history.append(reward)
        
        # Проверка условий завершения эпизода
        done = self.steps >= self.max_steps or self.balance <= 0
        
        # Если эпизод завершается, закрываем все позиции
        if done and self.in_position:
            if self.position_type == 'long':
                profit_percent = (current_price - self.entry_price) / self.entry_price * self.leverage
            else:  # short
                profit_percent = (self.entry_price - current_price) / self.entry_price * self.leverage
                
            profit = self.position_size * profit_percent
            
            # Обновляем баланс
            self.balance += profit - (self.position_size * self.fee_rate)
            self.balance_history.append(self.balance)
            
            # Информация о позиции
            info['position_closed_at_end'] = True
            info['position_type'] = self.position_type
            info['exit_price'] = current_price
            info['profit'] = profit
            info['profit_percent'] = profit_percent * 100
            
            # Очищаем позицию
            self.in_position = False
            self.position_type = None
            self.position_size = 0.0
            self.entry_price = 0.0
        
        # Дополнительная информация
        info['balance'] = self.balance
        info['price'] = current_price
        info['step'] = self.steps
        
        # Возвращаем результат шага
        return self._get_observation(), reward, done, info
    
    def _open_position(self, position_type: str, price: float) -> None:
        """
        Открыть позицию.
        
        Args:
            position_type: Тип позиции ('long' или 'short')
            price: Цена открытия
        """
        # Определяем размер позиции (упрощенно - 90% от баланса)
        position_size = self.balance * 0.9
        
        # Учитываем комиссию
        fee = position_size * self.fee_rate
        position_size -= fee
        
        # Обновляем состояние позиции
        self.in_position = True
        self.position_type = position_type
        self.position_size = position_size
        self.entry_price = price
        
        # Сохраняем информацию о позиции
        self.position_history.append({
            'type': position_type,
            'entry_price': price,
            'size': position_size,
            'entry_time': self.steps
        })
        
        self.logger.debug(f"Открыта позиция {position_type} по цене {price}")
    
    def _get_observation(self) -> np.ndarray:
        """
        Получить наблюдение для агента.
        
        Returns:
            np.ndarray: Наблюдение
        """
        # Нормализуем цены относительно последней цены
        price_history = np.array(self.price_history[-self.window_size:])
        last_price = price_history[-1]
        normalized_prices = price_history / last_price - 1.0
        
        # Информация о позиции в виде one-hot кодирования
        position_info = np.zeros(3)  # [нет позиции, длинная, короткая]
        
        if not self.in_position:
            position_info[0] = 1.0
        elif self.position_type == 'long':
            position_info[1] = 1.0
            
            # Добавляем информацию о P&L позиции
            current_price = self.price_history[-1]
            price_diff = (current_price - self.entry_price) / self.entry_price
            position_info = np.append(position_info, price_diff * self.leverage)
        else:  # short
            position_info[2] = 1.0
            
            # Добавляем информацию о P&L позиции
            current_price = self.price_history[-1]
            price_diff = (self.entry_price - current_price) / self.entry_price
            position_info = np.append(position_info, price_diff * self.leverage)
        
        # Добавление баланса (нормализованного относительно начального)
        normalized_balance = self.balance / self.initial_balance - 1.0
        
        # Объединяем все в одно наблюдение
        observation = np.concatenate([normalized_prices, position_info, [normalized_balance]])
        
        return observation


class DryTrader:
    """
    Трейдер для симуляции торговли без реальных сделок.
    Использует виртуальный баланс и историю сделок.
    """
    
    def __init__(self, 
                 symbol: str, 
                 exchange_client,
                 strategy, 
                 notification_service=None,
                 risk_controller=None,
                 initial_balance: float = 1000.0,
                 leverage: int = 1,
                 storage_file: Optional[str] = None,
                 price_update_interval: float = 5.0,
                 visualizer=None,
                 fee_rate: float = 0.001,  # 0.1% по умолчанию
                 slippage_model: str = "advanced"):  # "basic" или "advanced"
        """
        Инициализация трейдера для симуляции.
        
        Args:
            symbol: Торговая пара (например, "BTCUSDT")
            exchange_client: Клиент биржи для получения данных
            strategy: Торговая стратегия
            notification_service: Сервис уведомлений (опционально)
            risk_controller: Контроллер рисков (опционально)
            initial_balance: Начальный баланс
            leverage: Кредитное плечо
            storage_file: Файл для хранения данных симуляции
            price_update_interval: Интервал обновления цен в секундах
            visualizer: Визуализатор для отображения данных
            fee_rate: Комиссия за сделку
            slippage_model: Модель проскальзывания ('basic' или 'advanced')
        """
        # Core parameters from TraderBase
        self.symbol = symbol
        self.exchange_client = exchange_client
        self.strategy = strategy
        self.notification_service = notification_service
        self.risk_controller = risk_controller
        self.initial_balance = float(initial_balance)
        self.leverage = leverage
        
        # Менеджер данных симуляции
        self.dry_data = DryModeData(symbol, storage_file)
        
        # Параметры симуляции
        self.price_update_interval = price_update_interval
        self.visualizer = visualizer
        self.fee_rate = fee_rate
        self.slippage_model = slippage_model
        
        # Создаем симулятор рынка
        self.market_simulator = None  # Будет инициализирован позже
        
        # Текущая цена и задача обновления цены
        self.current_price = None
        self.price_update_task = None
        
        # Позиция
        self.in_position = False
        self.current_position = None
        
        # Флаг работы
        self.is_running = False
        
        # Логгер
        self.logger = logging.getLogger(f"DryTrader-{self.symbol}")
        
        # Менеджер визуализации
        self.visualization_manager = None
        
        # RL-среда для обучения моделей
        self.trading_environment = None
    
    async def initialize(self) -> bool:
        """
        Инициализация трейдера.
        
        Returns:
            bool: Успешность инициализации
        """
        self.logger.info(f"Инициализация dry трейдера для {self.symbol}")
        
        # Инициализация данных симуляции
        self.dry_data.initialize(self.initial_balance)
        
        # Получение текущей цены
        try:
            initial_price = await self.exchange_client.get_current_price(self.symbol)
            self.current_price = initial_price
            self.logger.info(f"Текущая цена {self.symbol}: {self.current_price}")
            
            # Инициализация симулятора рынка
            self.market_simulator = MarketSimulator(
                symbol=self.symbol,
                initial_price=initial_price,
                volatility=0.001,  # Базовая волатильность
                trend=0.0,  # Нейтральный тренд
                market_condition=MarketCondition.NORMAL
            )
            
            # Инициализация RL-среды
            self.trading_environment = TradingEnvironment(
                symbol=self.symbol,
                initial_balance=self.initial_balance,
                leverage=self.leverage,
                fee_rate=self.fee_rate
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка получения текущей цены: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
        
        # Проверка наличия открытых позиций
        open_positions = self.dry_data.get_open_positions()
        if open_positions:
            self.logger.info(f"Найдено {len(open_positions)} открытых позиций")
            self.in_position = True
            self.current_position = open_positions[0]  # Берем первую позицию
        
        # Инициализация визуализатора, если он предоставлен
        if self.visualizer:
            try:
                await self.visualizer.initialize(self.symbol)
                self.logger.info("Визуализатор инициализирован")
            except Exception as e:
                self.logger.error(f"Ошибка инициализации визуализатора: {str(e)}")
                self.logger.debug(traceback.format_exc())
        
        return True
    
    async def _update_price_loop(self) -> None:
        """
        Фоновая задача для периодического обновления цены.
        """
        self.logger.info(f"Запущен цикл обновления цены для {self.symbol}")
        
        while self.is_running:
            try:
                # Получение текущей цены с биржи или симулятора
                if random.random() < 0.1:  # 10% вероятность получения реальной цены для коррекции
                    try:
                        real_price = await self.exchange_client.get_current_price(self.symbol)
                        # Корректируем цену симулятора, чтобы не уходить далеко от реальности
                        self.market_simulator.current_price = real_price
                    except Exception:
                        # Если не удалось получить реальную цену, используем генерацию
                        pass
                
                # Используем симулятор рынка для генерации следующей цены
                new_price = self.market_simulator.get_current_price()
                self.current_price = new_price
                
                # Обновление цены в визуализаторе, если он есть
                if self.visualizer:
                    await self.visualizer.update_price(self.symbol, new_price)
                
                # Проверка условий для открытых позиций
                await self._check_positions()
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле обновления цены: {str(e)}")
                self.logger.debug(traceback.format_exc())
            
            # Пауза перед следующим обновлением
            await asyncio.sleep(self.price_update_interval)
    
    async def _check_positions(self) -> None:
        """
        Проверка условий для открытых позиций (стоп-лосс, тейк-профит).
        """
        if not self.in_position or not self.current_position:
            return
        
        position = self.current_position
        current_price = self.current_price
        
        # Проверка условий только если есть текущая цена
        if current_price is None:
            return
        
        # Проверка стоп-лосса и тейк-профита
        if position["direction"] == "LONG":
            # Для длинной позиции
            if position.get("stop_loss") and current_price <= position["stop_loss"]:
                self.logger.info(f"Сработал стоп-лосс для LONG позиции: {position['stop_loss']}")
                await self.exit_position(position["id"], current_price)
                
            elif position.get("take_profit") and current_price >= position["take_profit"]:
                self.logger.info(f"Сработал тейк-профит для LONG позиции: {position['take_profit']}")
                await self.exit_position(position["id"], current_price)
                
        elif position["direction"] == "SHORT":
            # Для короткой позиции
            if position.get("stop_loss") and current_price >= position["stop_loss"]:
                self.logger.info(f"Сработал стоп-лосс для SHORT позиции: {position['stop_loss']}")
                await self.exit_position(position["id"], current_price)
                
            elif position.get("take_profit") and current_price <= position["take_profit"]:
                self.logger.info(f"Сработал тейк-профит для SHORT позиции: {position['take_profit']}")
                await self.exit_position(position["id"], current_price)
    
    async def start(self) -> bool:
        """
        Запуск трейдера.
        
        Returns:
            bool: Успешность запуска
        """
        self.logger.info(f"Запуск dry трейдера для {self.symbol}")
        
        if self.is_running:
            self.logger.warning("Трейдер уже запущен")
            return True
        
        # Установка флага работы
        self.is_running = True
        
        # Запуск задачи обновления цены
        self.price_update_task = asyncio.create_task(self._update_price_loop())
        
        # Уведомление о запуске, если есть сервис уведомлений
        if self.notification_service:
            await self.notification_service.send_notification(
                f"Dry трейдер для {self.symbol} запущен. Начальный баланс: {self.dry_data.get_balance()}"
            )
        
        return True
    
    async def stop(self) -> bool:
        """
        Остановка трейдера.
        
        Returns:
            bool: Успешность остановки
        """
        self.logger.info(f"Остановка dry трейдера для {self.symbol}")
        
        if not self.is_running:
            self.logger.warning("Трейдер уже остановлен")
            return True
        
        # Сброс флага работы
        self.is_running = False
        
        # Отмена задачи обновления цены, если она существует
        if self.price_update_task:
            self.price_update_task.cancel()
            try:
                await self.price_update_task
            except asyncio.CancelledError:
                pass
            self.price_update_task = None
        
        # Уведомление об остановке, если есть сервис уведомлений
        if self.notification_service:
            stats = self.dry_data.get_performance_stats()
            await self.notification_service.send_notification(
                f"Dry трейдер для {self.symbol} остановлен. "
                f"Итоговый баланс: {stats['current_balance']:.2f} "
                f"(P&L: {stats['profit_loss_percent']:.2f}%)"
            )
        
        return True
    
    async def enter_position(self, 
                            direction: str, 
                            size: float, 
                            price: Optional[float] = None,
                            stop_loss: Optional[float] = None,
                            take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Вход в позицию.
        
        Args:
            direction: Направление ("LONG" или "SHORT")
            size: Размер позиции в базовой валюте
            price: Цена входа (опционально, по умолчанию текущая цена)
            stop_loss: Уровень стоп-лосса (опционально)
            take_profit: Уровень тейк-профита (опционально)
            
        Returns:
            Dict[str, Any]: Информация о созданной позиции
        """
        if self.in_position:
            self.logger.warning("Невозможно открыть позицию: уже есть открытая позиция")
            raise ValueError("Уже есть открытая позиция")
        
        # Использование текущей цены, если цена не указана
        entry_price = price if price is not None else self.current_price
        if entry_price is None:
            self.logger.error("Невозможно открыть позицию: неизвестна текущая цена")
            raise ValueError("Неизвестна текущая цена")
        
        # Проверка баланса
        balance = self.dry_data.get_balance()
        position_value = size * entry_price
        
        # Расчет комиссии
        commission = position_value * self.fee_rate
        
        # Расчет проскальзывания
        slippage = 0.0
        if self.slippage_model == "advanced" and self.market_simulator:
            # Используем симулятор рынка для реалистичного исполнения
            side = "buy" if direction == "LONG" else "sell"
            execution = self.market_simulator.execute_market_order(side, size)
            entry_price = execution['executed_price']
            size = execution['executed_volume']  # Может быть меньше запрошенного
            slippage = execution['slippage']
        elif self.slippage_model == "basic":
            # Простая модель проскальзывания
            slippage_percent = random.uniform(0, 0.1)  # 0-0.1% проскальзывания
            if direction == "LONG":
                # Для покупки цена входа увеличивается
                entry_price *= (1 + slippage_percent / 100)
            else:
                # Для продажи цена входа уменьшается
                entry_price *= (1 - slippage_percent / 100)
            slippage = slippage_percent
        
        # Учитываем комиссию при проверке баланса
        total_cost = position_value + commission
        
        if total_cost > balance:
            self.logger.warning(f"Недостаточно средств: {balance} < {total_cost} (включая комиссию {commission})")
            raise ValueError(f"Недостаточно средств: {balance} < {total_cost} (включая комиссию {commission})")
        
        # Вычитаем комиссию из баланса
        await self.dry_data.update_balance_async(-commission)
        
        # Создание идентификатора позиции
        position_id = f"{self.symbol}-{direction}-{int(time.time())}-{random.randint(1000, 9999)}"
        
        # Создание данных о позиции
        position = {
            "id": position_id,
            "symbol": self.symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "entry_time": datetime.now().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "commission": commission,
            "slippage": slippage,
            "status": "OPEN"
        }
        
        # Добавление позиции в список открытых
        await self.dry_data.add_position_async(position)
        
        # Обновление состояния трейдера
        self.in_position = True
        self.current_position = position
        
        # Логирование и уведомление
        self.logger.info(
            f"Открыта {direction} позиция: {size} {self.symbol} по цене {entry_price} "
            f"(комиссия: {commission:.2f}, проскальзывание: {slippage:.4f}%)"
        )
        
        if self.notification_service:
            await self.notification_service.send_notification(
                f"[DRY] Открыта {direction} позиция: {size} {self.symbol} по цене {entry_price} "
                f"(комиссия: {commission:.2f}, проскальзывание: {slippage:.4f}%)"
            )
        
        # Обновление визуализатора, если он есть
        if self.visualizer:
            await self.visualizer.add_trade_marker(
                self.symbol, 
                entry_price, 
                "BUY" if direction == "LONG" else "SELL"
            )
        
        return position
    
    async def exit_position(self, 
                           position_id: str, 
                           price: Optional[float] = None) -> Dict[str, Any]:
        """
        Закрытие позиции.
        
        Args:
            position_id: Идентификатор позиции
            price: Цена закрытия (если None, используется текущая рыночная цена)
            
        Returns:
            Информация о закрытой позиции
        """
        # Проверка наличия позиции
        position = None
        for pos in self.dry_data.get_open_positions():
            if pos["id"] == position_id:
                position = pos
                break
                
        if not position:
            self.logger.warning(f"Позиция с ID {position_id} не найдена")
            return {
                "success": False,
                "error": f"Позиция с ID {position_id} не найдена"
            }
            
        # Получение текущей цены, если не указана
        exit_price = price if price is not None else self.market_simulator.get_current_price()
        
        # Получение данных позиции
        entry_price = position["entry_price"]
        size = position["size"]
        direction = position["direction"]
        
        # Расчет проскальзывания
        slippage = self._calculate_slippage(size, direction == "SHORT")
        
        # Применение проскальзывания к цене
        if direction == "LONG":
            exit_price = exit_price * (1 - slippage / 100)
        else:  # SHORT
            exit_price = exit_price * (1 + slippage / 100)
            
        # Расчет комиссии
        exit_commission = size * exit_price * self.fee_rate
        
        # Расчет прибыли/убытка
        if direction == "LONG":
            profit = (exit_price - entry_price) * size - position["commission"] - exit_commission
            profit_percent = (exit_price / entry_price - 1) * 100 * self.leverage
        else:  # SHORT
            profit = (entry_price - exit_price) * size - position["commission"] - exit_commission
            profit_percent = (1 - exit_price / entry_price) * 100 * self.leverage
            
        # Обновление баланса
        new_balance = await self.dry_data.update_balance_async(profit)
        
        # Создание записи о сделке
        trade = {
            "id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "symbol": self.symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "profit": profit,
            "profit_percent": profit_percent,
            "commission": position["commission"] + exit_commission,
            "slippage": slippage,
            "position_id": position_id,
            "balance_after": new_balance
        }

        # Добавление сделки в историю и сохранение данных
        await self.dry_data.add_trade_async(trade)

        # Удаление позиции из списка открытых
        await self.dry_data.remove_position_async(position_id)

        # Обновление состояния трейдера
        self.in_position = False
        self.current_position = None

        # Определение типа исполнения (market, take_profit, stop_loss)
        execution_type = "market"
        if position.get("take_profit") and (
            (direction == "LONG" and exit_price >= position["take_profit"]) or
            (direction == "SHORT" and exit_price <= position["take_profit"])
        ):
            execution_type = "take_profit"
        elif position.get("stop_loss") and (
            (direction == "LONG" and exit_price <= position["stop_loss"]) or
            (direction == "SHORT" and exit_price >= position["stop_loss"])
        ):
            execution_type = "stop_loss"

        # Обновление типа исполнения в сделке
        trade["execution_type"] = execution_type

        # Логирование и уведомление
        profit_str = f"{profit:.2f} ({profit_percent:.2f}%)"
        execution_type_str = {
            "market": "по рынку", 
            "take_profit": "по тейк-профиту",
            "stop_loss": "по стоп-лоссу"
        }.get(execution_type, "по рынку")

        self.logger.info(
            f"Закрыта {direction} позиция {execution_type_str}: {size} {self.symbol} по цене {exit_price}. "
            f"P&L: {profit_str} (комиссия: {exit_commission:.2f}, проскальзывание: {slippage:.4f}%)"
        )

        if self.notification_service:
            await self.notification_service.send_notification(
                f"[DRY] Закрыта {direction} позиция {execution_type_str}: {size} {self.symbol} по цене {exit_price}. "
                f"P&L: {profit_str} (комиссия: {exit_commission:.2f}, проскальзывание: {slippage:.4f}%)"
            )

        # Обновление визуализатора, если он есть
        if self.visualizer:
            await self.visualizer.add_trade_marker(
                self.symbol, 
                exit_price, 
                "SELL" if direction == "LONG" else "BUY",
                profit > 0
            )

        return trade
    
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
        # Поиск позиции
        positions = self.dry_data.get_open_positions()
        for i, position in enumerate(positions):
            if position["id"] == position_id:
                # Обновление параметров
                if stop_loss is not None:
                    position["stop_loss"] = stop_loss
                if take_profit is not None:
                    position["take_profit"] = take_profit
                
                # Обновление текущей позиции, если это она
                if self.current_position and self.current_position["id"] == position_id:
                    self.current_position = position
                
                # Сохранение изменений
                await self.dry_data.save_data_async()
                
                # Логирование
                self.logger.info(
                    f"Обновлена позиция {position_id}: "
                    f"SL={stop_loss if stop_loss is not None else 'без изменений'}, "
                    f"TP={take_profit if take_profit is not None else 'без изменений'}"
                )
                
                return True
        
        self.logger.warning(f"Позиция с ID {position_id} не найдена")
        return False

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Получение списка открытых позиций.
        
        Returns:
            List[Dict[str, Any]]: Список открытых позиций
        """
        positions = self.dry_data.get_open_positions()
        
        # Обновляем текущую цену и P&L для каждой позиции
        for position in positions:
            # Обновление текущей цены
            position["current_price"] = self.current_price
            
            # Расчет нереализованного P&L
            if self.current_price is not None:
                entry_price = position["entry_price"]
                size = position["size"]
                direction = position["direction"]
                
                if direction == "LONG":
                    unrealized_pnl = (self.current_price - entry_price) * size
                    unrealized_pnl_percent = (self.current_price - entry_price) / entry_price * 100 * self.leverage
                else:  # SHORT
                    unrealized_pnl = (entry_price - self.current_price) * size
                    unrealized_pnl_percent = (entry_price - self.current_price) / entry_price * 100 * self.leverage
                
                position["unrealized_pnl"] = unrealized_pnl
                position["unrealized_pnl_percent"] = unrealized_pnl_percent
        
        return positions

    async def get_balance(self) -> float:
        """
        Получение текущего баланса.
        
        Returns:
            float: Текущий баланс
        """
        return self.dry_data.get_balance()

    async def get_current_price(self) -> float:
        """
        Получение текущей цены торговой пары.
        
        Returns:
            float: Текущая цена
        """
        return self.current_price

    async def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории сделок.
        
        Returns:
            List[Dict[str, Any]]: История сделок
        """
        return self.dry_data.get_trades()

    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Получение статистики производительности.
        
        Returns:
            Dict[str, Any]: Статистика производительности
        """
        return self.dry_data.get_performance_stats()

    async def execute_market_order(self, side: str, size: float, use_rl_model: bool = False) -> Dict[str, Any]:
        """
        Исполнение рыночного ордера.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            use_rl_model: Использовать RL-модель для принятия решений
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        # Преобразование стороны в верхний регистр для совместимости
        direction = "LONG" if side.lower() == "buy" else "SHORT"
        
        if use_rl_model and self.trading_environment:
            # Если используем RL-модель, имитируем шаг в среде
            action = 1 if side.lower() == "buy" else 2  # 1 - buy, 2 - sell
            observation, reward, done, info = self.trading_environment.step(action)
            
            # Используем информацию из RL-среды
            if 'position_opened' in info:
                # Создаем позицию на основе данных из RL-среды
                return await self.enter_position(
                    direction=direction,
                    size=size,
                    price=info.get('entry_price', self.current_price)
                )
            elif 'position_closed' in info:
                # Если у нас есть открытая позиция, закрываем ее
                if self.in_position and self.current_position:
                    return await self.exit_position(
                        position_id=self.current_position["id"],
                        price=info.get('exit_price', self.current_price)
                    )
        else:
            # Стандартное исполнение ордера
            if side.lower() == "buy":
                # Если это покупка
                if not self.in_position:
                    # Открываем новую длинную позицию
                    return await self.enter_position(
                        direction="LONG",
                        size=size
                    )
                elif self.in_position and self.current_position["direction"] == "SHORT":
                    # Закрываем короткую позицию
                    return await self.exit_position(
                        position_id=self.current_position["id"]
                    )
            else:
                # Если это продажа
                if not self.in_position:
                    # Открываем новую короткую позицию
                    return await self.enter_position(
                        direction="SHORT",
                        size=size
                    )
                elif self.in_position and self.current_position["direction"] == "LONG":
                    # Закрываем длинную позицию
                    return await self.exit_position(
                        position_id=self.current_position["id"]
                    )
        
        # Если ничего не сделали
        return {
            "success": False,
            "error": "Невозможно выполнить ордер в текущих условиях"
        }

    async def execute_limit_order(self, 
                                 side: str, 
                                 size: float, 
                                 price: float, 
                                 time_in_force: str = 'GTC', 
                                 post_only: bool = False,
                                 timeout: float = None) -> Dict[str, Any]:
        """
        Исполнение лимитного ордера.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            price: Цена ордера
            time_in_force: Время жизни ордера ('GTC', 'IOC', 'FOK')
            post_only: Только добавление в стакан (не исполнять как маркет)
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        if self.market_simulator:
            # Используем симулятор рынка для лимитного ордера
            return await AdvancedOrderTypes.limit_order(
                self.market_simulator,
                self.symbol,
                side,
                price,
                size,
                time_in_force,
                post_only,
                timeout
            )
        else:
            # Упрощенное исполнение лимитного ордера
            direction = "LONG" if side.lower() == "buy" else "SHORT"
            
            # Проверяем, может ли ордер быть исполнен как маркет
            current_price = self.current_price
            
            if (side.lower() == "buy" and current_price <= price) or \
               (side.lower() == "sell" and current_price >= price):
                # Исполняем как маркет ордер
                return await self.execute_market_order(side, size)
            else:
                # Возвращаем ошибку (упрощенно)
                return {
                    "success": False,
                    "error": "Лимитный ордер не может быть исполнен сразу, а ожидание не реализовано"
                }

    async def execute_stop_order(self, 
                                side: str, 
                                size: float, 
                                stop_price: float, 
                                timeout: float = None) -> Dict[str, Any]:
        """
        Исполнение стоп-ордера.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            stop_price: Цена активации ордера
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        if self.market_simulator:
            # Используем симулятор рынка для стоп-ордера
            return await AdvancedOrderTypes.stop_order(
                self.market_simulator,
                self.symbol,
                side,
                stop_price,
                size,
                timeout
            )
        else:
            # Упрощенное исполнение стоп-ордера
            direction = "LONG" if side.lower() == "buy" else "SHORT"
            
            # Проверяем, активировался ли стоп-ордер
            current_price = self.current_price
            
            if (side.lower() == "buy" and current_price >= stop_price) or \
               (side.lower() == "sell" and current_price <= stop_price):
                # Стоп-ордер активирован, исполняем как маркет ордер
                return await self.execute_market_order(side, size)
            else:
                # Возвращаем ошибку (упрощенно)
                return {
                    "success": False,
                    "error": "Стоп-ордер не активирован, а ожидание не реализовано"
                }

    async def execute_trailing_stop(self, 
                                   side: str, 
                                   size: float, 
                                   activation_price: float,
                                   trailing_offset: float,
                                   timeout: float = None) -> Dict[str, Any]:
        """
        Исполнение трейлинг-стоп ордера.
        
        Args:
            side: Сторона ордера ('buy' или 'sell')
            size: Размер ордера
            activation_price: Цена активации трейлинг-стопа
            trailing_offset: Отступ от максимальной/минимальной цены
            timeout: Таймаут ожидания исполнения (в секундах)
            
        Returns:
            Dict[str, Any]: Результат исполнения ордера
        """
        if self.market_simulator:
            # Используем симулятор рынка для трейлинг-стопа
            return await AdvancedOrderTypes.trailing_stop(
                self.market_simulator,
                self.symbol,
                side,
                activation_price,
                trailing_offset,
                size,
                timeout
            )
        else:
            # Возвращаем ошибку (упрощенно)
            return {
                "success": False,
                "error": "Трейлинг-стоп не реализован без симулятора рынка"
            }

    async def get_order_book(self, depth: int = 5) -> Dict[str, Any]:
        """
        Получение стакана заявок.
        
        Args:
            depth: Глубина стакана
            
        Returns:
            Dict[str, Any]: Стакан заявок
        """
        if self.market_simulator:
            # Получаем стакан из симулятора
            return {
                "symbol": self.symbol,
                "bids": self.market_simulator.order_book.get_bid_ladder()[:depth],
                "asks": self.market_simulator.order_book.get_ask_ladder()[:depth],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Возвращаем заглушку
            return {
                "symbol": self.symbol,
                "bids": [],
                "asks": [],
                "timestamp": datetime.now().isoformat()
            }

    async def get_market_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущем состоянии рынка.
        
        Returns:
            Dict[str, Any]: Информация о рынке
        """
        if self.market_simulator:
            return self.market_simulator.get_market_info()
        else:
            return {
                "symbol": self.symbol,
                "price": self.current_price,
                "volatility": 0.0,
                "trend_direction": 0.0,
                "timestamp": datetime.now().isoformat()
            }

    async def change_market_condition(self, condition: MarketCondition, duration: float = None) -> bool:
        """
        Изменение состояния рынка для симуляции различных условий.
        
        Args:
            condition: Новое состояние рынка
            duration: Продолжительность состояния в секундах (если None, состояние остается постоянным)
            
        Returns:
            bool: Успешность изменения состояния
        """
        if not self.market_simulator:
            self.logger.warning("Невозможно изменить состояние рынка без симулятора")
            return False
        
        if duration:
            # Временное изменение состояния
            self.market_simulator.simulate_special_condition(condition, duration)
            self.logger.info(f"Изменено состояние рынка на {condition.name} на {duration} секунд")
        else:
            # Постоянное изменение состояния
            self.market_simulator.update_market_condition(condition)
            self.logger.info(f"Изменено состояние рынка на {condition.name}")
        
        return True

    async def ml_training_step(self, action: int) -> Dict[str, Any]:
        """
        Выполнение шага обучения модели машинного обучения.
        
        Args:
            action: Действие (0 - держать, 1 - купить, 2 - продать)
            
        Returns:
            Dict[str, Any]: Результат шага
        """
        if not self.trading_environment:
            self.logger.warning("Невозможно выполнить шаг обучения без RL-среды")
            return {
                "success": False,
                "error": "RL-среда не инициализирована"
            }
        
        observation, reward, done, info = self.trading_environment.step(action)
        
        return {
            "success": True,
            "observation": observation.tolist(),
            "reward": reward,
            "done": done,
            "info": info
        }

    async def ml_reset_environment(self) -> Dict[str, Any]:
        """
        Сброс среды обучения.
        
        Returns:
            Dict[str, Any]: Начальное наблюдение
        """
        if not self.trading_environment:
            self.logger.warning("Невозможно сбросить среду без RL-среды")
            return {
                "success": False,
                "error": "RL-среда не инициализирована"
            }
        
        observation = self.trading_environment.reset()
        
        return {
            "success": True,
            "observation": observation.tolist()
        }

    # Методы форматирования данных для совместимости с исходным кодом

    def get_formatted_balance(self):
        """Получение форматированного баланса."""
        return f"{self.dry_data.get_balance():.2f} USDT"

    def get_formatted_symbol(self):
        """Получение форматированного символа торговой пары."""
        return self.symbol

    def get_formatted_mode(self):
        """Получение форматированного режима работы."""
        return "Симуляция (Dry Mode)"

    def get_formatted_leverage(self):
        """Получение форматированного кредитного плеча."""
        return f"{self.leverage}x"

    def get_formatted_risk(self):
        """Получение форматированного риска."""
        if self.risk_controller:
            return self.risk_controller.get_formatted_risk()
        return "Не определено"

    def get_formatted_stop_loss(self):
        """Получение форматированного стоп-лосса."""
        if self.risk_controller:
            return self.risk_controller.get_formatted_stop_loss()
        return "Не определено"

    def get_formatted_take_profit(self):
        """Получение форматированного тейк-профита."""
        if self.risk_controller:
            return self.risk_controller.get_formatted_take_profit()
        return "Не определено"

    def get_formatted_current_price(self):
        """Получение форматированной текущей цены."""
        if self.current_price:
            return f"{self.current_price:.2f}"
        return "Не определено"

    def get_formatted_open_positions(self):
        """Получение форматированного количества открытых позиций."""
        positions = self.dry_data.get_open_positions()
        return f"{len(positions)} позиции"

    def get_formatted_trade_history(self):
        """Получение форматированного количества сделок."""
        trades = self.dry_data.get_trades()
        return f"{len(trades)} сделок"

    def get_formatted_performance_stats(self):
        """Получение форматированной статистики производительности."""
        stats = self.dry_data.get_performance_stats()
        return f"P&L: {stats['profit_loss_percent']:.2f}%, {stats['total_trades']} сделок"

    def _calculate_slippage(self, size: float, is_sell: bool) -> float:
        """
        Расчет проскальзывания при закрытии позиции.
        
        Args:
            size: Размер позиции
            is_sell: Флаг продажи
            
        Returns:
            float: Проскальзывание в процентах
        """
        slippage_percent = random.uniform(0, 0.1)  # 0-0.1% проскальзывания
        slippage = slippage_percent * size
        return slippage

