"""
Модуль трейдера для работы в режиме симуляции (dry mode).
Позволяет тестировать стратегии без реальных сделок.
"""

import asyncio
import logging
import traceback
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import random
import time

from trading.traders.base import TraderBase


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
            amount: Сумма изменения (положительная или отрицательная)
            
        Returns:
            float: Новый баланс
        """
        self.data["balance"] += amount
        self.save_data()
        return self.data["balance"]
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Добавление сделки в историю.
        
        Args:
            trade_data: Данные о сделке
        """
        self.data["trades"].append(trade_data)
        self.save_data()
    
    def add_position(self, position_data: Dict[str, Any]) -> None:
        """
        Добавление открытой позиции.
        
        Args:
            position_data: Данные о позиции
        """
        self.data["open_positions"].append(position_data)
        self.save_data()
    
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
        
        return stats


class DryTrader(TraderBase):
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
                 visualizer=None):
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
        """
        super().__init__(symbol, exchange_client, strategy, notification_service, 
                         risk_controller, initial_balance, leverage)
        
        # Менеджер данных симуляции
        self.dry_data = DryModeData(symbol, storage_file)
        
        # Параметры симуляции
        self.price_update_interval = price_update_interval
        self.visualizer = visualizer
        
        # Текущая цена и задача обновления цены
        self.current_price = None
        self.price_update_task = None
        
        # Логгер
        self.logger = logging.getLogger(f"DryTrader-{self.symbol}")
    
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
            self.current_price = await self.exchange_client.get_current_price(self.symbol)
            self.logger.info(f"Текущая цена {self.symbol}: {self.current_price}")
        except Exception as e:
            self.logger.error(f"Ошибка получения текущей цены: {str(e)}")
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
        
        return True
    
    async def _update_price_loop(self):
        """
        Фоновая задача для периодического обновления цены.
        """
        self.logger.info(f"Запущен цикл обновления цены для {self.symbol}")
        
        while self.is_running:
            try:
                # Получение текущей цены с биржи
                new_price = await self.exchange_client.get_current_price(self.symbol)
                self.current_price = new_price
                
                # Обновление цены в визуализаторе, если он есть
                if self.visualizer:
                    await self.visualizer.update_price(self.symbol, new_price)
                
                # Проверка условий для открытых позиций
                await self._check_positions()
                
                # Логирование с меньшей частотой для уменьшения шума
                if random.random() < 0.05:  # ~5% вероятность логирования
                    self.logger.debug(f"Текущая цена {self.symbol}: {new_price}")
                
            except Exception as e:
                self.logger.error(f"Ошибка в цикле обновления цены: {str(e)}")
                self.logger.debug(traceback.format_exc())
            
            # Пауза перед следующим обновлением
            await asyncio.sleep(self.price_update_interval)
    
    async def _check_positions(self):
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
        
        if position_value > balance:
            self.logger.warning(f"Недостаточно средств: {balance} < {position_value}")
            raise ValueError(f"Недостаточно средств: {balance} < {position_value}")
        
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
            "status": "OPEN"
        }
        
        # Добавление позиции в список открытых
        self.dry_data.add_position(position)
        
        # Обновление состояния трейдера
        self.in_position = True
        self.current_position = position
        
        # Логирование и уведомление
        self.logger.info(f"Открыта {direction} позиция: {size} {self.symbol} по цене {entry_price}")
        
        if self.notification_service:
            await self.notification_service.send_notification(
                f"[DRY] Открыта {direction} позиция: {size} {self.symbol} по цене {entry_price}"
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
        Выход из позиции.
        
        Args:
            position_id: Идентификатор позиции
            price: Цена выхода (опционально, по умолчанию текущая цена)
            
        Returns:
            Dict[str, Any]: Информация о закрытой позиции
        """
        # Поиск позиции
        position = None
        for pos in self.dry_data.get_open_positions():
            if pos["id"] == position_id:
                position = pos
                break
        
        if not position:
            self.logger.warning(f"Позиция с ID {position_id} не найдена")
            raise ValueError(f"Позиция с ID {position_id} не найдена")
        
        # Использование текущей цены, если цена не указана
        exit_price = price if price is not None else self.current_price
        if exit_price is None:
            self.logger.error("Невозможно закрыть позицию: неизвестна текущая цена")
            raise ValueError("Неизвестна текущая цена")
        
        # Расчет прибыли/убытка
        entry_price = position["entry_price"]
        size = position["size"]
        direction = position["direction"]
        
        if direction == "LONG":
            profit = (exit_price - entry_price) * size
        else:  # SHORT
            profit = (entry_price - exit_price) * size
        
        # Обновление баланса
        new_balance = self.dry_data.update_balance(profit)
        
        # Создание данных о сделке
        trade = {
            "position_id": position_id,
            "symbol": self.symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "entry_time": position["entry_time"],
            "exit_price": exit_price,
            "exit_time": datetime.now().isoformat(),
            "profit": profit,
            "profit_percent": (profit / (entry_price * size)) * 100
        }
        
        # Добавление сделки в историю
        self.dry_data.add_trade(trade)
        
        # Удаление позиции из списка открытых
        self.dry_data.remove_position(position_id)
        
        # Обновление состояния трейдера
        self.in_position = False
        self.current_position = None
        
        # Логирование и уведомление
        profit_str = f"{profit:.2f} ({trade['profit_percent']:.2f}%)"
        self.logger.info(
            f"Закрыта {direction} позиция: {size} {self.symbol} по цене {exit_price}. "
            f"P&L: {profit_str}"
        )
        
        if self.notification_service:
            await self.notification_service.send_notification(
                f"[DRY] Закрыта {direction} позиция: {size} {self.symbol} по цене {exit_price}. "
                f"P&L: {profit_str}"
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
                self.dry_data.save_data()
                
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
        return self.dry_data.get_open_positions()
    
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