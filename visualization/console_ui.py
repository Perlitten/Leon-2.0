"""
Модуль консольного интерфейса для Leon Trading Bot.

Предоставляет компоненты для визуализации данных в консоли.
"""

import logging
import os
import time
import threading
import random
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime
import math

from visualization.base import BaseVisualizer
from core.localization import LocalizationManager


class ConsoleVisualizer(BaseVisualizer):
    """
    Консольный визуализатор для отображения данных в терминале.
    
    Отображает информацию о балансе, позициях, ценах и сигналах в консоли.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], localization: Any = None):
        """
        Инициализация консольного визуализатора.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
            localization: Менеджер локализации
        """
        super().__init__(name, config)
        self.logger = logging.getLogger("ConsoleVisualizer")
        self.data = {}
        self.last_render_time = 0
        self.render_interval = config.get("update_interval", 5)  # Интервал обновления в секундах
        self.localization = localization
        self.running = False
        self.render_task = None
        
        # Добавляем блокировку для потокобезопасного доступа к данным
        self._data_lock = threading.Lock()
        
        # Инициализация коллекций данных
        self.price_history = []
        self.indicators = {}
        self.signals = []
        self.positions = []
        self.balance = 0.0
        self.mode = "unknown"
        self.trading_pair = {"symbol": "BTCUSDT", "interval": "1h"}
        
    def start(self) -> bool:
        """
        Запускает визуализатор.
        
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        try:
            self.running = True
            self.render_task = asyncio.create_task(self._render_loop())
            self.logger.info("Консольный визуализатор запущен")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при запуске визуализатора: {str(e)}")
            return False
        
    def stop(self) -> bool:
        """
        Останавливает визуализатор.
        
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        try:
            self.running = False
            if self.render_task and not self.render_task.done():
                self.render_task.cancel()
            self.logger.info("Консольный визуализатор остановлен")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при остановке визуализатора: {str(e)}")
            return False
        
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновляет данные визуализатора.
        
        Args:
            data: Словарь с данными для отображения
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.data.update(data)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении данных: {str(e)}")
            return False
    
    # Методы-адаптеры для совместимости с интерфейсом рефакторенного ConsoleVisualizer
    
    def update_price(self, price: float) -> bool:
        """
        Обновляет данные о цене.
        
        Args:
            price: Текущая цена
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        if not hasattr(self, 'price_history'):
            self.price_history = []
            
        try:
            with self._data_lock:
                # Проверка на корректность данных
                if price is None or not isinstance(price, (int, float)) or math.isnan(price):
                    self.logger.warning(f"Некорректное значение цены: {price}")
                    return False
                    
                self.price_history.append(float(price))
                # Ограничиваем размер истории цен
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]
                self.data["price"] = price
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении цены: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
            
    def update_indicators(self, indicators: Dict[str, Any]) -> bool:
        """
        Обновляет данные индикаторов.
        
        Args:
            indicators: Словарь с индикаторами
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.indicators.update(indicators)
                self.data["indicators"] = self.indicators
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении индикаторов: {str(e)}")
            return False
            
    def update_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """
        Обновляет данные сигналов.
        
        Args:
            signals: Список сигналов
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.signals = signals
                self.data["signals"] = signals
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении сигналов: {str(e)}")
            return False
            
    def update_positions(self, positions: List[Dict[str, Any]]) -> bool:
        """
        Обновляет данные позиций.
        
        Args:
            positions: Список позиций
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.positions = positions
                self.data["positions"] = positions
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении позиций: {str(e)}")
            return False
            
    def update_balance(self, balance: float) -> bool:
        """
        Обновляет данные баланса.
        
        Args:
            balance: Текущий баланс
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.balance = float(balance)
                self.data["balance"] = self.balance
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении баланса: {str(e)}")
            return False
            
    def update_mode(self, mode: str) -> bool:
        """
        Обновляет режим работы.
        
        Args:
            mode: Режим работы
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.mode = mode
                self.data["mode"] = mode
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении режима: {str(e)}")
            return False
            
    def update_trading_pair(self, symbol: str, interval: str) -> bool:
        """
        Обновляет торговую пару.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            with self._data_lock:
                self.trading_pair = {"symbol": symbol, "interval": interval}
                self.data["trading_pair"] = self.trading_pair
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении торговой пары: {str(e)}")
            return False
    
    def update_recent_prices(self, prices: List[Dict[str, Any]]) -> bool:
        """
        Обновляет данные о последних ценах.
        
        Args:
            prices: Список последних цен
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        return self.update({"recent_prices": prices})
    
    def update_pnl(self, pnl: float, pnl_percent: float) -> bool:
        """
        Обновляет данные о прибыли/убытке.
        
        Args:
            pnl: Прибыль/убыток в абсолютном выражении
            pnl_percent: Прибыль/убыток в процентах
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        return self.update({"pnl": pnl, "pnl_percent": pnl_percent})
    
    def update_strategy_info(self, strategy_name: str) -> bool:
        """
        Обновляет информацию о стратегии.
        
        Args:
            strategy_name: Название стратегии
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        return self.update({"strategy_name": strategy_name})
    
    def update_signals_data(self, signals_dict: Dict[str, Dict[str, Any]]) -> bool:
        """
        Обновляет расширенные данные о сигналах.
        
        Args:
            signals_dict: Словарь с расширенными данными о сигналах
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        signals = []
        for signal_id, signal_data in signals_dict.items():
            signal_data["id"] = signal_id
            signals.append(signal_data)
        return self.update({"signals": signals})
        
    async def _render_loop(self) -> None:
        """Цикл отрисовки визуализации."""
        try:
            while self.running:
                current_time = time.time()
                if current_time - self.last_render_time >= self.render_interval:
                    self.render()
                    self.last_render_time = current_time
                await asyncio.sleep(0.1)  # Небольшая задержка для снижения нагрузки на CPU
        except asyncio.CancelledError:
            self.logger.info("Задача отрисовки отменена")
        except Exception as e:
            self.logger.error(f"Ошибка в цикле отрисовки: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
    def render(self) -> Any:
        """
        Отрисовывает данные в консоли.
        
        Returns:
            None
        """
        if not self.data:
            return None
            
        try:
            # Очищаем консоль
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Получаем локализованные строки
            loc = self._get_localized_strings()
            
            # Формируем заголовок
            header = self._create_header(loc)
            
            # Формируем информацию о балансе и позициях
            balance_info = self._create_balance_info(loc)
            
            # Формируем информацию о последних ценах
            prices_info = self._create_prices_info(loc)
            
            # Формируем информацию об индикаторах
            indicators_info = self._create_indicators_info(loc)
            
            # Формируем информацию о сигналах
            signals_info = self._create_signals_info(loc)
            
            # Формируем нижний колонтитул
            footer = self._create_footer(loc)
            
            # Объединяем все блоки и выводим на экран
            output = f"{header}\n\n{balance_info}\n\n{prices_info}\n{indicators_info}\n\n{signals_info}\n\n{footer}"
            print(output)
            
            return output
        except Exception as e:
            self.logger.error(f"Ошибка при отрисовке данных: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _get_localized_strings(self) -> Dict[str, str]:
        """
        Получает локализованные строки для интерфейса.
        
        Returns:
            Словарь с локализованными строками
        """
        # Значения по умолчанию на русском
        default_strings = {
            "title": "LEON TRADING BOT",
            "balance": "БАЛАНС",
            "trading_pair": "ТОРГОВАЯ ПАРА",
            "interval": "ИНТЕРВАЛ",
            "strategy": "СТРАТЕГИЯ",
            "pnl": "P&L",
            "positions": "ОТКРЫТЫЕ ПОЗИЦИИ",
            "no_positions": "Нет открытых позиций",
            "prices": "ПОСЛЕДНИЕ ЦЕНЫ",
            "no_prices": "Нет данных о ценах",
            "indicators": "ИНДИКАТОРЫ",
            "no_indicators": "Нет данных об индикаторах",
            "signals": "СИГНАЛЫ",
            "no_signals": "Нет активных сигналов",
            "footer": "Нажмите Ctrl+C для остановки бота. Обновление каждые {interval} сек."
        }
        
        # Если есть менеджер локализации, получаем строки из него
        if self.localization:
            try:
                for key in default_strings.keys():
                    localized = self.localization.get(f"visualization.console.{key}")
                    if localized:
                        default_strings[key] = localized
            except Exception as e:
                self.logger.warning(f"Ошибка при получении локализованных строк: {str(e)}")
                
        return default_strings
    
    def _create_header(self, loc: Dict[str, str]) -> str:
        """
        Создает заголовок визуализации.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с заголовком
        """
        mode = self.data.get("mode", "unknown").upper()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Определяем цвет режима
        mode_color = "\033[92m"  # Зеленый для real
        if mode == "BACKTEST":
            mode_color = "\033[94m"  # Синий для backtest
        elif mode == "DRY":
            mode_color = "\033[93m"  # Желтый для dry
            
        reset_color = "\033[0m"
        
        # Формируем заголовок с цветным режимом
        title = f"=== {loc['title']} [{mode_color}{mode}{reset_color}] ===  {current_time}"
        
        # Добавляем линию под заголовком
        header = f"{title}\n{'=' * len(title)}"
        
        return header
    
    def _create_balance_info(self, loc: Dict[str, str]) -> str:
        """
        Создает блок информации о балансе и позициях.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с информацией о балансе
        """
        # Получаем данные
        balance = self.data.get("balance", 0.0)
        symbol = self.data.get("symbol", "UNKNOWN")
        interval = self.data.get("interval", "1m")
        strategy_name = self.data.get("strategy_name", "Unknown")
        pnl = self.data.get("pnl", 0.0)
        pnl_percent = self.data.get("pnl_percent", 0.0)
        positions = self.data.get("positions", [])
        
        # Определяем цвет для P&L
        pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"  # Зеленый для положительного, красный для отрицательного
        reset_color = "\033[0m"
        
        # Формируем левую колонку
        left_column = [
            f"{loc['balance']}: {balance:.2f} USDT",
            f"{loc['trading_pair']}: {symbol}",
            f"{loc['interval']}: {interval}",
            f"{loc['strategy']}: {strategy_name}",
            f"{loc['pnl']}: {pnl_color}{pnl:.2f} USDT ({pnl_percent:.2f}%){reset_color}"
        ]
        
        # Формируем правую колонку (позиции)
        right_column = [f"=== {loc['positions']} ==="]
        
        if positions:
            for pos in positions:
                pos_type = pos.get("type", "UNKNOWN")
                pos_symbol = pos.get("symbol", "UNKNOWN")
                pos_amount = pos.get("amount", 0.0)
                pos_entry = pos.get("entry_price", 0.0)
                pos_current = pos.get("current_price", 0.0)
                pos_pnl = pos.get("pnl", 0.0)
                pos_pnl_percent = pos.get("pnl_percent", 0.0)
                
                # Определяем цвет для типа позиции и P&L
                type_color = "\033[92m" if pos_type == "LONG" else "\033[91m"  # Зеленый для LONG, красный для SHORT
                pnl_color = "\033[92m" if pos_pnl >= 0 else "\033[91m"
                
                pos_info = [
                    f"{type_color}{pos_type}{reset_color} {pos_symbol} {pos_amount}",
                    f"Вход: {pos_entry:.2f} | Текущая: {pos_current:.2f}",
                    f"P&L: {pnl_color}{pos_pnl:.2f} ({pos_pnl_percent:.2f}%){reset_color}"
                ]
                
                right_column.extend(pos_info)
                right_column.append("-" * 30)
        else:
            right_column.append(loc["no_positions"])
        
        # Объединяем колонки
        max_left_len = max(len(line) for line in left_column)
        combined = []
        
        for i in range(max(len(left_column), len(right_column))):
            left = left_column[i] if i < len(left_column) else ""
            right = right_column[i] if i < len(right_column) else ""
            
            # Добавляем отступ между колонками
            padding = " " * (max_left_len - len(left) + 4)
            combined.append(f"{left}{padding}{right}")
        
        return "\n".join(combined)
    
    def _create_prices_info(self, loc: Dict[str, str]) -> str:
        """
        Создает блок информации о последних ценах.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с информацией о ценах
        """
        prices = self.data.get("recent_prices", [])
        
        # Формируем заголовок
        prices_info = [f"=== {loc['prices']} ==="]
        
        if prices:
            # Отображаем только последние 4 цены
            display_prices = prices[-4:] if len(prices) > 4 else prices
            
            for price_data in display_prices:
                price = price_data.get("price", 0.0)
                direction = price_data.get("direction", "up")
                
                # Определяем символ и цвет направления
                direction_symbol = "🟢" if direction == "up" else "🔴"
                direction_color = "\033[92m" if direction == "up" else "\033[91m"
                reset_color = "\033[0m"
                
                prices_info.append(f"{direction_symbol} {direction_color}{price:.2f}{reset_color}")
        else:
            prices_info.append(loc["no_prices"])
        
        # Добавляем нижнюю границу
        prices_info.append("-" * 30)
        
        return "\n".join(prices_info)
    
    def _create_indicators_info(self, loc: Dict[str, str]) -> str:
        """
        Создает блок информации об индикаторах.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с информацией об индикаторах
        """
        indicators = self.data.get("indicators", {})
        
        # Формируем заголовок
        indicators_info = [f"=== {loc['indicators']} ==="]
        
        if indicators:
            # Отображаем только первые 4 индикатора
            display_indicators = list(indicators.items())[:4]
            
            for name, value in display_indicators:
                # Добавляем интерпретацию для некоторых индикаторов
                interpretation = ""
                
                if name == "RSI":
                    if value < 30:
                        interpretation = " (Перепродан)"
                    elif value > 70:
                        interpretation = " (Перекуплен)"
                elif name == "MACD":
                    if value > 0:
                        interpretation = " (Бычий)"
                    else:
                        interpretation = " (Медвежий)"
                
                indicators_info.append(f"{name}: {value}{interpretation}")
        else:
            indicators_info.append(loc["no_indicators"])
        
        return "\n".join(indicators_info)
    
    def _create_signals_info(self, loc: Dict[str, str]) -> str:
        """
        Создает блок информации о сигналах.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с информацией о сигналах
        """
        signals = self.data.get("signals", [])
        
        # Формируем заголовок
        signals_info = [f"=== {loc['signals']} ==="]
        
        if signals:
            # Отображаем только последние 3 сигнала
            display_signals = signals[-3:] if len(signals) > 3 else signals
            
            for signal in display_signals:
                signal_time = signal.get("time", "")
                signal_type = signal.get("type", "UNKNOWN")
                signal_price = signal.get("price", 0.0)
                signal_confidence = signal.get("confidence", 0.0)
                
                # Определяем символ и цвет типа сигнала
                type_symbol = "🟢" if signal_type == "BUY" else "🔴"
                type_color = "\033[92m" if signal_type == "BUY" else "\033[91m"
                reset_color = "\033[0m"
                
                signals_info.append(
                    f"{signal_time} {type_symbol} {type_color}{signal_type}{reset_color} "
                    f"@ {signal_price:.2f} (Уверенность: {signal_confidence:.2f})"
                )
        else:
            signals_info.append(loc["no_signals"])
        
        return "\n".join(signals_info)
    
    def _create_footer(self, loc: Dict[str, str]) -> str:
        """
        Создает нижний колонтитул визуализации.
        
        Args:
            loc: Словарь с локализованными строками
            
        Returns:
            Строка с нижним колонтитулом
        """
        footer = loc["footer"].format(interval=self.render_interval)
        return footer 