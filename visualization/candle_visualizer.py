"""
Модуль визуализации свечей для Leon Trading Bot.

Предоставляет консольную визуализацию свечей и индикаторов.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.box import Box, ROUNDED
from rich.columns import Columns

from visualization.base import ConsoleVisualizer
from core.constants import TRADING_MODES


class CandleVisualizer(ConsoleVisualizer):
    """
    Консольный визуализатор свечей и индикаторов.
    
    Отображает свечи, объемы и индикаторы в консоли с использованием
    библиотеки rich.
    """
    
    def __init__(self, name: str = "candle_visualizer", config: Dict[str, Any] = None):
        """
        Инициализация визуализатора свечей.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        super().__init__(name, config)
        
        # Инициализация консоли rich
        self.console = Console()
        
        # Данные для отображения
        self.data = {
            "symbol": self.config.get("symbol", "BTCUSDT"),
            "timeframe": self.config.get("timeframe", "1h"),
            "candles": [],
            "volumes": [],
            "indicators": {},
            "last_update": datetime.now(),
            "price_precision": self.config.get("price_precision", 2),
            "volume_precision": self.config.get("volume_precision", 2),
            "max_candles": self.config.get("max_candles", 20),
            "show_volumes": self.config.get("show_volumes", True),
            "show_indicators": self.config.get("show_indicators", True),
        }
        
        # Символы для отображения свечей
        self.candle_symbols = {
            "up": "▲",
            "down": "▼",
            "body_up": "█",
            "body_down": "█",
            "wick": "│",
        }
        
        # Цвета
        self.colors = {
            "up": "green",
            "down": "red",
            "neutral": "white",
            "volume": "blue",
            "ma": "yellow",
            "ema": "cyan",
            "rsi": "magenta",
            "bollinger_upper": "bright_green",
            "bollinger_lower": "bright_red",
            "bollinger_middle": "bright_blue",
        }
        
        # Настройка лайаута
        self._setup_layout()
        
        self.live = None
        self.logger.info(f"Визуализатор свечей {name} инициализирован")
    
    def _setup_layout(self) -> None:
        """Настройка лайаута визуализатора."""
        self.layout = Layout(name="root")
        
        # Разделение на верхнюю и нижнюю части
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Разделение основной части
        self.layout["main"].split_row(
            Layout(name="chart", ratio=3),
            Layout(name="info", ratio=1)
        )
        
        # Разделение информационной части
        self.layout["info"].split(
            Layout(name="price_info"),
            Layout(name="indicators")
        )
    
    def _header_panel(self) -> Panel:
        """Создание панели заголовка."""
        symbol = self.data["symbol"]
        timeframe = self.data["timeframe"]
        last_update = self.data["last_update"].strftime("%Y-%m-%d %H:%M:%S")
        
        title = Text(f"{symbol} - {timeframe}", style="bold white")
        subtitle = Text(f"Последнее обновление: {last_update}", style="dim white")
        
        return Panel(
            Align.center(Columns([title, Text("   "), subtitle])),
            box=ROUNDED,
            title="Leon Trading Bot - Визуализатор свечей",
            title_align="center",
            style="blue"
        )
    
    def _price_info_panel(self) -> Panel:
        """Создание панели с информацией о цене."""
        if not self.data["candles"]:
            return Panel(Text("Нет данных"), title="Информация о цене", box=ROUNDED)
        
        last_candle = self.data["candles"][-1]
        open_price = last_candle[1]
        high_price = last_candle[2]
        low_price = last_candle[3]
        close_price = last_candle[4]
        
        price_change = close_price - open_price
        price_change_percent = (price_change / open_price) * 100
        
        color = "green" if price_change >= 0 else "red"
        change_sign = "+" if price_change >= 0 else ""
        
        table = Table(box=None, show_header=False, show_edge=False, pad_edge=False)
        table.add_column("Параметр")
        table.add_column("Значение", justify="right")
        
        precision = self.data["price_precision"]
        
        table.add_row("Открытие", f"{open_price:.{precision}f}")
        table.add_row("Максимум", f"{high_price:.{precision}f}")
        table.add_row("Минимум", f"{low_price:.{precision}f}")
        table.add_row("Закрытие", f"{close_price:.{precision}f}")
        table.add_row("Изменение", Text(f"{change_sign}{price_change:.{precision}f} ({change_sign}{price_change_percent:.2f}%)", style=color))
        
        return Panel(table, title="Информация о цене", box=ROUNDED)
    
    def _indicators_panel(self) -> Panel:
        """Создание панели с индикаторами."""
        if not self.data["indicators"]:
            return Panel(Text("Нет данных"), title="Индикаторы", box=ROUNDED)
        
        table = Table(box=None, show_header=False, show_edge=False, pad_edge=False)
        table.add_column("Индикатор")
        table.add_column("Значение", justify="right")
        
        precision = self.data["price_precision"]
        
        for name, value in self.data["indicators"].items():
            if isinstance(value, (int, float)):
                table.add_row(name, f"{value:.{precision}f}")
            elif isinstance(value, dict):
                for subname, subvalue in value.items():
                    table.add_row(f"{name} {subname}", f"{subvalue:.{precision}f}")
            elif isinstance(value, list) and len(value) > 0:
                table.add_row(name, f"{value[-1]:.{precision}f}")
        
        return Panel(table, title="Индикаторы", box=ROUNDED)
    
    def _chart_panel(self) -> Panel:
        """Создание панели с графиком свечей."""
        if not self.data["candles"]:
            return Panel(Text("Нет данных для отображения"), title="График", box=ROUNDED)
        
        # Получение данных для отображения
        candles = self.data["candles"][-self.data["max_candles"]:]
        
        # Определение минимальной и максимальной цены для масштабирования
        min_price = min([candle[3] for candle in candles])  # Минимум из Low
        max_price = max([candle[2] for candle in candles])  # Максимум из High
        
        # Добавляем отступ для лучшего отображения
        price_range = max_price - min_price
        min_price -= price_range * 0.05
        max_price += price_range * 0.05
        
        # Высота графика
        chart_height = 15
        
        # Создание строк для графика
        chart_lines = ["" for _ in range(chart_height)]
        
        # Ширина одной свечи (включая отступы)
        candle_width = 3
        
        # Масштабирование цен для отображения
        def scale_price(price):
            return int((max_price - price) / (max_price - min_price) * (chart_height - 1))
        
        # Отображение свечей
        for i, candle in enumerate(candles):
            timestamp, open_price, high_price, low_price, close_price, volume = candle[:6]
            
            # Определение типа свечи
            is_bullish = close_price >= open_price
            
            # Масштабирование цен
            scaled_open = scale_price(open_price)
            scaled_close = scale_price(close_price)
            scaled_high = scale_price(high_price)
            scaled_low = scale_price(low_price)
            
            # Определение тела свечи
            body_top = min(scaled_open, scaled_close)
            body_bottom = max(scaled_open, scaled_close)
            
            # Цвет свечи
            color = self.colors["up"] if is_bullish else self.colors["down"]
            
            # Отображение свечи
            for j in range(chart_height):
                # Позиция в строке для текущей свечи
                pos = i * candle_width
                
                # Добавление символов в соответствующие строки
                if j == scaled_high and j < body_top:
                    # Верхний фитиль
                    chart_lines[j] += f"[{color}]{self.candle_symbols['wick']}[/{color}]" + " " * (candle_width - 1)
                elif j == scaled_low and j > body_bottom:
                    # Нижний фитиль
                    chart_lines[j] += f"[{color}]{self.candle_symbols['wick']}[/{color}]" + " " * (candle_width - 1)
                elif body_top <= j <= body_bottom:
                    # Тело свечи
                    body_symbol = self.candle_symbols["body_up"] if is_bullish else self.candle_symbols["body_down"]
                    chart_lines[j] += f"[{color}]{body_symbol}[/{color}]" + " " * (candle_width - 1)
                elif scaled_high <= j <= scaled_low:
                    # Фитиль внутри диапазона high-low
                    chart_lines[j] += f"[{color}]{self.candle_symbols['wick']}[/{color}]" + " " * (candle_width - 1)
                else:
                    # Пустое пространство
                    chart_lines[j] += " " * candle_width
        
        # Создание текстовых объектов для каждой строки
        chart_text = []
        for line in chart_lines:
            chart_text.append(Text.from_markup(line))
        
        # Добавление шкалы цен
        price_scale = []
        for i in range(chart_height):
            price = max_price - (i / (chart_height - 1)) * (max_price - min_price)
            price_scale.append(f"{price:.{self.data['price_precision']}f}")
        
        # Максимальная длина строки в шкале цен
        max_scale_len = max(len(p) for p in price_scale)
        
        # Объединение шкалы цен и графика
        combined_chart = []
        for i, (scale, line) in enumerate(zip(price_scale, chart_text)):
            # Выравнивание шкалы цен по правому краю
            padded_scale = scale.rjust(max_scale_len)
            combined_chart.append(Text(f"{padded_scale} │ ") + line)
        
        # Добавление временной шкалы
        time_scale = Text("─" * (len(candles) * candle_width + max_scale_len + 3))
        combined_chart.append(time_scale)
        
        # Добавление меток времени
        time_labels = Text(" " * (max_scale_len + 3))
        for i, candle in enumerate(candles):
            if i % 5 == 0:  # Отображаем каждую пятую метку
                timestamp = candle[0]
                time_str = datetime.fromtimestamp(timestamp / 1000).strftime("%H:%M")
                # Добавляем метку времени с учетом ширины свечи
                pos = i * candle_width
                if pos + len(time_str) <= len(time_labels) + candle_width * len(candles):
                    time_labels = Text(time_labels.plain[:pos] + time_str + time_labels.plain[pos+len(time_str):])
        
        combined_chart.append(time_labels)
        
        return Panel(
            Align.left(Columns(combined_chart)),
            title=f"График {self.data['symbol']} ({self.data['timeframe']})",
            box=ROUNDED
        )
    
    def _footer_panel(self) -> Panel:
        """Создание панели футера."""
        help_text = Text("Управление: [q] Выход | [r] Обновить | [+/-] Изменить масштаб | [←/→] Прокрутка", style="dim white")
        
        return Panel(
            Align.center(help_text),
            box=ROUNDED,
            style="blue"
        )
    
    def generate_layout(self) -> Layout:
        """Генерация полного лайаута."""
        # Обновление компонентов лайаута
        self.layout["header"].update(self._header_panel())
        self.layout["chart"].update(self._chart_panel())
        self.layout["price_info"].update(self._price_info_panel())
        self.layout["indicators"].update(self._indicators_panel())
        self.layout["footer"].update(self._footer_panel())
        
        return self.layout
    
    def start(self) -> bool:
        """
        Запуск визуализатора свечей.
        
        Returns:
            True, если визуализатор успешно запущен, иначе False
        """
        if self.is_running:
            self.logger.warning(f"Визуализатор {self.name} уже запущен")
            return False
        
        self.is_running = True
        self.logger.info(f"Визуализатор {self.name} запущен")
        
        # Создание Live объекта для обновления в реальном времени
        self.live = Live(
            self.generate_layout(),
            refresh_per_second=1.0 / self.refresh_rate,
            screen=True,
            console=self.console
        )
        
        return True
    
    def stop(self) -> bool:
        """
        Остановка визуализатора свечей.
        
        Returns:
            True, если визуализатор успешно остановлен, иначе False
        """
        if not self.is_running:
            self.logger.warning(f"Визуализатор {self.name} не запущен")
            return False
        
        self.is_running = False
        
        if self.live:
            self.live.stop()
            self.live = None
        
        self.logger.info(f"Визуализатор {self.name} остановлен")
        return True
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных визуализатора свечей.
        
        Args:
            data: Данные для обновления
                - symbol: Символ торговой пары
                - timeframe: Таймфрейм
                - candles: Список свечей
                - volumes: Список объемов
                - indicators: Словарь с индикаторами
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        if not self.is_running:
            self.logger.warning(f"Попытка обновить данные остановленного визуализатора {self.name}")
            return False
        
        # Обновление данных
        for key, value in data.items():
            if key in self.data:
                self.data[key] = value
        
        # Обновление времени последнего обновления
        self.data["last_update"] = datetime.now()
        
        # Обновление отображения, если Live объект существует
        if self.live:
            self.live.update(self.generate_layout())
        
        self.logger.debug(f"Данные визуализатора {self.name} обновлены")
        return True
    
    def render(self) -> Layout:
        """
        Отрисовка визуализатора свечей.
        
        Returns:
            Layout объект с визуализацией
        """
        return self.generate_layout()
    
    def run(self) -> None:
        """Запуск визуализатора в интерактивном режиме."""
        self.start()
        
        try:
            with self.live:
                while self.is_running:
                    time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.logger.info("Визуализатор остановлен пользователем")
        finally:
            self.stop()


def main():
    """Функция для тестового запуска визуализатора."""
    import random
    import numpy as np
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание тестовых данных
    def generate_candles(count=50, base_price=50000.0, volatility=0.02):
        candles = []
        current_price = base_price
        now = datetime.now().timestamp() * 1000
        interval = 60 * 60 * 1000  # 1 час в миллисекундах
        
        for i in range(count):
            timestamp = now - (count - i - 1) * interval
            
            # Случайное изменение цены
            price_change = current_price * volatility * (random.random() * 2 - 1)
            current_price += price_change
            
            # Генерация OHLCV
            open_price = current_price
            close_price = current_price * (1 + volatility * (random.random() * 2 - 1) * 0.5)
            high_price = max(open_price, close_price) * (1 + volatility * random.random() * 0.5)
            low_price = min(open_price, close_price) * (1 - volatility * random.random() * 0.5)
            volume = random.random() * 100 + 50
            
            candle = [timestamp, open_price, high_price, low_price, close_price, volume]
            candles.append(candle)
            
            current_price = close_price
        
        return candles
    
    # Генерация индикаторов
    def generate_indicators(candles):
        closes = [candle[4] for candle in candles]
        
        # Простые скользящие средние
        ma_9 = np.convolve(closes, np.ones(9)/9, mode='valid')
        ma_21 = np.convolve(closes, np.ones(21)/21, mode='valid')
        
        # RSI (упрощенный)
        rsi = random.randint(30, 70)
        
        # Bollinger Bands (упрощенные)
        ma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')[-1]
        std_20 = np.std(closes[-20:])
        upper_band = ma_20 + 2 * std_20
        lower_band = ma_20 - 2 * std_20
        
        return {
            "MA(9)": ma_9[-1] if len(ma_9) > 0 else 0,
            "MA(21)": ma_21[-1] if len(ma_21) > 0 else 0,
            "RSI(14)": rsi,
            "Bollinger": {
                "Upper": upper_band,
                "Middle": ma_20,
                "Lower": lower_band
            }
        }
    
    # Создание и запуск визуализатора
    visualizer = CandleVisualizer(config={
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "refresh_rate": 1.0,
        "max_candles": 20,
        "price_precision": 2
    })
    
    # Генерация тестовых данных
    candles = generate_candles(50)
    indicators = generate_indicators(candles)
    
    # Обновление данных визуализатора
    visualizer.update({
        "candles": candles,
        "indicators": indicators
    })
    
    # Запуск визуализатора
    visualizer.run()


if __name__ == "__main__":
    main() 