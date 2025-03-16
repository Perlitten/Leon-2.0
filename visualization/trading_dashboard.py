"""
Модуль консольной визуализации для Leon Trading Bot.

Предоставляет интерактивную консольную панель для отображения
торговой информации с использованием библиотеки rich.
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

from visualization.base import ConsoleVisualizer
from core.constants import TRADING_MODES


class TradingDashboard(ConsoleVisualizer):
    """
    Интерактивная консольная панель для отображения торговой информации.
    
    Использует библиотеку rich для создания красивого и информативного
    интерфейса в консоли.
    """
    
    def __init__(self, name: str = "trading_dashboard", config: Dict[str, Any] = None):
        """
        Инициализация торговой панели.
        
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
            "mode": self.config.get("mode", TRADING_MODES["DRY"]),
            "initial_balance": self.config.get("initial_balance", 1000.0),
            "current_balance": self.config.get("current_balance", 1000.0),
            "profit": 0.0,
            "profit_percent": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "positions": [],
            "trading_params": {
                "leverage": self.config.get("leverage", 1),
                "risk_per_trade": self.config.get("risk_per_trade", 1.0),
                "stop_loss": self.config.get("stop_loss", 0.5),
                "take_profit": self.config.get("take_profit", 1.0)
            },
            "signals": [],
            "recommendation": "Ожидание сигналов...",
            "recommendation_color": "yellow"
        }
        
        # Создание макета
        self.layout = Layout(name="root")
        self._setup_layout()
        
        # Объект Live для обновления в реальном времени
        self.live = None
        
        self.logger.info(f"Торговая панель {name} инициализирована")
    
    def _setup_layout(self) -> None:
        """Настройка макета панели."""
        # Основные секции
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Разделение body на колонки
        self.layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Детализация секций
        self.layout["left"].split(
            Layout(name="stats", size=10),
            Layout(name="positions", ratio=1)
        )
        
        self.layout["right"].split(
            Layout(name="params", size=8),
            Layout(name="signals", ratio=1)
        )
        
        self.logger.debug("Макет торговой панели настроен")
    
    def _update_header(self) -> Panel:
        """
        Обновление заголовка панели.
        
        Returns:
            Панель с заголовком
        """
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_color = "green" if self.data["mode"] == TRADING_MODES["REAL"] else "yellow" if self.data["mode"] == TRADING_MODES["BACKTEST"] else "blue"
        
        header_text = Text()
        header_text.append("LEON Trading Bot | ", style="bold white")
        header_text.append(f"{self.data['symbol']} | ", style="bold cyan")
        header_text.append(f"Режим: ", style="white")
        header_text.append(f"{self.data['mode'].upper()} | ", style=f"bold {mode_color}")
        header_text.append(f"Баланс: ", style="white")
        header_text.append(f"${self.data['current_balance']:.2f} | ", style="bold green")
        header_text.append(f"{time_str}", style="dim white")
        
        return Panel(header_text, style="white on blue")
    
    def _stats_table(self) -> Panel:
        """
        Создание таблицы статистики.
        
        Returns:
            Панель с таблицей статистики
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="magenta")
        
        profit_color = "green" if self.data["profit"] >= 0 else "red"
        profit_sign = "+" if self.data["profit"] > 0 else ""
        
        table.add_row("Начальный баланс", f"${self.data['initial_balance']:.2f}")
        table.add_row("Текущий баланс", f"${self.data['current_balance']:.2f}")
        table.add_row("Прибыль", f"[{profit_color}]{profit_sign}${self.data['profit']:.2f} ({profit_sign}{self.data['profit_percent']:.2f}%)[/{profit_color}]")
        table.add_row("Всего сделок", f"{self.data['total_trades']}")
        table.add_row("Выигрышных", f"[green]{self.data['winning_trades']}[/green]")
        table.add_row("Убыточных", f"[red]{self.data['losing_trades']}[/red]")
        
        win_rate = 0.0 if self.data['total_trades'] == 0 else (self.data['winning_trades'] / self.data['total_trades']) * 100
        table.add_row("Винрейт", f"{win_rate:.1f}%")
        
        return Panel(table, title="Статистика торговли", border_style="yellow")
    
    def _params_panel(self) -> Panel:
        """
        Создание панели параметров.
        
        Returns:
            Панель с параметрами
        """
        params = Text()
        params.append(f"Плечо: {self.data['trading_params']['leverage']}x\n", style="bright_white")
        params.append(f"Риск на сделку: {self.data['trading_params']['risk_per_trade']:.1f}%\n", style="bright_white")
        params.append(f"Стоп-лосс: {self.data['trading_params']['stop_loss']:.1f}%\n", style="bright_white")
        params.append(f"Тейк-профит: {self.data['trading_params']['take_profit']:.1f}%", style="bright_white")
        
        return Panel(params, title="Параметры", border_style="green")
    
    def _signals_table(self) -> Panel:
        """
        Создание таблицы сигналов.
        
        Returns:
            Панель с таблицей сигналов
        """
        table = Table(box=None, padding=(0, 1))
        table.add_column("Индикатор", style="bold")
        table.add_column("Значение")
        table.add_column("Сигнал")
        
        if not self.data["signals"]:
            # Если сигналов нет, добавляем заглушки
            default_signals = [
                ("RSI", "50.00", "NEUTRAL"),
                ("MACD", "0.00", "NEUTRAL"),
                ("Bollinger", "0.00", "NEUTRAL"),
                ("MA Cross", "0.00", "NEUTRAL")
            ]
            
            for ind, val, sig in default_signals:
                signal_style = "yellow"
                table.add_row(ind, val, f"[{signal_style}]{sig}[/{signal_style}]")
        else:
            # Добавляем реальные сигналы
            for signal in self.data["signals"]:
                ind = signal.get("indicator", "")
                val = signal.get("value", "0.00")
                sig = signal.get("signal", "NEUTRAL")
                
                signal_style = "yellow" if sig == "NEUTRAL" else "green" if "BUY" in sig else "red"
                table.add_row(ind, str(val), f"[{signal_style}]{sig}[/{signal_style}]")
        
        return Panel(table, title="Сигналы", border_style="blue")
    
    def _positions_table(self) -> Panel:
        """
        Создание таблицы позиций.
        
        Returns:
            Панель с таблицей позиций
        """
        if not self.data["positions"]:
            return Panel(Text("Нет активных позиций", style="italic"), title="Позиции", border_style="magenta")
        
        table = Table(box=ROUNDED)
        table.add_column("ID", style="dim")
        table.add_column("Символ", style="cyan")
        table.add_column("Тип", style="bold")
        table.add_column("Размер", justify="right")
        table.add_column("Цена входа", justify="right")
        table.add_column("Текущая цена", justify="right")
        table.add_column("P/L", justify="right")
        
        for position in self.data["positions"]:
            position_id = position.get("id", "")
            symbol = position.get("symbol", "")
            position_type = position.get("type", "")
            size = position.get("size", 0.0)
            entry_price = position.get("entry_price", 0.0)
            current_price = position.get("current_price", 0.0)
            pnl = position.get("pnl", 0.0)
            pnl_percent = position.get("pnl_percent", 0.0)
            
            type_style = "green" if position_type.upper() == "LONG" else "red"
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl > 0 else ""
            
            table.add_row(
                str(position_id),
                symbol,
                f"[{type_style}]{position_type.upper()}[/{type_style}]",
                f"{size:.4f}",
                f"${entry_price:.2f}",
                f"${current_price:.2f}",
                f"[{pnl_style}]{pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_percent:.2f}%)[/{pnl_style}]"
            )
        
        return Panel(table, title="Активные позиции", border_style="magenta")
    
    def _recommendation(self) -> Panel:
        """
        Создание панели рекомендаций.
        
        Returns:
            Панель с рекомендациями
        """
        return Panel(
            Text(self.data["recommendation"], style=f"bold {self.data['recommendation_color']}"),
            title="Рекомендация",
            border_style="red"
        )
    
    def generate_layout(self) -> Layout:
        """
        Генерация макета панели.
        
        Returns:
            Макет панели
        """
        self.layout["header"].update(self._update_header())
        self.layout["stats"].update(self._stats_table())
        self.layout["positions"].update(self._positions_table())
        self.layout["params"].update(self._params_panel())
        self.layout["signals"].update(self._signals_table())
        self.layout["footer"].update(self._recommendation())
        
        return self.layout
    
    def start(self) -> bool:
        """
        Запуск торговой панели.
        
        Returns:
            True, если панель успешно запущена, иначе False
        """
        try:
            super().start()
            self.live = Live(self.generate_layout(), refresh_per_second=1/self.refresh_rate, screen=True)
            self.live.start()
            self.logger.info("Торговая панель запущена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при запуске торговой панели: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Остановка торговой панели.
        
        Returns:
            True, если панель успешно остановлена, иначе False
        """
        try:
            if self.live:
                self.live.stop()
            super().stop()
            self.logger.info("Торговая панель остановлена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при остановке торговой панели: {e}")
            return False
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных торговой панели.
        
        Args:
            data: Данные для обновления
            
        Returns:
            True, если данные успешно обновлены, иначе False
        """
        try:
            # Обновляем только те данные, которые переданы
            for key, value in data.items():
                if key in self.data:
                    self.data[key] = value
            
            # Вычисляем производные данные
            if "current_balance" in data or "initial_balance" in data:
                self.data["profit"] = self.data["current_balance"] - self.data["initial_balance"]
                self.data["profit_percent"] = (self.data["profit"] / self.data["initial_balance"]) * 100 if self.data["initial_balance"] > 0 else 0.0
            
            # Обновляем макет, если панель запущена
            if self.live and self.is_running:
                self.live.update(self.generate_layout())
            
            self.logger.debug("Данные торговой панели обновлены")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении данных торговой панели: {e}")
            return False
    
    def render(self) -> Layout:
        """
        Отрисовка торговой панели.
        
        Returns:
            Макет панели
        """
        return self.generate_layout()
    
    def run(self) -> None:
        """Запуск торговой панели в интерактивном режиме."""
        self.start()
        try:
            while self.is_running:
                time.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.logger.info("Торговая панель остановлена пользователем")
        finally:
            self.stop()


def main():
    """Функция для запуска торговой панели в автономном режиме."""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/trading_dashboard.log')
        ]
    )
    
    # Создание и запуск торговой панели
    dashboard = TradingDashboard(config={
        "symbol": "BTCUSDT",
        "initial_balance": 1000.0,
        "current_balance": 1050.0,
        "refresh_rate": 1.0,
        "mode": TRADING_MODES["DRY"],
        "trading_params": {
            "leverage": 20,
            "risk_per_trade": 1.0,
            "stop_loss": 0.5,
            "take_profit": 1.0
        }
    })
    
    # Добавление тестовых данных
    dashboard.update({
        "total_trades": 10,
        "winning_trades": 7,
        "losing_trades": 3,
        "signals": [
            {"indicator": "RSI", "value": "44.03", "signal": "NEUTRAL"},
            {"indicator": "MACD", "value": "-168.6", "signal": "SELL"},
            {"indicator": "Bollinger", "value": "87488.5", "signal": "BUY"},
            {"indicator": "MA Cross", "value": "2432.1", "signal": "NEUTRAL"}
        ],
        "positions": [
            {
                "id": 1,
                "symbol": "BTCUSDT",
                "type": "LONG",
                "size": 0.01,
                "entry_price": 50000.0,
                "current_price": 51000.0,
                "pnl": 10.0,
                "pnl_percent": 2.0
            },
            {
                "id": 2,
                "symbol": "ETHUSDT",
                "type": "SHORT",
                "size": 0.1,
                "entry_price": 3000.0,
                "current_price": 3100.0,
                "pnl": -10.0,
                "pnl_percent": -3.33
            }
        ],
        "recommendation": "Рекомендуется открыть LONG позицию по BTCUSDT",
        "recommendation_color": "green"
    })
    
    # Запуск панели
    dashboard.run()


if __name__ == "__main__":
    main() 