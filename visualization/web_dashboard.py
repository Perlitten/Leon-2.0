"""
Модуль веб-визуализации для Leon Trading Bot.

Предоставляет веб-интерфейс для отображения торговой информации
с использованием Flask и Plotly.
"""

import logging
import threading
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.base import WebVisualizer
from core.constants import TRADING_MODES


class WebDashboard(WebVisualizer):
    """
    Веб-панель для отображения торговой информации.
    
    Использует Flask для создания веб-сервера и Plotly для
    визуализации графиков и диаграмм.
    """
    
    def __init__(self, name: str = "web_dashboard", config: Dict[str, Any] = None):
        """
        Инициализация веб-панели.
        
        Args:
            name: Имя визуализатора
            config: Конфигурация визуализатора
        """
        super().__init__(name, config)
        
        # Инициализация Flask
        self.app = Flask(__name__, 
                         template_folder="templates",
                         static_folder="static")
        
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
            "recommendation_color": "warning",
            "price_history": [],
            "balance_history": [],
            "trade_history": []
        }
        
        # Настройка маршрутов
        self._setup_routes()
        
        # Поток для веб-сервера
        self.server_thread = None
        
        self.logger.info(f"Веб-панель {name} инициализирована")
    
    def _setup_routes(self):
        """Настройка маршрутов Flask."""
        
        @self.app.route('/')
        def index():
            """Главная страница."""
            return render_template('index.html', 
                                  symbol=self.data["symbol"],
                                  mode=self.data["mode"])
        
        @self.app.route('/api/data')
        def get_data():
            """API для получения данных."""
            return jsonify(self.data)
        
        @self.app.route('/api/chart/price')
        def get_price_chart():
            """API для получения графика цены."""
            if not self.data["price_history"]:
                return jsonify({"error": "Нет данных о ценах"})
            
            # Создание графика
            fig = go.Figure()
            
            # Добавление линии цены
            timestamps = [item["timestamp"] for item in self.data["price_history"]]
            prices = [item["price"] for item in self.data["price_history"]]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=prices,
                mode='lines',
                name='Цена'
            ))
            
            # Добавление точек входа и выхода
            for trade in self.data["trade_history"]:
                if trade["type"] == "LONG":
                    fig.add_trace(go.Scatter(
                        x=[trade["entry_time"]],
                        y=[trade["entry_price"]],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name=f'Вход LONG #{trade["id"]}'
                    ))
                    
                    if "exit_time" in trade and "exit_price" in trade:
                        fig.add_trace(go.Scatter(
                            x=[trade["exit_time"]],
                            y=[trade["exit_price"]],
                            mode='markers',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            name=f'Выход LONG #{trade["id"]}'
                        ))
                else:  # SHORT
                    fig.add_trace(go.Scatter(
                        x=[trade["entry_time"]],
                        y=[trade["entry_price"]],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name=f'Вход SHORT #{trade["id"]}'
                    ))
                    
                    if "exit_time" in trade and "exit_price" in trade:
                        fig.add_trace(go.Scatter(
                            x=[trade["exit_time"]],
                            y=[trade["exit_price"]],
                            mode='markers',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            name=f'Выход SHORT #{trade["id"]}'
                        ))
            
            # Настройка макета
            fig.update_layout(
                title=f'График цены {self.data["symbol"]}',
                xaxis_title='Время',
                yaxis_title='Цена',
                template='plotly_dark'
            )
            
            return jsonify(fig.to_dict())
        
        @self.app.route('/api/chart/balance')
        def get_balance_chart():
            """API для получения графика баланса."""
            if not self.data["balance_history"]:
                return jsonify({"error": "Нет данных о балансе"})
            
            # Создание графика
            fig = go.Figure()
            
            # Добавление линии баланса
            timestamps = [item["timestamp"] for item in self.data["balance_history"]]
            balances = [item["balance"] for item in self.data["balance_history"]]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=balances,
                mode='lines',
                name='Баланс',
                line=dict(color='green')
            ))
            
            # Добавление начального баланса
            fig.add_trace(go.Scatter(
                x=[timestamps[0], timestamps[-1]],
                y=[self.data["initial_balance"], self.data["initial_balance"]],
                mode='lines',
                name='Начальный баланс',
                line=dict(color='gray', dash='dash')
            ))
            
            # Настройка макета
            fig.update_layout(
                title='График баланса',
                xaxis_title='Время',
                yaxis_title='Баланс',
                template='plotly_dark'
            )
            
            return jsonify(fig.to_dict())
        
        @self.app.route('/api/positions')
        def get_positions():
            """API для получения позиций."""
            return jsonify(self.data["positions"])
        
        @self.app.route('/api/signals')
        def get_signals():
            """API для получения сигналов."""
            return jsonify(self.data["signals"])
        
        @self.app.route('/api/stats')
        def get_stats():
            """API для получения статистики."""
            win_rate = 0.0
            if self.data["total_trades"] > 0:
                win_rate = (self.data["winning_trades"] / self.data["total_trades"]) * 100
            
            return jsonify({
                "initial_balance": self.data["initial_balance"],
                "current_balance": self.data["current_balance"],
                "profit": self.data["profit"],
                "profit_percent": self.data["profit_percent"],
                "total_trades": self.data["total_trades"],
                "winning_trades": self.data["winning_trades"],
                "losing_trades": self.data["losing_trades"],
                "win_rate": win_rate
            })
        
        @self.app.route('/api/recommendation')
        def get_recommendation():
            """API для получения рекомендации."""
            return jsonify({
                "recommendation": self.data["recommendation"],
                "color": self.data["recommendation_color"]
            })
        
        @self.app.route('/api/trading_params')
        def get_trading_params():
            """API для получения параметров торговли."""
            return jsonify(self.data["trading_params"])
        
        self.logger.debug("Маршруты Flask настроены")
    
    def start(self) -> bool:
        """
        Запуск веб-панели.
        
        Returns:
            True, если панель успешно запущена, иначе False
        """
        try:
            super().start()
            
            # Запуск Flask в отдельном потоке
            self.server_thread = threading.Thread(
                target=self.app.run,
                kwargs={
                    'host': self.host,
                    'port': self.port,
                    'debug': self.debug,
                    'use_reloader': False
                }
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.logger.info(f"Веб-панель запущена на http://{self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при запуске веб-панели: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Остановка веб-панели.
        
        Returns:
            True, если панель успешно остановлена, иначе False
        """
        try:
            # Остановка Flask не требуется, так как поток демон
            super().stop()
            self.logger.info("Веб-панель остановлена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при остановке веб-панели: {e}")
            return False
    
    def update(self, data: Dict[str, Any]) -> bool:
        """
        Обновление данных веб-панели.
        
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
            
            # Обновление истории цен
            if "current_price" in data and "symbol" in data:
                self.data["price_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": data["symbol"],
                    "price": data["current_price"]
                })
                
                # Ограничение истории цен
                max_history = self.config.get("max_price_history", 1000)
                if len(self.data["price_history"]) > max_history:
                    self.data["price_history"] = self.data["price_history"][-max_history:]
            
            # Обновление истории баланса
            if "current_balance" in data:
                self.data["balance_history"].append({
                    "timestamp": datetime.now().isoformat(),
                    "balance": data["current_balance"]
                })
                
                # Ограничение истории баланса
                max_history = self.config.get("max_balance_history", 1000)
                if len(self.data["balance_history"]) > max_history:
                    self.data["balance_history"] = self.data["balance_history"][-max_history:]
            
            # Обновление истории сделок
            if "new_trade" in data:
                self.data["trade_history"].append(data["new_trade"])
                
                # Ограничение истории сделок
                max_history = self.config.get("max_trade_history", 100)
                if len(self.data["trade_history"]) > max_history:
                    self.data["trade_history"] = self.data["trade_history"][-max_history:]
            
            self.logger.debug("Данные веб-панели обновлены")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка при обновлении данных веб-панели: {e}")
            return False
    
    def render(self) -> Dict[str, Any]:
        """
        Отрисовка веб-панели.
        
        Returns:
            Данные для отправки клиенту
        """
        return self.data


def main():
    """Функция для запуска веб-панели в автономном режиме."""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/web_dashboard.log')
        ]
    )
    
    # Создание и запуск веб-панели
    dashboard = WebDashboard(config={
        "symbol": "BTCUSDT",
        "initial_balance": 1000.0,
        "current_balance": 1050.0,
        "refresh_rate": 1.0,
        "mode": TRADING_MODES["DRY"],
        "host": "127.0.0.1",
        "port": 8080,
        "debug": True,
        "trading_params": {
            "leverage": 20,
            "risk_per_trade": 1.0,
            "stop_loss": 0.5,
            "take_profit": 1.0
        }
    })
    
    # Запуск панели
    dashboard.start()
    
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
        "recommendation_color": "success"
    })
    
    # Симуляция обновления цены
    for i in range(100):
        price = 50000.0 + i * 10
        dashboard.update({
            "symbol": "BTCUSDT",
            "current_price": price,
            "current_balance": 1000.0 + i
        })
        time.sleep(0.1)
    
    # Ожидание завершения
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()


if __name__ == "__main__":
    main() 