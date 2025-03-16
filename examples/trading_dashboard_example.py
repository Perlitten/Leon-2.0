"""
Пример использования торговой панели Leon Trading Bot.
"""

import logging
import time
import sys
import os
import random
from datetime import datetime

# Добавление корневой директории проекта в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.trading_dashboard import TradingDashboard
from core.constants import TRADING_MODES


def setup_logging():
    """Настройка логирования."""
    # Создание директории для логов, если она не существует
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/trading_dashboard_example.log')
        ]
    )


def simulate_trading(dashboard):
    """
    Симуляция торговли для демонстрации работы панели.
    
    Args:
        dashboard: Экземпляр торговой панели
    """
    logger = logging.getLogger(__name__)
    
    # Начальные данные
    initial_balance = 1000.0
    current_balance = initial_balance
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    positions = []
    
    # Обновление начальных данных
    dashboard.update({
        "initial_balance": initial_balance,
        "current_balance": current_balance,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "positions": positions
    })
    
    # Симуляция торговли
    try:
        while dashboard.is_running:
            # Обновление сигналов
            signals = [
                {
                    "indicator": "RSI",
                    "value": f"{random.uniform(30, 70):.2f}",
                    "signal": random.choice(["BUY", "SELL", "NEUTRAL"])
                },
                {
                    "indicator": "MACD",
                    "value": f"{random.uniform(-200, 200):.2f}",
                    "signal": random.choice(["BUY", "SELL", "NEUTRAL"])
                },
                {
                    "indicator": "Bollinger",
                    "value": f"{random.uniform(50000, 100000):.2f}",
                    "signal": random.choice(["BUY", "SELL", "NEUTRAL"])
                },
                {
                    "indicator": "MA Cross",
                    "value": f"{random.uniform(2000, 3000):.2f}",
                    "signal": random.choice(["BUY", "SELL", "NEUTRAL"])
                }
            ]
            
            # Обновление рекомендации
            buy_signals = sum(1 for s in signals if s["signal"] == "BUY")
            sell_signals = sum(1 for s in signals if s["signal"] == "SELL")
            
            if buy_signals > sell_signals:
                recommendation = "Рекомендуется открыть LONG позицию"
                recommendation_color = "green"
            elif sell_signals > buy_signals:
                recommendation = "Рекомендуется открыть SHORT позицию"
                recommendation_color = "red"
            else:
                recommendation = "Лучше воздержаться от входа в рынок"
                recommendation_color = "yellow"
            
            # Случайное изменение баланса
            if random.random() < 0.2:  # 20% шанс на новую сделку
                trade_profit = random.uniform(-50, 100)
                current_balance += trade_profit
                total_trades += 1
                
                if trade_profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                
                logger.info(f"Новая сделка: {'прибыль' if trade_profit > 0 else 'убыток'} ${abs(trade_profit):.2f}")
            
            # Случайное добавление/удаление позиций
            if random.random() < 0.1:  # 10% шанс на новую позицию
                position_type = random.choice(["LONG", "SHORT"])
                symbol = random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
                size = random.uniform(0.01, 0.5)
                entry_price = random.uniform(1000, 60000)
                
                position = {
                    "id": len(positions) + 1,
                    "symbol": symbol,
                    "type": position_type,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": entry_price,
                    "pnl": 0.0,
                    "pnl_percent": 0.0
                }
                
                positions.append(position)
                logger.info(f"Открыта новая позиция: {position_type} {symbol}")
            
            # Обновление существующих позиций
            for position in positions:
                # Случайное изменение текущей цены
                price_change_percent = random.uniform(-2, 2)
                new_price = position["entry_price"] * (1 + price_change_percent / 100)
                
                # Расчет P/L
                if position["type"] == "LONG":
                    pnl = (new_price - position["entry_price"]) * position["size"]
                    pnl_percent = (new_price / position["entry_price"] - 1) * 100
                else:  # SHORT
                    pnl = (position["entry_price"] - new_price) * position["size"]
                    pnl_percent = (position["entry_price"] / new_price - 1) * 100
                
                # Обновление позиции
                position["current_price"] = new_price
                position["pnl"] = pnl
                position["pnl_percent"] = pnl_percent
            
            # Удаление закрытых позиций
            if positions and random.random() < 0.1:  # 10% шанс на закрытие позиции
                closed_position = positions.pop(0)
                logger.info(f"Закрыта позиция: {closed_position['type']} {closed_position['symbol']}")
            
            # Обновление данных на панели
            dashboard.update({
                "current_balance": current_balance,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "positions": positions,
                "signals": signals,
                "recommendation": recommendation,
                "recommendation_color": recommendation_color
            })
            
            # Пауза перед следующим обновлением
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Симуляция торговли остановлена пользователем")


def main():
    """Основная функция примера."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Запуск примера использования торговой панели")
    
    # Создание торговой панели
    dashboard = TradingDashboard(config={
        "symbol": "BTCUSDT",
        "mode": TRADING_MODES["DRY"],
        "refresh_rate": 1.0,
        "trading_params": {
            "leverage": 20,
            "risk_per_trade": 1.0,
            "stop_loss": 0.5,
            "take_profit": 1.0
        }
    })
    
    # Запуск панели
    dashboard.start()
    
    try:
        # Запуск симуляции торговли
        simulate_trading(dashboard)
    except Exception as e:
        logger.error(f"Ошибка при выполнении примера: {e}")
    finally:
        # Остановка панели
        dashboard.stop()
    
    logger.info("Пример использования торговой панели завершен")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 