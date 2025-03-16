"""
Пример использования менеджера визуализаторов.

Этот скрипт демонстрирует, как использовать VisualizationManager для управления
различными визуализаторами в системе.
"""

import asyncio
import logging
import time
import random
import numpy as np
from datetime import datetime

from visualization.manager import VisualizationManager
from exchange.binance.client import BinanceClient
from core.config_manager import ConfigManager


async def generate_test_data(count=50, base_price=50000.0, volatility=0.02):
    """
    Генерация тестовых данных для визуализатора.
    
    Args:
        count: Количество свечей
        base_price: Базовая цена
        volatility: Волатильность
        
    Returns:
        Кортеж (свечи, индикаторы)
    """
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
    
    # Генерация индикаторов
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
    
    indicators = {
        "MA(9)": ma_9[-1] if len(ma_9) > 0 else 0,
        "MA(21)": ma_21[-1] if len(ma_21) > 0 else 0,
        "RSI(14)": rsi,
        "Bollinger": {
            "Upper": upper_band,
            "Middle": ma_20,
            "Lower": lower_band
        }
    }
    
    return candles, indicators


async def generate_test_trading_data():
    """
    Генерация тестовых данных для торговой панели.
    
    Returns:
        Словарь с данными для торговой панели
    """
    # Генерация баланса
    balance = random.uniform(5000.0, 15000.0)
    
    # Генерация позиций
    positions = []
    for i in range(random.randint(0, 3)):
        position = {
            "id": f"pos-{i+1}",
            "symbol": "BTCUSDT",
            "direction": "BUY" if random.random() > 0.5 else "SELL",
            "entry_price": random.uniform(45000.0, 55000.0),
            "size": random.uniform(0.001, 0.01),
            "current_price": random.uniform(45000.0, 55000.0),
            "pnl": random.uniform(-500.0, 500.0),
            "pnl_percent": random.uniform(-5.0, 5.0),
            "timestamp": datetime.now().timestamp() * 1000 - random.randint(0, 86400000)
        }
        positions.append(position)
    
    # Генерация ордеров
    orders = []
    for i in range(random.randint(0, 2)):
        order = {
            "id": f"order-{i+1}",
            "symbol": "BTCUSDT",
            "direction": "BUY" if random.random() > 0.5 else "SELL",
            "type": "LIMIT",
            "price": random.uniform(45000.0, 55000.0),
            "size": random.uniform(0.001, 0.01),
            "status": "OPEN",
            "timestamp": datetime.now().timestamp() * 1000 - random.randint(0, 86400000)
        }
        orders.append(order)
    
    # Генерация сигналов
    signals = []
    for i in range(random.randint(0, 3)):
        signal = {
            "id": f"signal-{i+1}",
            "symbol": "BTCUSDT",
            "action": "BUY" if random.random() > 0.5 else "SELL",
            "price": random.uniform(45000.0, 55000.0),
            "confidence": random.uniform(0.5, 1.0),
            "timestamp": datetime.now().timestamp() * 1000 - random.randint(0, 86400000)
        }
        signals.append(signal)
    
    return {
        "balance": balance,
        "positions": positions,
        "orders": orders,
        "signals": signals
    }


async def run_with_real_data():
    """Запуск визуализаторов с реальными данными от Binance."""
    # Загрузка конфигурации
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Создание клиента Binance
    client = BinanceClient(
        api_key=config["binance"]["api_key"],
        api_secret=config["binance"]["api_secret"],
        testnet=config["binance"]["testnet"]
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Получение исторических данных
        symbol = config["trading"]["symbol"]
        timeframe = config["trading"]["timeframe"]
        klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=50)
        
        # Создание менеджера визуализаторов
        visualization_manager = VisualizationManager()
        
        # Создание и запуск визуализатора свечей
        visualization_manager.create_and_start_visualizer(
            visualizer_type="candle_visualizer",
            name="candles",
            config={
                "symbol": symbol,
                "timeframe": timeframe,
                "refresh_rate": 1.0,
                "max_candles": 20,
                "price_precision": 2
            },
            background=True
        )
        
        # Создание и запуск торговой панели
        visualization_manager.create_and_start_visualizer(
            visualizer_type="trading_dashboard",
            name="trading",
            config={
                "symbol": symbol,
                "mode": config["trading"]["mode"],
                "refresh_rate": 1.0
            },
            background=True
        )
        
        # Расчет индикаторов
        closes = [float(candle[4]) for candle in klines]
        
        # Простые скользящие средние
        ma_9 = np.convolve(closes, np.ones(9)/9, mode='valid')
        ma_21 = np.convolve(closes, np.ones(21)/21, mode='valid')
        
        # Bollinger Bands
        ma_20 = np.convolve(closes, np.ones(20)/20, mode='valid')[-1] if len(closes) >= 20 else 0
        std_20 = np.std(closes[-20:]) if len(closes) >= 20 else 0
        upper_band = ma_20 + 2 * std_20
        lower_band = ma_20 - 2 * std_20
        
        # Обновление данных визуализаторов
        visualization_manager.update_visualizer("candles", {
            "candles": klines,
            "indicators": {
                "MA(9)": ma_9[-1] if len(ma_9) > 0 else 0,
                "MA(21)": ma_21[-1] if len(ma_21) > 0 else 0,
                "Bollinger": {
                    "Upper": upper_band,
                    "Middle": ma_20,
                    "Lower": lower_band
                }
            }
        })
        
        # Получение информации об аккаунте
        account_info = await client.get_account_info()
        
        # Получение баланса USDT
        usdt_balance = next((float(asset["free"]) + float(asset["locked"]) 
                           for asset in account_info["balances"] 
                           if asset["asset"] == "USDT"), 0.0)
        
        # Обновление торговой панели
        visualization_manager.update_visualizer("trading", {
            "balance": usdt_balance,
            "positions": [],
            "orders": [],
            "signals": []
        })
        
        # Ожидание 30 секунд для просмотра визуализаций
        for i in range(30):
            # Обновление данных каждые 5 секунд
            if i % 5 == 0:
                # Получение новых данных
                new_klines = await client.get_klines(symbol=symbol, interval=timeframe, limit=50)
                
                # Обновление визуализаторов
                visualization_manager.update_visualizer("candles", {
                    "candles": new_klines
                })
                
                # Генерация тестовых данных для торговой панели
                trading_data = await generate_test_trading_data()
                trading_data["balance"] = usdt_balance  # Использование реального баланса
                
                visualization_manager.update_visualizer("trading", trading_data)
            
            time.sleep(1)
        
        # Остановка всех визуализаторов
        visualization_manager.stop_all_visualizers()
        
    finally:
        # Закрытие клиента
        await client.close()


async def run_with_test_data():
    """Запуск визуализаторов с тестовыми данными."""
    # Генерация тестовых данных
    candles, indicators = await generate_test_data(50)
    trading_data = await generate_test_trading_data()
    
    # Создание менеджера визуализаторов
    visualization_manager = VisualizationManager()
    
    # Создание и запуск визуализатора свечей
    visualization_manager.create_and_start_visualizer(
        visualizer_type="candle_visualizer",
        name="candles",
        config={
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "refresh_rate": 1.0,
            "max_candles": 20,
            "price_precision": 2
        },
        background=True
    )
    
    # Создание и запуск торговой панели
    visualization_manager.create_and_start_visualizer(
        visualizer_type="trading_dashboard",
        name="trading",
        config={
            "symbol": "BTCUSDT",
            "mode": "DRY",
            "refresh_rate": 1.0
        },
        background=True
    )
    
    # Обновление данных визуализаторов
    visualization_manager.update_visualizer("candles", {
        "candles": candles,
        "indicators": indicators
    })
    
    visualization_manager.update_visualizer("trading", trading_data)
    
    # Ожидание 30 секунд для просмотра визуализаций
    for i in range(30):
        # Обновление данных каждые 5 секунд
        if i % 5 == 0:
            # Генерация новых тестовых данных
            new_candles, new_indicators = await generate_test_data(50)
            new_trading_data = await generate_test_trading_data()
            
            # Обновление визуализаторов
            visualization_manager.update_visualizer("candles", {
                "candles": new_candles,
                "indicators": new_indicators
            })
            
            visualization_manager.update_visualizer("trading", new_trading_data)
        
        time.sleep(1)
    
    # Остановка всех визуализаторов
    visualization_manager.stop_all_visualizers()


async def main():
    """Основная функция."""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Выбор режима работы
    use_real_data = False
    
    if use_real_data:
        await run_with_real_data()
    else:
        await run_with_test_data()


if __name__ == "__main__":
    asyncio.run(main()) 