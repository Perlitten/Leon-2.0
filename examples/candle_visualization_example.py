"""
Пример использования визуализатора свечей.

Этот скрипт демонстрирует, как использовать CandleVisualizer для отображения
свечей и индикаторов в консоли.
"""

import asyncio
import logging
import random
import numpy as np
from datetime import datetime

from visualization.candle_visualizer import CandleVisualizer
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


async def run_with_real_data():
    """Запуск визуализатора с реальными данными от Binance."""
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
        
        # Создание визуализатора
        visualizer = CandleVisualizer(config={
            "symbol": symbol,
            "timeframe": timeframe,
            "refresh_rate": 1.0,
            "max_candles": 20,
            "price_precision": 2
        })
        
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
        
        # Обновление данных визуализатора
        visualizer.update({
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
        
        # Запуск визуализатора
        visualizer.run()
        
    finally:
        # Закрытие клиента
        await client.close()


async def run_with_test_data():
    """Запуск визуализатора с тестовыми данными."""
    # Генерация тестовых данных
    candles, indicators = await generate_test_data(50)
    
    # Создание визуализатора
    visualizer = CandleVisualizer(config={
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "refresh_rate": 1.0,
        "max_candles": 20,
        "price_precision": 2
    })
    
    # Обновление данных визуализатора
    visualizer.update({
        "candles": candles,
        "indicators": indicators
    })
    
    # Запуск визуализатора
    visualizer.run()


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