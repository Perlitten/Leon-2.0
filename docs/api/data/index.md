# Модуль данных

Модуль `data` предоставляет функциональность для работы с данными в системе Leon Trading Bot. Он включает в себя компоненты для хранения, загрузки и обработки исторических данных, а также для получения и анализа рыночных данных в реальном времени.

## Компоненты

### [Хранилище данных](storage.md)

Класс `DataStorage` отвечает за хранение и управление данными в системе. Он обеспечивает сохранение и загрузку исторических данных, кэширование, а также управление метаданными.

### [Исторические данные](historical_data.md)

Класс `HistoricalDataManager` предоставляет функциональность для загрузки, обработки и анализа исторических торговых данных. Он позволяет получать данные с биржи, обновлять их, экспортировать в CSV и анализировать рыночные тренды.

### [Рыночные данные](market_data.md)

Класс `MarketDataManager` отвечает за получение, обработку и распространение рыночных данных в реальном времени. Он позволяет подписываться на обновления данных по различным символам, получать данные через WebSocket, а также анализировать рыночные данные.

### [Процессор данных](data_processor.md)

Класс `DataProcessor` отвечает за предобработку, трансформацию и подготовку данных для анализа и машинного обучения. Он позволяет нормализовать и стандартизировать данные, создавать признаки и целевые переменные, обрабатывать пропущенные значения и выбросы, а также подготавливать данные для машинного обучения.

## Примеры использования

### Базовый пример работы с данными

```python
import asyncio
import logging
from data.storage import DataStorage
from data.historical_data import HistoricalDataManager
from data.market_data import MarketDataManager
from exchange.binance.client import BinanceClient
from core.config_manager import ConfigManager

async def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    client = BinanceClient(
        api_key=config["binance"]["api_key"],
        api_secret=config["binance"]["api_secret"],
        testnet=config["binance"]["testnet"]
    )
    
    # Инициализация клиента
    await client.initialize()
    
    try:
        # Создание менеджера исторических данных
        historical_data_manager = HistoricalDataManager(storage, client)
        
        # Загрузка исторических данных
        symbol = "BTCUSDT"
        interval = "1h"
        df = await historical_data_manager.load_historical_data(
            symbol=symbol,
            interval=interval,
            start_date="2023-01-01",
            end_date="2023-01-31",
            use_cache=True,
            add_indicators=True
        )
        
        print(f"Загружено {len(df)} свечей для {symbol} {interval}")
        
        # Создание менеджера рыночных данных
        market_data_manager = MarketDataManager(storage)
        
        # Запуск менеджера
        await market_data_manager.start()
        
        # Получение текущего тикера
        ticker = await market_data_manager.get_ticker(client, symbol)
        print(f"Текущая цена {symbol}: {ticker['lastPrice']}")
        
        # Анализ настроения рынка
        sentiment = await market_data_manager.analyze_market_sentiment(symbol)
        print(f"Настроение рынка: {sentiment['sentiment']} (уверенность: {sentiment['confidence']})")
        
        # Остановка менеджера
        await market_data_manager.stop()
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Работа с историческими данными для бэктестинга

```python
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from data.storage import DataStorage
from data.historical_data import HistoricalDataManager
from exchange.binance.client import BinanceClient

async def backtest_data():
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    client = BinanceClient(testnet=True)
    await client.initialize()
    
    try:
        # Создание менеджера исторических данных
        historical_data_manager = HistoricalDataManager(storage, client)
        
        # Получение данных для бэктестинга
        symbol = "BTCUSDT"
        interval = "1h"
        df = await historical_data_manager.get_data_for_backtesting(
            symbol=symbol,
            interval=interval,
            start_date="2023-01-01",
            end_date="2023-03-31",
            indicators=["sma", "ema", "rsi", "macd"]
        )
        
        print(f"Получено {len(df)} свечей для бэктестинга")
        
        # Визуализация данных
        plt.figure(figsize=(12, 8))
        
        # График цены и скользящих средних
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Цена закрытия')
        plt.plot(df.index, df['sma_20'], label='SMA 20')
        plt.plot(df.index, df['ema_50'], label='EMA 50')
        plt.title(f'{symbol} - Цена и индикаторы')
        plt.legend()
        plt.grid(True)
        
        # График RSI
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['rsi_14'], label='RSI 14')
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        plt.title('RSI')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_data.png')
        print("График сохранен в backtest_data.png")
        
        # Экспорт данных в CSV
        historical_data_manager.export_data_to_csv(df, f"{symbol}_{interval}_backtest.csv")
        print(f"Данные экспортированы в {symbol}_{interval}_backtest.csv")
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(backtest_data())
```

### Мониторинг рыночных данных в реальном времени

```python
import asyncio
import logging
import json
from data.storage import DataStorage
from data.market_data import MarketDataManager
from exchange.binance.client import BinanceClient

async def monitor_market():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    client = BinanceClient(testnet=True)
    await client.initialize()
    
    # Создание менеджера рыночных данных
    market_data_manager = MarketDataManager(storage)
    
    # Запуск менеджера
    await market_data_manager.start()
    
    try:
        # Символы для мониторинга
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        # Функция обратного вызова для обновлений
        def on_data_update(data):
            symbol = data['symbol']
            close_price = data['close']
            timestamp = data['timestamp']
            print(f"[{timestamp}] {symbol}: {close_price}")
            
            # Проверка аномалий
            asyncio.create_task(check_anomalies(symbol))
        
        # Функция для проверки аномалий
        async def check_anomalies(symbol):
            anomalies = await market_data_manager.detect_market_anomalies(symbol)
            if anomalies:
                print(f"ВНИМАНИЕ! Обнаружены аномалии для {symbol}:")
                for anomaly in anomalies:
                    print(f"  - {anomaly['type']}: {anomaly['description']}")
        
        # Подписка на обновления
        for symbol in symbols:
            market_data_manager.subscribe(symbol, on_data_update)
            
            # Получение книги ордеров
            order_book = await market_data_manager.get_order_book(client, symbol)
            print(f"\nКнига ордеров {symbol}:")
            print(f"Лучшая цена покупки: {order_book['bids'][0][0]}")
            print(f"Лучшая цена продажи: {order_book['asks'][0][0]}")
            
            # Расчет рыночных метрик
            metrics = await market_data_manager.calculate_market_metrics(symbol)
            print(f"\nМетрики {symbol}:")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        
        # Запуск WebSocket
        await market_data_manager.start_websocket(client, symbols)
        
        # Мониторинг в течение 5 минут
        print("\nНачало мониторинга рынка (5 минут)...")
        await asyncio.sleep(300)
        
    finally:
        # Остановка менеджера
        await market_data_manager.stop()
        
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(monitor_market())

### Подготовка данных для машинного обучения

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from data.storage import DataStorage
from data.historical_data import HistoricalDataManager
from data.data_processor import DataProcessor
from exchange.binance.client import BinanceClient

async def prepare_ml_data():
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    client = BinanceClient(testnet=True)
    await client.initialize()
    
    try:
        # Создание менеджера исторических данных
        historical_data_manager = HistoricalDataManager(storage, client)
        
        # Создание процессора данных
        data_processor = DataProcessor(storage)
        
        # Получение исторических данных
        symbol = "BTCUSDT"
        interval = "1h"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = await historical_data_manager.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=True
        )
        
        print(f"Загружено {len(df)} свечей для {symbol} {interval}")
        
        # Обработка пропущенных значений
        df_clean = data_processor.handle_missing_values(df, method='interpolate')
        
        # Создание дополнительных признаков
        df_features = data_processor.create_features(df_clean)
        
        # Создание целевой переменной (прогноз на 12 часов вперед)
        df_target = data_processor.create_target(df_features, periods=12, threshold=0.01)
        
        # Нормализация данных
        price_columns = ['open', 'high', 'low', 'close']
        df_norm = data_processor.normalize_data(df_target, columns=price_columns)
        
        # Разделение данных на обучающую и тестовую выборки
        train_df, test_df = data_processor.split_data(df_norm, train_size=0.8, shuffle=False)
        
        print(f"Данные разделены: {len(train_df)} строк для обучения, {len(test_df)} строк для тестирования")
        
        # Подготовка данных для машинного обучения
        target_column = 'target_cls_12'  # Целевая переменная для классификации
        
        # Выбор признаков (исключаем целевые переменные и некоторые технические столбцы)
        feature_columns = [col for col in df_norm.columns if not col.startswith('target_') 
                          and not col in ['open_time', 'close_time']]
        
        X_train, y_train = data_processor.prepare_data_for_ml(
            train_df, target_column=target_column, feature_columns=feature_columns
        )
        
        X_test, y_test = data_processor.prepare_data_for_ml(
            test_df, target_column=target_column, feature_columns=feature_columns
        )
        
        print(f"Подготовлены данные для ML:")
        print(f"  - Обучающая выборка: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
        print(f"  - Тестовая выборка: {X_test.shape[0]} строк, {X_test.shape[1]} признаков")
        
        # Распределение классов в целевой переменной
        class_distribution = y_train.value_counts()
        print(f"\nРаспределение классов в обучающей выборке:")
        for cls, count in class_distribution.items():
            print(f"  - Класс {cls}: {count} ({count/len(y_train)*100:.2f}%)")
        
        # Визуализация некоторых признаков
        plt.figure(figsize=(12, 8))
        
        # График цены и индикаторов
        plt.subplot(2, 1, 1)
        plt.plot(df_norm.index, df_norm['close'], label='Цена закрытия')
        plt.plot(df_norm.index, df_norm['sma_20'], label='SMA 20')
        plt.plot(df_norm.index, df_norm['ema_7'], label='EMA 7')
        plt.title(f'{symbol} - Цена и индикаторы')
        plt.legend()
        plt.grid(True)
        
        # График целевой переменной
        plt.subplot(2, 1, 2)
        plt.scatter(df_target.index, df_target[target_column], label=target_column, alpha=0.5)
        plt.title('Целевая переменная')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ml_data_preparation.png')
        print("\nГрафик сохранен в ml_data_preparation.png")
        
        # Сохранение подготовленных данных
        train_df.to_csv(f"{symbol}_{interval}_train.csv")
        test_df.to_csv(f"{symbol}_{interval}_test.csv")
        print(f"\nДанные сохранены в CSV-файлы для дальнейшего использования")
        
    finally:
        # Закрытие клиента
        await client.close()

if __name__ == "__main__":
    asyncio.run(prepare_ml_data()) 