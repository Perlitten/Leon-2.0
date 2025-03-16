"""
Модуль для работы с историческими данными.

Предоставляет функциональность для загрузки, обработки и анализа
исторических данных для торговли.
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from data.storage import DataStorage
from exchange.binance.client import BinanceClient


class HistoricalDataManager:
    """
    Менеджер исторических данных.
    
    Отвечает за загрузку, обработку и анализ исторических данных для торговли.
    """
    
    def __init__(self, storage: Optional[DataStorage] = None, client: Optional[BinanceClient] = None):
        """
        Инициализация менеджера исторических данных.
        
        Args:
            storage: Хранилище данных
            client: Клиент Binance
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = storage or DataStorage()
        self.client = client
        self.logger.info("Инициализирован менеджер исторических данных")
    
    async def set_client(self, client: BinanceClient) -> None:
        """
        Установка клиента Binance.
        
        Args:
            client: Клиент Binance
        """
        self.client = client
        self.logger.info("Установлен клиент Binance")
    
    async def load_historical_data(self, symbol: str, interval: str, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None,
                                 use_cache: bool = True,
                                 add_indicators: bool = True,
                                 detect_patterns: bool = False) -> pd.DataFrame:
        """
        Загрузка исторических данных.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            use_cache: Использовать кэш
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            
        Returns:
            DataFrame с историческими данными
        """
        if use_cache:
            # Попытка загрузки данных из кэша
            try:
                df = await self.storage.load_klines_from_sqlite(symbol, interval, start_date, end_date)
                if not df.empty:
                    self.logger.info(f"Загружены данные из кэша для {symbol} {interval}")
                    
                    # Проверка на пропущенные интервалы
                    missing_intervals = self.storage.get_missing_intervals(df, interval)
                    if missing_intervals:
                        self.logger.info(f"Обнаружены пропущенные интервалы: {missing_intervals}")
                        # Загрузка пропущенных данных
                        for start, end in missing_intervals:
                            await self._load_and_save_klines(symbol, interval, start, end)
                        
                        # Повторная загрузка данных из кэша
                        df = await self.storage.load_klines_from_sqlite(symbol, interval, start_date, end_date)
                    
                    # Добавление индикаторов и паттернов
                    if add_indicators:
                        df = self.storage.add_technical_indicators(df)
                    if detect_patterns:
                        df = self.storage.detect_patterns(df)
                    
                    return df
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке данных из кэша: {e}")
        
        # Если данные не найдены в кэше или use_cache=False, загрузка с Binance
        if not self.client:
            raise ValueError("Клиент Binance не установлен")
        
        return await self._load_and_save_klines(symbol, interval, start_date, end_date, add_indicators, detect_patterns)
    
    async def _load_and_save_klines(self, symbol: str, interval: str,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  add_indicators: bool = True,
                                  detect_patterns: bool = False) -> pd.DataFrame:
        """
        Загрузка и сохранение свечей.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            
        Returns:
            DataFrame с историческими данными
        """
        # Преобразование дат в миллисекунды для Binance API
        start_ms = int(start_date.timestamp() * 1000) if start_date else None
        end_ms = int(end_date.timestamp() * 1000) if end_date else None
        
        # Загрузка данных с Binance
        klines = await self.client.get_historical_klines(symbol, interval, start_ms, end_ms)
        
        if not klines:
            self.logger.warning(f"Нет данных для {symbol} {interval}")
            return pd.DataFrame()
        
        # Преобразование в DataFrame
        df = self.storage.klines_to_dataframe(klines)
        
        # Сохранение данных в кэш
        await self.storage.save_klines_to_sqlite(symbol, interval, df)
        
        # Добавление индикаторов и паттернов
        if add_indicators:
            df = self.storage.add_technical_indicators(df)
        if detect_patterns:
            df = self.storage.detect_patterns(df)
        
        self.logger.info(f"Загружены и сохранены данные для {symbol} {interval}")
        return df
    
    async def update_historical_data(self, symbol: str, interval: str,
                                   days_to_update: int = 1,
                                   add_indicators: bool = True,
                                   detect_patterns: bool = False) -> pd.DataFrame:
        """
        Обновление исторических данных.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            days_to_update: Количество дней для обновления
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            
        Returns:
            DataFrame с обновленными историческими данными
        """
        # Определение начальной даты для обновления
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_update)
        
        # Загрузка данных
        return await self.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=False, add_indicators=add_indicators, detect_patterns=detect_patterns
        )
    
    async def get_latest_data(self, symbol: str, interval: str, limit: int = 100,
                            add_indicators: bool = True,
                            detect_patterns: bool = False) -> pd.DataFrame:
        """
        Получение последних данных.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            limit: Количество свечей
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            
        Returns:
            DataFrame с последними данными
        """
        if not self.client:
            raise ValueError("Клиент Binance не установлен")
        
        # Загрузка последних свечей
        klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not klines:
            self.logger.warning(f"Нет данных для {symbol} {interval}")
            return pd.DataFrame()
        
        # Преобразование в DataFrame
        df = self.storage.klines_to_dataframe(klines)
        
        # Добавление индикаторов и паттернов
        if add_indicators:
            df = self.storage.add_technical_indicators(df)
        if detect_patterns:
            df = self.storage.detect_patterns(df)
        
        self.logger.info(f"Получены последние данные для {symbol} {interval}")
        return df
    
    async def get_data_for_backtesting(self, symbol: str, interval: str,
                                     start_date: datetime, end_date: datetime,
                                     add_indicators: bool = True,
                                     detect_patterns: bool = False) -> pd.DataFrame:
        """
        Получение данных для бэктестинга.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            
        Returns:
            DataFrame с данными для бэктестинга
        """
        # Загрузка данных
        df = await self.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=add_indicators, detect_patterns=detect_patterns
        )
        
        if df.empty:
            self.logger.warning(f"Нет данных для бэктестинга {symbol} {interval}")
            return df
        
        # Проверка на достаточное количество данных
        if len(df) < 100:
            self.logger.warning(f"Недостаточно данных для бэктестинга {symbol} {interval}")
        
        self.logger.info(f"Получены данные для бэктестинга {symbol} {interval}")
        return df
    
    async def get_data_for_training(self, symbol: str, interval: str,
                                  start_date: datetime, end_date: datetime,
                                  add_indicators: bool = True,
                                  detect_patterns: bool = False,
                                  train_test_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Получение данных для обучения ML моделей.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            add_indicators: Добавить технические индикаторы
            detect_patterns: Обнаружить паттерны
            train_test_split: Соотношение обучающей и тестовой выборок
            
        Returns:
            Кортеж (обучающая выборка, тестовая выборка)
        """
        # Загрузка данных
        df = await self.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=add_indicators, detect_patterns=detect_patterns
        )
        
        if df.empty:
            self.logger.warning(f"Нет данных для обучения {symbol} {interval}")
            return df, pd.DataFrame()
        
        # Разделение на обучающую и тестовую выборки
        train_size = int(len(df) * train_test_split)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        self.logger.info(f"Получены данные для обучения {symbol} {interval}")
        return train_df, test_df
    
    async def export_data_to_csv(self, symbol: str, interval: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> str:
        """
        Экспорт данных в CSV.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Путь к CSV файлу
        """
        # Загрузка данных
        df = await self.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=True, detect_patterns=True
        )
        
        if df.empty:
            self.logger.warning(f"Нет данных для экспорта {symbol} {interval}")
            return ""
        
        # Сохранение в CSV
        csv_path = await self.storage.save_klines_to_csv(symbol, interval, df)
        
        self.logger.info(f"Данные экспортированы в CSV: {csv_path}")
        return csv_path
    
    async def import_data_from_csv(self, csv_path: str, symbol: str, interval: str) -> pd.DataFrame:
        """
        Импорт данных из CSV.
        
        Args:
            csv_path: Путь к CSV файлу
            symbol: Символ торговой пары
            interval: Интервал свечей
            
        Returns:
            DataFrame с импортированными данными
        """
        try:
            # Загрузка данных из CSV
            df = pd.read_csv(csv_path)
            
            # Преобразование столбца времени
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Сохранение в SQLite
            await self.storage.save_klines_to_sqlite(symbol, interval, df)
            
            self.logger.info(f"Данные импортированы из CSV: {csv_path}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка при импорте данных из CSV: {e}")
            return pd.DataFrame()
    
    async def analyze_market_data(self, symbol: str, interval: str, days: int = 30) -> Dict[str, Any]:
        """
        Анализ рыночных данных.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            days: Количество дней для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        # Определение дат
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Загрузка данных
        df = await self.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=True, detect_patterns=True
        )
        
        if df.empty:
            self.logger.warning(f"Нет данных для анализа {symbol} {interval}")
            return {}
        
        # Анализ данных
        analysis = {
            "symbol": symbol,
            "interval": interval,
            "period": f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
            "data_points": len(df),
            "price": {
                "current": df['close'].iloc[-1],
                "change_1d": (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) > 1 else 0,
                "change_7d": (df['close'].iloc[-1] / df['close'].iloc[-7] - 1) * 100 if len(df) > 7 else 0,
                "change_30d": (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                "high": df['high'].max(),
                "low": df['low'].min(),
                "volatility": df['close'].pct_change().std() * 100
            },
            "volume": {
                "current": df['volume'].iloc[-1],
                "average": df['volume'].mean(),
                "max": df['volume'].max()
            },
            "indicators": {
                "rsi": df['rsi'].iloc[-1] if 'rsi' in df.columns else None,
                "macd": df['macd'].iloc[-1] if 'macd' in df.columns else None,
                "macd_signal": df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else None,
                "macd_hist": df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else None,
                "ma_50": df['ma_50'].iloc[-1] if 'ma_50' in df.columns else None,
                "ma_200": df['ma_200'].iloc[-1] if 'ma_200' in df.columns else None
            },
            "patterns": {}
        }
        
        # Добавление информации о паттернах
        pattern_columns = [col for col in df.columns if col.startswith('pattern_')]
        for col in pattern_columns:
            pattern_name = col.replace('pattern_', '')
            pattern_count = df[col].sum()
            if pattern_count > 0:
                analysis["patterns"][pattern_name] = pattern_count
        
        self.logger.info(f"Выполнен анализ рыночных данных для {symbol} {interval}")
        return analysis


async def main():
    """Пример использования менеджера исторических данных."""
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание хранилища данных
    storage = DataStorage()
    
    # Создание клиента Binance
    from core.config_manager import ConfigManager
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
        data_manager = HistoricalDataManager(storage, client)
        
        # Загрузка исторических данных
        symbol = "BTCUSDT"
        interval = "1h"
        days = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await data_manager.load_historical_data(
            symbol, interval, start_date, end_date, 
            use_cache=True, add_indicators=True, detect_patterns=True
        )
        
        print(f"Загружено {len(df)} свечей для {symbol} {interval}")
        print(df.head())
        
        # Анализ рыночных данных
        analysis = await data_manager.analyze_market_data(symbol, interval, days)
        print("\nАнализ рыночных данных:")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        
    finally:
        # Закрытие клиента
        await client.close()


if __name__ == "__main__":
    import json
    asyncio.run(main()) 