"""
Тесты для модуля storage.

Тестирование функциональности класса DataStorage.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import logging

# Отключение логирования для тестов
logging.disable(logging.CRITICAL)

# Добавление корневой директории проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.storage import DataStorage


class TestDataStorage(unittest.TestCase):
    """Тесты для класса DataStorage."""

    def setUp(self):
        """Настройка тестового окружения."""
        # Создание временной директории для тестов
        self.test_dir = tempfile.mkdtemp()
        
        # Создание поддиректорий для хранения данных
        self.csv_dir = os.path.join(self.test_dir, 'csv')
        self.sqlite_dir = os.path.join(self.test_dir, 'sqlite')
        self.json_dir = os.path.join(self.test_dir, 'json')
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.sqlite_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Создание экземпляра хранилища данных
        self.storage = DataStorage(
            csv_dir=self.csv_dir,
            sqlite_dir=self.sqlite_dir,
            json_dir=self.json_dir
        )
        
        # Создание тестовых данных
        self.symbol = "BTCUSDT"
        self.interval = "1h"
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        self.data = {
            'open_time': [int(d.timestamp() * 1000) for d in dates],
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(102, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates)),
            'close_time': [int((d + timedelta(hours=1)).timestamp() * 1000) for d in dates],
            'quote_asset_volume': np.random.normal(100000, 20000, len(dates)),
            'number_of_trades': np.random.randint(100, 1000, len(dates)),
            'taker_buy_base_asset_volume': np.random.normal(500, 100, len(dates)),
            'taker_buy_quote_asset_volume': np.random.normal(50000, 10000, len(dates)),
            'ignore': np.zeros(len(dates))
        }
        
        self.df = pd.DataFrame(self.data)

    def tearDown(self):
        """Очистка после тестов."""
        # Удаление временной директории
        shutil.rmtree(self.test_dir)

    def test_save_load_kline_data_csv(self):
        """Тест сохранения и загрузки данных в формате CSV."""
        # Сохранение данных в CSV
        self.storage.save_kline_data_csv(self.df, self.symbol, self.interval)
        
        # Проверка, что файл создан
        csv_path = os.path.join(self.csv_dir, f"{self.symbol}_{self.interval}.csv")
        self.assertTrue(os.path.exists(csv_path))
        
        # Загрузка данных из CSV
        loaded_df = self.storage.load_kline_data_csv(self.symbol, self.interval)
        
        # Проверка, что данные загружены корректно
        self.assertEqual(len(loaded_df), len(self.df))
        self.assertTrue(all(col in loaded_df.columns for col in self.df.columns))

    def test_save_load_kline_data_sqlite(self):
        """Тест сохранения и загрузки данных в формате SQLite."""
        # Сохранение данных в SQLite
        self.storage.save_kline_data_sqlite(self.df, self.symbol, self.interval)
        
        # Проверка, что файл создан
        sqlite_path = os.path.join(self.sqlite_dir, f"{self.symbol}.db")
        self.assertTrue(os.path.exists(sqlite_path))
        
        # Загрузка данных из SQLite
        loaded_df = self.storage.load_kline_data_sqlite(self.symbol, self.interval)
        
        # Проверка, что данные загружены корректно
        self.assertEqual(len(loaded_df), len(self.df))
        self.assertTrue(all(col in loaded_df.columns for col in self.df.columns))

    def test_save_load_kline_data_json(self):
        """Тест сохранения и загрузки данных в формате JSON."""
        # Сохранение данных в JSON
        self.storage.save_kline_data_json(self.df, self.symbol, self.interval)
        
        # Проверка, что файл создан
        json_path = os.path.join(self.json_dir, f"{self.symbol}_{self.interval}.json")
        self.assertTrue(os.path.exists(json_path))
        
        # Загрузка данных из JSON
        loaded_df = self.storage.load_kline_data_json(self.symbol, self.interval)
        
        # Проверка, что данные загружены корректно
        self.assertEqual(len(loaded_df), len(self.df))
        self.assertTrue(all(col in loaded_df.columns for col in self.df.columns))

    def test_save_kline(self):
        """Тест сохранения одной свечи."""
        # Создание данных для одной свечи
        kline = {
            'open_time': int(datetime.now().timestamp() * 1000),
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000.0,
            'close_time': int((datetime.now() + timedelta(hours=1)).timestamp() * 1000),
            'quote_asset_volume': 100000.0,
            'number_of_trades': 500,
            'taker_buy_base_asset_volume': 500.0,
            'taker_buy_quote_asset_volume': 50000.0,
            'ignore': 0.0
        }
        
        # Сохранение свечи
        self.storage.save_kline(kline, self.symbol, self.interval)
        
        # Проверка, что файл создан
        sqlite_path = os.path.join(self.sqlite_dir, f"{self.symbol}.db")
        self.assertTrue(os.path.exists(sqlite_path))
        
        # Загрузка данных из SQLite
        loaded_df = self.storage.load_kline_data_sqlite(self.symbol, self.interval)
        
        # Проверка, что данные загружены корректно
        self.assertEqual(len(loaded_df), 1)
        self.assertTrue(all(col in loaded_df.columns for col in kline.keys()))

    def test_get_average_volume(self):
        """Тест получения среднего объема."""
        # Сохранение данных в SQLite
        self.storage.save_kline_data_sqlite(self.df, self.symbol, self.interval)
        
        # Получение среднего объема
        periods = 5
        avg_volume = self.storage.get_average_volume(self.symbol, self.interval, periods)
        
        # Проверка, что средний объем рассчитан корректно
        expected_avg = self.df['volume'].tail(periods).mean()
        self.assertAlmostEqual(avg_volume, expected_avg, delta=0.1)

    def test_add_technical_indicators(self):
        """Тест добавления технических индикаторов."""
        # Добавление технических индикаторов
        df_with_indicators = self.storage.add_technical_indicators(self.df)
        
        # Проверка, что индикаторы добавлены
        expected_indicators = [
            'sma_7', 'sma_25', 'sma_99',
            'ema_7', 'ema_25', 'ema_99',
            'rsi_14', 'macd', 'macdsignal', 'macdhist',
            'upper_band', 'middle_band', 'lower_band'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, df_with_indicators.columns)

    def test_detect_patterns(self):
        """Тест обнаружения паттернов."""
        # Обнаружение паттернов
        df_with_patterns = self.storage.detect_patterns(self.df)
        
        # Проверка, что паттерны добавлены
        expected_patterns = [
            'CDL_DOJI', 'CDL_HAMMER', 'CDL_SHOOTING_STAR',
            'CDL_ENGULFING', 'CDL_MORNING_STAR', 'CDL_EVENING_STAR'
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, df_with_patterns.columns)

    def test_save_load_metadata(self):
        """Тест сохранения и загрузки метаданных."""
        # Создание метаданных
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'start_date': '2023-01-01',
            'end_date': '2023-01-10',
            'count': len(self.df),
            'last_update': datetime.now().isoformat()
        }
        
        # Сохранение метаданных
        self.storage.save_metadata(metadata, self.symbol, self.interval)
        
        # Проверка, что файл создан
        metadata_path = os.path.join(self.json_dir, f"{self.symbol}_{self.interval}_metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        
        # Загрузка метаданных
        loaded_metadata = self.storage.load_metadata(self.symbol, self.interval)
        
        # Проверка, что метаданные загружены корректно
        self.assertEqual(loaded_metadata['symbol'], metadata['symbol'])
        self.assertEqual(loaded_metadata['interval'], metadata['interval'])
        self.assertEqual(loaded_metadata['count'], metadata['count'])


if __name__ == '__main__':
    unittest.main() 