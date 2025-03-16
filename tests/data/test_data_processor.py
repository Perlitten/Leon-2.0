"""
Тесты для модуля data_processor.

Тестирование функциональности класса DataProcessor.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging

# Отключение логирования для тестов
logging.disable(logging.CRITICAL)

# Добавление корневой директории проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.data_processor import DataProcessor
from data.storage import DataStorage


class TestDataProcessor(unittest.TestCase):
    """Тесты для класса DataProcessor."""

    def setUp(self):
        """Настройка тестового окружения."""
        self.storage = DataStorage()
        self.processor = DataProcessor(self.storage)
        
        # Создание тестовых данных
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        self.data = {
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(102, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        }
        
        self.df = pd.DataFrame(self.data, index=dates)
        
        # Добавление пропущенных значений для тестирования
        self.df.loc[self.df.index[10:15], 'close'] = np.nan

    def test_normalize_data(self):
        """Тест нормализации данных."""
        # Тест с указанными столбцами
        columns = ['close', 'volume']
        df_norm = self.processor.normalize_data(self.df, columns=columns)
        
        # Проверка, что нормализованные столбцы добавлены
        self.assertIn('close_norm', df_norm.columns)
        self.assertIn('volume_norm', df_norm.columns)
        
        # Проверка, что значения в диапазоне [0, 1]
        self.assertTrue((df_norm['close_norm'] >= 0).all())
        self.assertTrue((df_norm['close_norm'] <= 1).all())
        self.assertTrue((df_norm['volume_norm'] >= 0).all())
        self.assertTrue((df_norm['volume_norm'] <= 1).all())
        
        # Тест без указания столбцов (все числовые)
        df_norm_all = self.processor.normalize_data(self.df)
        
        # Проверка, что все числовые столбцы нормализованы
        for col in self.df.select_dtypes(include=['number']).columns:
            self.assertIn(f'{col}_norm', df_norm_all.columns)

    def test_standardize_data(self):
        """Тест стандартизации данных."""
        # Тест с указанными столбцами
        columns = ['close', 'volume']
        df_std = self.processor.standardize_data(self.df, columns=columns)
        
        # Проверка, что стандартизованные столбцы добавлены
        self.assertIn('close_std', df_std.columns)
        self.assertIn('volume_std', df_std.columns)
        
        # Проверка, что среднее близко к 0, а стандартное отклонение близко к 1
        # (с учетом пропущенных значений)
        close_values = df_std['close_std'].dropna()
        self.assertAlmostEqual(close_values.mean(), 0, delta=0.1)
        self.assertAlmostEqual(close_values.std(), 1, delta=0.1)
        
        # Тест без указания столбцов (все числовые)
        df_std_all = self.processor.standardize_data(self.df)
        
        # Проверка, что все числовые столбцы стандартизованы
        for col in self.df.select_dtypes(include=['number']).columns:
            self.assertIn(f'{col}_std', df_std_all.columns)

    def test_create_features(self):
        """Тест создания признаков."""
        # Заполнение пропущенных значений для корректной работы
        df_clean = self.df.fillna(method='ffill')
        
        # Создание признаков
        df_features = self.processor.create_features(df_clean)
        
        # Проверка, что признаки добавлены
        expected_features = [
            'price_range', 'price_change', 'price_change_pct',
            'volume_change', 'volume_ma_5', 'volume_ma_10',
            'close_lag_1', 'close_lag_2', 'close_lag_3',
            'sma_5', 'sma_10', 'sma_20',
            'volatility_5', 'volatility_10',
            'sma_5_10_diff', 'sma_10_20_diff',
            'volume_price_ratio'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_features.columns)
        
        # Проверка корректности расчета некоторых признаков
        self.assertTrue((df_features['price_range'] == df_features['high'] - df_features['low']).all())
        self.assertTrue((df_features['price_change'] == df_features['close'] - df_features['open']).all())

    def test_create_target(self):
        """Тест создания целевой переменной."""
        # Заполнение пропущенных значений для корректной работы
        df_clean = self.df.fillna(method='ffill')
        
        # Создание целевой переменной
        periods = 5
        threshold = 0.01
        df_target = self.processor.create_target(df_clean, periods=periods, threshold=threshold)
        
        # Проверка, что целевые переменные добавлены
        self.assertIn(f'target_reg_{periods}', df_target.columns)
        self.assertIn(f'target_cls_{periods}', df_target.columns)
        
        # Проверка, что классификационная переменная имеет только значения -1, 0, 1
        unique_values = df_target[f'target_cls_{periods}'].dropna().unique()
        self.assertTrue(all(val in [-1, 0, 1] for val in unique_values))
        
        # Проверка корректности классификации
        for i in range(len(df_target) - periods):
            if df_target.iloc[i][f'target_reg_{periods}'] > threshold:
                self.assertEqual(df_target.iloc[i][f'target_cls_{periods}'], 1)
            elif df_target.iloc[i][f'target_reg_{periods}'] < -threshold:
                self.assertEqual(df_target.iloc[i][f'target_cls_{periods}'], -1)
            else:
                self.assertEqual(df_target.iloc[i][f'target_cls_{periods}'], 0)

    def test_handle_missing_values(self):
        """Тест обработки пропущенных значений."""
        # Проверка метода 'ffill'
        df_ffill = self.processor.handle_missing_values(self.df, method='ffill')
        self.assertEqual(df_ffill.isna().sum().sum(), 0)
        
        # Проверка метода 'bfill'
        df_bfill = self.processor.handle_missing_values(self.df, method='bfill')
        self.assertEqual(df_bfill.isna().sum().sum(), 0)
        
        # Проверка метода 'interpolate'
        df_interpolate = self.processor.handle_missing_values(self.df, method='interpolate')
        self.assertEqual(df_interpolate.isna().sum().sum(), 0)
        
        # Проверка метода 'drop'
        df_drop = self.processor.handle_missing_values(self.df, method='drop')
        self.assertEqual(df_drop.isna().sum().sum(), 0)
        self.assertLess(len(df_drop), len(self.df))

    def test_remove_outliers(self):
        """Тест удаления выбросов."""
        # Добавление выбросов
        df_with_outliers = self.df.copy()
        df_with_outliers.loc[df_with_outliers.index[5], 'volume'] = 10000  # Большое значение
        df_with_outliers.loc[df_with_outliers.index[10], 'volume'] = 10  # Малое значение
        
        # Удаление выбросов методом IQR
        df_no_outliers_iqr = self.processor.remove_outliers(
            df_with_outliers, columns=['volume'], method='iqr', threshold=1.5
        )
        
        # Проверка, что выбросы удалены
        self.assertLess(len(df_no_outliers_iqr), len(df_with_outliers))
        
        # Удаление выбросов методом Z-score
        df_no_outliers_zscore = self.processor.remove_outliers(
            df_with_outliers, columns=['volume'], method='zscore', threshold=2.0
        )
        
        # Проверка, что выбросы удалены
        self.assertLess(len(df_no_outliers_zscore), len(df_with_outliers))

    def test_split_data(self):
        """Тест разделения данных."""
        # Разделение без перемешивания
        train_size = 0.8
        train_df, test_df = self.processor.split_data(self.df, train_size=train_size, shuffle=False)
        
        # Проверка размеров выборок
        expected_train_size = int(len(self.df) * train_size)
        self.assertEqual(len(train_df), expected_train_size)
        self.assertEqual(len(test_df), len(self.df) - expected_train_size)
        
        # Проверка, что данные не перемешаны
        self.assertTrue((train_df.index == self.df.index[:expected_train_size]).all())
        self.assertTrue((test_df.index == self.df.index[expected_train_size:]).all())
        
        # Разделение с перемешиванием
        train_df_shuffled, test_df_shuffled = self.processor.split_data(
            self.df, train_size=train_size, shuffle=True
        )
        
        # Проверка размеров выборок
        self.assertEqual(len(train_df_shuffled), expected_train_size)
        self.assertEqual(len(test_df_shuffled), len(self.df) - expected_train_size)
        
        # Проверка, что данные перемешаны
        self.assertFalse((train_df_shuffled.index == self.df.index[:expected_train_size]).all())

    def test_prepare_data_for_ml(self):
        """Тест подготовки данных для машинного обучения."""
        # Создание данных с целевой переменной
        df_clean = self.df.fillna(method='ffill')
        df_target = self.processor.create_target(df_clean, periods=5, threshold=0.01)
        
        # Подготовка данных для ML
        target_column = 'target_cls_5'
        X, y = self.processor.prepare_data_for_ml(df_target, target_column=target_column)
        
        # Проверка размеров выборок
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)
        
        # Проверка, что целевая переменная не входит в признаки
        self.assertNotIn(target_column, X.columns)
        
        # Проверка с указанием признаков
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        X_selected, y_selected = self.processor.prepare_data_for_ml(
            df_target, target_column=target_column, feature_columns=feature_columns
        )
        
        # Проверка, что только указанные признаки включены
        self.assertEqual(set(X_selected.columns), set(feature_columns))
        self.assertEqual(len(X_selected), len(y_selected))


if __name__ == '__main__':
    unittest.main() 