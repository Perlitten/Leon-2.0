"""
Модуль для обработки и преобразования данных.

Предоставляет функциональность для предобработки, трансформации
и подготовки данных для анализа и машинного обучения.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from data.storage import DataStorage


class DataProcessor:
    """
    Класс для обработки и преобразования данных.
    
    Отвечает за предобработку, трансформацию и подготовку данных
    для анализа и машинного обучения.
    """
    
    def __init__(self, storage: Optional[DataStorage] = None):
        """
        Инициализация процессора данных.
        
        Args:
            storage: Хранилище данных
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = storage or DataStorage()
        self.logger.info("Инициализирован процессор данных")
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Нормализация данных (масштабирование в диапазон [0, 1]).
        
        Args:
            df: DataFrame с данными
            columns: Список столбцов для нормализации. Если None, нормализуются все числовые столбцы.
            
        Returns:
            DataFrame с нормализованными данными
        """
        result = df.copy()
        
        if columns is None:
            # Выбор всех числовых столбцов
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if col in result.columns:
                min_val = result[col].min()
                max_val = result[col].max()
                
                if max_val > min_val:
                    result[f"{col}_norm"] = (result[col] - min_val) / (max_val - min_val)
                else:
                    result[f"{col}_norm"] = 0.5  # Если все значения одинаковые
                    
                self.logger.debug(f"Нормализован столбец {col}")
            else:
                self.logger.warning(f"Столбец {col} не найден в DataFrame")
        
        return result
    
    def standardize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Стандартизация данных (z-score нормализация).
        
        Args:
            df: DataFrame с данными
            columns: Список столбцов для стандартизации. Если None, стандартизуются все числовые столбцы.
            
        Returns:
            DataFrame со стандартизованными данными
        """
        result = df.copy()
        
        if columns is None:
            # Выбор всех числовых столбцов
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if col in result.columns:
                mean_val = result[col].mean()
                std_val = result[col].std()
                
                if std_val > 0:
                    result[f"{col}_std"] = (result[col] - mean_val) / std_val
                else:
                    result[f"{col}_std"] = 0  # Если стандартное отклонение равно 0
                    
                self.logger.debug(f"Стандартизован столбец {col}")
            else:
                self.logger.warning(f"Столбец {col} не найден в DataFrame")
        
        return result
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков для машинного обучения.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        result = df.copy()
        
        # Проверка наличия необходимых столбцов
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in result.columns]
        
        if missing_columns:
            self.logger.warning(f"Отсутствуют столбцы: {missing_columns}")
            return result
        
        # Создание признаков на основе цен
        result['price_range'] = result['high'] - result['low']
        result['price_change'] = result['close'] - result['open']
        result['price_change_pct'] = (result['close'] - result['open']) / result['open'] * 100
        
        # Создание признаков на основе объема
        result['volume_change'] = result['volume'].pct_change()
        result['volume_ma_5'] = result['volume'].rolling(window=5).mean()
        result['volume_ma_10'] = result['volume'].rolling(window=10).mean()
        
        # Создание признаков на основе временных рядов
        result['close_lag_1'] = result['close'].shift(1)
        result['close_lag_2'] = result['close'].shift(2)
        result['close_lag_3'] = result['close'].shift(3)
        
        # Создание признаков на основе скользящих средних
        result['sma_5'] = result['close'].rolling(window=5).mean()
        result['sma_10'] = result['close'].rolling(window=10).mean()
        result['sma_20'] = result['close'].rolling(window=20).mean()
        
        # Создание признаков на основе волатильности
        result['volatility_5'] = result['close'].rolling(window=5).std()
        result['volatility_10'] = result['close'].rolling(window=10).std()
        
        # Создание признаков на основе разницы скользящих средних
        result['sma_5_10_diff'] = result['sma_5'] - result['sma_10']
        result['sma_10_20_diff'] = result['sma_10'] - result['sma_20']
        
        # Создание признаков на основе отношения объема к диапазону цен
        result['volume_price_ratio'] = result['volume'] / result['price_range']
        
        self.logger.info(f"Созданы признаки для машинного обучения")
        return result
    
    def create_target(self, df: pd.DataFrame, periods: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """
        Создание целевой переменной для машинного обучения.
        
        Args:
            df: DataFrame с данными
            periods: Количество периодов для прогнозирования
            threshold: Порог изменения цены для классификации
            
        Returns:
            DataFrame с добавленной целевой переменной
        """
        result = df.copy()
        
        # Проверка наличия столбца 'close'
        if 'close' not in result.columns:
            self.logger.warning("Отсутствует столбец 'close'")
            return result
        
        # Создание целевой переменной для регрессии
        result[f'target_reg_{periods}'] = result['close'].shift(-periods) / result['close'] - 1
        
        # Создание целевой переменной для классификации
        result[f'target_cls_{periods}'] = 0  # Нейтральный класс по умолчанию
        
        # Если изменение цены выше порога, то 1 (рост), если ниже отрицательного порога, то -1 (падение)
        result.loc[result[f'target_reg_{periods}'] > threshold, f'target_cls_{periods}'] = 1
        result.loc[result[f'target_reg_{periods}'] < -threshold, f'target_cls_{periods}'] = -1
        
        self.logger.info(f"Создана целевая переменная для прогнозирования на {periods} периодов")
        return result
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        Обработка пропущенных значений.
        
        Args:
            df: DataFrame с данными
            method: Метод обработки пропущенных значений ('ffill', 'bfill', 'interpolate', 'drop')
            
        Returns:
            DataFrame с обработанными пропущенными значениями
        """
        result = df.copy()
        
        # Проверка наличия пропущенных значений
        missing_count = result.isna().sum().sum()
        
        if missing_count == 0:
            self.logger.info("Пропущенные значения отсутствуют")
            return result
        
        # Обработка пропущенных значений
        if method == 'ffill':
            result = result.ffill()
            self.logger.info(f"Пропущенные значения заполнены методом forward fill")
        elif method == 'bfill':
            result = result.bfill()
            self.logger.info(f"Пропущенные значения заполнены методом backward fill")
        elif method == 'interpolate':
            result = result.interpolate()
            self.logger.info(f"Пропущенные значения заполнены методом интерполяции")
        elif method == 'drop':
            result = result.dropna()
            self.logger.info(f"Строки с пропущенными значениями удалены")
        else:
            self.logger.warning(f"Неизвестный метод обработки пропущенных значений: {method}")
        
        # Проверка оставшихся пропущенных значений
        remaining_missing = result.isna().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Осталось {remaining_missing} пропущенных значений")
        
        return result
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Удаление выбросов.
        
        Args:
            df: DataFrame с данными
            columns: Список столбцов для обработки. Если None, обрабатываются все числовые столбцы.
            method: Метод обнаружения выбросов ('iqr', 'zscore')
            threshold: Порог для обнаружения выбросов
            
        Returns:
            DataFrame с удаленными выбросами
        """
        result = df.copy()
        
        if columns is None:
            # Выбор всех числовых столбцов
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        # Создание маски для строк без выбросов
        mask = pd.Series(True, index=result.index)
        
        for col in columns:
            if col in result.columns:
                if method == 'iqr':
                    # Метод межквартильного размаха
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    
                    col_mask = (result[col] >= lower_bound) & (result[col] <= upper_bound)
                    mask = mask & col_mask
                    
                    outliers_count = (~col_mask).sum()
                    self.logger.debug(f"Обнаружено {outliers_count} выбросов в столбце {col} методом IQR")
                    
                elif method == 'zscore':
                    # Метод z-score
                    mean_val = result[col].mean()
                    std_val = result[col].std()
                    
                    if std_val > 0:
                        z_scores = (result[col] - mean_val) / std_val
                        col_mask = z_scores.abs() <= threshold
                        mask = mask & col_mask
                        
                        outliers_count = (~col_mask).sum()
                        self.logger.debug(f"Обнаружено {outliers_count} выбросов в столбце {col} методом Z-score")
                    else:
                        self.logger.warning(f"Стандартное отклонение для столбца {col} равно 0")
                
                else:
                    self.logger.warning(f"Неизвестный метод обнаружения выбросов: {method}")
            else:
                self.logger.warning(f"Столбец {col} не найден в DataFrame")
        
        # Применение маски
        result_filtered = result[mask]
        
        removed_rows = len(result) - len(result_filtered)
        self.logger.info(f"Удалено {removed_rows} строк с выбросами")
        
        return result_filtered
    
    def split_data(self, df: pd.DataFrame, train_size: float = 0.8, shuffle: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на обучающую и тестовую выборки.
        
        Args:
            df: DataFrame с данными
            train_size: Доля обучающей выборки
            shuffle: Перемешивать ли данные перед разделением
            
        Returns:
            Кортеж (обучающая выборка, тестовая выборка)
        """
        if shuffle:
            # Перемешивание данных
            df_shuffled = df.sample(frac=1, random_state=42)
        else:
            df_shuffled = df.copy()
        
        # Разделение на обучающую и тестовую выборки
        train_count = int(len(df_shuffled) * train_size)
        
        train_df = df_shuffled.iloc[:train_count]
        test_df = df_shuffled.iloc[train_count:]
        
        self.logger.info(f"Данные разделены на обучающую ({len(train_df)} строк) и тестовую ({len(test_df)} строк) выборки")
        
        return train_df, test_df
    
    def prepare_data_for_ml(self, df: pd.DataFrame, target_column: str, feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Подготовка данных для машинного обучения.
        
        Args:
            df: DataFrame с данными
            target_column: Имя столбца с целевой переменной
            feature_columns: Список столбцов с признаками. Если None, используются все числовые столбцы кроме целевой.
            
        Returns:
            Кортеж (признаки, целевая переменная)
        """
        # Проверка наличия целевой переменной
        if target_column not in df.columns:
            self.logger.error(f"Столбец с целевой переменной {target_column} не найден")
            return pd.DataFrame(), pd.Series()
        
        # Выбор признаков
        if feature_columns is None:
            # Выбор всех числовых столбцов кроме целевой переменной
            feature_columns = df.select_dtypes(include=['number']).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        # Проверка наличия признаков
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Отсутствуют столбцы с признаками: {missing_columns}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Удаление строк с пропущенными значениями
        df_clean = df.dropna(subset=[target_column] + feature_columns)
        
        # Выделение признаков и целевой переменной
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        self.logger.info(f"Данные подготовлены для машинного обучения: {len(X)} строк, {len(feature_columns)} признаков")
        
        return X, y


# Пример использования
if __name__ == "__main__":
    import asyncio
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    async def main():
        # Создание хранилища данных
        storage = DataStorage()
        
        # Создание процессора данных
        processor = DataProcessor(storage)
        
        # Создание тестовых данных
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
        data = {
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(102, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Добавление пропущенных значений для тестирования
        df.loc[df.index[10:15], 'close'] = np.nan
        
        # Обработка пропущенных значений
        df_clean = processor.handle_missing_values(df, method='interpolate')
        print(f"Обработка пропущенных значений: {df_clean.isna().sum().sum()} пропущенных значений осталось")
        
        # Создание признаков
        df_features = processor.create_features(df_clean)
        print(f"Созданы признаки: {df_features.columns.tolist()}")
        
        # Создание целевой переменной
        df_target = processor.create_target(df_features, periods=5, threshold=0.01)
        print(f"Создана целевая переменная: {df_target['target_cls_5'].value_counts()}")
        
        # Нормализация данных
        df_norm = processor.normalize_data(df_target, columns=['close', 'volume'])
        print(f"Нормализованы данные: {[col for col in df_norm.columns if '_norm' in col]}")
        
        # Стандартизация данных
        df_std = processor.standardize_data(df_target, columns=['close', 'volume'])
        print(f"Стандартизованы данные: {[col for col in df_std.columns if '_std' in col]}")
        
        # Удаление выбросов
        df_no_outliers = processor.remove_outliers(df_target, columns=['volume'], method='iqr')
        print(f"Удалены выбросы: {len(df_target) - len(df_no_outliers)} строк удалено")
        
        # Разделение данных
        train_df, test_df = processor.split_data(df_target, train_size=0.8, shuffle=False)
        print(f"Разделение данных: {len(train_df)} строк в обучающей выборке, {len(test_df)} строк в тестовой выборке")
        
        # Подготовка данных для машинного обучения
        X, y = processor.prepare_data_for_ml(df_target, target_column='target_cls_5')
        print(f"Подготовка данных для ML: {X.shape[0]} строк, {X.shape[1]} признаков")
    
    # Запуск асинхронной функции
    asyncio.run(main()) 