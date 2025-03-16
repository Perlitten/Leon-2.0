import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import aiofiles
import asyncio

class DataStorage:
    """
    Класс для хранения и обработки исторических данных.
    Поддерживает сохранение данных в CSV, SQLite и JSON форматах.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Инициализация хранилища данных.
        
        Args:
            base_dir: Базовая директория для хранения данных
        """
        self.logger = logging.getLogger("DataStorage")
        self.base_dir = base_dir
        
        # Создание директорий для хранения данных
        self.csv_dir = os.path.join(base_dir, "csv")
        self.sqlite_dir = os.path.join(base_dir, "sqlite")
        self.json_dir = os.path.join(base_dir, "json")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.sqlite_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        self.logger.info(f"Инициализировано хранилище данных в директории {base_dir}")
    
    def klines_to_dataframe(self, klines: List[List[Any]]) -> pd.DataFrame:
        """
        Преобразование данных свечей Binance в DataFrame.
        
        Args:
            klines: Список свечей от Binance API
            
        Returns:
            DataFrame с данными свечей
        """
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # Преобразование типов данных
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Преобразование временных меток в datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Установка индекса
        df.set_index('open_time', inplace=True)
        
        return df
    
    async def save_klines_to_csv(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        """
        Асинхронное сохранение данных свечей в CSV файл.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            df: DataFrame с данными свечей
            
        Returns:
            Путь к сохраненному файлу
        """
        # Создание директории для символа, если она не существует
        symbol_dir = os.path.join(self.csv_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Формирование имени файла
        filename = f"{symbol}_{interval}_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(symbol_dir, filename)
        
        # Сохранение в CSV
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: df.to_csv(filepath))
        
        self.logger.info(f"Данные сохранены в CSV: {filepath}")
        return filepath
    
    async def load_klines_from_csv(self, symbol: str, interval: str, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Асинхронная загрузка данных свечей из CSV файлов.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)
            
        Returns:
            DataFrame с данными свечей
        """
        symbol_dir = os.path.join(self.csv_dir, symbol)
        
        if not os.path.exists(symbol_dir):
            self.logger.warning(f"Директория {symbol_dir} не существует")
            return pd.DataFrame()
        
        # Получение списка файлов
        files = [f for f in os.listdir(symbol_dir) if f.startswith(f"{symbol}_{interval}") and f.endswith(".csv")]
        
        if not files:
            self.logger.warning(f"Файлы для {symbol}_{interval} не найдены")
            return pd.DataFrame()
        
        # Фильтрация файлов по датам, если указаны
        if start_date or end_date:
            filtered_files = []
            for file in files:
                parts = file.split('_')
                if len(parts) >= 4:
                    file_start = datetime.strptime(parts[2], '%Y%m%d')
                    file_end = datetime.strptime(parts[3].split('.')[0], '%Y%m%d')
                    
                    if start_date and file_end < start_date:
                        continue
                    if end_date and file_start > end_date:
                        continue
                    
                    filtered_files.append(file)
            
            files = filtered_files
        
        # Загрузка и объединение данных из файлов
        dfs = []
        loop = asyncio.get_event_loop()
        
        for file in files:
            filepath = os.path.join(symbol_dir, file)
            df = await loop.run_in_executor(None, lambda: pd.read_csv(filepath))
            
            # Преобразование столбца open_time в datetime и установка его в качестве индекса
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df.set_index('open_time', inplace=True)
            
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Объединение всех DataFrame
        result_df = pd.concat(dfs)
        
        # Фильтрация по датам
        if start_date:
            result_df = result_df[result_df.index >= pd.Timestamp(start_date)]
        if end_date:
            result_df = result_df[result_df.index <= pd.Timestamp(end_date)]
        
        # Сортировка по времени
        result_df.sort_index(inplace=True)
        
        # Удаление дубликатов
        result_df = result_df[~result_df.index.duplicated(keep='first')]
        
        self.logger.info(f"Загружено {len(result_df)} строк данных для {symbol}_{interval}")
        return result_df
    
    async def save_klines_to_sqlite(self, symbol: str, interval: str, df: pd.DataFrame) -> str:
        """
        Асинхронное сохранение данных свечей в SQLite базу данных.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            df: DataFrame с данными свечей
            
        Returns:
            Путь к базе данных
        """
        # Формирование имени файла базы данных
        db_filename = f"{symbol}.db"
        db_path = os.path.join(self.sqlite_dir, db_filename)
        
        # Подготовка данных для сохранения
        df_copy = df.reset_index()
        table_name = f"klines_{interval.lower()}"
        
        # Асинхронное сохранение в SQLite
        loop = asyncio.get_event_loop()
        
        def save_to_sqlite():
            conn = sqlite3.connect(db_path)
            df_copy.to_sql(table_name, conn, if_exists='append', index=False)
            
            # Удаление дубликатов
            cursor = conn.cursor()
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM {table_name}
                    GROUP BY open_time
                )
            """)
            conn.commit()
            conn.close()
        
        await loop.run_in_executor(None, save_to_sqlite)
        
        self.logger.info(f"Данные сохранены в SQLite: {db_path}, таблица: {table_name}")
        return db_path
    
    async def load_klines_from_sqlite(self, symbol: str, interval: str,
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Асинхронная загрузка данных свечей из SQLite базы данных.
        
        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)
            
        Returns:
            DataFrame с данными свечей
        """
        db_filename = f"{symbol}.db"
        db_path = os.path.join(self.sqlite_dir, db_filename)
        
        if not os.path.exists(db_path):
            self.logger.warning(f"База данных {db_path} не существует")
            return pd.DataFrame()
        
        table_name = f"klines_{interval.lower()}"
        loop = asyncio.get_event_loop()
        
        def load_from_sqlite():
            conn = sqlite3.connect(db_path)
            
            # Формирование SQL запроса с учетом фильтрации по датам
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            if start_date:
                start_timestamp = int(start_date.timestamp() * 1000)
                conditions.append(f"open_time >= '{start_date}'")
            
            if end_date:
                end_timestamp = int(end_date.timestamp() * 1000)
                conditions.append(f"open_time <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY open_time"
            
            try:
                df = pd.read_sql_query(query, conn)
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'])
                    df.set_index('open_time', inplace=True)
                return df
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке данных из SQLite: {e}")
                return pd.DataFrame()
            finally:
                conn.close()
        
        df = await loop.run_in_executor(None, load_from_sqlite)
        
        self.logger.info(f"Загружено {len(df)} строк данных для {symbol}_{interval} из SQLite")
        return df
    
    async def save_to_json(self, filename: str, data: Any) -> str:
        """
        Асинхронное сохранение данных в JSON файл.
        
        Args:
            filename: Имя файла
            data: Данные для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        filepath = os.path.join(self.json_dir, filename)
        
        # Преобразование DataFrame в список словарей
        if isinstance(data, pd.DataFrame):
            data = data.reset_index().to_dict(orient='records')
        
        # Асинхронная запись в файл
        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, default=str, indent=2))
        
        self.logger.info(f"Данные сохранены в JSON: {filepath}")
        return filepath
    
    async def load_from_json(self, filename: str) -> Any:
        """
        Асинхронная загрузка данных из JSON файла.
        
        Args:
            filename: Имя файла
            
        Returns:
            Загруженные данные
        """
        filepath = os.path.join(self.json_dir, filename)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"Файл {filepath} не существует")
            return None
        
        # Асинхронное чтение файла
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)
        
        self.logger.info(f"Данные загружены из JSON: {filepath}")
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов к DataFrame.
        
        Args:
            df: DataFrame с данными свечей
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # Копирование DataFrame для избежания предупреждений
        result = df.copy()
        
        # Простые скользящие средние
        result['sma_7'] = result['close'].rolling(window=7).mean()
        result['sma_25'] = result['close'].rolling(window=25).mean()
        result['sma_99'] = result['close'].rolling(window=99).mean()
        
        # Экспоненциальные скользящие средние
        result['ema_7'] = result['close'].ewm(span=7, adjust=False).mean()
        result['ema_25'] = result['close'].ewm(span=25, adjust=False).mean()
        result['ema_99'] = result['close'].ewm(span=99, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        # RSI (14)
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # ATR (14)
        high_low = result['high'] - result['low']
        high_close = (result['high'] - result['close'].shift()).abs()
        low_close = (result['low'] - result['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['atr_14'] = true_range.rolling(14).mean()
        
        # Стохастический осциллятор
        result['stoch_k'] = 100 * ((result['close'] - result['low'].rolling(14).min()) / 
                                  (result['high'].rolling(14).max() - result['low'].rolling(14).min()))
        result['stoch_d'] = result['stoch_k'].rolling(3).mean()
        
        return result
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение паттернов свечей.
        
        Args:
            df: DataFrame с данными свечей
            
        Returns:
            DataFrame с добавленными паттернами
        """
        result = df.copy()
        
        # Doji (тело свечи очень маленькое)
        body_size = abs(result['close'] - result['open'])
        avg_body_size = body_size.rolling(window=14).mean()
        result['doji'] = body_size < (0.1 * avg_body_size)
        
        # Молот (маленькое тело, длинная нижняя тень, маленькая верхняя тень)
        upper_shadow = result['high'] - result[['open', 'close']].max(axis=1)
        lower_shadow = result[['open', 'close']].min(axis=1) - result['low']
        
        result['hammer'] = (body_size < (0.3 * (result['high'] - result['low']))) & \
                          (lower_shadow > (2 * body_size)) & \
                          (upper_shadow < (0.1 * (result['high'] - result['low'])))
        
        # Поглощение
        prev_body_size = body_size.shift(1)
        prev_open = result['open'].shift(1)
        prev_close = result['close'].shift(1)
        
        result['bullish_engulfing'] = (result['open'] < result['close']) & \
                                     (prev_open > prev_close) & \
                                     (result['open'] <= prev_close) & \
                                     (result['close'] >= prev_open)
        
        result['bearish_engulfing'] = (result['open'] > result['close']) & \
                                     (prev_open < prev_close) & \
                                     (result['open'] >= prev_close) & \
                                     (result['close'] <= prev_open)
        
        return result
    
    def get_missing_intervals(self, df: pd.DataFrame, interval: str) -> List[Tuple[datetime, datetime]]:
        """
        Определение отсутствующих интервалов в данных.
        
        Args:
            df: DataFrame с данными свечей
            interval: Интервал свечей
            
        Returns:
            Список кортежей (начало, конец) отсутствующих интервалов
        """
        if df.empty:
            return []
        
        # Определение ожидаемого интервала между свечами
        interval_map = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30)  # Приблизительно
        }
        
        expected_delta = interval_map.get(interval)
        if not expected_delta:
            self.logger.warning(f"Неизвестный интервал: {interval}")
            return []
        
        # Сортировка индекса
        sorted_df = df.sort_index()
        
        # Поиск пропусков
        missing_intervals = []
        prev_time = None
        
        for time_idx in sorted_df.index:
            if prev_time is not None:
                # Проверка, есть ли пропуск больше, чем ожидаемый интервал
                if time_idx - prev_time > expected_delta * 1.5:
                    missing_intervals.append((prev_time + expected_delta, time_idx - expected_delta))
            
            prev_time = time_idx
        
        return missing_intervals
    
    async def get_average_volume(self, symbol: str, interval: str, periods: int = 24) -> float:
        """
        Получение среднего объема за указанное количество периодов.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечей
            periods: Количество периодов для расчета среднего
            
        Returns:
            Средний объем
        """
        try:
            # Загрузка данных из SQLite
            df = await self.load_klines_from_sqlite(symbol, interval)
            
            if df.empty:
                self.logger.warning(f"Нет данных для расчета среднего объема {symbol} {interval}")
                return 0.0
            
            # Сортировка по времени и выбор последних периодов
            df = df.sort_index().tail(periods)
            
            # Расчет среднего объема
            avg_volume = df['volume'].mean()
            
            self.logger.debug(f"Средний объем за {periods} периодов для {symbol} {interval}: {avg_volume}")
            return avg_volume
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете среднего объема: {e}")
            return 0.0
    
    async def save_kline(self, symbol: str, interval: str, kline: List[Any]) -> bool:
        """
        Сохранение одной свечи в хранилище.
        
        Args:
            symbol: Символ торговой пары
            interval: Интервал свечи
            kline: Данные свечи
            
        Returns:
            True, если свеча успешно сохранена, иначе False
        """
        try:
            # Преобразование свечи в DataFrame
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame([kline], columns=columns)
            
            # Преобразование типов данных
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'taker_buy_base_asset_volume', 
                              'taker_buy_quote_asset_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Преобразование временных меток в datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Установка индекса
            df.set_index('open_time', inplace=True)
            
            # Сохранение в SQLite
            await self.save_klines_to_sqlite(symbol, interval, df)
            
            self.logger.debug(f"Сохранена свеча для {symbol} {interval}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении свечи: {e}")
            return False


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
        
        # Создание тестовых данных
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        data = {
            'open': np.random.normal(100, 5, len(dates)),
            'high': np.random.normal(105, 5, len(dates)),
            'low': np.random.normal(95, 5, len(dates)),
            'close': np.random.normal(102, 5, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Сохранение данных в разных форматах
        await storage.save_klines_to_csv('BTCUSDT', '1h', df)
        await storage.save_klines_to_sqlite('BTCUSDT', '1h', df)
        await storage.save_to_json('BTCUSDT_1h_test.json', df)
        
        # Загрузка данных
        csv_df = await storage.load_klines_from_csv('BTCUSDT', '1h')
        sqlite_df = await storage.load_klines_from_sqlite('BTCUSDT', '1h')
        json_data = await storage.load_from_json('BTCUSDT_1h_test.json')
        
        print(f"CSV data shape: {csv_df.shape}")
        print(f"SQLite data shape: {sqlite_df.shape}")
        print(f"JSON data length: {len(json_data)}")
        
        # Добавление технических индикаторов
        df_with_indicators = storage.add_technical_indicators(df)
        print(f"Columns with indicators: {df_with_indicators.columns.tolist()}")
        
        # Обнаружение паттернов
        df_with_patterns = storage.detect_patterns(df)
        print(f"Columns with patterns: {df_with_patterns.columns.tolist()}")
        
        # Поиск пропущенных интервалов
        # Создаем DataFrame с пропусками
        dates_with_gaps = pd.date_range(start='2023-01-01', end='2023-01-05', freq='1h').tolist()
        # Добавляем пропуск
        dates_with_gaps.extend(pd.date_range(start='2023-01-06', end='2023-01-10', freq='1h').tolist())
        
        df_with_gaps = pd.DataFrame(data, index=dates_with_gaps[:len(data['open'])])
        missing = storage.get_missing_intervals(df_with_gaps, '1h')
        print(f"Missing intervals: {missing}")
    
    # Запуск асинхронной функции
    asyncio.run(main()) 