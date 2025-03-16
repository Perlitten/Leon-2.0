"""
Модуль извлечения признаков для машинного обучения.

Предоставляет класс для извлечения признаков из исторических данных.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

class FeatureExtractor:
    """
    Класс для извлечения признаков из исторических данных.
    
    Отвечает за:
    - Преобразование сырых данных в признаки для моделей
    - Нормализацию и масштабирование признаков
    - Создание временных рядов для обучения
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация экстрактора признаков.
        
        Args:
            config: Конфигурация экстрактора признаков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Параметры по умолчанию
        self.window_size = self.config.get("window_size", 20)
        self.features = self.config.get("features", ["close", "volume", "ma_7", "ma_14", "rsi_14", "bb_upper", "bb_lower"])
        self.target = self.config.get("target", "future_return")
        self.future_periods = self.config.get("future_periods", 5)
        self.normalize = self.config.get("normalize", True)
        
        # Статистика для нормализации
        self.feature_stats = {}
    
    def transform(self, data: List[List[Union[int, str, float]]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Преобразование сырых данных в признаки и целевые значения.
        
        Args:
            data: Исторические данные в формате Binance
                [timestamp, open, high, low, close, volume, ...]
            
        Returns:
            Кортеж (признаки, целевые значения)
        """
        try:
            # Преобразование данных в DataFrame
            df = self._convert_to_dataframe(data)
            
            # Расчет технических индикаторов
            df = self._calculate_indicators(df)
            
            # Создание целевой переменной
            if self.target == "future_return":
                df["future_return"] = df["close"].pct_change(self.future_periods).shift(-self.future_periods)
            elif self.target == "direction":
                df["future_close"] = df["close"].shift(-self.future_periods)
                df["direction"] = (df["future_close"] > df["close"]).astype(int)
            
            # Удаление строк с NaN
            df = df.dropna()
            
            # Выбор признаков и целевой переменной
            X = df[self.features].values
            y = df[self.target].values if self.target in df.columns else None
            
            # Нормализация признаков
            if self.normalize:
                X = self._normalize_features(X)
            
            # Создание окон для временных рядов
            X_windows, y_windows = self._create_windows(X, y)
            
            return X_windows, y_windows
        
        except Exception as e:
            self.logger.error(f"Ошибка при преобразовании данных: {e}")
            return np.array([]), None
    
    def _convert_to_dataframe(self, data: List[List[Union[int, str, float]]]) -> pd.DataFrame:
        """
        Преобразование данных в формате Binance в DataFrame.
        
        Args:
            data: Исторические данные в формате Binance
                [timestamp, open, high, low, close, volume, ...]
            
        Returns:
            DataFrame с данными
        """
        # Создание DataFrame
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", 
                                         "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
                                         "taker_buy_quote_asset_volume", "ignore"])
        
        # Преобразование типов
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        
        # Установка индекса
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет технических индикаторов.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # Скользящие средние
        df["ma_7"] = df["close"].rolling(window=7).mean()
        df["ma_14"] = df["close"].rolling(window=14).mean()
        df["ma_21"] = df["close"].rolling(window=21).mean()
        
        # Относительная сила (RSI)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Полосы Боллинджера
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        
        # MACD
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Стохастический осциллятор
        df["stoch_k"] = 100 * ((df["close"] - df["low"].rolling(window=14).min()) / 
                              (df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min()))
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        # Возвращаемые изменения
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        
        # Волатильность
        df["volatility_14"] = df["return_1"].rolling(window=14).std()
        
        return df
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Нормализация признаков.
        
        Args:
            X: Массив признаков
            
        Returns:
            Нормализованный массив признаков
        """
        # Расчет статистики для нормализации
        if not self.feature_stats:
            self.feature_stats["mean"] = np.mean(X, axis=0)
            self.feature_stats["std"] = np.std(X, axis=0)
            
            # Замена нулевых стандартных отклонений на 1
            self.feature_stats["std"] = np.where(self.feature_stats["std"] == 0, 1, self.feature_stats["std"])
        
        # Нормализация
        X_norm = (X - self.feature_stats["mean"]) / self.feature_stats["std"]
        
        return X_norm
    
    def _create_windows(self, X: np.ndarray, y: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Создание окон для временных рядов.
        
        Args:
            X: Массив признаков
            y: Массив целевых значений
            
        Returns:
            Кортеж (окна признаков, окна целевых значений)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Проверка наличия достаточного количества данных
        if n_samples < self.window_size:
            self.logger.warning(f"Недостаточно данных для создания окон: {n_samples} < {self.window_size}")
            return np.array([]), None
        
        # Создание окон для признаков
        X_windows = np.zeros((n_samples - self.window_size + 1, self.window_size, n_features))
        
        for i in range(n_samples - self.window_size + 1):
            X_windows[i] = X[i:i+self.window_size]
        
        # Создание окон для целевых значений
        y_windows = None
        if y is not None:
            y_windows = y[self.window_size-1:]
        
        return X_windows, y_windows
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование нормализованных признаков.
        
        Args:
            X: Нормализованный массив признаков
            
        Returns:
            Исходный массив признаков
        """
        if not self.feature_stats:
            self.logger.warning("Статистика для обратного преобразования не найдена")
            return X
        
        # Обратное преобразование
        X_orig = X * self.feature_stats["std"] + self.feature_stats["mean"]
        
        return X_orig
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновление конфигурации экстрактора признаков.
        
        Args:
            config: Новая конфигурация
        """
        self.config.update(config)
        
        # Обновление параметров
        self.window_size = self.config.get("window_size", self.window_size)
        self.features = self.config.get("features", self.features)
        self.target = self.config.get("target", self.target)
        self.future_periods = self.config.get("future_periods", self.future_periods)
        self.normalize = self.config.get("normalize", self.normalize)
        
        # Сброс статистики для нормализации
        self.feature_stats = {}
        
        self.logger.info("Конфигурация экстрактора признаков обновлена") 