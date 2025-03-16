"""
Модуль инженерии признаков для машинного обучения.

Предоставляет классы и функции для создания новых признаков
на основе существующих данных.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer:
    """
    Класс для создания новых признаков на основе существующих данных.
    
    Предоставляет методы для создания различных типов признаков,
    включая полиномиальные, взаимодействия, временные и текстовые признаки.
    
    Attributes:
        config (Dict[str, Any]): Конфигурация инженерии признаков
        transformers (Dict[str, Callable]): Словарь трансформеров признаков
        feature_names (List[str]): Список имен признаков
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация инженера признаков.
        
        Args:
            config: Конфигурация инженерии признаков
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.config = config or {}
        self.transformers = {}
        self.feature_names = []
        
        self.logger.info("Инициализирован инженер признаков")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           feature_names: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Обучение инженера признаков.
        
        Args:
            X: Матрица признаков
            feature_names: Список имен признаков
            
        Returns:
            FeatureEngineer: Обученный инженер признаков
        """
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = X_df.columns.tolist()
        
        # Сохранение имен признаков
        self.feature_names = feature_names
        
        # Обучение трансформеров
        for transformer_name, transformer in self.transformers.items():
            if hasattr(transformer, 'fit'):
                transformer.fit(X_df)
                self.logger.info(f"Обучен трансформер {transformer_name}")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Преобразование данных с созданием новых признаков.
        
        Args:
            X: Матрица признаков
            
        Returns:
            pd.DataFrame: Преобразованные данные с новыми признаками
        """
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            if len(self.feature_names) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        # Применение трансформеров
        result_df = X_df.copy()
        
        for transformer_name, transformer in self.transformers.items():
            try:
                if hasattr(transformer, 'transform'):
                    transformed = transformer.transform(X_df)
                    
                    # Если результат - DataFrame, объединяем его с результатом
                    if isinstance(transformed, pd.DataFrame):
                        result_df = pd.concat([result_df, transformed], axis=1)
                    # Если результат - массив, создаем DataFrame и объединяем
                    elif isinstance(transformed, np.ndarray):
                        if hasattr(transformer, 'get_feature_names_out'):
                            new_feature_names = transformer.get_feature_names_out()
                        else:
                            new_feature_names = [f"{transformer_name}_{i}" for i in range(transformed.shape[1])]
                        
                        transformed_df = pd.DataFrame(transformed, index=X_df.index, columns=new_feature_names)
                        result_df = pd.concat([result_df, transformed_df], axis=1)
                
                self.logger.info(f"Применен трансформер {transformer_name}, добавлено {result_df.shape[1] - X_df.shape[1]} признаков")
            
            except Exception as e:
                self.logger.error(f"Ошибка при применении трансформера {transformer_name}: {e}")
        
        return result_df
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Обучение и преобразование данных.
        
        Args:
            X: Матрица признаков
            feature_names: Список имен признаков
            
        Returns:
            pd.DataFrame: Преобразованные данные с новыми признаками
        """
        return self.fit(X, feature_names).transform(X)
    
    def add_transformer(self, name: str, transformer: Any) -> 'FeatureEngineer':
        """
        Добавление трансформера признаков.
        
        Args:
            name: Имя трансформера
            transformer: Трансформер признаков
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        self.transformers[name] = transformer
        self.logger.info(f"Добавлен трансформер {name}")
        
        return self
    
    def remove_transformer(self, name: str) -> 'FeatureEngineer':
        """
        Удаление трансформера признаков.
        
        Args:
            name: Имя трансформера
            
        Returns:
            FeatureEngineer: Инженер признаков
            
        Raises:
            ValueError: Если трансформер с указанным именем не найден
        """
        if name not in self.transformers:
            raise ValueError(f"Трансформер с именем {name} не найден")
        
        del self.transformers[name]
        self.logger.info(f"Удален трансформер {name}")
        
        return self
    
    def add_polynomial_features(self, degree: int = 2, 
                              include_bias: bool = False,
                              interaction_only: bool = False,
                              columns: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Добавление полиномиальных признаков.
        
        Args:
            degree: Степень полинома
            include_bias: Включать ли константный признак
            interaction_only: Включать только взаимодействия (без степеней)
            columns: Список колонок для преобразования (если None, используются все)
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        class PolynomialTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, degree=2, include_bias=False, interaction_only=False, columns=None):
                self.degree = degree
                self.include_bias = include_bias
                self.interaction_only = interaction_only
                self.columns = columns
                self.poly = PolynomialFeatures(
                    degree=degree, 
                    include_bias=include_bias, 
                    interaction_only=interaction_only
                )
                self.feature_names_in_ = None
                self.feature_names_out_ = None
            
            def fit(self, X, y=None):
                if isinstance(X, pd.DataFrame):
                    if self.columns is not None:
                        self.feature_names_in_ = [col for col in X.columns if col in self.columns]
                    else:
                        self.feature_names_in_ = X.columns.tolist()
                    
                    X_subset = X[self.feature_names_in_]
                    self.poly.fit(X_subset)
                    
                    # Генерация имен признаков
                    self.feature_names_out_ = self.poly.get_feature_names_out(self.feature_names_in_)
                else:
                    self.poly.fit(X)
                    self.feature_names_out_ = self.poly.get_feature_names_out()
                
                return self
            
            def transform(self, X):
                if isinstance(X, pd.DataFrame):
                    if self.columns is not None:
                        X_subset = X[self.feature_names_in_]
                    else:
                        X_subset = X
                    
                    transformed = self.poly.transform(X_subset)
                    
                    # Удаление первого столбца (константа), если include_bias=True
                    if self.include_bias:
                        transformed = transformed[:, 1:]
                        self.feature_names_out_ = self.feature_names_out_[1:]
                    
                    return transformed
                else:
                    return self.poly.transform(X)
            
            def get_feature_names_out(self):
                return self.feature_names_out_
        
        transformer = PolynomialTransformer(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
            columns=columns
        )
        
        self.add_transformer(f"polynomial_degree_{degree}", transformer)
        
        return self
    
    def add_interaction_features(self, columns: List[str]) -> 'FeatureEngineer':
        """
        Добавление признаков взаимодействия между указанными колонками.
        
        Args:
            columns: Список колонок для создания взаимодействий
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        class InteractionTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, columns):
                self.columns = columns
                self.feature_names_out_ = None
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Проверка наличия колонок
                for col in self.columns:
                    if col not in X.columns:
                        raise ValueError(f"Колонка {col} не найдена в данных")
                
                # Генерация имен признаков
                self.feature_names_out_ = []
                for i in range(len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        self.feature_names_out_.append(f"{self.columns[i]}_{self.columns[j]}_interaction")
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Создание DataFrame для результата
                result = pd.DataFrame(index=X.index)
                
                # Создание признаков взаимодействия
                for i in range(len(self.columns)):
                    for j in range(i+1, len(self.columns)):
                        col1 = self.columns[i]
                        col2 = self.columns[j]
                        result[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
                
                return result
            
            def get_feature_names_out(self):
                return self.feature_names_out_
        
        transformer = InteractionTransformer(columns=columns)
        
        self.add_transformer(f"interaction_{'_'.join(columns)}", transformer)
        
        return self
    
    def add_date_features(self, column: str, 
                        features: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Добавление признаков на основе даты.
        
        Args:
            column: Имя колонки с датой
            features: Список признаков для создания (если None, создаются все)
                Возможные значения: 'year', 'month', 'day', 'dayofweek', 'hour', 
                'minute', 'second', 'quarter', 'dayofyear', 'weekofyear', 'is_weekend'
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        class DateTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, column, features=None):
                self.column = column
                self.features = features or [
                    'year', 'month', 'day', 'dayofweek', 'hour', 
                    'minute', 'second', 'quarter', 'dayofyear', 'weekofyear', 'is_weekend'
                ]
                self.feature_names_out_ = None
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Проверка наличия колонки
                if self.column not in X.columns:
                    raise ValueError(f"Колонка {self.column} не найдена в данных")
                
                # Генерация имен признаков
                self.feature_names_out_ = [f"{self.column}_{feature}" for feature in self.features]
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Создание копии колонки с датой
                date_col = pd.to_datetime(X[self.column])
                
                # Создание DataFrame для результата
                result = pd.DataFrame(index=X.index)
                
                # Создание признаков на основе даты
                for feature in self.features:
                    if feature == 'year':
                        result[f"{self.column}_year"] = date_col.dt.year
                    elif feature == 'month':
                        result[f"{self.column}_month"] = date_col.dt.month
                    elif feature == 'day':
                        result[f"{self.column}_day"] = date_col.dt.day
                    elif feature == 'dayofweek':
                        result[f"{self.column}_dayofweek"] = date_col.dt.dayofweek
                    elif feature == 'hour':
                        result[f"{self.column}_hour"] = date_col.dt.hour
                    elif feature == 'minute':
                        result[f"{self.column}_minute"] = date_col.dt.minute
                    elif feature == 'second':
                        result[f"{self.column}_second"] = date_col.dt.second
                    elif feature == 'quarter':
                        result[f"{self.column}_quarter"] = date_col.dt.quarter
                    elif feature == 'dayofyear':
                        result[f"{self.column}_dayofyear"] = date_col.dt.dayofyear
                    elif feature == 'weekofyear':
                        result[f"{self.column}_weekofyear"] = date_col.dt.isocalendar().week
                    elif feature == 'is_weekend':
                        result[f"{self.column}_is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)
                
                return result
            
            def get_feature_names_out(self):
                return self.feature_names_out_
        
        transformer = DateTransformer(column=column, features=features)
        
        self.add_transformer(f"date_{column}", transformer)
        
        return self
    
    def add_time_features(self, column: str, 
                        lag_values: Optional[List[int]] = None,
                        window_sizes: Optional[List[int]] = None,
                        functions: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Добавление признаков на основе временного ряда.
        
        Args:
            column: Имя колонки с временным рядом
            lag_values: Список значений лага
            window_sizes: Список размеров окна для скользящих статистик
            functions: Список функций для скользящих статистик
                Возможные значения: 'mean', 'std', 'min', 'max', 'median', 'sum'
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, column, lag_values=None, window_sizes=None, functions=None):
                self.column = column
                self.lag_values = lag_values or [1, 2, 3, 5, 7, 14, 30]
                self.window_sizes = window_sizes or [3, 5, 7, 14, 30]
                self.functions = functions or ['mean', 'std', 'min', 'max', 'median', 'sum']
                self.feature_names_out_ = None
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Проверка наличия колонки
                if self.column not in X.columns:
                    raise ValueError(f"Колонка {self.column} не найдена в данных")
                
                # Генерация имен признаков
                self.feature_names_out_ = []
                
                # Имена признаков для лагов
                for lag in self.lag_values:
                    self.feature_names_out_.append(f"{self.column}_lag_{lag}")
                
                # Имена признаков для скользящих статистик
                for window in self.window_sizes:
                    for func in self.functions:
                        self.feature_names_out_.append(f"{self.column}_window_{window}_{func}")
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Создание копии данных
                X_copy = X.copy()
                
                # Создание DataFrame для результата
                result = pd.DataFrame(index=X.index)
                
                # Создание признаков на основе лагов
                for lag in self.lag_values:
                    result[f"{self.column}_lag_{lag}"] = X_copy[self.column].shift(lag)
                
                # Создание признаков на основе скользящих статистик
                for window in self.window_sizes:
                    for func in self.functions:
                        if func == 'mean':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).mean()
                        elif func == 'std':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).std()
                        elif func == 'min':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).min()
                        elif func == 'max':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).max()
                        elif func == 'median':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).median()
                        elif func == 'sum':
                            result[f"{self.column}_window_{window}_{func}"] = X_copy[self.column].rolling(window=window).sum()
                
                return result
            
            def get_feature_names_out(self):
                return self.feature_names_out_
        
        transformer = TimeSeriesTransformer(
            column=column, 
            lag_values=lag_values, 
            window_sizes=window_sizes, 
            functions=functions
        )
        
        self.add_transformer(f"time_series_{column}", transformer)
        
        return self
    
    def add_text_features(self, column: str, 
                        features: Optional[List[str]] = None) -> 'FeatureEngineer':
        """
        Добавление признаков на основе текста.
        
        Args:
            column: Имя колонки с текстом
            features: Список признаков для создания (если None, создаются все)
                Возможные значения: 'char_count', 'word_count', 'sentence_count', 
                'avg_word_length', 'special_char_count', 'stopword_count', 
                'uppercase_count', 'lowercase_count', 'digit_count'
            
        Returns:
            FeatureEngineer: Инженер признаков
        """
        class TextTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, column, features=None):
                self.column = column
                self.features = features or [
                    'char_count', 'word_count', 'sentence_count', 
                    'avg_word_length', 'special_char_count', 
                    'uppercase_count', 'lowercase_count', 'digit_count'
                ]
                self.feature_names_out_ = None
                
                # Список стоп-слов (если нужен)
                self.stopwords = None
                if 'stopword_count' in self.features:
                    try:
                        from nltk.corpus import stopwords
                        self.stopwords = set(stopwords.words('english'))
                    except:
                        self.stopwords = set()
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Проверка наличия колонки
                if self.column not in X.columns:
                    raise ValueError(f"Колонка {self.column} не найдена в данных")
                
                # Генерация имен признаков
                self.feature_names_out_ = [f"{self.column}_{feature}" for feature in self.features]
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    raise ValueError("X должен быть DataFrame")
                
                # Проверка, что колонка содержит строки
                if not pd.api.types.is_string_dtype(X[self.column]):
                    X[self.column] = X[self.column].astype(str)
                
                # Создание DataFrame для результата
                result = pd.DataFrame(index=X.index)
                
                # Создание признаков на основе текста
                for feature in self.features:
                    if feature == 'char_count':
                        result[f"{self.column}_char_count"] = X[self.column].str.len()
                    
                    elif feature == 'word_count':
                        result[f"{self.column}_word_count"] = X[self.column].str.split().str.len()
                    
                    elif feature == 'sentence_count':
                        result[f"{self.column}_sentence_count"] = X[self.column].str.count(r'[.!?]+')
                    
                    elif feature == 'avg_word_length':
                        # Вычисление средней длины слова
                        result[f"{self.column}_avg_word_length"] = X[self.column].apply(
                            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
                        )
                    
                    elif feature == 'special_char_count':
                        result[f"{self.column}_special_char_count"] = X[self.column].str.count(r'[^\w\s]')
                    
                    elif feature == 'stopword_count' and self.stopwords:
                        # Подсчет стоп-слов
                        result[f"{self.column}_stopword_count"] = X[self.column].apply(
                            lambda x: len([word for word in x.lower().split() if word in self.stopwords])
                        )
                    
                    elif feature == 'uppercase_count':
                        result[f"{self.column}_uppercase_count"] = X[self.column].str.count(r'[A-Z]')
                    
                    elif feature == 'lowercase_count':
                        result[f"{self.column}_lowercase_count"] = X[self.column].str.count(r'[a-z]')
                    
                    elif feature == 'digit_count':
                        result[f"{self.column}_digit_count"] = X[self.column].str.count(r'[0-9]')
                
                return result
            
            def get_feature_names_out(self):
                return self.feature_names_out_
        
        transformer = TextTransformer(column=column, features=features)
        
        self.add_transformer(f"text_{column}", transformer)
        
        return self
    
    def add_custom_transformer(self, name: str, 
                             transformer: BaseEstimator) -> 'FeatureEngineer':
        """
        Добавление пользовательского трансформера.
        
        Args:
            name: Имя трансформера
            transformer: Трансформер (должен иметь методы fit и transform)
            
        Returns:
            FeatureEngineer: Инженер признаков
            
        Raises:
            ValueError: Если трансформер не имеет методов fit и transform
        """
        if not hasattr(transformer, 'fit') or not hasattr(transformer, 'transform'):
            raise ValueError("Трансформер должен иметь методы fit и transform")
        
        self.add_transformer(name, transformer)
        
        return self
    
    def save(self, path: str) -> str:
        """
        Сохранение инженера признаков.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            str: Путь к сохраненному инженеру признаков
        """
        import pickle
        import os
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение инженера признаков
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Инженер признаков сохранен в {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """
        Загрузка инженера признаков.
        
        Args:
            path: Путь к сохраненному инженеру признаков
            
        Returns:
            FeatureEngineer: Загруженный инженер признаков
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        import pickle
        
        # Проверка существования файла
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")
        
        # Загрузка инженера признаков
        with open(path, 'rb') as f:
            feature_engineer = pickle.load(f)
        
        feature_engineer.logger.info(f"Инженер признаков загружен из {path}")
        
        return feature_engineer
    
    def get_feature_names(self) -> List[str]:
        """
        Получение списка имен признаков.
        
        Returns:
            List[str]: Список имен признаков
        """
        return self.feature_names
    
    def get_transformers(self) -> Dict[str, Any]:
        """
        Получение словаря трансформеров.
        
        Returns:
            Dict[str, Any]: Словарь трансформеров
        """
        return self.transformers.copy()
    
    def reset(self) -> 'FeatureEngineer':
        """
        Сброс инженера признаков.
        
        Returns:
            FeatureEngineer: Инженер признаков
        """
        self.transformers = {}
        self.feature_names = []
        
        self.logger.info("Инженер признаков сброшен")
        
        return self 