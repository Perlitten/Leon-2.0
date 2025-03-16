"""
Модуль предобработки данных для машинного обучения.

Предоставляет классы и функции для очистки, трансформации и подготовки данных
к обучению моделей машинного обучения.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime
import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    Класс для предобработки данных для машинного обучения.
    
    Предоставляет методы для очистки, трансформации и подготовки данных
    к обучению моделей машинного обучения.
    
    Attributes:
        config (Dict[str, Any]): Конфигурация предобработки данных
        transformers (Dict[str, Any]): Словарь трансформеров данных
        pipelines (Dict[str, Pipeline]): Словарь пайплайнов предобработки
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация предобработчика данных.
        
        Args:
            config: Конфигурация предобработки данных
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.config = config or {}
        self.transformers = {}
        self.pipelines = {}
        
        self.logger.info("Инициализирован предобработчик данных")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
           y: Optional[Union[np.ndarray, pd.Series]] = None) -> 'DataPreprocessor':
        """
        Обучение предобработчика данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная (опционально)
            
        Returns:
            DataPreprocessor: Обученный предобработчик данных
        """
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Обучение трансформеров
        for transformer_name, transformer in self.transformers.items():
            if hasattr(transformer, 'fit'):
                if transformer_name.startswith('target_'):
                    if y is not None:
                        transformer.fit(y.reshape(-1, 1) if isinstance(y, np.ndarray) else y)
                        self.logger.info(f"Обучен трансформер {transformer_name} для целевой переменной")
                else:
                    transformer.fit(X_df)
                    self.logger.info(f"Обучен трансформер {transformer_name}")
        
        # Обучение пайплайнов
        for pipeline_name, pipeline in self.pipelines.items():
            if hasattr(pipeline, 'fit'):
                if pipeline_name.startswith('target_'):
                    if y is not None:
                        pipeline.fit(y.reshape(-1, 1) if isinstance(y, np.ndarray) else y)
                        self.logger.info(f"Обучен пайплайн {pipeline_name} для целевой переменной")
                else:
                    pipeline.fit(X_df)
                    self.logger.info(f"Обучен пайплайн {pipeline_name}")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Преобразование данных.
        
        Args:
            X: Матрица признаков
            
        Returns:
            pd.DataFrame: Преобразованные данные
        """
        # Преобразование входных данных в DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Применение трансформеров
        result_df = X_df.copy()
        
        for transformer_name, transformer in self.transformers.items():
            if hasattr(transformer, 'transform') and not transformer_name.startswith('target_'):
                try:
                    result_df = transformer.transform(result_df)
                    self.logger.info(f"Применен трансформер {transformer_name}")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении трансформера {transformer_name}: {e}")
        
        # Применение пайплайнов
        for pipeline_name, pipeline in self.pipelines.items():
            if hasattr(pipeline, 'transform') and not pipeline_name.startswith('target_'):
                try:
                    result_df = pipeline.transform(result_df)
                    self.logger.info(f"Применен пайплайн {pipeline_name}")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении пайплайна {pipeline_name}: {e}")
        
        return result_df
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Optional[Union[np.ndarray, pd.Series]] = None) -> pd.DataFrame:
        """
        Обучение и преобразование данных.
        
        Args:
            X: Матрица признаков
            y: Целевая переменная (опционально)
            
        Returns:
            pd.DataFrame: Преобразованные данные
        """
        return self.fit(X, y).transform(X)
    
    def transform_target(self, y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Преобразование целевой переменной.
        
        Args:
            y: Целевая переменная
            
        Returns:
            Union[np.ndarray, pd.Series]: Преобразованная целевая переменная
        """
        # Преобразование входных данных
        if isinstance(y, np.ndarray):
            y_array = y.reshape(-1, 1)
        else:
            y_array = y.values.reshape(-1, 1)
        
        # Применение трансформеров для целевой переменной
        result = y_array.copy()
        
        for transformer_name, transformer in self.transformers.items():
            if hasattr(transformer, 'transform') and transformer_name.startswith('target_'):
                try:
                    result = transformer.transform(result)
                    self.logger.info(f"Применен трансформер {transformer_name} для целевой переменной")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении трансформера {transformer_name} для целевой переменной: {e}")
        
        # Применение пайплайнов для целевой переменной
        for pipeline_name, pipeline in self.pipelines.items():
            if hasattr(pipeline, 'transform') and pipeline_name.startswith('target_'):
                try:
                    result = pipeline.transform(result)
                    self.logger.info(f"Применен пайплайн {pipeline_name} для целевой переменной")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении пайплайна {pipeline_name} для целевой переменной: {e}")
        
        # Возвращение результата в исходном формате
        if isinstance(y, pd.Series):
            return pd.Series(result.ravel(), index=y.index)
        else:
            return result.ravel()
    
    def inverse_transform_target(self, y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Обратное преобразование целевой переменной.
        
        Args:
            y: Преобразованная целевая переменная
            
        Returns:
            Union[np.ndarray, pd.Series]: Исходная целевая переменная
        """
        # Преобразование входных данных
        if isinstance(y, np.ndarray):
            y_array = y.reshape(-1, 1)
        else:
            y_array = y.values.reshape(-1, 1)
        
        # Применение обратного преобразования для целевой переменной
        result = y_array.copy()
        
        # Применение обратного преобразования пайплайнов в обратном порядке
        for pipeline_name in reversed(list(self.pipelines.keys())):
            pipeline = self.pipelines[pipeline_name]
            if hasattr(pipeline, 'inverse_transform') and pipeline_name.startswith('target_'):
                try:
                    result = pipeline.inverse_transform(result)
                    self.logger.info(f"Применено обратное преобразование пайплайна {pipeline_name} для целевой переменной")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении обратного преобразования пайплайна {pipeline_name} для целевой переменной: {e}")
        
        # Применение обратного преобразования трансформеров в обратном порядке
        for transformer_name in reversed(list(self.transformers.keys())):
            transformer = self.transformers[transformer_name]
            if hasattr(transformer, 'inverse_transform') and transformer_name.startswith('target_'):
                try:
                    result = transformer.inverse_transform(result)
                    self.logger.info(f"Применено обратное преобразование трансформера {transformer_name} для целевой переменной")
                except Exception as e:
                    self.logger.error(f"Ошибка при применении обратного преобразования трансформера {transformer_name} для целевой переменной: {e}")
        
        # Возвращение результата в исходном формате
        if isinstance(y, pd.Series):
            return pd.Series(result.ravel(), index=y.index)
        else:
            return result.ravel()
    
    def add_transformer(self, name: str, transformer: Any) -> 'DataPreprocessor':
        """
        Добавление трансформера данных.
        
        Args:
            name: Имя трансформера
            transformer: Трансформер данных
            
        Returns:
            DataPreprocessor: Предобработчик данных
        """
        self.transformers[name] = transformer
        self.logger.info(f"Добавлен трансформер {name}")
        
        return self
    
    def add_pipeline(self, name: str, pipeline: Pipeline) -> 'DataPreprocessor':
        """
        Добавление пайплайна предобработки.
        
        Args:
            name: Имя пайплайна
            pipeline: Пайплайн предобработки
            
        Returns:
            DataPreprocessor: Предобработчик данных
        """
        self.pipelines[name] = pipeline
        self.logger.info(f"Добавлен пайплайн {name}")
        
        return self
    
    def remove_transformer(self, name: str) -> 'DataPreprocessor':
        """
        Удаление трансформера данных.
        
        Args:
            name: Имя трансформера
            
        Returns:
            DataPreprocessor: Предобработчик данных
            
        Raises:
            ValueError: Если трансформер с указанным именем не найден
        """
        if name not in self.transformers:
            raise ValueError(f"Трансформер с именем {name} не найден")
        
        del self.transformers[name]
        self.logger.info(f"Удален трансформер {name}")
        
        return self
    
    def remove_pipeline(self, name: str) -> 'DataPreprocessor':
        """
        Удаление пайплайна предобработки.
        
        Args:
            name: Имя пайплайна
            
        Returns:
            DataPreprocessor: Предобработчик данных
            
        Raises:
            ValueError: Если пайплайн с указанным именем не найден
        """
        if name not in self.pipelines:
            raise ValueError(f"Пайплайн с именем {name} не найден")
        
        del self.pipelines[name]
        self.logger.info(f"Удален пайплайн {name}")
        
        return self
    
    def add_scaler(self, name: str, scaler_type: str = 'standard', 
                 columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Добавление масштабирования признаков.
        
        Args:
            name: Имя трансформера
            scaler_type: Тип масштабирования ('standard', 'minmax', 'robust')
            columns: Список колонок для масштабирования (если None, масштабируются все)
            
        Returns:
            DataPreprocessor: Предобработчик данных
            
        Raises:
            ValueError: Если указан неизвестный тип масштабирования
        """
        class ColumnScaler(BaseEstimator, TransformerMixin):
            def __init__(self, scaler_type='standard', columns=None):
                self.scaler_type = scaler_type
                self.columns = columns
                
                if scaler_type == 'standard':
                    self.scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    self.scaler = MinMaxScaler()
                elif scaler_type == 'robust':
                    self.scaler = RobustScaler()
                else:
                    raise ValueError(f"Неизвестный тип масштабирования: {scaler_type}")
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Определение колонок для масштабирования
                if self.columns is None:
                    self.columns = X.select_dtypes(include=np.number).columns.tolist()
                else:
                    # Проверка наличия колонок
                    for col in self.columns:
                        if col not in X.columns:
                            raise ValueError(f"Колонка {col} не найдена в данных")
                
                # Обучение масштабирования
                if self.columns:
                    self.scaler.fit(X[self.columns])
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Копирование данных
                X_transformed = X.copy()
                
                # Применение масштабирования
                if self.columns:
                    X_transformed[self.columns] = self.scaler.transform(X[self.columns])
                
                return X_transformed
            
            def inverse_transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Копирование данных
                X_inverse = X.copy()
                
                # Применение обратного масштабирования
                if self.columns:
                    X_inverse[self.columns] = self.scaler.inverse_transform(X[self.columns])
                
                return X_inverse
        
        transformer = ColumnScaler(scaler_type=scaler_type, columns=columns)
        
        self.add_transformer(name, transformer)
        
        return self
    
    def add_encoder(self, name: str, encoder_type: str = 'onehot', 
                  columns: Optional[List[str]] = None,
                  drop: str = 'first') -> 'DataPreprocessor':
        """
        Добавление кодирования категориальных признаков.
        
        Args:
            name: Имя трансформера
            encoder_type: Тип кодирования ('onehot', 'label', 'ordinal')
            columns: Список колонок для кодирования (если None, кодируются все категориальные)
            drop: Стратегия удаления колонок при one-hot кодировании ('first', 'if_binary', None)
            
        Returns:
            DataPreprocessor: Предобработчик данных
            
        Raises:
            ValueError: Если указан неизвестный тип кодирования
        """
        class ColumnEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, encoder_type='onehot', columns=None, drop='first'):
                self.encoder_type = encoder_type
                self.columns = columns
                self.drop = drop
                self.encoders = {}
                self.encoded_columns = {}
                
                if encoder_type not in ['onehot', 'label', 'ordinal']:
                    raise ValueError(f"Неизвестный тип кодирования: {encoder_type}")
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Определение колонок для кодирования
                if self.columns is None:
                    self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
                else:
                    # Проверка наличия колонок
                    for col in self.columns:
                        if col not in X.columns:
                            raise ValueError(f"Колонка {col} не найдена в данных")
                
                # Обучение кодирования для каждой колонки
                for col in self.columns:
                    if self.encoder_type == 'onehot':
                        encoder = OneHotEncoder(drop=self.drop, sparse=False)
                        encoder.fit(X[[col]])
                        
                        # Сохранение имен закодированных колонок
                        if hasattr(encoder, 'get_feature_names_out'):
                            self.encoded_columns[col] = encoder.get_feature_names_out([col]).tolist()
                        else:
                            categories = encoder.categories_[0]
                            if self.drop == 'first':
                                categories = categories[1:]
                            self.encoded_columns[col] = [f"{col}_{cat}" for cat in categories]
                    
                    elif self.encoder_type == 'label':
                        encoder = LabelEncoder()
                        encoder.fit(X[col])
                        self.encoded_columns[col] = [col]
                    
                    elif self.encoder_type == 'ordinal':
                        encoder = OrdinalEncoder()
                        encoder.fit(X[[col]])
                        self.encoded_columns[col] = [col]
                    
                    self.encoders[col] = encoder
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Копирование данных
                X_transformed = X.copy()
                
                # Применение кодирования для каждой колонки
                for col in self.columns:
                    if col in X_transformed.columns:
                        if self.encoder_type == 'onehot':
                            # One-hot кодирование
                            encoded = self.encoders[col].transform(X_transformed[[col]])
                            
                            # Создание DataFrame с закодированными колонками
                            encoded_df = pd.DataFrame(
                                encoded, 
                                index=X_transformed.index, 
                                columns=self.encoded_columns[col]
                            )
                            
                            # Удаление исходной колонки и добавление закодированных
                            X_transformed = X_transformed.drop(col, axis=1)
                            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                        
                        elif self.encoder_type == 'label':
                            # Label кодирование
                            X_transformed[col] = self.encoders[col].transform(X_transformed[col])
                        
                        elif self.encoder_type == 'ordinal':
                            # Ordinal кодирование
                            X_transformed[col] = self.encoders[col].transform(X_transformed[[col]]).ravel()
                
                return X_transformed
            
            def inverse_transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Копирование данных
                X_inverse = X.copy()
                
                # Применение обратного кодирования для каждой колонки
                for col in self.columns:
                    if self.encoder_type == 'onehot':
                        # Проверка наличия всех закодированных колонок
                        if all(encoded_col in X_inverse.columns for encoded_col in self.encoded_columns[col]):
                            # Извлечение закодированных значений
                            encoded_values = X_inverse[self.encoded_columns[col]].values
                            
                            # Обратное one-hot кодирование
                            decoded = self.encoders[col].inverse_transform(encoded_values)
                            
                            # Удаление закодированных колонок и добавление исходной
                            X_inverse = X_inverse.drop(self.encoded_columns[col], axis=1)
                            X_inverse[col] = decoded.ravel()
                    
                    elif self.encoder_type == 'label':
                        if col in X_inverse.columns:
                            # Обратное label кодирование
                            X_inverse[col] = self.encoders[col].inverse_transform(X_inverse[col].astype(int))
                    
                    elif self.encoder_type == 'ordinal':
                        if col in X_inverse.columns:
                            # Обратное ordinal кодирование
                            X_inverse[col] = self.encoders[col].inverse_transform(X_inverse[[col]])
                
                return X_inverse
        
        transformer = ColumnEncoder(encoder_type=encoder_type, columns=columns, drop=drop)
        
        self.add_transformer(name, transformer)
        
        return self
    
    def add_imputer(self, name: str, imputer_type: str = 'simple', 
                  strategy: str = 'mean',
                  columns: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Добавление заполнения пропущенных значений.
        
        Args:
            name: Имя трансформера
            imputer_type: Тип заполнения ('simple', 'knn')
            strategy: Стратегия заполнения для SimpleImputer ('mean', 'median', 'most_frequent', 'constant')
            columns: Список колонок для заполнения (если None, заполняются все с пропусками)
            
        Returns:
            DataPreprocessor: Предобработчик данных
            
        Raises:
            ValueError: Если указан неизвестный тип заполнения
        """
        class ColumnImputer(BaseEstimator, TransformerMixin):
            def __init__(self, imputer_type='simple', strategy='mean', columns=None):
                self.imputer_type = imputer_type
                self.strategy = strategy
                self.columns = columns
                self.imputers = {}
                
                if imputer_type not in ['simple', 'knn']:
                    raise ValueError(f"Неизвестный тип заполнения: {imputer_type}")
            
            def fit(self, X, y=None):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Определение колонок для заполнения
                if self.columns is None:
                    self.columns = X.columns[X.isna().any()].tolist()
                else:
                    # Проверка наличия колонок
                    for col in self.columns:
                        if col not in X.columns:
                            raise ValueError(f"Колонка {col} не найдена в данных")
                
                # Группировка колонок по типу данных
                numeric_cols = [col for col in self.columns if pd.api.types.is_numeric_dtype(X[col])]
                categorical_cols = [col for col in self.columns if col not in numeric_cols]
                
                # Обучение заполнения для числовых колонок
                if numeric_cols:
                    if self.imputer_type == 'simple':
                        imputer = SimpleImputer(strategy=self.strategy)
                    else:  # 'knn'
                        imputer = KNNImputer()
                    
                    imputer.fit(X[numeric_cols])
                    self.imputers['numeric'] = (imputer, numeric_cols)
                
                # Обучение заполнения для категориальных колонок
                if categorical_cols:
                    imputer = SimpleImputer(strategy='most_frequent')
                    imputer.fit(X[categorical_cols])
                    self.imputers['categorical'] = (imputer, categorical_cols)
                
                return self
            
            def transform(self, X):
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)
                
                # Копирование данных
                X_transformed = X.copy()
                
                # Применение заполнения для каждой группы колонок
                for group, (imputer, cols) in self.imputers.items():
                    # Проверка наличия колонок
                    cols_present = [col for col in cols if col in X_transformed.columns]
                    
                    if cols_present:
                        X_transformed[cols_present] = imputer.transform(X_transformed[cols_present])
                
                return X_transformed
        
        transformer = ColumnImputer(imputer_type=imputer_type, strategy=strategy, columns=columns)
        
        self.add_transformer(name, transformer)
        
        return self
    
    def add_target_transformer(self, name: str, transformer: Any) -> 'DataPreprocessor':
        """
        Добавление трансформера для целевой переменной.
        
        Args:
            name: Имя трансформера
            transformer: Трансформер данных
            
        Returns:
            DataPreprocessor: Предобработчик данных
        """
        self.add_transformer(f"target_{name}", transformer)
        
        return self
    
    def save(self, path: str) -> str:
        """
        Сохранение предобработчика данных.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            str: Путь к сохраненному предобработчику данных
        """
        import pickle
        
        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение предобработчика данных
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Предобработчик данных сохранен в {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Загрузка предобработчика данных.
        
        Args:
            path: Путь к сохраненному предобработчику данных
            
        Returns:
            DataPreprocessor: Загруженный предобработчик данных
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        import pickle
        
        # Проверка существования файла
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")
        
        # Загрузка предобработчика данных
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        preprocessor.logger.info(f"Предобработчик данных загружен из {path}")
        
        return preprocessor 