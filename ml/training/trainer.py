"""
Модуль для обучения моделей машинного обучения.

Предоставляет класс для обучения и валидации моделей на исторических данных.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import json
from datetime import datetime
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

class ModelTrainer:
    """
    Класс для обучения моделей машинного обучения.
    
    Отвечает за:
    - Подготовку данных для обучения
    - Обучение моделей
    - Валидацию моделей
    - Сохранение обученных моделей
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация тренера моделей.
        
        Args:
            config: Конфигурация тренера моделей
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Параметры по умолчанию
        self.models_dir = self.config.get("models_dir", "ml/models")
        self.task_type = self.config.get("task_type", "classification")  # classification или regression
        self.test_size = self.config.get("test_size", 0.2)
        self.random_state = self.config.get("random_state", 42)
        self.batch_size = self.config.get("batch_size", 32)
        self.epochs = self.config.get("epochs", 50)
        self.early_stopping = self.config.get("early_stopping", True)
        self.patience = self.config.get("patience", 10)
        
        # Создание директории для моделей, если она не существует
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train(self, model_type: str, X: np.ndarray, y: np.ndarray, 
             model_params: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Обучение модели на данных.
        
        Args:
            model_type: Тип модели (lstm, cnn, mlp, xgboost, random_forest)
            X: Признаки для обучения
            y: Целевые значения
            model_params: Параметры модели
            metadata: Метаданные модели
            
        Returns:
            Словарь с результатами обучения и путем к сохраненной модели
        """
        try:
            # Подготовка данных
            X_train, X_val, y_train, y_val = self._prepare_data(X, y)
            
            # Создание модели
            model = self._create_model(model_type, X_train.shape, model_params)
            
            # Обучение модели
            if model_type in ["lstm", "cnn", "mlp"]:
                history = self._train_tf_model(model, X_train, y_train, X_val, y_val)
                training_history = history.history
            else:  # xgboost, random_forest
                model.fit(X_train, y_train)
                training_history = None
            
            # Оценка модели
            metrics = self._evaluate_model(model, X_val, y_val)
            
            # Создание метаданных
            model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_metadata = self._create_metadata(model_id, model_type, metrics, training_history, metadata)
            
            # Сохранение модели
            model_path = self._save_model(model, model_id, model_metadata)
            
            # Результаты обучения
            results = {
                "model_id": model_id,
                "model_path": model_path,
                "metrics": metrics,
                "metadata": model_metadata
            }
            
            self.logger.info(f"Обучение модели {model_id} завершено. Метрики: {metrics}")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели {model_type}: {e}")
            return {"error": str(e)}
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения.
        
        Args:
            X: Признаки для обучения
            y: Целевые значения
            
        Returns:
            Кортеж (X_train, X_val, y_train, y_val)
        """
        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        return X_train, X_val, y_train, y_val
    
    def _create_model(self, model_type: str, input_shape: Tuple[int, ...], 
                     model_params: Dict[str, Any] = None) -> Any:
        """
        Создание модели машинного обучения.
        
        Args:
            model_type: Тип модели (lstm, cnn, mlp, xgboost, random_forest)
            input_shape: Форма входных данных
            model_params: Параметры модели
            
        Returns:
            Модель машинного обучения
        """
        model_params = model_params or {}
        
        if model_type == "lstm":
            return self._create_lstm_model(input_shape, model_params)
        elif model_type == "cnn":
            return self._create_cnn_model(input_shape, model_params)
        elif model_type == "mlp":
            return self._create_mlp_model(input_shape, model_params)
        elif model_type == "xgboost":
            return self._create_xgboost_model(model_params)
        elif model_type == "random_forest":
            return self._create_random_forest_model(model_params)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    def _create_lstm_model(self, input_shape: Tuple[int, ...], model_params: Dict[str, Any]) -> tf.keras.Model:
        """
        Создание LSTM модели.
        
        Args:
            input_shape: Форма входных данных
            model_params: Параметры модели
            
        Returns:
            LSTM модель
        """
        # Параметры модели
        units = model_params.get("units", [64, 32])
        dropout = model_params.get("dropout", 0.2)
        learning_rate = model_params.get("learning_rate", 0.001)
        
        # Создание модели
        model = tf.keras.Sequential()
        
        # Добавление LSTM слоев
        model.add(tf.keras.layers.LSTM(units[0], return_sequences=len(units) > 1, 
                                      input_shape=(input_shape[1], input_shape[2])))
        model.add(tf.keras.layers.Dropout(dropout))
        
        for i in range(1, len(units) - 1):
            model.add(tf.keras.layers.LSTM(units[i], return_sequences=True))
            model.add(tf.keras.layers.Dropout(dropout))
        
        if len(units) > 1:
            model.add(tf.keras.layers.LSTM(units[-1]))
            model.add(tf.keras.layers.Dropout(dropout))
        
        # Выходной слой
        if self.task_type == "classification":
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:  # regression
            model.add(tf.keras.layers.Dense(1))
            loss = "mse"
            metrics = ["mae"]
        
        # Компиляция модели
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _create_cnn_model(self, input_shape: Tuple[int, ...], model_params: Dict[str, Any]) -> tf.keras.Model:
        """
        Создание CNN модели.
        
        Args:
            input_shape: Форма входных данных
            model_params: Параметры модели
            
        Returns:
            CNN модель
        """
        # Параметры модели
        filters = model_params.get("filters", [64, 128, 128])
        kernel_size = model_params.get("kernel_size", 3)
        pool_size = model_params.get("pool_size", 2)
        dense_units = model_params.get("dense_units", [64, 32])
        dropout = model_params.get("dropout", 0.2)
        learning_rate = model_params.get("learning_rate", 0.001)
        
        # Создание модели
        model = tf.keras.Sequential()
        
        # Добавление сверточных слоев
        model.add(tf.keras.layers.Conv1D(filters[0], kernel_size, activation="relu", 
                                        input_shape=(input_shape[1], input_shape[2])))
        model.add(tf.keras.layers.MaxPooling1D(pool_size))
        
        for f in filters[1:]:
            model.add(tf.keras.layers.Conv1D(f, kernel_size, activation="relu"))
            model.add(tf.keras.layers.MaxPooling1D(pool_size))
        
        # Преобразование в плоский вектор
        model.add(tf.keras.layers.Flatten())
        
        # Добавление полносвязных слоев
        for units in dense_units:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout))
        
        # Выходной слой
        if self.task_type == "classification":
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:  # regression
            model.add(tf.keras.layers.Dense(1))
            loss = "mse"
            metrics = ["mae"]
        
        # Компиляция модели
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _create_mlp_model(self, input_shape: Tuple[int, ...], model_params: Dict[str, Any]) -> tf.keras.Model:
        """
        Создание MLP модели.
        
        Args:
            input_shape: Форма входных данных
            model_params: Параметры модели
            
        Returns:
            MLP модель
        """
        # Параметры модели
        units = model_params.get("units", [128, 64, 32])
        dropout = model_params.get("dropout", 0.2)
        learning_rate = model_params.get("learning_rate", 0.001)
        
        # Создание модели
        model = tf.keras.Sequential()
        
        # Преобразование входных данных
        model.add(tf.keras.layers.Flatten(input_shape=(input_shape[1], input_shape[2])))
        
        # Добавление полносвязных слоев
        for units_i in units:
            model.add(tf.keras.layers.Dense(units_i, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout))
        
        # Выходной слой
        if self.task_type == "classification":
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:  # regression
            model.add(tf.keras.layers.Dense(1))
            loss = "mse"
            metrics = ["mae"]
        
        # Компиляция модели
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def _create_xgboost_model(self, model_params: Dict[str, Any]) -> Any:
        """
        Создание XGBoost модели.
        
        Args:
            model_params: Параметры модели
            
        Returns:
            XGBoost модель
        """
        try:
            import xgboost as xgb
            
            # Параметры модели
            params = {
                "n_estimators": model_params.get("n_estimators", 100),
                "max_depth": model_params.get("max_depth", 6),
                "learning_rate": model_params.get("learning_rate", 0.1),
                "subsample": model_params.get("subsample", 0.8),
                "colsample_bytree": model_params.get("colsample_bytree", 0.8),
                "random_state": self.random_state
            }
            
            # Создание модели
            if self.task_type == "classification":
                model = xgb.XGBClassifier(**params)
            else:  # regression
                model = xgb.XGBRegressor(**params)
            
            return model
        
        except ImportError:
            self.logger.error("Библиотека XGBoost не установлена")
            raise ImportError("Для использования XGBoost необходимо установить библиотеку: pip install xgboost")
    
    def _create_random_forest_model(self, model_params: Dict[str, Any]) -> Any:
        """
        Создание Random Forest модели.
        
        Args:
            model_params: Параметры модели
            
        Returns:
            Random Forest модель
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Параметры модели
        params = {
            "n_estimators": model_params.get("n_estimators", 100),
            "max_depth": model_params.get("max_depth", None),
            "min_samples_split": model_params.get("min_samples_split", 2),
            "min_samples_leaf": model_params.get("min_samples_leaf", 1),
            "random_state": self.random_state
        }
        
        # Создание модели
        if self.task_type == "classification":
            model = RandomForestClassifier(**params)
        else:  # regression
            model = RandomForestRegressor(**params)
        
        return model
    
    def _train_tf_model(self, model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
        """
        Обучение TensorFlow модели.
        
        Args:
            model: TensorFlow модель
            X_train: Обучающие признаки
            y_train: Обучающие целевые значения
            X_val: Валидационные признаки
            y_val: Валидационные целевые значения
            
        Returns:
            История обучения
        """
        # Подготовка колбэков
        callbacks = []
        
        if self.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели на валидационных данных.
        
        Args:
            model: Модель машинного обучения
            X_val: Валидационные признаки
            y_val: Валидационные целевые значения
            
        Returns:
            Словарь с метриками производительности
        """
        # Получение предсказаний
        if isinstance(model, tf.keras.Model):
            y_pred = model.predict(X_val)
        else:
            y_pred = model.predict(X_val)
        
        # Расчет метрик в зависимости от типа задачи
        if self.task_type == "classification":
            # Преобразование вероятностей в классы
            if isinstance(model, tf.keras.Model):
                y_pred_class = (y_pred > 0.5).astype(int)
            else:
                try:
                    y_pred_class = model.predict_proba(X_val)[:, 1]
                    y_pred_class = (y_pred_class > 0.5).astype(int)
                except:
                    y_pred_class = y_pred
            
            # Расчет метрик
            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred_class)),
                "precision": float(precision_score(y_val, y_pred_class, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_val, y_pred_class, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_val, y_pred_class, average="weighted", zero_division=0))
            }
        else:  # regression
            # Расчет метрик
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred)
            mae = np.mean(np.abs(y_val - y_pred))
            
            metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2),
                "mae": float(mae)
            }
        
        return metrics
    
    def _create_metadata(self, model_id: str, model_type: str, metrics: Dict[str, float], 
                        training_history: Dict[str, List[float]] = None, 
                        additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Создание метаданных модели.
        
        Args:
            model_id: Идентификатор модели
            model_type: Тип модели
            metrics: Метрики производительности
            training_history: История обучения
            additional_metadata: Дополнительные метаданные
            
        Returns:
            Словарь с метаданными
        """
        # Создание метаданных
        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "task_type": self.task_type,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        # Добавление истории обучения
        if training_history:
            # Преобразование массивов numpy в списки
            history_dict = {}
            for key, value in training_history.items():
                history_dict[key] = [float(v) for v in value]
            
            metadata["training_history"] = history_dict
        
        # Добавление дополнительных метаданных
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Сохранение модели и метаданных.
        
        Args:
            model: Модель машинного обучения
            model_id: Идентификатор модели
            metadata: Метаданные модели
            
        Returns:
            Путь к сохраненной модели
        """
        # Создание директории для модели
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохранение модели
        if isinstance(model, tf.keras.Model):
            model_path = os.path.join(model_dir, "model.h5")
            model.save(model_path)
        else:
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        
        # Сохранение метаданных
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        return model_path
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновление конфигурации тренера моделей.
        
        Args:
            config: Новая конфигурация
        """
        self.config.update(config)
        
        # Обновление параметров
        self.models_dir = self.config.get("models_dir", self.models_dir)
        self.task_type = self.config.get("task_type", self.task_type)
        self.test_size = self.config.get("test_size", self.test_size)
        self.random_state = self.config.get("random_state", self.random_state)
        self.batch_size = self.config.get("batch_size", self.batch_size)
        self.epochs = self.config.get("epochs", self.epochs)
        self.early_stopping = self.config.get("early_stopping", self.early_stopping)
        self.patience = self.config.get("patience", self.patience)
        
        # Создание директории для моделей, если она не существует
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.logger.info("Конфигурация тренера моделей обновлена") 