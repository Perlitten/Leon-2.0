"""
Модуль ансамблевых моделей машинного обучения.

Предоставляет реализацию ансамблевых моделей на основе базового класса.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import logging
from ml.models.base_model import BaseModel


class EnsembleModel(BaseModel):
    """
    Класс для ансамблевых моделей машинного обучения.
    
    Объединяет несколько моделей для улучшения качества предсказаний.
    Поддерживает различные методы агрегации результатов: голосование, 
    усреднение, взвешенное усреднение и пользовательские функции агрегации.
    """
    
    AGGREGATION_METHODS = {
        'voting': lambda predictions, weights=None: np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), weights=weights).argmax(), 
            axis=0, 
            arr=predictions
        ),
        'mean': lambda predictions, weights=None: np.average(predictions, axis=0, weights=weights),
        'median': lambda predictions, weights=None: np.median(predictions, axis=0),
        'max': lambda predictions, weights=None: np.max(predictions, axis=0),
        'min': lambda predictions, weights=None: np.min(predictions, axis=0)
    }
    
    def __init__(self, name: str, models: List[BaseModel], 
                 aggregation_method: str = 'mean', 
                 weights: Optional[List[float]] = None,
                 custom_aggregation_func: Optional[Callable] = None,
                 version: str = "1.0.0", **kwargs):
        """
        Инициализация ансамблевой модели.
        
        Args:
            name: Название модели
            models: Список моделей для ансамбля
            aggregation_method: Метод агрегации результатов ('voting', 'mean', 'median', 'max', 'min')
            weights: Веса моделей для взвешенной агрегации
            custom_aggregation_func: Пользовательская функция агрегации
            version: Версия модели
            **kwargs: Дополнительные параметры модели
        
        Raises:
            ValueError: Если указан неподдерживаемый метод агрегации
            ValueError: Если количество весов не соответствует количеству моделей
        """
        super().__init__(name=name, version=version, **kwargs)
        
        self.models = models
        self.model_count = len(models)
        
        if self.model_count == 0:
            raise ValueError("Список моделей не может быть пустым")
        
        # Проверка и установка метода агрегации
        if custom_aggregation_func is not None:
            self.aggregation_func = custom_aggregation_func
            self.aggregation_method = 'custom'
        else:
            if aggregation_method not in self.AGGREGATION_METHODS:
                raise ValueError(f"Неподдерживаемый метод агрегации: {aggregation_method}. "
                                f"Доступные методы: {', '.join(self.AGGREGATION_METHODS.keys())}")
            self.aggregation_method = aggregation_method
            self.aggregation_func = self.AGGREGATION_METHODS[aggregation_method]
        
        # Проверка и установка весов
        if weights is not None:
            if len(weights) != self.model_count:
                raise ValueError(f"Количество весов ({len(weights)}) не соответствует "
                                f"количеству моделей ({self.model_count})")
            self.weights = np.array(weights) / np.sum(weights)  # Нормализация весов
        else:
            self.weights = None
        
        self.logger.info(f"Создана ансамблевая модель с {self.model_count} моделями и методом агрегации '{self.aggregation_method}'")
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], 
             y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """
        Обучение всех моделей ансамбля на данных.
        
        Args:
            X: Признаки для обучения
            y: Целевые значения
            **kwargs: Дополнительные параметры обучения
            
        Returns:
            Dict[str, Any]: Метрики обучения
        """
        self.logger.info(f"Начало обучения ансамблевой модели {self.name}")
        
        all_metrics = {}
        
        # Обучение каждой модели
        for i, model in enumerate(self.models):
            self.logger.info(f"Обучение модели {i+1}/{self.model_count}: {model.name}")
            metrics = model.train(X, y, **kwargs)
            all_metrics[f"model_{i+1}"] = metrics
        
        self.is_trained = True
        
        # Оценка ансамбля на обучающих данных
        ensemble_metrics = self.evaluate(X, y)
        all_metrics["ensemble"] = ensemble_metrics
        
        self.logger.info(f"Ансамблевая модель {self.name} обучена. Метрики ансамбля: {ensemble_metrics}")
        return all_metrics
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение предсказаний ансамблевой модели.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            RuntimeError: Если хотя бы одна модель не обучена
        """
        if not self.is_trained:
            raise RuntimeError("Ансамблевая модель не обучена. Сначала вызовите метод train().")
        
        # Получение предсказаний от каждой модели
        predictions = []
        for i, model in enumerate(self.models):
            if not model.is_trained:
                raise RuntimeError(f"Модель {i+1} ({model.name}) не обучена")
            
            pred = model.predict(X)
            predictions.append(pred)
        
        # Преобразование списка предсказаний в массив numpy
        predictions_array = np.array(predictions)
        
        # Агрегация предсказаний
        aggregated_predictions = self.aggregation_func(predictions_array, self.weights)
        
        return aggregated_predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Получение вероятностей классов для классификационных моделей.
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            np.ndarray: Вероятности принадлежности к каждому классу
            
        Raises:
            RuntimeError: Если хотя бы одна модель не обучена
            AttributeError: Если хотя бы одна модель не поддерживает вероятностные предсказания
        """
        if not self.is_trained:
            raise RuntimeError("Ансамблевая модель не обучена. Сначала вызовите метод train().")
        
        # Проверка, что все модели поддерживают predict_proba
        for i, model in enumerate(self.models):
            if not hasattr(model, 'predict_proba'):
                raise AttributeError(f"Модель {i+1} ({model.name}) не поддерживает метод predict_proba")
        
        # Получение вероятностей от каждой модели
        probabilities = []
        for i, model in enumerate(self.models):
            if not model.is_trained:
                raise RuntimeError(f"Модель {i+1} ({model.name}) не обучена")
            
            prob = model.predict_proba(X)
            probabilities.append(prob)
        
        # Преобразование списка вероятностей в массив numpy
        probabilities_array = np.array(probabilities)
        
        # Агрегация вероятностей (всегда используем усреднение для вероятностей)
        aggregated_probabilities = np.average(probabilities_array, axis=0, weights=self.weights)
        
        return aggregated_probabilities
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Оценка производительности ансамблевой модели на данных.
        
        Args:
            X: Признаки для оценки
            y: Истинные значения
            
        Returns:
            Dict[str, float]: Метрики оценки
            
        Raises:
            RuntimeError: Если хотя бы одна модель не обучена
        """
        if not self.is_trained:
            raise RuntimeError("Ансамблевая модель не обучена. Сначала вызовите метод train().")
        
        # Получение предсказаний ансамбля
        predictions = self.predict(X)
        
        # Определение типа задачи по первой модели
        first_model = self.models[0]
        
        # Используем метод оценки первой модели для оценки ансамбля
        # Это позволяет использовать правильные метрики для конкретного типа задачи
        metrics = first_model._calculate_metrics(y, predictions) if hasattr(first_model, '_calculate_metrics') else {}
        
        # Если у первой модели нет метода _calculate_metrics, пытаемся использовать общий метод evaluate
        if not metrics:
            # Преобразование входных данных в numpy массивы, если необходимо
            X_eval = X.values if isinstance(X, pd.DataFrame) else X
            y_true = y.values if isinstance(y, pd.Series) else y
            
            # Создаем временную копию модели для оценки
            temp_model = first_model.__class__(name="temp_model")
            temp_model.model = first_model.model
            temp_model.is_trained = True
            if hasattr(first_model, 'classes_'):
                temp_model.classes_ = first_model.classes_
            
            # Заменяем метод predict, чтобы он возвращал предсказания ансамбля
            original_predict = temp_model.predict
            temp_model.predict = lambda x: predictions
            
            # Оцениваем с помощью метода evaluate первой модели
            metrics = temp_model.evaluate(X_eval, y_true)
            
            # Восстанавливаем оригинальный метод predict
            temp_model.predict = original_predict
        
        self.logger.info(f"Оценка ансамблевой модели {self.name}. Метрики: {metrics}")
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Получение важности признаков, усредненной по всем моделям.
        
        Returns:
            Optional[np.ndarray]: Массив важности признаков или None, если не поддерживается
        """
        if not self.is_trained:
            raise RuntimeError("Ансамблевая модель не обучена. Сначала вызовите метод train().")
        
        # Получение важности признаков от каждой модели
        importances = []
        for model in self.models:
            try:
                importance = model.get_feature_importance()
                if importance is not None:
                    importances.append(importance)
            except (NotImplementedError, RuntimeError, AttributeError):
                continue
        
        if not importances:
            self.logger.warning("Ни одна из моделей не поддерживает получение важности признаков")
            return None
        
        # Преобразование списка важностей в массив numpy
        importances_array = np.array(importances)
        
        # Усреднение важностей
        avg_importance = np.mean(importances_array, axis=0)
        
        return avg_importance
    
    def save(self, path: str) -> str:
        """
        Сохранение ансамблевой модели и всех входящих в нее моделей.
        
        Args:
            path: Путь для сохранения модели
            
        Returns:
            str: Полный путь к сохраненной модели
        """
        import os
        
        # Создание директории для сохранения моделей ансамбля
        ensemble_dir = os.path.splitext(path)[0] + "_ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Сохранение каждой модели
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = os.path.join(ensemble_dir, f"model_{i+1}.pkl")
            saved_path = model.save(model_path)
            model_paths.append(saved_path)
        
        # Сохранение информации о моделях в метаданных
        self.metadata["model_paths"] = model_paths
        self.metadata["aggregation_method"] = self.aggregation_method
        if self.weights is not None:
            self.metadata["weights"] = self.weights.tolist()
        
        # Сохранение базовой информации о модели
        return super().save(path)
    
    def load(self, path: str) -> bool:
        """
        Загрузка ансамблевой модели и всех входящих в нее моделей.
        
        Args:
            path: Путь к сохраненной модели
            
        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        import os
        
        # Загрузка базовой информации о модели
        if not super().load(path):
            return False
        
        # Проверка наличия информации о моделях в метаданных
        if "model_paths" not in self.metadata:
            self.logger.error("В метаданных отсутствует информация о путях к моделям")
            return False
        
        # Загрузка каждой модели
        model_paths = self.metadata["model_paths"]
        self.models = []
        
        for model_path in model_paths:
            # Определение типа модели по метаданным
            model_metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
            if not os.path.exists(model_metadata_path):
                self.logger.error(f"Файл метаданных модели не найден: {model_metadata_path}")
                return False
            
            import json
            with open(model_metadata_path, 'r') as f:
                model_metadata = json.load(f)
            
            # Создание экземпляра модели соответствующего типа
            model_type = model_metadata.get("type", "")
            model_name = model_metadata.get("name", "")
            
            # Импорт соответствующего класса модели
            if "Regression" in model_type:
                from ml.models.regression_model import RegressionModel
                model = RegressionModel(name=model_name)
            elif "Classification" in model_type:
                from ml.models.classification_model import ClassificationModel
                model = ClassificationModel(name=model_name)
            else:
                self.logger.error(f"Неизвестный тип модели: {model_type}")
                return False
            
            # Загрузка модели
            if not model.load(model_path):
                self.logger.error(f"Не удалось загрузить модель: {model_path}")
                return False
            
            self.models.append(model)
        
        # Установка метода агрегации
        self.aggregation_method = self.metadata.get("aggregation_method", "mean")
        if self.aggregation_method in self.AGGREGATION_METHODS:
            self.aggregation_func = self.AGGREGATION_METHODS[self.aggregation_method]
        else:
            self.logger.warning(f"Неизвестный метод агрегации: {self.aggregation_method}. Используется 'mean'")
            self.aggregation_method = "mean"
            self.aggregation_func = self.AGGREGATION_METHODS["mean"]
        
        # Установка весов
        if "weights" in self.metadata:
            self.weights = np.array(self.metadata["weights"])
        else:
            self.weights = None
        
        self.model_count = len(self.models)
        self.is_trained = all(model.is_trained for model in self.models)
        
        self.logger.info(f"Ансамблевая модель загружена: {self.model_count} моделей, метод агрегации: {self.aggregation_method}")
        return True 